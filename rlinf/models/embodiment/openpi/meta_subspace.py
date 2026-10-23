# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native inference support for LIBERO-Plus Meta-Subspace checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

META_CHECKPOINT_METHOD = "libero-plus-support-conditioned-demix-mask"
SITE_TO_LAYER_INDEX = {
    "pre_joint_layer_9": 8,
    "pre_joint_layer_12": 11,
    "pre_joint_layer_15": 14,
}
COMPONENT_MODES = ("full", "demix_only", "mask_only")
ACTION_MODES = ("all_meta", "meta_action6_base_gripper")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SupportTaskEncoderConfig:
    feature_dim: int
    hidden_dim: int
    task_code_dim: int


@dataclass(frozen=True)
class DemixMaskConfig:
    site: str
    hidden_dim: int
    rank: int
    task_code_dim: int
    conditioner_hidden_dim: int
    rms_norm_eps: float = 1e-6
    capture_regularization: bool = False


class SupportTaskEncoder(nn.Module):
    """Permutation-invariant encoder copied from the PCD training contract."""

    def __init__(self, config: SupportTaskEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.episode_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.task_code_dim),
        )
        self.output_norm = nn.LayerNorm(config.task_code_dim)

    def forward(
        self,
        support_features: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if support_features.ndim != 3:
            raise ValueError(
                "support_features must have shape [batch, support, feature]"
            )
        if support_features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"expected support feature size {self.config.feature_dim}, "
                f"got {support_features.shape[-1]}"
            )
        batch_size, support_size = support_features.shape[:2]
        if support_size == 0:
            raise ValueError("at least one support episode is required")
        if support_mask is None:
            support_mask = torch.ones(
                (batch_size, support_size),
                dtype=torch.bool,
                device=support_features.device,
            )
        else:
            expected = (batch_size, support_size)
            if tuple(support_mask.shape) != expected:
                raise ValueError(
                    f"support_mask must have shape {expected}, "
                    f"got {tuple(support_mask.shape)}"
                )
            support_mask = support_mask.to(
                device=support_features.device, dtype=torch.bool
            )
        counts = support_mask.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("every batch item needs at least one valid support")
        encoded = self.episode_encoder(self.input_norm(support_features))
        pooled = (encoded * support_mask[..., None]).sum(dim=1)
        pooled = pooled / counts.to(dtype=encoded.dtype)
        return self.output_norm(pooled)


class DemixMaskIntervention(nn.Module):
    """PCD Demix -> FiLM -> Mask intervention with per-sample bypass."""

    def __init__(self, config: DemixMaskConfig) -> None:
        super().__init__()
        if config.site not in SITE_TO_LAYER_INDEX:
            raise ValueError(f"unknown intervention site: {config.site}")
        self.config = config
        self.down_proj = nn.Linear(config.hidden_dim, config.rank, bias=False)
        self.up_proj = nn.Linear(config.rank, config.hidden_dim, bias=False)
        self.film = nn.Sequential(
            nn.Linear(config.task_code_dim, config.conditioner_hidden_dim),
            nn.GELU(),
            nn.Linear(config.conditioner_hidden_dim, 2 * config.rank),
        )
        self.mask = nn.Sequential(
            nn.Linear(config.task_code_dim, config.conditioner_hidden_dim),
            nn.GELU(),
            nn.Linear(config.conditioner_hidden_dim, config.hidden_dim),
        )
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        nn.init.zeros_(self.mask[-1].weight)
        nn.init.zeros_(self.mask[-1].bias)
        self.intervention_alpha = 1.0
        self.component_mode = "full"

    def configure(self, *, alpha: float, component_mode: str) -> None:
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("intervention alpha must be finite and in [0, 1]")
        if component_mode not in COMPONENT_MODES:
            raise ValueError(
                f"component_mode must be one of {COMPONENT_MODES}, "
                f"got {component_mode!r}"
            )
        self.intervention_alpha = alpha
        self.component_mode = component_mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        task_code: torch.Tensor,
        image_token_mask: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden]")
        batch_size, token_count, hidden_dim = hidden_states.shape
        if hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"expected hidden size {self.config.hidden_dim}, got {hidden_dim}"
            )
        if tuple(task_code.shape) != (batch_size, self.config.task_code_dim):
            raise ValueError(
                "task_code must match the hidden-state batch and configured "
                f"code dimension, got {tuple(task_code.shape)}"
            )
        if tuple(image_token_mask.shape) != (batch_size, token_count):
            raise ValueError(
                "image_token_mask must match hidden-state batch/token axes"
            )
        if sample_mask is None:
            sample_mask = torch.ones(
                batch_size, dtype=torch.bool, device=hidden_states.device
            )
        elif tuple(sample_mask.shape) != (batch_size,):
            raise ValueError("sample_mask must have shape [batch]")

        if self.intervention_alpha == 0.0 or not bool(sample_mask.any()):
            return hidden_states

        task_code = task_code.to(
            device=hidden_states.device, dtype=self.down_proj.weight.dtype
        )
        image_token_mask = image_token_mask.to(
            device=hidden_states.device, dtype=torch.bool
        )
        sample_mask = sample_mask.to(device=hidden_states.device, dtype=torch.bool)

        if self.component_mode == "mask_only":
            delta = torch.zeros_like(hidden_states, dtype=torch.float32)
        else:
            normalized = hidden_states.float()
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True)
                + self.config.rms_norm_eps
            )
            low_rank = self.down_proj(normalized)
            gamma, beta = self.film(task_code).chunk(2, dim=-1)
            conditioned = F.gelu(
                (1.0 + gamma[:, None, :]) * low_rank + beta[:, None, :]
            )
            delta = self.up_proj(conditioned)

        demixed = hidden_states.float() + delta
        if self.component_mode == "demix_only":
            channel_mask = torch.ones_like(hidden_states[:, 0], dtype=torch.float32)
        else:
            channel_mask = 2.0 * torch.sigmoid(self.mask(task_code))
        adapted = demixed * channel_mask[:, None, :]
        if self.intervention_alpha != 1.0:
            adapted = hidden_states.float() + self.intervention_alpha * (
                adapted - hidden_states.float()
            )

        selected = image_token_mask & sample_mask[:, None]
        return torch.where(
            selected[..., None],
            adapted.to(dtype=hidden_states.dtype),
            hidden_states,
        )


class SupportConditionedDemixMask(nn.Module):
    def __init__(
        self,
        encoder_config: SupportTaskEncoderConfig,
        intervention_config: DemixMaskConfig,
    ) -> None:
        super().__init__()
        self.task_encoder = SupportTaskEncoder(encoder_config)
        self.intervention = DemixMaskIntervention(intervention_config)
        if encoder_config.task_code_dim != intervention_config.task_code_dim:
            raise ValueError("task encoder and intervention code dimensions differ")

    def encode_support(self, support_features: torch.Tensor) -> torch.Tensor:
        return self.task_encoder(support_features.float())


@dataclass(frozen=True)
class TaskIdentity:
    suite: str
    task_id: int
    task_name: str
    base_task_id: str
    perturbation_category: str
    variant_id: str


class MetaSubspaceRuntime(nn.Module):
    """Own a frozen PCD adapter, deterministic support provider, and VLM hook."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        base_checkpoint_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.runtime_config = dict(config)
        self.enabled = bool(self.runtime_config.get("enabled", False))
        self.action_mode = str(self.runtime_config.get("action_mode", "all_meta"))
        if self.action_mode not in ACTION_MODES:
            raise ValueError(
                f"meta_subspace.action_mode must be one of {ACTION_MODES}, "
                f"got {self.action_mode!r}"
            )
        self._hook_handle = None
        self._active_context: dict[str, torch.Tensor] | None = None
        self._codes: dict[tuple[str, str], torch.Tensor | None] = {}
        self._code_details: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.enabled:
            self.adapter = None
            return

        checkpoint_path = Path(self._required("checkpoint_path")).expanduser().resolve()
        manifest_path = Path(self._required("manifest_path")).expanduser().resolve()
        feature_dir = Path(self._required("feature_dir")).expanduser().resolve()
        mapping_path = Path(self._required("task_mapping_path")).expanduser().resolve()
        for path in (checkpoint_path, manifest_path, feature_dir, mapping_path):
            if not path.exists():
                raise FileNotFoundError(f"Meta-Subspace input does not exist: {path}")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != 1
            or payload.get("method") != META_CHECKPOINT_METHOD
        ):
            raise ValueError(f"Unknown Meta-Subspace checkpoint: {checkpoint_path}")
        adapter_config = payload["adapter_config"]
        encoder_config = SupportTaskEncoderConfig(**adapter_config["encoder"])
        intervention_config = DemixMaskConfig(**adapter_config["intervention"])
        self.adapter = SupportConditionedDemixMask(encoder_config, intervention_config)
        if set(payload["adapter"]) != set(self.adapter.state_dict()):
            raise ValueError("Meta-Subspace checkpoint tensor keys do not match")
        self.adapter.load_state_dict(payload["adapter"], strict=True)
        self.adapter.requires_grad_(False)
        self.adapter.eval()
        self.adapter.intervention.configure(
            alpha=float(self.runtime_config.get("intervention_alpha", 1.0)),
            component_mode=str(self.runtime_config.get("component_mode", "full")),
        )

        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        self.payload = payload
        self.site = intervention_config.site
        self.layer_index = SITE_TO_LAYER_INDEX[self.site]
        self.task_code_dim = encoder_config.task_code_dim
        self.support_size = int(
            self.runtime_config.get("support_size", payload["config"]["support_size"])
        )
        self.support_seed = int(self.runtime_config.get("support_seed", 7))
        self.strict = bool(self.runtime_config.get("strict", True))
        if self.support_size <= 0:
            raise ValueError("support_size must be positive")

        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema_version") != 1
            or self.manifest.get("kind") != "libero-plus-meta-subspace-pool"
        ):
            raise ValueError("Unsupported Meta-Subspace manifest")
        self.rows = self.manifest["episodes"]

        self.feature_dir = feature_dir
        self.feature_index_path = feature_dir / "index.json"
        self.feature_index = json.loads(
            self.feature_index_path.read_text(encoding="utf-8")
        )
        if (
            self.feature_index.get("schema_version") != 1
            or self.feature_index.get("kind") != "libero-plus-mid-vlm-features"
        ):
            raise ValueError("Unsupported Meta-Subspace feature cache")
        if self.site not in self.feature_index["sites"]:
            raise ValueError(f"Feature cache does not contain {self.site}")
        if self.feature_index["feature_dim"] != encoder_config.feature_dim:
            raise ValueError("Feature cache and task encoder dimensions differ")

        if bool(self.runtime_config.get("verify_artifact_hashes", True)):
            metadata = payload.get("metadata", {})
            if metadata.get("manifest_sha256") != _sha256(manifest_path):
                raise ValueError("Training checkpoint/manifest SHA256 mismatch")
            if metadata.get("feature_index_sha256") != _sha256(self.feature_index_path):
                raise ValueError("Training checkpoint/feature-index SHA256 mismatch")
            if self.feature_index.get("manifest_sha256") != _sha256(manifest_path):
                raise ValueError("Feature cache/manifest SHA256 mismatch")

        self.mapping_sha256 = _sha256(mapping_path)
        if self.manifest.get("mapping_sha256") != self.mapping_sha256:
            raise ValueError("Task mapping/manifest SHA256 mismatch")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if mapping.get("schema_version") != 1:
            raise ValueError("Unsupported task-mapping schema")
        self.mapping_by_id: dict[tuple[str, int], TaskIdentity] = {}
        for row in mapping.get("tasks", []):
            identity = TaskIdentity(
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                task_name=str(row["task_name"]),
                base_task_id=str(row["base_task_id"]),
                perturbation_category=str(row["perturbation_category"]),
                variant_id=str(row["variant_id"]),
            )
            key = (identity.suite, identity.task_id)
            if key in self.mapping_by_id:
                raise ValueError(f"Duplicate task mapping: {key}")
            self.mapping_by_id[key] = identity

        if bool(self.runtime_config.get("verify_base_identity", True)):
            self._verify_base_identity(base_checkpoint_dir)

    @property
    def uses_base_gripper(self) -> bool:
        """Whether evaluation keeps the base policy's seventh action dimension."""
        return self.action_mode == "meta_action6_base_gripper"

    def combine_actions(
        self,
        meta_actions: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Combine model-space actions according to the evaluation protocol."""
        if not self.uses_base_gripper:
            return meta_actions
        if meta_actions.shape != base_actions.shape:
            raise ValueError("Meta and base action chunks must have equal shapes")
        if meta_actions.ndim < 1 or meta_actions.shape[-1] < 7:
            raise ValueError(
                "Base-gripper mixing requires at least seven action dimensions"
            )
        if meta_actions.device != base_actions.device:
            raise ValueError("Meta and base action chunks must be on the same device")

        combined = meta_actions.clone()
        combined[..., 6] = base_actions[..., 6].to(dtype=combined.dtype)
        return combined

    def _required(self, key: str) -> str:
        value = self.runtime_config.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"meta_subspace.{key} is required when enabled")
        return str(value)

    def _verify_base_identity(self, base_checkpoint_dir: str | Path | None) -> None:
        if base_checkpoint_dir is None:
            raise ValueError(
                "base checkpoint directory is required for identity verification"
            )
        root = Path(base_checkpoint_dir).expanduser().resolve()
        identity = self.payload["base_identity"]
        expected_config = str(identity.get("base_config", ""))
        configured = str(self.runtime_config.get("base_config", "pi0_libero"))
        if expected_config != configured:
            raise ValueError(
                f"Meta checkpoint expects {expected_config}, configured {configured}"
            )
        weights = root / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(
                "Base identity verification requires model.safetensors"
            )
        if _sha256(weights) != identity.get("weights_sha256"):
            raise ValueError("Meta checkpoint/base model weights SHA256 mismatch")
        for relative, expected_sha in identity.get("norm_stats", {}).items():
            path = root / relative
            if not path.is_file() or _sha256(path) != expected_sha:
                raise ValueError(
                    f"Meta checkpoint/base normalization mismatch: {relative}"
                )

    def install(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        layers = model.paligemma_with_expert.paligemma.language_model.layers
        if self.layer_index >= len(layers):
            raise ValueError(
                f"Model has {len(layers)} VLM layers, cannot install {self.site}"
            )

        def pre_hook(module, args, kwargs):
            del module
            if self._active_context is None:
                return None
            if args:
                hidden_states = args[0]
                adapted = self.adapter.intervention(
                    hidden_states, **self._active_context
                )
                return (adapted, *args[1:]), kwargs
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError(
                    "Cannot locate hidden_states in Gemma decoder layer call"
                )
            updated_kwargs = dict(kwargs)
            updated_kwargs["hidden_states"] = self.adapter.intervention(
                hidden_states, **self._active_context
            )
            return args, updated_kwargs

        self._hook_handle = layers[self.layer_index].register_forward_pre_hook(
            pre_hook, with_kwargs=True, prepend=True
        )

    @contextmanager
    def activate(self, context: dict[str, torch.Tensor] | None):
        if not self.enabled or context is None:
            yield
            return
        if self._active_context is not None:
            raise RuntimeError("Nested Meta-Subspace inference is not supported")
        self._active_context = context
        try:
            yield
        finally:
            self._active_context = None

    def _identity(self, suite: str, task_id: int, task_name: str) -> TaskIdentity:
        key = (suite, int(task_id))
        if key not in self.mapping_by_id:
            raise KeyError(f"Task is absent from Meta-Subspace mapping: {key}")
        identity = self.mapping_by_id[key]
        if task_name != identity.task_name:
            raise ValueError(
                f"Task-name mismatch for {key}: env={task_name!r}, "
                f"mapping={identity.task_name!r}"
            )
        return identity

    def _feature(self, row: Mapping[str, Any]) -> np.ndarray:
        episode_sha = str(row["sha256"])
        path = self.feature_dir / f"{episode_sha}.npz"
        expected_sha = self.feature_index.get("shards", {}).get(episode_sha)
        if expected_sha is None or not path.is_file():
            raise ValueError(f"Missing support feature shard: {path}")
        if bool(self.runtime_config.get("verify_artifact_hashes", True)):
            if _sha256(path) != expected_sha:
                raise ValueError(f"Modified support feature shard: {path}")
        with np.load(path, allow_pickle=False) as data:
            if str(data["episode_sha256"].item()) != episode_sha:
                raise ValueError("Feature shard/episode mismatch")
            features = np.asarray(data[self.site], dtype=np.float32)
        if (
            features.ndim != 2
            or not len(features)
            or features.shape[1] != self.adapter.task_encoder.config.feature_dim
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"Invalid support features: {path}")
        return features.mean(axis=0)

    @torch.no_grad()
    def _code_for(self, base_task_id: str, query_category: str) -> torch.Tensor | None:
        key = (base_task_id, query_category)
        if key in self._codes:
            return self._codes[key]
        candidates = [
            row
            for row in self.rows
            if row["split"] == "train"
            and row["metadata"].get("success") is True
            and row["metadata"]["base_task_id"] == base_task_id
            and row["metadata"]["perturbation_category"] != query_category
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{self.support_seed}:{key}:{row['path']}".encode()
            ).hexdigest()
        )
        supports = (
            candidates[: self.support_size]
            if len(candidates) >= self.support_size
            else []
        )
        details = {
            "base_task_id": base_task_id,
            "query_category": query_category,
            "support_seed": self.support_seed,
            "support_size": len(supports),
            "support_episodes": [row["path"] for row in supports],
            "identity_fallback": not bool(supports),
        }
        code = None
        if supports:
            values = np.stack([self._feature(row) for row in supports])[None]
            parameter = next(self.adapter.task_encoder.parameters())
            tensor = torch.as_tensor(
                values, device=parameter.device, dtype=torch.float32
            )
            code = self.adapter.encode_support(tensor).detach()
            details["task_code_sha256"] = hashlib.sha256(
                code.float().cpu().numpy().tobytes()
            ).hexdigest()
        else:
            details["fallback_reason"] = "insufficient_cross_category_training_support"
        self._codes[key] = code
        self._code_details[key] = details
        return code

    @torch.no_grad()
    def prepare_context(
        self,
        env_obs: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        if not self.enabled:
            return None
        required = (
            "meta_suite_names",
            "meta_task_ids",
            "meta_task_names",
            "meta_perturbation_categories",
        )
        missing = [key for key in required if key not in env_obs]
        if missing:
            raise KeyError(f"Meta-Subspace environment metadata is missing: {missing}")

        suites = [str(value) for value in env_obs["meta_suite_names"]]
        task_names = [str(value) for value in env_obs["meta_task_names"]]
        categories = [str(value) for value in env_obs["meta_perturbation_categories"]]
        task_ids_value = env_obs["meta_task_ids"]
        if torch.is_tensor(task_ids_value):
            task_ids = task_ids_value.detach().cpu().tolist()
        else:
            task_ids = list(task_ids_value)
        batch_size = len(task_ids)
        if not (len(suites) == len(task_names) == len(categories) == batch_size):
            raise ValueError("Meta-Subspace environment metadata batch mismatch")

        codes = []
        sample_mask = []
        for suite, task_id, task_name, category in zip(
            suites, task_ids, task_names, categories, strict=True
        ):
            try:
                identity = self._identity(suite, int(task_id), task_name)
                if category and category != identity.perturbation_category:
                    raise ValueError(
                        "Environment/task-mapping category mismatch: "
                        f"{category!r} != {identity.perturbation_category!r}"
                    )
                code = self._code_for(
                    identity.base_task_id,
                    identity.perturbation_category,
                )
            except (KeyError, ValueError):
                if self.strict:
                    raise
                code = None
            if code is None:
                if self.strict:
                    raise ValueError(
                        "No eligible cross-category training supports for "
                        f"base task {identity.base_task_id!r} and category "
                        f"{identity.perturbation_category!r}"
                    )
                codes.append(torch.zeros(self.task_code_dim, device=device))
                sample_mask.append(False)
            else:
                codes.append(code[0].to(device=device, dtype=torch.float32))
                sample_mask.append(True)

        return {
            "task_code": torch.stack(codes),
            "sample_mask": torch.tensor(sample_mask, device=device, dtype=torch.bool),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "checkpoint": str(getattr(self, "checkpoint_path", "")),
            "checkpoint_sha256": getattr(self, "checkpoint_sha256", ""),
            "site": getattr(self, "site", ""),
            "support_size": getattr(self, "support_size", 0),
            "support_seed": getattr(self, "support_seed", 0),
            "action_mode": self.action_mode,
            "cached_task_codes": len(self._codes),
            "task_codes": list(self._code_details.values()),
        }


def install_meta_subspace(
    model: nn.Module,
    config: Mapping[str, Any] | None,
    *,
    base_checkpoint_dir: str | Path | None = None,
) -> MetaSubspaceRuntime | None:
    runtime_config = dict(config or {})
    if not bool(runtime_config.get("enabled", False)):
        model.meta_subspace_runtime = None
        return None
    if str(runtime_config.get("runtime", "legacy")) == "residual_mask_factorial":
        from rlinf.models.embodiment.openpi.factorial_meta_subspace import (
            FactorialMetaSubspaceRuntime,
        )

        runtime = FactorialMetaSubspaceRuntime(
            runtime_config, base_checkpoint_dir=base_checkpoint_dir
        )
    else:
        runtime = MetaSubspaceRuntime(
            runtime_config, base_checkpoint_dir=base_checkpoint_dir
        )
    model.meta_subspace_runtime = runtime
    runtime.install(model)
    return runtime
