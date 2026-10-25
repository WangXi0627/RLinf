"""Native RLinf inference for PCD B/C1/C2/C3 factorial checkpoints.

This module intentionally mirrors the PCD inference contract.  In particular,
C2/C3 perform the same deterministic one-step Mask update from two successful
cross-category support episodes before an evaluation task is rolled out.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from torch import nn
from torch.utils._pytree import tree_map

from rlinf.utils.pytree import register_pytree_dataclasses


CHECKPOINT_METHOD = "libero-plus-task-residual-mask-factorial"
CHECKPOINT_PROTOCOL = "libero-plus-task-residual-mask-factorial-v1"
MASK_GUIDED_METHOD = "libero-plus-mask-guided-residual"
MASK_GUIDED_PROTOCOL = "libero-plus-mask-guided-residual-v1"
SITE_TO_LAYER_INDEX = {"pre_joint_layer_9": 8, "pre_joint_layer_12": 11, "pre_joint_layer_15": 14}
ACTION_MODES = ("all_meta", "meta_action6_base_gripper", "conservative_gate")
ARM_SPECS = {
    "b_residual": {"intervention_type": "task_residual_mlp", "inner_steps": 0, "second_order": False, "delta_condition": False},
    "c1_joint_mask": {"intervention_type": "task_residual_mask", "inner_steps": 0, "second_order": False, "delta_condition": False},
    "c2_bilevel": {"intervention_type": "task_residual_mask", "inner_steps": 1, "second_order": True, "delta_condition": False},
    "c3_full": {"intervention_type": "task_residual_mask_c3", "inner_steps": 1, "second_order": True, "delta_condition": True},
}
MASK_GUIDED_ARM_SPECS = {
    "importance_fo": {"intervention_type": "mask_guided_residual", "inner_steps": 1, "guide_signal": "importance", "second_order": False},
    "importance_so": {"intervention_type": "mask_guided_residual", "inner_steps": 1, "guide_signal": "importance", "second_order": True},
    "delta_fo": {"intervention_type": "mask_guided_residual", "inner_steps": 1, "guide_signal": "delta", "second_order": False},
    "delta_so": {"intervention_type": "mask_guided_residual", "inner_steps": 1, "guide_signal": "delta", "second_order": True},
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _private_seed(value: Any) -> int:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


@contextmanager
def _private_rng(device: torch.device, seed: int):
    device = torch.device(device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield


@dataclass(frozen=True)
class SupportTaskEncoderConfig:
    feature_dim: int
    hidden_dim: int
    task_code_dim: int


@dataclass(frozen=True)
class TaskResidualMaskConfig:
    intervention_type: str
    site: str
    hidden_dim: int
    task_code_dim: int
    conditioner_hidden_dim: int
    residual_hidden_dim: int
    guide_signal: str | None = None
    rms_norm_eps: float = 1e-6
    strength: float = 1.0
    capture_regularization: bool = False


class SupportTaskEncoder(nn.Module):
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

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        if support_features.ndim != 3 or support_features.shape[-1] != self.config.feature_dim:
            raise ValueError("Invalid support feature shape")
        if support_features.shape[1] == 0:
            raise ValueError("At least one support episode is required")
        encoded = self.episode_encoder(self.input_norm(support_features.float()))
        return self.output_norm(encoded.mean(dim=1))


class TaskResidualMask(nn.Module):
    def __init__(self, config: TaskResidualMaskConfig) -> None:
        super().__init__()
        self.config = config
        self.down_proj = nn.Linear(config.hidden_dim + config.task_code_dim, config.residual_hidden_dim)
        self.up_proj = nn.Linear(config.residual_hidden_dim, config.hidden_dim)
        if config.intervention_type != "task_residual_mlp":
            self.mask = nn.Sequential(
                nn.Linear(config.task_code_dim, config.conditioner_hidden_dim),
                nn.GELU(),
                nn.Linear(config.conditioner_hidden_dim, config.hidden_dim),
            )
            nn.init.zeros_(self.mask[-1].weight)
            nn.init.zeros_(self.mask[-1].bias)
        if config.intervention_type == "task_residual_mask_c3":
            self.delta_condition_proj = nn.Linear(config.hidden_dim, config.residual_hidden_dim, bias=False)
            nn.init.zeros_(self.delta_condition_proj.weight)
        if config.intervention_type == "mask_guided_residual":
            self.guide_condition_proj = nn.Linear(config.hidden_dim, config.residual_hidden_dim, bias=False)
            nn.init.zeros_(self.guide_condition_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        self._last_relative_residual_rms: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        task_code: torch.Tensor,
        image_token_mask: torch.Tensor,
        strength: float,
        mask_logits: torch.Tensor | None = None,
        delta_u: torch.Tensor | None = None,
        guide_condition: torch.Tensor | None = None,
        mask_importance: torch.Tensor | None = None,
        diagnostic_mask: bool = False,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, hidden = hidden_states.shape
        if hidden != self.config.hidden_dim:
            raise ValueError("Factorial hidden dimension mismatch")
        if tuple(task_code.shape) != (batch, self.config.task_code_dim):
            raise ValueError("Factorial task-code shape mismatch")
        if tuple(image_token_mask.shape) != (batch, tokens):
            raise ValueError("Factorial image-token mask shape mismatch")
        strength = float(strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("intervention_strength must be finite and in [0, 1]")
        if sample_mask is None:
            sample_mask = torch.ones(batch, device=hidden_states.device, dtype=torch.bool)
        if strength == 0.0 or not bool(sample_mask.any()):
            return hidden_states

        task_code = task_code.to(device=hidden_states.device, dtype=torch.float32)
        raw = hidden_states.float()
        normalized = raw * torch.rsqrt(
            raw.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        )
        code = task_code[:, None, :].expand(-1, tokens, -1)
        preactivation = self.down_proj(torch.cat((normalized, code), dim=-1))
        if delta_u is not None:
            if self.config.intervention_type != "task_residual_mask_c3":
                raise ValueError("delta_u is only valid for C3")
            device_type = hidden_states.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                projected = self.delta_condition_proj(
                    delta_u.to(device=hidden_states.device, dtype=torch.float32)
                )
            preactivation = preactivation + projected[:, None].to(preactivation.dtype)
        if guide_condition is not None:
            if self.config.intervention_type != "mask_guided_residual" or diagnostic_mask:
                raise ValueError("guide_condition is valid only on mask-guided Query forwards")
            if tuple(guide_condition.shape) != (batch, hidden):
                raise ValueError("Mask-guided condition shape mismatch")
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                projected = self.guide_condition_proj(
                    guide_condition.to(device=hidden_states.device, dtype=torch.float32)
                )
            preactivation = preactivation + projected[:, None].to(preactivation.dtype)
        residual = self.up_proj(F.gelu(preactivation)).float()

        if self.config.intervention_type == "mask_guided_residual" and not diagnostic_mask:
            if guide_condition is None or mask_importance is None:
                raise ValueError("Mask-guided Query requires guide_condition and mask_importance")
            if tuple(mask_importance.shape) != (batch, hidden):
                raise ValueError("Mask-guided importance shape mismatch")
            adapted = raw + strength * residual
            selected = image_token_mask.bool() & sample_mask[:, None].bool()
            counts = selected.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
            residual_power = (
                residual.float().square() * selected[..., None]
            ).sum(dim=(1, 2)) / (counts * hidden)
            raw_power = (
                raw.square() * selected[..., None]
            ).sum(dim=(1, 2)) / (counts * hidden)
            self._last_relative_residual_rms = (
                residual_power.sqrt() / raw_power.sqrt().clamp_min(1e-12)
            ).detach()
            return torch.where(selected[..., None], adapted.to(hidden_states.dtype), hidden_states)

        if self.config.intervention_type == "task_residual_mlp":
            if mask_logits is not None:
                raise ValueError("B has no Mask logits")
            channels = torch.ones((batch, hidden), device=raw.device, dtype=torch.float32)
        else:
            logits = self.mask(task_code) if mask_logits is None else mask_logits
            channels = 2.0 * torch.sigmoid(logits.float())
        adapted = channels[:, None, :] * (raw + residual)
        adapted = raw + strength * (adapted - raw)
        selected = image_token_mask.bool() & sample_mask[:, None].bool()
        return torch.where(selected[..., None], adapted.to(hidden_states.dtype), hidden_states)


class SupportConditionedResidualMask(nn.Module):
    def __init__(self, encoder: SupportTaskEncoderConfig, intervention: TaskResidualMaskConfig) -> None:
        super().__init__()
        self.task_encoder = SupportTaskEncoder(encoder)
        self.intervention = TaskResidualMask(intervention)

    def encode_support(self, features: torch.Tensor) -> torch.Tensor:
        return self.task_encoder(features)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        site: str,
        context: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if site != self.intervention.config.site:
            return hidden_states
        payload = dict(context or {})
        layer_index = payload.pop("layer_index", None)
        expected_layer = SITE_TO_LAYER_INDEX[site]
        if layer_index is not None and int(layer_index) != expected_layer:
            raise ValueError(
                f"Intervention site/layer mismatch: {site} != layer {layer_index}"
            )
        allowed = {
            "task_code",
            "image_token_mask",
            "strength",
            "mask_logits",
            "delta_u",
            "guide_condition",
            "mask_importance",
            "diagnostic_mask",
            "sample_mask",
        }
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise TypeError(f"Unexpected factorial intervention context: {unexpected}")
        return self.intervention(hidden_states, **payload)


@dataclass(frozen=True)
class TaskIdentity:
    suite: str
    task_id: int
    task_name: str
    base_task_id: str
    perturbation_category: str
    variant_id: str


class FactorialMetaSubspaceRuntime(nn.Module):
    """Frozen factorial adapter plus deterministic test-time Mask adaptation."""

    def __init__(self, config: Mapping[str, Any], *, base_checkpoint_dir: str | Path | None = None) -> None:
        super().__init__()
        self.runtime_config = dict(config)
        self.enabled = bool(self.runtime_config.get("enabled", False))
        self.action_mode = str(self.runtime_config.get("action_mode", "all_meta"))
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"Unknown factorial action_mode: {self.action_mode}")
        self._active_context: dict[str, torch.Tensor] | None = None
        self._hook_handle = None
        self._states: dict[tuple[str, str], dict[str, torch.Tensor] | None] = {}
        self._state_details: dict[tuple[str, str], dict[str, Any]] = {}
        self._model_ref: weakref.ReferenceType[nn.Module] | None = None
        if not self.enabled:
            self.adapter = None
            return

        checkpoint_path = Path(self._required("checkpoint_path")).expanduser().resolve()
        manifest_path = Path(self._required("manifest_path")).expanduser().resolve()
        feature_dir = Path(self._required("feature_dir")).expanduser().resolve()
        mapping_path = Path(self._required("task_mapping_path")).expanduser().resolve()
        for path in (checkpoint_path, manifest_path, feature_dir, mapping_path):
            if not path.exists():
                raise FileNotFoundError(f"Factorial evaluation input does not exist: {path}")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 2:
            raise ValueError(f"Unknown meta-subspace checkpoint: {checkpoint_path}")
        identity = (payload.get("method"), payload.get("protocol"))
        self.mask_guided = identity == (MASK_GUIDED_METHOD, MASK_GUIDED_PROTOCOL)
        if not self.mask_guided and identity != (CHECKPOINT_METHOD, CHECKPOINT_PROTOCOL):
            raise ValueError(f"Unknown meta-subspace checkpoint: {checkpoint_path}")
        arm = str(payload["config"]["arm"])
        specs = MASK_GUIDED_ARM_SPECS if self.mask_guided else ARM_SPECS
        if arm not in specs:
            raise ValueError(f"Unknown checkpoint arm: {arm}")
        spec = specs[arm]
        checked = ("inner_steps", "second_order")
        if self.mask_guided:
            checked += ("guide_signal",)
        else:
            checked += ("intervention_type",)
        for key in checked:
            actual = (
                payload["adapter_config"]["intervention"].get("intervention_type")
                if key == "intervention_type" else payload["config"].get(key)
            )
            if actual != spec[key]:
                raise ValueError(f"Checkpoint arm/config mismatch for {key}")

        adapter_config = payload["adapter_config"]
        encoder_config = SupportTaskEncoderConfig(**adapter_config["encoder"])
        intervention_config = TaskResidualMaskConfig(**adapter_config["intervention"])
        if intervention_config.intervention_type != spec["intervention_type"]:
            raise ValueError("Checkpoint intervention type does not match its arm")
        if self.mask_guided and intervention_config.guide_signal != spec["guide_signal"]:
            raise ValueError("Checkpoint guide signal does not match its arm")
        self.adapter = SupportConditionedResidualMask(encoder_config, intervention_config)
        if set(payload["adapter"]) != set(self.adapter.state_dict()):
            raise ValueError("Factorial checkpoint tensor keys do not match")
        self.adapter.load_state_dict(payload["adapter"], strict=True)
        self.adapter.requires_grad_(False)
        self.adapter.eval()

        self.payload = payload
        self.protocol = str(payload["protocol"])
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        self.arm = arm
        self.spec = spec
        self.site = intervention_config.site
        self.layer_index = SITE_TO_LAYER_INDEX[self.site]
        self.task_code_dim = encoder_config.task_code_dim
        self.hidden_dim = intervention_config.hidden_dim
        self.support_size = int(self.runtime_config.get("support_size", payload["config"]["support_size"]))
        self.support_seed = int(self.runtime_config.get("support_seed", 7))
        self.training_seed = int(payload["config"]["seed"])
        self.intervention_strength = float(self.runtime_config.get("intervention_strength", 0.5))
        self.strict = bool(self.runtime_config.get("strict", True))
        no_inner = self.runtime_config.get("no_inner_update", False)
        if isinstance(no_inner, str):
            normalized = no_inner.strip().lower()
            if normalized not in {"0", "1", "false", "true"}:
                raise ValueError("no_inner_update must be a boolean")
            no_inner = normalized in {"1", "true"}
        self.no_inner_update = bool(no_inner)
        if self.no_inner_update and not self.mask_guided:
            raise ValueError("no_inner_update is valid only for mask-guided checkpoints")
        self.inner_update_scale = float(
            self.runtime_config.get("inner_update_scale", 1.0)
        )
        self.support_gate_min_relative_improvement = float(
            self.runtime_config.get(
                "support_gate_min_relative_improvement", -1.0
            )
        )
        self.residual_gate_max_relative_rms = float(
            self.runtime_config.get("residual_gate_max_relative_rms", -1.0)
        )
        self.action_gate_max_mean_abs_delta = float(
            self.runtime_config.get("action_gate_max_mean_abs_delta", -1.0)
        )
        self.action_gate_horizon = int(
            self.runtime_config.get("action_gate_horizon", 5)
        )
        if not 0.0 <= self.inner_update_scale <= 1.0:
            raise ValueError("inner_update_scale must be in [0, 1]")
        if self.action_mode == "conservative_gate" and not self.mask_guided:
            raise ValueError("conservative_gate requires a mask-guided checkpoint")
        if self.action_mode == "conservative_gate" and (
            self.residual_gate_max_relative_rms < 0
            or self.action_gate_max_mean_abs_delta < 0
        ):
            raise ValueError(
                "conservative_gate requires residual/action gate thresholds"
            )
        if not 0 < self.action_gate_horizon <= int(payload["config"]["action_horizon"]):
            raise ValueError("Invalid action_gate_horizon")
        self._last_gate_metrics: dict[str, Any] = {}
        self.verify_hashes = bool(self.runtime_config.get("verify_artifact_hashes", True))
        self.inner_lr = float(payload["config"]["inner_lr"])
        self.flow_prefix_steps = int(payload["config"]["flow_prefix_steps"])
        self.mask_weight = float(payload["config"]["mask_weight"])
        if self.support_size != 2:
            raise ValueError("Strict factorial inference requires support_size=2")
        if not 0.0 <= self.intervention_strength <= 1.0:
            raise ValueError("intervention_strength must be in [0, 1]")

        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("kind") != "libero-plus-meta-subspace-pool":
            raise ValueError("Unsupported factorial manifest")
        self.rows = self.manifest["episodes"]
        self.episodes_root = Path(self.manifest["episodes_root"]).expanduser().resolve()

        self.feature_dir = feature_dir
        self.feature_index_path = feature_dir / "index.json"
        self.feature_index = json.loads(self.feature_index_path.read_text(encoding="utf-8"))
        if self.feature_index.get("schema_version") != 1 or self.feature_index.get("kind") != "libero-plus-mid-vlm-features":
            raise ValueError("Unsupported factorial feature cache")
        if self.site not in self.feature_index["sites"]:
            raise ValueError(f"Feature cache does not contain {self.site}")
        if self.feature_index["feature_dim"] != encoder_config.feature_dim:
            raise ValueError("Feature cache/task encoder dimension mismatch")
        if self.verify_hashes:
            metadata = payload.get("metadata", {})
            if metadata.get("manifest_sha256") != _sha256(manifest_path):
                raise ValueError("Checkpoint/manifest SHA256 mismatch")
            if metadata.get("feature_index_sha256") != _sha256(self.feature_index_path):
                raise ValueError("Checkpoint/feature-index SHA256 mismatch")
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
                suite=str(row["suite"]), task_id=int(row["task_id"]),
                task_name=str(row["task_name"]), base_task_id=str(row["base_task_id"]),
                perturbation_category=str(row["perturbation_category"]), variant_id=str(row["variant_id"]),
            )
            key = (identity.suite, identity.task_id)
            if key in self.mapping_by_id:
                raise ValueError(f"Duplicate task mapping: {key}")
            self.mapping_by_id[key] = identity
        if bool(self.runtime_config.get("verify_base_identity", True)):
            self._verify_base_identity(base_checkpoint_dir)

    @property
    def uses_base_gripper(self) -> bool:
        return self.action_mode == "meta_action6_base_gripper"

    @property
    def requires_base_actions(self) -> bool:
        return self.uses_base_gripper or self.action_mode == "conservative_gate"

    def reset_query_diagnostics(self) -> None:
        self._last_gate_metrics = {}
        if self.adapter is not None:
            self.adapter.intervention._last_relative_residual_rms = None

    def combine_actions(self, meta_actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
        if not self.requires_base_actions:
            return meta_actions
        if meta_actions.shape != base_actions.shape or meta_actions.shape[-1] < 7:
            raise ValueError("Invalid actions for base-policy comparison")
        if self.uses_base_gripper:
            combined = meta_actions.clone()
            combined[..., 6] = base_actions[..., 6].to(dtype=combined.dtype)
            return combined

        action_delta = (
            meta_actions[:, : self.action_gate_horizon, :7].float()
            - base_actions[:, : self.action_gate_horizon, :7].float()
        ).abs().mean(dim=(1, 2))
        residual_rms = self.adapter.intervention._last_relative_residual_rms
        if residual_rms is None or residual_rms.shape != action_delta.shape:
            accepted = torch.zeros_like(action_delta, dtype=torch.bool)
            residual_rms = torch.full_like(action_delta, float("inf"))
        else:
            accepted = (
                action_delta <= self.action_gate_max_mean_abs_delta
            ) & (
                residual_rms.to(action_delta.device)
                <= self.residual_gate_max_relative_rms
            )
        self._last_gate_metrics = {
            "accepted": accepted.detach().cpu().tolist(),
            "action_delta_mean_abs": action_delta.detach().cpu().tolist(),
            "relative_residual_rms": residual_rms.detach().cpu().tolist(),
            "action_threshold": self.action_gate_max_mean_abs_delta,
            "residual_threshold": self.residual_gate_max_relative_rms,
        }
        return torch.where(
            accepted[:, None, None],
            meta_actions,
            base_actions.to(dtype=meta_actions.dtype),
        )

    def _required(self, key: str) -> str:
        value = self.runtime_config.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"meta_subspace.{key} is required when enabled")
        return str(value)

    def _verify_base_identity(self, base_checkpoint_dir: str | Path | None) -> None:
        if base_checkpoint_dir is None:
            raise ValueError("Base checkpoint directory is required")
        root = Path(base_checkpoint_dir).expanduser().resolve()
        identity = self.payload["base_identity"]
        configured = str(self.runtime_config.get("base_config", "pi0_libero"))
        if identity.get("base_config") != configured:
            raise ValueError("Factorial checkpoint/base config mismatch")
        weights = root / "model.safetensors"
        if not weights.is_file() or _sha256(weights) != identity.get("weights_sha256"):
            raise ValueError("Factorial checkpoint/base weights SHA256 mismatch")
        for relative, expected in identity.get("norm_stats", {}).items():
            path = root / relative
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"Factorial checkpoint/base normalization mismatch: {relative}")

    @property
    def model(self) -> nn.Module:
        if self._model_ref is None:
            raise RuntimeError("Factorial runtime is not installed on a model")
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Factorial runtime model has been released")
        return model

    def _bind_model(self, model: nn.Module) -> None:
        # The runtime is already registered below the policy model. Keeping the
        # policy as a normal nn.Module attribute here would create the cycle
        # policy -> runtime -> policy and recurse forever in model.to/state_dict.
        self._model_ref = weakref.ref(model)

    def install(self, model: nn.Module) -> None:
        self._bind_model(model)
        layers = model.paligemma_with_expert.paligemma.language_model.layers
        if self.layer_index >= len(layers):
            raise ValueError(f"Cannot install factorial hook at {self.site}")

        def pre_hook(module, args, kwargs):
            del module
            if self._active_context is None:
                return None
            if args:
                adapted = self.adapter.intervention(args[0], **self._active_context)
                return (adapted, *args[1:]), kwargs
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Cannot locate Gemma hidden_states")
            updated = dict(kwargs)
            updated["hidden_states"] = self.adapter.intervention(hidden_states, **self._active_context)
            return args, updated

        self._hook_handle = layers[self.layer_index].register_forward_pre_hook(
            pre_hook, with_kwargs=True, prepend=True
        )

    @contextmanager
    def activate(self, context: dict[str, torch.Tensor] | None):
        if context is None:
            yield
            return
        if self._active_context is not None:
            raise RuntimeError("Nested factorial inference is not supported")
        self._active_context = context
        try:
            yield
        finally:
            self._active_context = None

    def _identity(self, suite: str, task_id: int, task_name: str) -> TaskIdentity:
        key = (suite, int(task_id))
        if key not in self.mapping_by_id:
            raise KeyError(f"Task absent from factorial mapping: {key}")
        identity = self.mapping_by_id[key]
        if task_name != identity.task_name:
            raise ValueError(f"Task-name mismatch for {key}")
        return identity

    def _supports(self, key: tuple[str, str]) -> list[Mapping[str, Any]]:
        candidates = [
            row for row in self.rows
            if row["split"] == "train"
            and row["metadata"].get("success") is True
            and row["metadata"]["base_task_id"] == key[0]
            and row["metadata"]["perturbation_category"] != key[1]
        ]
        candidates.sort(key=lambda row: hashlib.sha256(f"{self.support_seed}:{key}:{row['path']}".encode()).hexdigest())
        return candidates[: self.support_size] if len(candidates) >= self.support_size else []

    def _feature(self, row: Mapping[str, Any]) -> np.ndarray:
        episode_sha = str(row["sha256"])
        path = self.feature_dir / f"{episode_sha}.npz"
        expected = self.feature_index.get("shards", {}).get(episode_sha)
        if expected is None or not path.is_file():
            raise ValueError(f"Missing support feature shard: {path}")
        if self.verify_hashes and _sha256(path) != expected:
            raise ValueError(f"Modified support feature shard: {path}")
        with np.load(path, allow_pickle=False) as data:
            if str(data["episode_sha256"].item()) != episode_sha:
                raise ValueError("Feature shard/episode mismatch")
            values = np.asarray(data[self.site], dtype=np.float32)
        if values.ndim != 2 or not len(values) or values.shape[1] != self.adapter.task_encoder.config.feature_dim:
            raise ValueError(f"Invalid support features: {path}")
        return values.mean(axis=0)

    def _episode_path(self, row: Mapping[str, Any]) -> Path:
        path = (self.episodes_root / row["path"]).resolve()
        if not path.is_relative_to(self.episodes_root):
            raise ValueError("Support rollout escapes collection root")
        if not path.is_file():
            raise FileNotFoundError(f"Missing support rollout: {path}")
        if self.verify_hashes and _sha256(path) != row["sha256"]:
            raise ValueError(f"Modified support rollout: {path}")
        return path

    def _raw_support(self, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not row["query_indices"]:
            raise ValueError("Support episode has no full action window")
        seed = _private_seed([self.protocol, self.training_seed, row["path"], "support-chunk"])
        index = row["query_indices"][seed % len(row["query_indices"])]
        with np.load(self._episode_path(row), allow_pickle=False) as data:
            start = int(data["query_steps"][index])
            raw = {
                "observation/image": data["primary_images"][index].copy(),
                "observation/wrist_image": data["wrist_images"][index].copy(),
                "observation/state": data["states"][index].copy(),
                "prompt": row["metadata"]["instruction"],
                "actions": data["executed_actions"][start : start + self.manifest["action_horizon"]].copy(),
            }
        record = {
            "path": row["path"], "episode_key": row["metadata"]["episode_key"],
            "category": row["metadata"]["perturbation_category"], "query_index": index,
        }
        return raw, record

    def _support_batch(self, rows: list[dict[str, Any]], device: torch.device):
        batch = {}
        for key in rows[0]:
            batch[key] = [row[key] for row in rows] if key == "prompt" else np.stack([row[key] for row in rows])
        processed = self.model.input_transform(copy.deepcopy(batch), transpose=False)
        processed = self.model.precision_processor(processed)
        actions = processed.pop("actions").to(device=device, dtype=torch.float32)
        observation = _model.Observation.from_dict(processed)
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda value: value.to(device=device).contiguous() if torch.is_tensor(value) else value,
            observation,
        )
        return observation, actions

    def _support_flow_loss(
        self,
        raw_rows: list[dict[str, Any]],
        code: torch.Tensor,
        logits: torch.Tensor,
        *,
        seed: int,
    ) -> torch.Tensor:
        device = code.device
        observation, actions = self._support_batch(raw_rows, device)
        images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(observation, train=False)
        images = [value.to(device) for value in images]
        img_masks = [value.to(device) for value in img_masks]
        if lang_tokens is not None:
            lang_tokens = lang_tokens.to(device)
        if lang_masks is not None:
            lang_masks = lang_masks.to(device)
        state = state.to(device)
        with _private_rng(device, seed):
            noise = self.model.sample_noise(actions.shape, device)
            time = self.model.sample_time(actions.shape[0], device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        target = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks, image_token_mask = (
            self.model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                adapter_enabled=False,
                return_image_token_mask=True,
            )
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.model.embed_suffix(state, x_t, time)
        )
        backbone_dtype = self.model.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention = self.model._prepare_attention_masks_4d(
            make_att_2d_masks(pad_masks, att_masks)
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        context = {
            "image_token_mask": image_token_mask,
            "task_code": code.expand(len(raw_rows), -1),
            "mask_logits": logits.expand(len(raw_rows), -1),
            "strength": 1.0,
            "diagnostic_mask": self.mask_guided,
        }
        (_, suffix_out), _ = self.model.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            feature_intervention=self.adapter,
            intervention_context=context,
        )
        suffix_out = suffix_out[:, -self.model.config.action_horizon :].float()
        predicted = self.model.action_out_proj(suffix_out)
        return (
            predicted[:, : self.flow_prefix_steps, :7].float()
            - target[:, : self.flow_prefix_steps, :7].float()
        ).square().mean()

    def _adapt(self, key: tuple[str, str]) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
        supports = self._supports(key)
        details = {
            "base_task_id": key[0], "query_category": key[1],
            "support_seed": self.support_seed, "support_size": len(supports),
            "support_episodes": [row["path"] for row in supports],
            "intervention_strength": self.intervention_strength,
            "factorial_arm": self.arm, "factorial_spec": self.spec,
            "identity_fallback": not bool(supports) or self.intervention_strength == 0,
        }
        if not supports or self.intervention_strength == 0:
            details["fallback_reason"] = "intervention_strength_zero" if self.intervention_strength == 0 else "insufficient_cross_category_training_support"
            return None, details
        parameter = next(self.adapter.parameters())
        features = np.stack([self._feature(row) for row in supports])[None]
        with torch.no_grad():
            code = self.adapter.encode_support(torch.as_tensor(features, device=parameter.device, dtype=torch.float32)).detach()
        state = {"task_code": code, "strength": self.intervention_strength}
        details.update(task_code_sha256=_tensor_hash(code), inner_update_computed=False, delta_condition_retained=False)

        if self.spec["inner_steps"]:
            raw_rows, records = zip(*(self._raw_support(row) for row in supports), strict=True)
            noise_seed = _private_seed([self.protocol, self.training_seed, list(records), "support-flow"])
            with torch.inference_mode(False), torch.enable_grad():
                logits = self.adapter.intervention.mask(code).float().detach().requires_grad_(True)
                flow = self._support_flow_loss(list(raw_rows), code, logits, seed=noise_seed)
                identity = (2.0 * torch.sigmoid(logits) - 1.0).square().mean()
                gradient, = torch.autograd.grad(flow + self.mask_weight * identity, logits)
                raw_delta_u = -self.inner_lr * gradient
                delta_u = (
                    torch.zeros_like(raw_delta_u)
                    if self.no_inner_update
                    else self.inner_update_scale * raw_delta_u
                )
                updated = logits + delta_u
            if not torch.isfinite(updated).all():
                raise FloatingPointError("Non-finite meta-subspace inner update")
            if self.mask_guided:
                initial_mask = 2.0 * torch.sigmoid(logits)
                adapted_mask = 2.0 * torch.sigmoid(updated)
                importance = adapted_mask - 1.0
                delta_mask = adapted_mask - initial_mask
                guide = importance if self.spec["guide_signal"] == "importance" else delta_mask
                state["mask_importance"] = importance.detach()
                state["guide_condition"] = guide.detach()
            else:
                state["mask_logits"] = updated.detach()
                if self.spec["delta_condition"]:
                    state["delta_u"] = delta_u.detach()
            details.update(
                adapted_logits_sha256=_tensor_hash(updated), delta_u_sha256=_tensor_hash(delta_u),
                inner_grad_rms=float(gradient.detach().square().mean().sqrt().cpu()),
                inner_logit_max_abs_delta=float(delta_u.detach().abs().max().cpu()),
                inner_support_samples=[{"supports": list(records), "flow_seed": noise_seed, "flow_loss": float(flow.detach().cpu())}],
                inner_update_computed=True,
                inner_update_applied=not self.no_inner_update,
                inner_update_scale=self.inner_update_scale,
                delta_condition_retained=bool(self.spec.get("delta_condition", False)),
                query_direct_mask=False if self.mask_guided else True,
            )
            if self.mask_guided:
                details.update(
                    guide_signal=self.spec["guide_signal"],
                    importance_sha256=_tensor_hash(importance),
                    delta_mask_sha256=_tensor_hash(delta_mask),
                    guide_condition_sha256=_tensor_hash(guide),
                    importance_rms=float(importance.detach().square().mean().sqrt().cpu()),
                    delta_mask_rms=float(delta_mask.detach().square().mean().sqrt().cpu()),
                )
            if self.support_gate_min_relative_improvement >= 0:
                with torch.inference_mode(False), torch.enable_grad():
                    adapted_flow = self._support_flow_loss(
                        list(raw_rows), code, updated.detach(), seed=noise_seed
                    )
                relative_improvement = float(
                    ((flow.detach() - adapted_flow.detach())
                    / flow.detach().abs().clamp_min(1e-12)).cpu()
                )
                passed = (
                    relative_improvement
                    >= self.support_gate_min_relative_improvement
                )
                details.update(
                    support_adapted_flow_loss=float(adapted_flow.detach().cpu()),
                    support_relative_improvement=relative_improvement,
                    support_gate_min_relative_improvement=(
                        self.support_gate_min_relative_improvement
                    ),
                    support_gate_passed=passed,
                )
                if not passed:
                    details.update(
                        identity_fallback=True,
                        fallback_reason="support_loss_gate",
                    )
                    return None, details
        return state, details

    def _state_for(self, base_task_id: str, category: str):
        key = (base_task_id, category)
        if key not in self._states:
            self._states[key], self._state_details[key] = self._adapt(key)
        return self._states[key]

    def prepare_context(self, env_obs: Mapping[str, Any], *, device: torch.device) -> dict[str, torch.Tensor] | None:
        required = ("meta_suite_names", "meta_task_ids", "meta_task_names", "meta_perturbation_categories")
        missing = [key for key in required if key not in env_obs]
        if missing:
            raise KeyError(f"Factorial environment metadata is missing: {missing}")
        suites = [str(value) for value in env_obs["meta_suite_names"]]
        names = [str(value) for value in env_obs["meta_task_names"]]
        categories = [str(value) for value in env_obs["meta_perturbation_categories"]]
        ids_value = env_obs["meta_task_ids"]
        ids = ids_value.detach().cpu().tolist() if torch.is_tensor(ids_value) else list(ids_value)
        if not (len(suites) == len(names) == len(categories) == len(ids)):
            raise ValueError("Factorial environment metadata batch mismatch")

        states = []
        for suite, task_id, name, category in zip(suites, ids, names, categories, strict=True):
            try:
                identity = self._identity(suite, int(task_id), name)
                if category and category != identity.perturbation_category:
                    raise ValueError("Environment/task-mapping category mismatch")
                state = self._state_for(identity.base_task_id, identity.perturbation_category)
            except (KeyError, ValueError):
                if self.strict:
                    raise
                state = None
            if state is None and self.strict:
                raise ValueError(f"No eligible supports for task {name!r}")
            states.append(state)

        context: dict[str, torch.Tensor] = {
            "task_code": torch.stack([
                state["task_code"][0].to(device=device, dtype=torch.float32) if state is not None
                else torch.zeros(self.task_code_dim, device=device)
                for state in states
            ]),
            "sample_mask": torch.tensor([state is not None for state in states], device=device, dtype=torch.bool),
            "strength": self.intervention_strength,
        }
        if self.mask_guided:
            context["guide_condition"] = torch.stack([
                state["guide_condition"][0].to(device=device, dtype=torch.float32) if state is not None
                else torch.zeros(self.hidden_dim, device=device)
                for state in states
            ])
            context["mask_importance"] = torch.stack([
                state["mask_importance"][0].to(device=device, dtype=torch.float32) if state is not None
                else torch.zeros(self.hidden_dim, device=device)
                for state in states
            ])
            context["diagnostic_mask"] = False
        elif self.spec["inner_steps"]:
            context["mask_logits"] = torch.stack([
                state["mask_logits"][0].to(device=device, dtype=torch.float32) if state is not None
                else torch.zeros(self.hidden_dim, device=device)
                for state in states
            ])
        if self.spec.get("delta_condition", False):
            context["delta_u"] = torch.stack([
                state["delta_u"][0].to(device=device, dtype=torch.float32) if state is not None
                else torch.zeros(self.hidden_dim, device=device)
                for state in states
            ])
        return context

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256, "site": self.site,
            "factorial_arm": self.arm, "factorial_spec": self.spec,
            "support_size": self.support_size, "support_seed": self.support_seed,
            "intervention_strength": self.intervention_strength,
            "mask_guided": self.mask_guided, "no_inner_update": self.no_inner_update,
            "action_mode": self.action_mode, "cached_task_states": len(self._states),
            "task_states": list(self._state_details.values()),
        }
