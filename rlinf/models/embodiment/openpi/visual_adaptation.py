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

"""Parameter-efficient visual adaptation modules for the PyTorch pi0 model.

The visual backbone choice and FTM are intentionally orthogonal:

    SigLIP(+FLA) -> optional last-layer adapter -> projector -> optional FTM

FLA and the last-layer adapter are alternative backbone adapters, while FTM
can be enabled with either one (or on its own).
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


def _get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


class FLALinear(nn.Linear):
    """Linear layer with an identity-at-initialization LoRA branch.

    The original ``weight`` and ``bias`` parameter names are retained. This
    lets an unmodified pi0 safetensors checkpoint load without rewriting its
    keys; only ``lora_A`` and ``lora_B`` are new checkpoint entries.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        bias: bool,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if rank <= 0:
            raise ValueError(f"FLA rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(
            in_features, rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            rank, out_features, bias=False, device=device, dtype=dtype
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "FLALinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Reuse the exact Parameters so the original checkpoint keys stay valid.
        layer.weight = linear.weight
        layer.bias = linear.bias
        layer.train(linear.training)
        return layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = nn.functional.linear(inputs, self.weight, self.bias)
        update = self.lora_B(self.lora_A(self.lora_dropout(inputs)))
        return base + update * self.scaling


class LastLayerVisualAdapter(nn.Module):
    """Residual bottleneck adapter applied to final SigLIP tokens."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError(
                f"Adapter bottleneck_dim must be positive, got {bottleneck_dim}"
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.residual_scale = residual_scale
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(
            self.dropout(self.activation(self.down(self.norm(features))))
        )
        return features + self.residual_scale * residual


class FeatureTransformModulation(nn.Module):
    """Channel-wise affine modulation of projected visual tokens."""

    def __init__(
        self,
        hidden_dim: int,
        init_gamma: float = 0.0,
        init_beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((hidden_dim,), init_gamma))
        self.beta = nn.Parameter(torch.full((hidden_dim,), init_beta))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * (1.0 + self.gamma) + self.beta


def _replace_fla_linears(
    module: nn.Module,
    target_modules: set[str],
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
) -> list[str]:
    replaced: list[str] = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and child_name in target_modules:
            setattr(
                module,
                child_name,
                FLALinear.from_linear(child, rank, alpha, dropout),
            )
            replaced.append(full_name)
        else:
            replaced.extend(
                _replace_fla_linears(
                    child,
                    target_modules,
                    rank,
                    alpha,
                    dropout,
                    full_name,
                )
            )
    return replaced


def _first_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise TypeError(
        "Could not infer multi_modal_projector dimensions: no nn.Linear found"
    )


def visual_adaptation_enabled(config: Any) -> bool:
    backbone = _get(config, "backbone", {})
    mode = str(_get(backbone, "type", "none")).lower()
    ftm = _get(config, "ftm", {})
    return mode != "none" or bool(_get(ftm, "enabled", False))


def install_visual_adaptation(
    owner: nn.Module,
    paligemma_model: nn.Module,
    config: Any,
) -> list[Any]:
    """Install configured modules and return removable forward-hook handles."""

    owner.visual_last_layer_adapter = None
    owner.visual_ftm = None
    owner._fla_module_names = ()

    backbone = _get(config, "backbone", {})
    backbone_type = str(_get(backbone, "type", "none")).lower()
    valid_types = {"none", "fla", "last_layer_adapter"}
    if backbone_type not in valid_types:
        raise ValueError(
            "visual_adaptation.backbone.type must be one of "
            f"{sorted(valid_types)}, got {backbone_type!r}"
        )

    handles: list[Any] = []
    vision_tower = paligemma_model.vision_tower
    projector = paligemma_model.multi_modal_projector
    projector_linear = _first_linear(projector)

    if backbone_type == "fla":
        fla = _get(backbone, "fla", {})
        targets = set(
            _get(
                fla,
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "out_proj"],
            )
        )
        names = _replace_fla_linears(
            vision_tower,
            targets,
            rank=int(_get(fla, "rank", 16)),
            alpha=float(_get(fla, "alpha", 16.0)),
            dropout=float(_get(fla, "dropout", 0.0)),
        )
        if not names:
            raise ValueError(
                "FLA was enabled, but none of target_modules matched a vision "
                f"nn.Linear. Requested targets: {sorted(targets)}"
            )
        expected_modules = _get(fla, "expected_modules", None)
        if expected_modules is not None and len(names) != int(expected_modules):
            raise ValueError(
                "FLA module count mismatch: "
                f"expected {int(expected_modules)}, replaced {len(names)}"
            )
        owner._fla_module_names = tuple(names)

    if backbone_type == "last_layer_adapter":
        adapter_cfg = _get(backbone, "last_layer_adapter", {})
        adapter = LastLayerVisualAdapter(
            hidden_dim=projector_linear.in_features,
            bottleneck_dim=int(_get(adapter_cfg, "bottleneck_dim", 64)),
            dropout=float(_get(adapter_cfg, "dropout", 0.0)),
            residual_scale=float(_get(adapter_cfg, "residual_scale", 1.0)),
        )
        owner.visual_last_layer_adapter = adapter

        def adapter_pre_hook(_module: nn.Module, args: tuple[Any, ...]):
            if not args:
                return args
            return (adapter(args[0]), *args[1:])

        handles.append(projector.register_forward_pre_hook(adapter_pre_hook))

    ftm_cfg = _get(config, "ftm", {})
    if bool(_get(ftm_cfg, "enabled", False)):
        ftm = FeatureTransformModulation(
            hidden_dim=projector_linear.out_features,
            init_gamma=float(_get(ftm_cfg, "init_gamma", 0.0)),
            init_beta=float(_get(ftm_cfg, "init_beta", 0.0)),
        )
        owner.visual_ftm = ftm

        def ftm_hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            return ftm(output)

        handles.append(projector.register_forward_hook(ftm_hook))

    return handles


def iter_visual_adaptation_parameters(owner: nn.Module) -> Iterable[nn.Parameter]:
    """Yield visual adaptation parameters without yielding frozen base weights."""

    for module in owner.modules():
        if isinstance(module, FLALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()
    for name in ("visual_last_layer_adapter", "visual_ftm"):
        module = getattr(owner, name, None)
        if module is not None:
            yield from module.parameters()


def align_visual_adaptation_dtype(
    owner: nn.Module,
    paligemma_model: nn.Module,
) -> None:
    """Align new modules with the already-loaded bf16 pi0 backbone."""

    for module in owner.modules():
        if isinstance(module, FLALinear):
            module.lora_A.to(device=module.weight.device, dtype=module.weight.dtype)
            module.lora_B.to(device=module.weight.device, dtype=module.weight.dtype)

    projector_linear = _first_linear(paligemma_model.multi_modal_projector)
    for name in ("visual_last_layer_adapter", "visual_ftm"):
        module = getattr(owner, name, None)
        if module is not None:
            module.to(
                device=projector_linear.weight.device,
                dtype=projector_linear.weight.dtype,
            )

