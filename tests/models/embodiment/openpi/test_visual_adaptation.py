import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openpi import _pcd_spatial_adaptation_state
from rlinf.models.embodiment.openpi.visual_adaptation import (
    FeatureTransformModulation,
    FLALinear,
    install_visual_adaptation,
    iter_visual_adaptation_parameters,
)


class _VisionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.unrelated = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.q_proj(inputs) + self.k_proj(inputs)


class _PaliGemmaModel(nn.Module):
    def __init__(self, vision_width: int, language_width: int) -> None:
        super().__init__()
        self.vision_tower = _VisionBlock(vision_width)
        self.multi_modal_projector = nn.Linear(vision_width, language_width)


class _Owner(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


@pytest.mark.parametrize("backbone_type", ["fla", "last_layer_adapter"])
def test_ftm_composes_with_backbone_adapter_and_starts_as_identity(
    backbone_type: str,
) -> None:
    torch.manual_seed(0)
    model = _PaliGemmaModel(vision_width=8, language_width=12)
    owner = _Owner(model)
    inputs = torch.randn(2, 4, 8)
    expected = model.multi_modal_projector(model.vision_tower(inputs))
    config = {
        "backbone": {
            "type": backbone_type,
            "fla": {
                "rank": 2,
                "alpha": 2.0,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj"],
            },
            "last_layer_adapter": {
                "bottleneck_dim": 4,
                "dropout": 0.0,
                "residual_scale": 1.0,
            },
        },
        "ftm": {"enabled": True, "init_gamma": 0.0, "init_beta": 0.0},
    }

    handles = install_visual_adaptation(owner, model, config)
    actual = model.multi_modal_projector(model.vision_tower(inputs))

    torch.testing.assert_close(actual, expected)
    assert len(handles) >= 1
    assert owner.visual_ftm is not None
    if backbone_type == "fla":
        assert isinstance(model.vision_tower.q_proj, FLALinear)
        assert isinstance(model.vision_tower.k_proj, FLALinear)
        assert not isinstance(model.vision_tower.unrelated, FLALinear)
    else:
        assert owner.visual_last_layer_adapter is not None

    adaptation_parameters = list(iter_visual_adaptation_parameters(owner))
    assert adaptation_parameters
    assert len({id(parameter) for parameter in adaptation_parameters}) == len(
        adaptation_parameters
    )


def test_fla_preserves_base_checkpoint_parameter_names() -> None:
    model = _PaliGemmaModel(vision_width=8, language_width=12)
    owner = _Owner(model)
    old_weight = model.vision_tower.q_proj.weight
    config = {
        "backbone": {
            "type": "fla",
            "fla": {
                "rank": 2,
                "alpha": 2.0,
                "target_modules": ["q_proj"],
            },
        },
        "ftm": {"enabled": False},
    }

    install_visual_adaptation(owner, model, config)
    state_keys = set(model.state_dict())

    assert model.vision_tower.q_proj.weight is old_weight
    assert "vision_tower.q_proj.weight" in state_keys
    assert "vision_tower.q_proj.lora_A.weight" in state_keys
    assert "vision_tower.q_proj.lora_B.weight" in state_keys


def test_native_pcd_ftm_checkpoint_maps_to_existing_rlinf_keys() -> None:
    owner = _Owner(_PaliGemmaModel(vision_width=8, language_width=12))
    owner.visual_ftm = FeatureTransformModulation(12)
    gamma = torch.randn(12)
    beta = torch.randn(12)
    payload = {
        "adaptation": {
            "method": "ftm",
            "spec": {"feature_dim": 12, "view_indices": [0, 1]},
        },
        "adaptation_state": {"gamma": gamma, "beta": beta},
    }

    state = _pcd_spatial_adaptation_state(owner, payload)

    assert set(state) == {"visual_ftm.gamma", "visual_ftm.beta"}
    assert state["visual_ftm.gamma"] is gamma
    assert state["visual_ftm.beta"] is beta


def test_native_pcd_fla_checkpoint_maps_fc1_fc2_and_checks_scaling() -> None:
    owner = nn.Module()
    owner.first = FLALinear(8, 16, rank=2, alpha=2.0, dropout=0.0, bias=True)
    owner.second = FLALinear(16, 8, rank=2, alpha=2.0, dropout=0.0, bias=True)
    state = {
        "layers.0.mlp.fc1.lora_a": torch.randn(2, 8),
        "layers.0.mlp.fc1.lora_b": torch.randn(16, 2),
        "layers.0.mlp.fc2.lora_a": torch.randn(2, 16),
        "layers.0.mlp.fc2.lora_b": torch.randn(8, 2),
    }
    payload = {
        "adaptation": {
            "method": "fla",
            "spec": {"rank": 2, "scale": 1.0, "expected_layers": 1},
        },
        "adaptation_state": state,
    }

    translated = _pcd_spatial_adaptation_state(owner, payload)

    prefix = (
        "paligemma_with_expert.paligemma.model."
        "vision_tower.vision_model.encoder."
    )
    assert set(translated) == {
        prefix + "layers.0.mlp.fc1.lora_A.weight",
        prefix + "layers.0.mlp.fc1.lora_B.weight",
        prefix + "layers.0.mlp.fc2.lora_A.weight",
        prefix + "layers.0.mlp.fc2.lora_B.weight",
    }


def test_fla_expected_module_count_is_strict() -> None:
    model = _PaliGemmaModel(vision_width=8, language_width=12)
    owner = _Owner(model)
    config = {
        "backbone": {
            "type": "fla",
            "fla": {
                "rank": 2,
                "alpha": 2.0,
                "target_modules": ["q_proj", "k_proj"],
                "expected_modules": 3,
            },
        },
        "ftm": {"enabled": False},
    }

    with pytest.raises(ValueError, match="module count mismatch"):
        install_visual_adaptation(owner, model, config)

