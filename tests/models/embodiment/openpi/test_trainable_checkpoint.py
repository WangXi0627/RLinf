from pathlib import Path

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openpi import (
    _load_trainable_checkpoint,
    _resolve_trainable_checkpoint,
)


class _TinyAdapterModel(nn.Module):
    def __init__(self, add_value_head: bool = True) -> None:
        super().__init__()
        self.frozen_base = nn.Linear(4, 4)
        self.visual_last_layer_adapter = nn.Linear(4, 4)
        if add_value_head:
            self.value_head = nn.Linear(4, 1)


def _write_checkpoint(path: Path, model: nn.Module) -> dict[str, torch.Tensor]:
    state_dict = {
        name: torch.full_like(value, 0.25)
        for name, value in model.state_dict().items()
        if name.startswith("visual_last_layer_adapter.")
        or name.startswith("value_head.")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "global_step": 10,
            "state_dict": state_dict,
        },
        path,
    )
    return state_dict


def test_resolve_checkpoint_from_global_step_directory(tmp_path: Path) -> None:
    checkpoint_path = (
        tmp_path
        / "actor"
        / "trainable_model_state_dict"
        / "trainable_weights.pt"
    )
    _write_checkpoint(checkpoint_path, _TinyAdapterModel())

    assert _resolve_trainable_checkpoint(tmp_path) == checkpoint_path.resolve()


def test_load_trainable_checkpoint_without_overwriting_base(tmp_path: Path) -> None:
    source_model = _TinyAdapterModel()
    checkpoint_path = tmp_path / "trainable_weights.pt"
    expected = _write_checkpoint(checkpoint_path, source_model)

    target_model = _TinyAdapterModel()
    frozen_before = {
        name: value.clone()
        for name, value in target_model.frozen_base.state_dict().items()
    }
    _load_trainable_checkpoint(target_model, str(checkpoint_path))

    for name, value in target_model.frozen_base.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name])
    for name, value in target_model.state_dict().items():
        if name in expected:
            torch.testing.assert_close(value, expected[name])


def test_value_head_is_optional_during_evaluation(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "trainable_weights.pt"
    expected = _write_checkpoint(checkpoint_path, _TinyAdapterModel())
    target_model = _TinyAdapterModel(add_value_head=False)

    _load_trainable_checkpoint(target_model, str(checkpoint_path))

    for name, value in target_model.state_dict().items():
        if name in expected:
            torch.testing.assert_close(value, expected[name])


def test_mismatched_adapter_configuration_fails(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "trainable_weights.pt"
    torch.save(
        {"state_dict": {"visual_ftm.gamma": torch.zeros(4)}},
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="configuration matches training"):
        _load_trainable_checkpoint(
            _TinyAdapterModel(add_value_head=False),
            str(checkpoint_path),
        )
