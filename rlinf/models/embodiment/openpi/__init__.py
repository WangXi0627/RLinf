# Copyright 2025 The RLinf Authors.
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
# openpi model configs

import os
import pathlib

import torch
from omegaconf import DictConfig


def _resolve_trainable_checkpoint(path: str | os.PathLike[str]) -> pathlib.Path:
    requested_path = pathlib.Path(path).expanduser()
    candidates = [
        requested_path,
        requested_path / "trainable_weights.pt",
        requested_path
        / "trainable_model_state_dict"
        / "trainable_weights.pt",
        requested_path
        / "actor"
        / "trainable_model_state_dict"
        / "trainable_weights.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find a trainable-only checkpoint. Checked:\n  - " + checked
    )


def _load_trainable_checkpoint(model: torch.nn.Module, path: str) -> pathlib.Path:
    checkpoint_path = _resolve_trainable_checkpoint(path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Trainable checkpoint must be a dictionary, got {type(payload)!r}."
        )
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            f"No state_dict tensors found in trainable checkpoint {checkpoint_path}."
        )

    model_keys = set(model.state_dict())
    loadable_state_dict = {
        name: value for name, value in state_dict.items() if name in model_keys
    }
    skipped_keys = sorted(set(state_dict) - set(loadable_state_dict))
    disallowed_skipped_keys = [
        name for name in skipped_keys if not name.startswith("value_head.")
    ]
    if disallowed_skipped_keys:
        raise RuntimeError(
            "Trainable checkpoint contains parameters absent from the configured "
            f"model: {disallowed_skipped_keys[:20]}. Ensure the visual adapter "
            "configuration matches training."
        )
    if not loadable_state_dict:
        raise RuntimeError(
            f"No checkpoint parameters match the configured model: {checkpoint_path}"
        )

    model.load_state_dict(loadable_state_dict, strict=False)
    logger = getattr(model, "logger", None)
    message = (
        f"Loaded {len(loadable_state_dict)} trainable tensors from "
        f"{checkpoint_path}"
    )
    if skipped_keys:
        message += f"; skipped {len(skipped_keys)} value-head tensors"
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    return checkpoint_path


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.meta_subspace import (
        install_meta_subspace,
    )
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        model.load_state_dict(all_state_dict, strict=False)

    trainable_checkpoint_path = getattr(cfg, "trainable_checkpoint_path", None)
    if trainable_checkpoint_path is not None:
        _load_trainable_checkpoint(model, str(trainable_checkpoint_path))

    meta_runtime = install_meta_subspace(
        model,
        getattr(actor_model_config, "meta_subspace", None),
        base_checkpoint_dir=checkpoint_dir,
    )
    if meta_runtime is not None:
        message = (
            "Loaded Meta-Subspace checkpoint "
            f"{meta_runtime.checkpoint_path} at {meta_runtime.site}; "
            f"support_size={meta_runtime.support_size}, "
            f"support_seed={meta_runtime.support_seed}, "
            f"action_mode={meta_runtime.action_mode}"
        )
        logger = getattr(model, "logger", None)
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.align_visual_adaptation_dtype()
    model.configure_adaptation_trainability()
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
