#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: $0 original|original_no_inner|preserved|preserved_no_inner|preserved_gate|preserved_gate_no_inner}"
case "${variant}" in
  original|original_no_inner|preserved|preserved_no_inner|preserved_gate|preserved_gate_no_inner) ;;
  *) echo "unknown priority variant: ${variant}" >&2; exit 2 ;;
esac

export RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"
export PCD_ROOT="${PCD_ROOT:-/data/Xixixi/VLA1/PCD-meta-subspace}"
export INTERVENTION_STRENGTH=0.25
export MASK_GUIDED_INNER_UPDATE_SCALE=1.0
export MASK_GUIDED_ACTION_MODE=all_meta
export MASK_GUIDED_SUPPORT_GATE_MIN_RELATIVE_IMPROVEMENT=-1.0
export MASK_GUIDED_RESIDUAL_GATE_MAX_RELATIVE_RMS=-1.0
export MASK_GUIDED_ACTION_GATE_MAX_MEAN_ABS_DELTA=-1.0
export MASK_GUIDED_ACTION_GATE_HORIZON="${MASK_GUIDED_ACTION_GATE_HORIZON:-5}"

case "${variant}" in
  original|original_no_inner)
    export MASK_GUIDED_CHECKPOINT="${ORIGINAL_IMPORTANCE_SO_CHECKPOINT:-${PCD_ROOT}/results/libero_plus_meta_subspace/v4/importance_so_layer15_seed7/latest.pt}"
    ;;
  *)
    export MASK_GUIDED_CHECKPOINT="${PRESERVED_IMPORTANCE_SO_CHECKPOINT:-${PCD_ROOT}/results/libero_plus_meta_subspace/v5/importance_so_preserved_layer15_seed7/latest.pt}"
    ;;
esac

if [[ "${variant}" == preserved_gate* ]]; then
  export MASK_GUIDED_ACTION_MODE=conservative_gate
  export MASK_GUIDED_SUPPORT_GATE_MIN_RELATIVE_IMPROVEMENT="${SUPPORT_GATE_MIN_RELATIVE_IMPROVEMENT:-0.0}"
  export MASK_GUIDED_RESIDUAL_GATE_MAX_RELATIVE_RMS="${RESIDUAL_GATE_MAX_RELATIVE_RMS:-0.10}"
  export MASK_GUIDED_ACTION_GATE_MAX_MEAN_ABS_DELTA="${ACTION_GATE_MAX_MEAN_ABS_DELTA:-0.05}"
fi

python - "${variant}" <<'PY'
import os
import sys

import torch

variant = sys.argv[1]
payload = torch.load(
    os.environ["MASK_GUIDED_CHECKPOINT"],
    map_location="cpu",
    weights_only=True,
)
config = payload["config"]
assert config["arm"] == "importance_so"
assert config["second_order"] is True
assert config["inner_lr"] == 100
assert config["strength"] == 1
if variant.startswith("original"):
    assert config.get("preservation_variant", "none") == "none"
    assert config["behavior_preservation_weight"] == 0
    assert config["action_preservation_weight"] == 0
else:
    assert config.get("preservation_variant") == "base_success_behavior_action_v1"
    assert config["behavior_preservation_weight"] == 1
    assert config["action_preservation_weight"] == 0.1
print("Priority checkpoint preflight passed:", variant)
PY

eval_arm=importance_so
[[ "${variant}" == *_no_inner ]] && eval_arm=importance_so_no_inner
export RESULT_ROOT="${RESULT_ROOT_BASE:-${RLINF_ROOT}/results/pi0/importance_so_priority_strength025/goal5}/${variant}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/eval_mask_guided_arm.sh" "${eval_arm}"
