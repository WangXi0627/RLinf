#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arm in b_residual c1_joint_mask c2_bilevel c3_full; do
  bash "${script_dir}/eval_factorial_arm.sh" "${arm}"
done
