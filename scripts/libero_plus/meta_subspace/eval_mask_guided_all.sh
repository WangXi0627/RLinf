#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arm in   importance_fo importance_fo_no_inner   importance_so importance_so_no_inner   delta_fo delta_fo_no_inner   delta_so delta_so_no_inner; do
  bash "${script_dir}/eval_mask_guided_arm.sh" "${arm}"
done
