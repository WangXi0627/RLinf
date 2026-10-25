#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a variants <<< "${PRIORITY_VARIANTS:-original original_no_inner preserved preserved_no_inner preserved_gate preserved_gate_no_inner}"

for variant in "${variants[@]}"; do
  echo "=== importance_so priority evaluation: ${variant} ==="
  bash "${script_dir}/eval_importance_so_priority_arm.sh" "${variant}"
done
