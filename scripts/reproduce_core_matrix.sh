#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_SERVER:?Set LLAMA_SERVER to the llama-server binary}"
: "${MODEL:?Set MODEL to the GGUF file}"

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bench="$repo_root/scripts/benchmark_llama.sh"
context=${CONTEXT:-262144}

common=(-sm tensor -fa on --jinja -c "$context" --spec-type draft-mtp)

run() {
  local devices=$1
  local label=$2
  shift 2
  echo
  echo "### $label (CUDA_VISIBLE_DEVICES=$devices)"
  CUDA_VISIBLE_DEVICES=$devices "$bench" "$label" -- "$@"
}

if [[ ${RUN_THREE_GPU:-1} == 1 ]]; then
  run 0,1,2 three-even-mtp3 "${common[@]}" --spec-draft-n-max 3
  run 0,1,2 three-weighted-mtp3 "${common[@]}" --spec-draft-n-max 3 -ts 0.4,0.4,0.2
fi

run 0,1 two-mtp3 "${common[@]}" --spec-draft-n-max 3 -ts 1,1
run 0,1 two-mtp4 "${common[@]}" --spec-draft-n-max 4 -ts 1,1
run 0,1 two-mtp5 "${common[@]}" --spec-draft-n-max 5 -ts 1,1
run 0,1 two-mtp3-kvq8 "${common[@]}" --spec-draft-n-max 3 -ts 1,1 -ctk q8_0 -ctv q8_0
run 0,1 two-mtp3-pmin05 "${common[@]}" --spec-draft-n-max 3 --spec-draft-p-min 0.5 -ts 1,1

if [[ ${RUN_LONG:-1} == 1 ]]; then
  N_PREDICT=2048 run 0,1 two-mtp3-long "${common[@]}" --spec-draft-n-max 3 -ts 1,1
fi
