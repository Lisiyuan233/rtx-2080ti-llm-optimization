#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: LLAMA_SERVER=... MODEL=... $0 LABEL [-- llama-server arguments...]" >&2
  exit 2
fi

: "${LLAMA_SERVER:?Set LLAMA_SERVER to the llama-server binary}"
: "${MODEL:?Set MODEL to the GGUF file}"

label=$1
shift
if [[ ${1:-} == "--" ]]; then
  shift
fi

if [[ ! $label =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "LABEL may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
prompt_file=${PROMPT_FILE:-"$repo_root/prompts/english-token-stream.txt"}
result_dir=${RESULT_DIR:-"$repo_root/results/raw"}
port=${PORT:-8099}
n_predict=${N_PREDICT:-256}
startup_timeout=${STARTUP_TIMEOUT:-180}

mkdir -p "$result_dir"
server_log="$result_dir/${label}.server.log"
response_json="$result_dir/${label}.response.json"
request_json="$result_dir/${label}.request.json"
summary_jsonl="$result_dir/runs.jsonl"

if [[ ! -x $LLAMA_SERVER ]]; then
  echo "Not executable: $LLAMA_SERVER" >&2
  exit 2
fi
if [[ ! -f $MODEL ]]; then
  echo "Model not found: $MODEL" >&2
  exit 2
fi
if [[ ! -f $prompt_file ]]; then
  echo "Prompt not found: $prompt_file" >&2
  exit 2
fi

python3 - "$prompt_file" "$request_json" "$n_predict" <<'PY'
import json
import pathlib
import sys

prompt_path, output_path, n_predict = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = pathlib.Path(prompt_path).read_text(encoding="utf-8")
payload = {
    "prompt": prompt,
    "n_predict": n_predict,
    "temperature": 0,
    "cache_prompt": False,
}
pathlib.Path(output_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
PY

server_pid=""
cleanup() {
  if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# An explicitly empty value disables the override; an unset value selects the
# internal all-reduce path used by the tested configuration.
if [[ -z ${GGML_CUDA_ALLREDUCE+x} ]]; then
  export GGML_CUDA_ALLREDUCE=internal
elif [[ -z ${GGML_CUDA_ALLREDUCE} ]]; then
  unset GGML_CUDA_ALLREDUCE
fi

"$LLAMA_SERVER" -m "$MODEL" --host 127.0.0.1 --port "$port" -ngl 999 "$@" \
  >"$server_log" 2>&1 &
server_pid=$!

ready=0
for ((i = 0; i < startup_timeout; i++)); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "$label: server exited during startup" >&2
    tail -n 20 "$server_log" >&2 || true
    exit 1
  fi
  sleep 1
done

if [[ $ready -ne 1 ]]; then
  echo "$label: health check timed out after ${startup_timeout}s" >&2
  exit 1
fi

curl -fsS --max-time 120 "http://127.0.0.1:${port}/completion" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"warmup","n_predict":8,"temperature":0}' >/dev/null

curl -fsS --max-time "${REQUEST_TIMEOUT:-900}" \
  "http://127.0.0.1:${port}/completion" \
  -H 'Content-Type: application/json' \
  --data-binary "@$request_json" >"$response_json"

cleanup
server_pid=""
trap - EXIT INT TERM

python3 "$repo_root/scripts/parse_llama.py" \
  "$label" "$response_json" "$server_log" --jsonl "$summary_jsonl"
