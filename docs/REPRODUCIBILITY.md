# Reproducibility Notes

## What is pinned

- llama.cpp base commit: `9723942`
- patch commit represented by the format-patch: `b4e45e05e0edee0b32875a70b4e451eb2b4029a5`
- CUDA architecture: `sm_75`
- model family and format: Qwen3.8-27B, Q4_K_M GGUF with embedded MTP head
- context: 262,144 tokens
- KV type: F16
- split mode: tensor, 1/1 on the NVLink pair
- sampling: temperature 0 for performance and bit-identity checks

The model weights are not redistributed. Record the exact model filename and SHA-256 hash in your private run log.

## Recommended run discipline

1. Stop other processes using the target GPUs.
2. Record `nvidia-smi -q`, `nvidia-smi topo -m`, the driver version, CPU model, kernel, and llama.cpp commit.
3. Lock the prompt and output token count.
4. Warm up the server before recording a run.
5. Run at least three repetitions for short generations.
6. Include a 2048-token generation to reduce startup noise.
7. Save both the server log and full JSON response.
8. Compare output bytes at temperature 0 when validating an all-reduce change.

Use `scripts/check_topology.sh` for the environment snapshot and `scripts/benchmark_llama.sh` for isolated llama.cpp runs. Raw results are written under `results/raw/`, which is intentionally ignored by Git because responses may contain private prompts.

## P2P patch validation checklist

- The patch applies cleanly to `9723942` with `git am`.
- The project builds for `sm_75`.
- Two-way peer access is reported for the selected devices.
- A 2048-token temperature-0 response is byte-identical to the unpatched build.
- At least three consecutive long requests complete.
- A no-peer topology falls back without crashing.
- Throughput is compared on the same day and with the same clock/thermal conditions.

## vLLM comparison caveat

The recorded comparison used FP8 safetensors in vLLM and Q4_K_M GGUF in llama.cpp. It answers “which deployable stack is faster on this host?” rather than “which engine is faster with identical kernels and weights?”. Do not use it as a model-quality comparison.
