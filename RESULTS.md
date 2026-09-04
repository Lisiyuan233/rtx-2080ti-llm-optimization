# Benchmark Results

All figures below were measured on the host documented in [the report](docs/REPORT.zh-CN.md). `tg` is generated-token throughput and `pp` is prompt-processing throughput. Runs used a fixed prompt, temperature 0, and an otherwise idle target GPU set.

## Topology and split-mode sweep

| Configuration | pp (tok/s) | tg (tok/s) | Notes |
|---|---:|---:|---|
| 3 GPUs, even tensor split, MTP3 | 214 | 41.8 | Original baseline |
| 3 GPUs, split 0.4/0.4/0.2, MTP3 | 246 | 53.4 | Better, still pays non-NVLink synchronization |
| 3 GPUs, weighted, all-reduce env disabled | 234 | 52.5 | No material benefit |
| 2 GPUs, NVLink pair, split 1/1, MTP3 | 560 | 66.2 | Topology winner |
| 3 GPUs, layer split | 553 | 34.4 | Prefill good, decode poor |
| 1 GPU | 533 | 35.3 | Capacity-limited comparison |

## MTP, KV, and drafter sweep

| Configuration | tg (tok/s) | Relative conclusion |
|---|---:|---|
| 3 GPUs, MTP3 | 41.8 | Best three-GPU draft length |
| 3 GPUs, MTP4 / MTP5 / MTP6 | 34.7 / 35.6 / 32.2 | Longer is worse |
| 3 GPUs, no speculative decoding | 25.6 | MTP3 adds about 63% |
| 2 GPUs, MTP3 | 66.9 | Best two-GPU baseline |
| 2 GPUs, MTP4 / MTP5 | 49.3 / 43.4 | Rejected |
| 2 GPUs, MTP4 + p-min 0.6 | 29.3 | Rejected |
| 2 GPUs, MTP6 + p-min 0.7 | 27.0 | Rejected |
| 2 GPUs, MTP3 + p-min 0.5 | 33.5 | Rejected |
| 2 GPUs, MTP3, Q8_0 KV | 53.0 | F16 KV is faster here |
| 2 GPUs, MTP3 + ngram | 66.6 | No benefit |
| 2 GPUs, ngram8 only | 41.7 | Rejected |
| 2 GPUs, DFlash2 draft 3 | 52.9 | 21% below same-tree MTP3 (67.1) |
| 2 GPUs, DFlash2 draft 4 | 43.0 | Rejected |

## End-to-end workloads

| Workload | Original 3-GPU route | Tuned 2-GPU route | P2P-patched route |
|---|---:|---:|---:|
| English, 256 generated tokens | 41.8 | 66.6–67.1 | **69.2–69.8** |
| English, 2048 generated tokens | — | 81.3 | **82.4–83.9** |
| Chinese prose, 1024 generated tokens | 39.0 | 47.5 | **48.1** |
| Long prompt prefill | 175 | 335–699 | Not separately isolated |

## vLLM comparison

This is a deployment comparison, not a pure engine A/B: llama.cpp used Q4_K_M GGUF while vLLM used the official FP8 checkpoint. The prompt and wall-clock measurement method were aligned.

| Workload | llama.cpp Q4_K_M | vLLM FP8 MTP3 | vLLM delta |
|---|---:|---:|---:|
| English, 256 generated tokens | 66.6 | 34.5 | -48% |
| English, 2048 generated tokens | 81.3 | 54.0 | -34% |
| Chinese prose, 1024 generated tokens | 47.5 | 33.2 | -30% |
| ~14.5K-token prefill | 335–699 | ~908 | +30% or more |

The old Ivy Bridge Xeon host is the likely cause of much of the decode regression: vLLM's Python/service control plane is more sensitive to single-core latency than llama.cpp's C++ loop. vLLM remains an attractive route for prefill-heavy or batched workloads.

## Profiling summary

For a steady 2048-token decode run, one speculative pass produced about 3.85 accepted tokens and took roughly 49 ms:

| Component | Time per pass | Approx. wall share |
|---|---:|---:|
| 62-layer compute graph | ~13.5 ms | ~28% |
| Internal all-reduce kernels | ~3.7 ms | ~7.5% |
| CPU sampling and draft/verify orchestration gaps | ~16–29 ms | ~33–40% |
| MTP draft graph and other work | ~2–3 ms | ~5% |

Layer compute was already near the weight-bandwidth roofline. This ruled out custom GEMM/dequantization work and motivated the smaller, measurable P2P all-reduce patch.
