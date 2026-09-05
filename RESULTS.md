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

## CPU orchestration follow-up

More than 40 affinity, thread, poll, HTTP-thread, and governor configurations were tested in randomized five-round matrices. The best candidate was re-run against the default in an interleaved ABAB confirmation:

| Workload | Default | Candidate | Delta | Decision |
|---|---:|---:|---:|---|
| English, 2048 generated tokens | 81.53 | 81.62 | +0.11% | Do not deploy |
| Chinese prose, 1024 generated tokens | 49.63 | 49.66 | +0.06% | Do not deploy |

A 30-minute Chinese soak completed 86 requests with zero errors and -0.04% first-half to second-half drift. Source-level control-thread affinity and CUDA wait-mode experiments were also below 0.2%.

## CUDA Graph P0: corrected profiling

An earlier kernel-table-only reading understated GPU work because Nsight Systems records kernels executed inside CUDA Graph replay in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, not the ordinary kernel table. The corrected steady-state accounting is:

| Component | Time per pass | Interpretation |
|---|---:|---|
| Whole speculative pass | ~45.2 ms | 100% wall budget |
| Target decode / synchronization window | ~35.5 ms | Host is waiting for genuinely busy GPUs |
| GPU0 graph execution | ~33.3 ms | Must be read from graph-trace rows |
| P2P all-reduce kernels | ~1.7 ms | ~137 calls/pass, about 13.3µs each |
| GPU0 idle | ~11.0 ms | Mostly adjacent to the draft path |
| `spec_draft` window | ~7.5 ms | Only ~1 ms GPU work; remainder is fixed host work |

CUDA Graph on/off ABAB×5 produced 82.73 vs 81.59 tok/s for English and 50.18 vs 49.50 tok/s for Chinese: graphs are correct and useful, but their entire end-to-end value on this host is only about 1.4%. Steady-state replay was already saturated at 97.8%.

## CUDA Graph P1: per-shape cache keys

The MTP nextn-layer subgraph alternates between catch-up `n_tokens=4` and three draft steps at `n_tokens=1`. Both shapes shared one graph-cache key, so the stored properties alternated `4,1,1,1,4,...` and never completed the two-identical-call warmup requirement.

With `GGML_CUDA_SHAPE_KEYS=1`, each leading shape receives a separate cache key. Draft-bucket replay changed from none to 506/508 calls, warmup resets fell from 870 to 303, and direct evaluations fell from 2352 to 1779. Temperature-0 output hashes matched in all six runs.

| Workload | Shape keys on | Shape keys off | Delta |
|---|---:|---:|---:|
| English, 2048 generated tokens | 82.76 | 82.51 | **+0.30%** |
| Chinese prose, 1024 generated tokens | 50.29 | 50.08 | **+0.42%** |

Both configurations slowed with thermal/run-order drift. The cleanest first round showed +1.24% English and +0.88% Chinese, still below the 3% deployment threshold. The mechanism is valid, but `spec_draft` remained 7.5 ms; launch fragmentation was a symptom rather than the limiting cost. See the sanitized raw rows in `results/cuda-graph-shape-key-abab.csv`.
