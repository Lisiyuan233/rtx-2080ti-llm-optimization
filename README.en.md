# RTX 2080 Ti LLM Inference Optimization

English | [简体中文](README.md)

A measured optimization study for Qwen3.8-27B inference on modified RTX 2080 Ti 22GB / SM75 hardware. The test host has three GPUs, but only GPU0 and GPU1 share NVLink. Topology-aware selection, MTP tuning, and a direct P2P all-reduce patch increased single-request generation from roughly **41.8 tok/s to 69–84 tok/s**, depending on workload length.

## Highlights

- Two NVLink-connected GPUs beat an even three-GPU tensor split by about 59%.
- MTP draft length 3 was optimal; longer drafts and probability gating regressed.
- F16 KV beat Q8_0 KV for this MTP workload.
- A BF16 DFlash2 drafter was 21% slower than the embedded MTP head on Turing.
- vLLM FP8 had 30%+ faster long-prompt prefill, but 30–48% slower decode on the old dual-Xeon host.
- The included llama.cpp patch replaces host-staged internal all-reduce with direct peer staging when bidirectional peer access is available. It improved short English generation by 4–5% and long generation by 1.5–3% on the NVLink pair.
- More than 40 CPU/NUMA/thread/poll configurations converged to a +0.11% English and +0.06% Chinese ABAB result, so none were deployed.
- CUDA Graph replay already covered 97.8% of steady-state compute. Disabling graphs cost only 1.4% end to end.
- A per-shape graph-cache experiment fixed permanent direct evaluation in the MTP draft path, but honest ABAB gains were only +0.30% English and +0.42% Chinese. It remains an opt-in experiment.

Representative final results:

| Workload | Result |
|---|---:|
| English, 256 generated tokens | 69.2–69.8 tok/s |
| English, 2048 generated tokens | 82.4–83.9 tok/s |
| Chinese prose, 1024 generated tokens | 48.1 tok/s |
| ~14.5K-token prefill, llama.cpp | 335–699 tok/s |
| ~14.5K-token prefill, vLLM FP8 | ~908 tok/s |

## Reproduce

The patch targets llama.cpp commit `9723942`:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 9723942
git am /path/to/rtx-2080ti-llm-optimization/patches/llama.cpp/0001-ggml-cuda-add-NVLink-P2P-allreduce-path-for-2-GPU-t.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release -j
```

Run a single benchmark:

```bash
LLAMA_SERVER=/opt/llama.cpp/build/bin/llama-server \
MODEL=/models/Qwen3.8-27B-Q4_K_M.gguf \
CUDA_VISIBLE_DEVICES=0,1 \
./scripts/benchmark_llama.sh two-gpu-mtp3 -- \
  -sm tensor -ts 1,1 -fa on --jinja -c 262144 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

See the [Chinese technical report](docs/REPORT.zh-CN.md), [results](RESULTS.md), and [patch notes](patches/llama.cpp/README.md) for the full methodology and limitations.

The 2026-09-05 follow-up is documented in [CPU orchestration](docs/CPU-ORCHESTRATION.zh-CN.md), [CUDA Graph P0](docs/CUDA-GRAPH-P0.zh-CN.md), [CUDA Graph P1](docs/CUDA-GRAPH-P1.zh-CN.md), and the English [Nsight Systems CUDA Graph note](docs/NSYS-CUDA-GRAPH-BUSYTIME.md). The key measurement correction is that Nsight Systems' kernel table omits kernels executed inside CUDA Graph replay; GPU busy time must merge kernel and graph-trace intervals.

## Scope and licensing

Numbers are specific to the documented host, model, prompt, and pinned software versions. The vLLM comparison uses [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) as an external dependency; its source is not redistributed here.

Original material in this repository is MIT licensed. The patch is derived from MIT-licensed [llama.cpp](https://github.com/ggml-org/llama.cpp) source context.
