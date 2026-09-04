# llama.cpp two-GPU P2P all-reduce patch

## Version

- Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Tested base commit: `9723942`
- Original local commit: `b4e45e05e0edee0b32875a70b4e451eb2b4029a5`
- Touched file: `ggml/src/ggml-cuda/allreduce.cu`
- Target path: internal all-reduce with exactly two CUDA backends

## Why it exists

At the pinned revision, the internal all-reduce path stages partial values in pinned host memory and coordinates ranks with host-visible flags. That fallback is useful on systems without GPU peer access, but it causes avoidable PCIe round trips when the selected two GPUs can directly access one another over NVLink or PCIe P2P.

The patch adds a direct peer path:

- per-rank device staging rings;
- per-block device arrival tokens;
- peer reads after the matching token is visible;
- rotating slots to avoid overwrite between calls;
- bidirectional `cudaDeviceCanAccessPeer` probing;
- automatic fallback to the original host-staged implementation.

It preserves the same wire-type conversion on both paths. A temperature-0 2048-token response was byte-identical in the tested build.

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 9723942
git am /path/to/0001-ggml-cuda-add-NVLink-P2P-allreduce-path-for-2-GPU-t.patch
```

If you do not want to preserve commit metadata, use `git apply` instead. Newer llama.cpp revisions may require a manual rebase because `allreduce.cu` changes over time.

## Enable

Select exactly two peer-capable GPUs and the internal all-reduce implementation:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export GGML_CUDA_ALLREDUCE=internal
```

The direct path is selected only when both devices report peer access and peer mapping can be enabled. Otherwise it falls back automatically.

## Observed gain

On dual modified RTX 2080 Ti 22GB cards connected by NVLink:

| Workload | Before | After | Gain |
|---|---:|---:|---:|
| English, 256 generated tokens | 66.6–67.1 tok/s | 69.2–69.8 tok/s | +4–5% |
| English, 2048 generated tokens | 81.3 tok/s | 82.4–83.9 tok/s | +1.5–3% |
| Chinese prose, 1024 generated tokens | 47.5 tok/s | 48.1 tok/s | +1.3% |

Results depend on topology, message size, CPU scheduling, model, and speculative acceptance rate. A PCIe-P2P system may behave differently from NVLink.
