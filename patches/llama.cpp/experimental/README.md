# Experimental CUDA Graph shape-key patch

This patch is a minimal, independently generated experiment against llama.cpp commit `9723942`:

```bash
git checkout 9723942
git am /path/to/rtx-2080ti-llm-optimization/patches/llama.cpp/experimental/0001-ggml-cuda-bucket-graph-cache-by-leading-shape.patch
```

It adds the opt-in environment switch:

```bash
GGML_CUDA_SHAPE_KEYS=1
```

The default is off, so applying the patch alone does not change graph-cache behavior.

## Why it exists

An MTP nextn subgraph alternated between `n_tokens=4` catch-up calls and three `n_tokens=1` draft calls. The topology-only graph key put both shapes in one cache entry. Because the entry held one property/warmup state and required two consecutive identical property sets, the repeating `4,1,1,1` sequence never warmed up.

The patch includes the leading token shape in the cache key while retaining full per-call node-property validation. In the measured workload, draft replay changed from none to 506/508 calls, with byte-identical temperature-0 output in all six comparisons.

## Performance decision

| Workload | Enabled | Disabled | Delta |
|---|---:|---:|---:|
| English, 2048 generated tokens | 82.76 tok/s | 82.51 tok/s | +0.30% |
| Chinese prose, 1024 generated tokens | 50.29 tok/s | 50.08 tok/s | +0.42% |

The cleanest first round showed +1.24% and +0.88%, respectively, while both configurations declined with thermal/run-order drift. The `spec_draft` window remained about 7.5ms. The mechanism is valid, but the end-to-end gain is below this project's 3% deployment threshold, so it was not enabled in production.

The patch is independent of the P2P all-reduce patch because they touch different source files. Its format-patch commit ID is `f29066d`; the published file's SHA-256 is `6f18e424ee000909a00beea3bb813f004d0016902ed0de131a00e9342f8b247a` (the author email in the public patch header is intentionally anonymized).
