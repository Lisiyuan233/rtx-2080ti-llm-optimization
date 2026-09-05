# CUDA Graph P0：覆盖率与计量纠错

日期：2026-09-05

## 结论

CUDA Graph 在当前 llama.cpp decode 路径中已经接近饱和：稳态 replay 覆盖率为 97.8%，graph on/off 的端到端差异约 1.4%。继续扩大 capture、构建跨设备 mega-graph 或专门消除 host rendezvous 的收益上限不足，生产不做改动。

本阶段最重要的产出是修正性能计量方法。Nsight Systems 的普通 CUDA kernel 表不会列出 graph replay 内执行的 kernel；必须合并 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` 的执行区间。只看 kernel 表会把大量真实 GPU 工作误判成 idle。

## ABAB×5

| 配置 | 英文 2048 token | 中文正文 1024 token | 内容 hash |
|---|---:|---:|---|
| graph on | 82.73 tok/s | 50.18 tok/s | 一致 |
| graph off | 81.59 tok/s | 49.50 tok/s | 一致 |

五组配对都同向，`off/on ≈ 0.9862`。Graph 是正确且有用的，但它在本机上的全部用户可见价值只有约 1.4%。逐轮脱敏数据见 [`results/cuda-graph-on-off-abab.csv`](../results/cuda-graph-on-off-abab.csv)。

## Graph 计数器

在约 1000 个 speculative pass 的 trace 中，稳态 512-call 采样窗为 512 replay、0 capture、0 direct。全程 compute 事件包含 137826 次 replay，占 97.8%；其余主要是启动、prefill 和 draft shape 交替导致的 warmup/direct 路径。

平均每次捕获约 28 个节点，说明 graph 子图本身很小。即使把边界进一步合并，剩余可消除的 launch-gap 也很有限。

## 修正后的每-pass 时间线

| 分量 | 约 ms/pass | 解释 |
|---|---:|---|
| 整个 speculative pass | 45.2 | 墙钟预算 |
| target decode / sync | 35.5 | 宿主主要等待真实 GPU 工作 |
| GPU0 graph execution | 33.3 | 来自 graph-trace 表 |
| P2P all-reduce kernel | 1.7 | 约 137 次，每次约 13.3µs |
| GPU0 idle | 11.0 | 主要邻接 draft 路径 |
| `spec_draft` | 7.5 | GPU 约 1ms，其余为宿主固定成本 |

GPU0 合并后的 busy 占比约 76.4%，而 kernel-only 口径只有约 4.6%。旧的“GPU 实际只忙约 17ms”结论因此作废。详细 SQL 与 interval-union 方法见 [英文方法论笔记](NSYS-CUDA-GRAPH-BUSYTIME.md)。

## 方向判定

- 扩大 capture：稳态已有 97.8% replay，不投入；
- mega-graph / 消 host rendezvous：graph on/off 已给出约 1.4% 总上限，不投入；
- 深挖 P2P all-reduce：当前约 1.7ms/pass，且多数等待与 GPU 工作重叠，不投入；
- 检查 draft shape：GPU idle 大气泡集中在 draft 邻域，进入 P1 做最小机制实验。

P1 最终成功修复了 replay 机制，但端到端只提升 0.3–0.4%，见 [CUDA Graph P1](CUDA-GRAPH-P1.zh-CN.md)。
