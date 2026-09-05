# CUDA Graph P1：draft shape 分桶实验

日期：2026-09-05

## 结论

P1 找到了 draft 子图永久 direct evaluation 的具体根因，并用最小、环境变量门控的改动修复了机制。但端到端 ABAB 只有英文 +0.30%、中文 +0.42%，低于 3% 上线门槛，因此不部署，实验补丁默认关闭。

## 两个高频 shape

对 draft 侧约 3840 次 decode 的统计恰好得到两个高频 token shape：

| 类型 | `n_tokens` | 次数 | 含义 |
|---|---:|---:|---|
| draft step | 1 | 2874 | 每个目标 pass 三步 |
| catch-up | 4 | 958 | `n_draft_max + 1` |

其余 shape 都是 prefill 或 warmup 边缘的一次性输入。

## 为什么同 shape 仍不命中

MTP nextn 层子图同时服务 catch-up `nt=4` 和 draft step `nt=1`。两者拓扑相同，所以原实现给出同一个 graph cache key。每个 key 只保存一份 `node_props` 和 warmup 状态，并要求连续两次属性相同才进入稳定 replay。

实际输入顺序反复为 `4,1,1,1,4,...`。`node_props` 在两个 shape 之间交替，warmup 每轮都被重置，导致整个 draft 路径永久退化成逐 kernel direct evaluation。reset 的首个差异字段也都指向 token 维及其派生 stride/source shape，形成完整旁证。

## 最小修复

实验开关 `GGML_CUDA_SHAPE_KEYS=1` 把 leading token shape hash 进 cache key，让 `nt=1` 与 `nt=4` 分别 warmup。同时，完整 `node_props` 比较仍在每次调用时执行，避免错误配对。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| draft bucket replay | 0 | 506 / 508 calls |
| warmup reset | 870 | 303 |
| direct evaluation | 2352 | 1779 |
| 温度 0 内容 hash | 生产参照 | 6/6 逐字节一致 |
| `spec_draft` p50 | 7.5ms | 7.5ms |

机制成功，但 draft 墙钟没有下降。原先看到的 CUDA API 发射碎片是 direct evaluation 的症状，不是窗口的主要限制。

## 诚实的 ABAB×3

| 场景 | shape keys on | shape keys off | 差值 |
|---|---:|---:|---:|
| 英文 2048 token | 82.76 tok/s | 82.51 tok/s | +0.30% |
| 中文正文 1024 token | 50.29 tok/s | 50.08 tok/s | +0.42% |

两组配置都随轮次升温而下降。热状态最干净的第一轮差值为英文 +1.24%、中文 +0.88%，仍低于 3% 门槛。不能用单次 trace 中接近 84 tok/s 的峰值替代交错对照均值。逐轮数据见 [`results/cuda-graph-shape-key-abab.csv`](../results/cuda-graph-shape-key-abab.csv)。

## 决策与后续边界

- 生产不设置 `GGML_CUDA_SHAPE_KEYS`，保持默认关闭；
- 保留[独立 format-patch](../patches/llama.cpp/experimental/README.md)，便于上游讨论和未来平台复测；
- 不再继续 shape bucket、graph capture 或 mega-graph 方向；
- 唯一仍有量级依据的候选是 draft 链的宿主固定成本：约 7.5ms/pass 中 GPU 仅约 1ms，其余涉及三步 CPU sampling 和每步构建/调度开销。

把 draft sampling 下放 GPU 或合并每步宿主工作，目标是把 `spec_draft` 降到不高于 5ms，理论上约对应 +3–6%。这是下一阶段假设，不是已经取得的优化结果。
