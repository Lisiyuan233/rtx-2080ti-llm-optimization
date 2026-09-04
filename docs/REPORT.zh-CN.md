# 双 RTX 2080 Ti 上的 Qwen3.8-27B 推理提速报告

## 摘要

本项目在一台三卡 RTX 2080 Ti 22GB 主机上，围绕 Qwen3.8-27B 的单请求生成速度做了逐项隔离测试。机器上只有 GPU0 与 GPU1 通过 NVLink 相连，GPU2 没有桥；CPU 是双路 Xeon E5-2680 v2。原三卡均匀 tensor split 配置在英文短生成上约 41.8 tok/s。

最大的收益来自拓扑选择，而不是新增算子：改用 GPU0+1 的 NVLink 对后达到 66.2–66.9 tok/s。随后通过 nsys 剖析定位出 llama.cpp 内部 all-reduce 仍走 pin 住的主机内存中转，于是实现双卡直接 P2P staging 路径，最终短英文生成达到 69.2–69.8 tok/s，长英文生成达到 82.4–83.9 tok/s。

与此同时，MTP draft 长度、KV 精度、DFlash2、ngram、三卡加权切分和 vLLM FP8 路线都做了对照。许多在更新 GPU 或更新 CPU 上合理的经验，在这台 Turing + 老 Xeon 主机上并不成立。

## 1. 实验目标与约束

目标是在以下条件不变的前提下提高单请求生成速度：

- 保留 262,144 token 的最大上下文配置；
- 不降低模型输出质量；
- 继续使用现有 OpenAI 兼容服务接口；
- 任何实验都必须能快速回退；
- 不把“能启动”当成优化成功，必须用同一提示词和温度做墙钟或服务端 timing 对照。

测试模型是 Qwen3.8-27B 的 Q4_K_M GGUF，包含内嵌 MTP 头。除 vLLM 对照外，测试使用同一模型文件。vLLM 使用官方 FP8 safetensors，因此该组只代表两套可部署方案的性能对照，并非严格的引擎微基准。

## 2. 硬件与软件

| 项目 | 配置 |
|---|---|
| GPU | 3× RTX 2080 Ti 22GB（SM75） |
| GPU 互联 | GPU0↔GPU1 有 NVLink；GPU2 无桥 |
| CPU | 2× Xeon E5-2680 v2（Ivy Bridge） |
| 内存 | 256GB |
| 系统 | Ubuntu 24.04，Linux 6.8 |
| llama.cpp 基线 | `9723942` |
| llama.cpp P2P 补丁 | `b4e45e0`，本仓库提供 format-patch |
| vLLM 对照 | vLLM-2080Ti-Definitive `v0.1.17`，base vLLM `0.21.0` |

需要特别说明：一张卡所在根端口只支持 PCIe Gen2。因为最终数据路径主要使用 GPU0↔GPU1 的 NVLink，双卡方案仍然获胜；在其他主板上结果可能不同。

## 3. 方法

### 3.1 单变量原则

每轮启动一个独立的 `llama-server`，固定：

- 提示词；
- `temperature=0`；
- 输出 token 数；
- 上下文与 offload 参数；
- 测试期间的 GPU 集合。

改变的变量包括 GPU 拓扑、split mode、tensor split 权重、MTP draft 长度、p-min、KV 类型、投机解码实现和 all-reduce 路径。结果优先读取 llama.cpp 响应中的 `timings.prompt_per_second` 与 `timings.predicted_per_second`。vLLM 使用同一提示词并以墙钟除以返回的 completion token 数。

仓库中的 [英文固定提示词](../prompts/english-token-stream.txt) 是原测试输入；它由大量中性英文 token 构成，目的是减少内容安全、事实性和随机语义对接受率的影响。[中文提示词](../prompts/chinese-prose.txt) 用于更贴近实际写作负载的长生成对照。

### 3.2 结果解释边界

短测试容易受启动预热、CUDAGraph、频率和 MTP 接受率影响，因此最终候选又跑了 2048-token 长生成、中文 1024-token 正文和长提示词预填充。所有结论都要求方向在多个负载上可以解释，而不是只追逐一次最高值。

## 4. 第一阶段：拓扑比卡数重要

原配置把模型均匀切到三张卡。GPU2 没有 NVLink，tensor parallel 的逐层同步必须经过较慢路径，抵消了第三张卡增加的计算和显存带宽。

| 配置 | pp (tok/s) | tg (tok/s) |
|---|---:|---:|
| 三卡均匀 tensor split | 214 | 41.8 |
| 三卡 0.4/0.4/0.2 | 246 | 53.4 |
| 三卡加权，关闭 all-reduce 环境设置 | 234 | 52.5 |
| 双卡 GPU0+1，NVLink，1/1 | 560 | 66.2 |
| 三卡 layer split | 553 | 34.4 |
| 单卡 GPU1 | 533 | 35.3 |

结论是明确的：该机器上双卡 NVLink 对比三卡均匀切分的解码吞吐高约 59%。第三张卡适合承担独立工作负载，而不是加入同一个低并发 tensor-parallel 实例。

## 5. 第二阶段：投机解码和 KV

### 5.1 MTP draft 长度

三卡上 MTP3 为 41.8 tok/s，关闭投机解码只有 25.6 tok/s，说明 MTP 贡献约 63%。但是增加 draft 长度并不会继续加速：三卡 MTP4/5/6 分别为 34.7/35.6/32.2 tok/s；双卡 MTP4/5 也只有 49.3/43.4 tok/s，而 MTP3 是 66.9 tok/s。

原因是更长草稿只有在额外 draft 成本小、接受率足够高时才划算。2080 Ti 上每多一步都有可见代价，固定更大的 `n_max` 反而拉长验证循环。

### 5.2 概率门控与 ngram

MTP4+p-min0.6、MTP6+p-min0.7、MTP3+p-min0.5 分别只有 29.3、27.0、33.5 tok/s。MTP3 叠加 ngram 为 66.6 tok/s，纯 ngram8 为 41.7 tok/s。该模型的内嵌 MTP3 已经是更合适的组合。

### 5.3 KV 精度

F16 KV 改为 Q8_0 后从约 66.9 降到 53.0 tok/s。量化 KV 节省显存，但量化/反量化成本和 tensor 模式兼容性使当前负载变慢。由于双 22GB 卡已经容纳 256K 配置，本项目选择 F16 KV。

### 5.4 DFlash2

上游修复 tensor split 崩溃后，DFlash2 已经可以运行，但 BF16 drafter 在 Turing 上太贵。同一代码树中 MTP3 为 67.1 tok/s，DFlash2 draft=3 为 52.9，draft=4 为 43.0。其在更新 GPU 上可能有收益，但不适合本项目硬件。

## 6. vLLM FP8 对照

对照运行时是 [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) `v0.1.17`。环境使用 CUDA 12.8、PyTorch 2.11 cu128、TP=2、FP16 KV 和 MTP3。

| 场景 | llama.cpp Q4_K_M | vLLM FP8 | vLLM 差距 |
|---|---:|---:|---:|
| 英文 256-token 生成 | 66.6 | 34.5 | -48% |
| 英文 2048-token 生成 | 81.3 | 54.0 | -34% |
| 中文 1024-token 生成 | 47.5 | 33.2 | -30% |
| 约 14.5K-token 预填充 | 335–699 | ~908 | +30% 以上 |

结果符合预先风险判断：vLLM 的 Python/服务控制面更依赖现代 CPU 单核性能，E5-2680 v2 放大了逐 token 编排成本；它的批处理和长提示词预填充路径仍有明显优势。生产场景以单请求长生成为主，因此保留 llama.cpp；如果工作负载转为批量预填充，应重新评估 vLLM。

## 7. nsys 剖析与算子取舍

在 2048-token 稳态 decode 中，MTP 接受率约 0.95，平均每个 speculative pass 产出约 3.85 token，周期约 49 ms。估算时间分布：

| 项 | 每 pass | 墙钟占比 |
|---|---:|---:|
| 62 层计算 graph | ~13.5 ms | ~28% |
| internal all-reduce | ~3.7 ms | ~7.5% |
| CPU 采样与 draft/verify 编排空隙 | ~16–29 ms | ~33–40% |
| MTP draft graph 等 | ~2–3 ms | ~5% |

16.8GB 权重除以两卡合计约 1232GB/s 的显存带宽，理论读一遍约 13.6 ms，与层计算实测非常接近。权重 GEMM 已贴近带宽 roofline，自写 GEMM 或反量化核没有足够空间。真正的大头是 CPU 编排，但那需要更深的 llama.cpp 调度重构。

all-reduce 占比虽小，却是一个边界清楚、可验证的局部优化。原 internal 路径面向没有 P2P 的机器，写入 pin 住的主机内存、用主机 flag 握手，再从主机读取对端数据；在已经有 NVLink 的两张卡上，这会绕远路。

## 8. P2P all-reduce 补丁

补丁新增双卡直接 peer 路径：

1. 每个 rank 把 partial cast 到本地显存 staging ring；
2. 每个 block 写入本地设备 arrival token；
3. 对端通过 peer mapping 轮询 arrival token；
4. token 到达后经 P2P/NVLink 读取对端 staging 并求和；
5. slot 轮换避免下一次调用覆盖尚未消费的数据。

初始化时会双向调用 `cudaDeviceCanAccessPeer`。任何方向不可用、卡数不等于 2 或 peer enable 失败时，都回退原有 host-staged 路径。两侧保持相同 wire-type 舍入；温度 0 下 2048-token 输出与旧路径逐字节一致。

补丁前后的同日对照：

| 场景 | host-staged | P2P | 提升 |
|---|---:|---:|---:|
| 英文 256 token | 66.6–67.1 | 69.2–69.8 | +4–5% |
| 英文 2048 token | 81.3 | 82.4–83.9 | +1.5–3% |
| 中文正文 1024 token | 47.5 | 48.1 | +1.3% |

连续三轮 2048-token 生成和中文正文测试未发现异常。补丁只固定验证在 llama.cpp `9723942`；上游文件继续演进时应 rebase 并重新跑位精确与回退测试。

## 9. 最终建议

对类似“多张旧卡、只有部分卡有高速互联”的机器：

1. 先画清 GPU 拓扑并测两卡/三卡，而不是默认卡越多越快；
2. 固定提示词逐项扫描 MTP draft，别照搬其他架构的最优值；
3. 显存够用时比较 F16 与量化 KV 的实际吞吐；
4. 把预填充与解码分开看，vLLM 和 llama.cpp 可能各胜一段；
5. 用 profiler 判断是显存带宽、通信还是 CPU 空隙，再决定是否写算子；
6. 发布结果时保留模型格式、CPU、上下文、接受率和测量口径，避免只报一个峰值。

本项目最终生产路线是双卡 NVLink、F16 KV、MTP3、llama.cpp internal all-reduce 加 P2P 补丁。无桥的第三张 GPU 留给独立的图像/视频生成任务，比加入同一推理实例更有价值。
