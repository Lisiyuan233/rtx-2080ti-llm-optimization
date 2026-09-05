# llama.cpp CPU 编排优化结果

日期：2026-09-05

## 结论

NUMA、线程池、poll、HTTP 线程、governor、控制线程绑核和 CUDA wait mode 都没有达到预先设定的 3% 上线门槛。生产服务保持原参数，不部署任何 CPU 编排改动。

这是一项有价值的负结果：40 多组配置经过五轮随机/交错测试后全部收敛到默认值附近，说明继续扫同类参数的预期回报已经很低。

## 测试范围

- NUMA0 本地核、跨 socket 分布与负对照绑核；
- target/draft 生成线程与 batch 线程；
- `poll`、batch poll、draft poll 及其组合；
- HTTP worker 数量与 4 并发短请求；
- `schedutil` 与 `performance` governor；
- 源码级控制线程 affinity；
- CUDA `spin`、`yield` 与默认 wait mode。

测试始终使用同一双卡 NVLink 拓扑、MTP3、F16 KV、固定提示词和温度 0。候选先在随机五轮矩阵中筛选，再以交错 ABAB 确认，避免把热漂移或运行顺序误判为收益。

## 关键结果

NUMA0 本地核在初筛中约快 0.3%，但没有在最终确认中保持可部署收益。线程、poll、HTTP 和 governor 家族均与默认值打平。最终默认配置与候选配置对照如下：

| 场景 | 默认 | 候选 | 差值 | 判定 |
|---|---:|---:|---:|---|
| 英文 2048 token | 81.53 tok/s | 81.62 tok/s | +0.11% | 不上线 |
| 中文正文 1024 token | 49.63 tok/s | 49.66 tok/s | +0.06% | 不上线 |

两组的 TPOT 与内容 hash 一致，变异系数都低于 0.5%。候选配置还进行了 30 分钟中文长稳测试：86 个请求、零错误，前半 49.49 tok/s、后半 49.47 tok/s，漂移 -0.04%。它稳定，但不更快。

源码级实验同样给出负结果：控制线程固定到不同核的差异低于 0.2%，CUDA 默认 wait mode 略优于 `spin` 和 `yield`。

## 与 CUDA Graph P0 的关系

早期仅看 Nsight Systems 普通 kernel 表时，曾把 target decode 同步窗口解释成大段 CPU 空洞。后续 [CUDA Graph P0](CUDA-GRAPH-P0.zh-CN.md) 纠正了这个口径：graph replay 内核位于 graph-trace 表，合并后 GPU 在该窗口内实际约忙 35.5ms/pass。CPU 编排实验的负结果因此有了更直接的解释——宿主大多是在等待真正执行中的 GPU，而不是被简单绑核或 poll 参数拖慢。

这里仍需区分 target sampling 与 draft sampling。CPU 编排阶段否定的是 target 路径采样作为主要瓶颈；P1 后发现约 7.5ms 的 draft 窗口具有不同的三步固定宿主成本，仍可作为独立研究假设，但尚未证明收益。

## 运维决策

- 不创建 systemd drop-in，不改变 governor，不部署实验环境变量；
- 保留默认 llama.cpp 线程和 wait 策略；
- 未来只有在 CPU、驱动或推理引擎发生明显变化时，才值得重跑这套矩阵；
- 任何新候选仍需同时通过温度 0 输出一致性、稳定性和至少 3% 的端到端收益门槛。
