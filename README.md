# RTX 2080 Ti LLM 推理优化实录

[English](README.en.md) | 简体中文

这是一套面向 **RTX 2080 Ti 22GB / Turing SM75** 的实测型 LLM 推理优化项目。它记录了在双路老 Xeon、三张 2080 Ti（只有其中两张以 NVLink 相连）的机器上，如何把 Qwen3.8-27B 的单请求生成速度从约 **41.8 tok/s 提升到 69–84 tok/s**，并提供可复现脚本、部署配置和 llama.cpp 双卡 P2P all-reduce 补丁。

项目的核心不是“堆更多显卡”，而是先识别拓扑、投机解码和 CPU 编排的真实瓶颈，再用数据排除无效路线。

## 关键结论

| 调整 | 实测结果 | 决策 |
|---|---:|---|
| 三卡均匀 tensor split → 双卡 NVLink 对 | 41.8 → 66.2–66.9 tok/s | 使用 GPU0+1，闲置无桥 GPU |
| MTP draft 3 → 4/5/6 | 66.9 → 49.3/43.4/更低 | 保持 draft=3 |
| F16 KV → Q8_0 KV | 66.9 → 53.0 tok/s | 保持 F16 KV |
| DFlash2 BF16 drafter | 67.1 → 52.9 tok/s | 2080 Ti 上不启用 |
| vLLM FP8 MTP3 | 解码比 llama.cpp 慢 30–48%，预填充快 30%+ | 老 CPU 主机保留 llama.cpp |
| 主机中转 all-reduce → P2P/NVLink | 英文短生成 +4–5%，长生成 +1.5–3% | 补丁投入生产 |

最终配置在同一台机器上的代表性结果：

- 英文 256-token 生成：**69.2–69.8 tok/s**（P2P 补丁后）
- 英文 2048-token 生成：**82.4–83.9 tok/s**
- 中文正文 1024-token 生成：**48.1 tok/s**
- 约 14.5K-token 预填充：llama.cpp **335–699 tok/s**；vLLM FP8 约 **908 tok/s**

完整实验条件、对照表和限制见 [中文技术报告](docs/REPORT.zh-CN.md) 与 [结果表](RESULTS.md)。这些数字只代表本项目的硬件、模型和软件版本，不应直接外推到其他模型或平台。

## 测试平台

- GPU：3× RTX 2080 Ti 22GB；GPU0↔GPU1 有 NVLink，GPU2 无桥
- CPU：2× Xeon E5-2680 v2
- 系统：Ubuntu 24.04，CUDA / SM75
- 生产引擎：llama.cpp，基线提交 `9723942`
- 模型：Qwen3.8-27B Q4_K_M GGUF，内嵌 MTP 头
- 上下文：262,144，F16 KV，tensor split 1,1

## 快速复现

### 1. 检查拓扑

```bash
./scripts/check_topology.sh
```

只有 `nvidia-smi topo -m` 显示两张目标卡之间存在 NVLink（或确认可用的双向 PCIe P2P）时，才应期待补丁走直接 peer 路径。

### 2. 应用 llama.cpp 补丁

补丁固定基于 `9723942`，建议先在该提交上复现，再自行移植到新版本：

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 9723942
git am /path/to/rtx-2080ti-llm-optimization/patches/llama.cpp/0001-ggml-cuda-add-NVLink-P2P-allreduce-path-for-2-GPU-t.patch

cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release -j
```

补丁会在双向 peer access 可用时使用显存 staging ring 和设备端 arrival flag；不满足条件时自动回退到原有主机内存中转路径。设计说明见 [补丁文档](patches/llama.cpp/README.md)。

### 3. 启动服务

先把 [systemd 示例](configs/llama-server.service.example) 中的运行用户、工作目录、二进制和模型路径改成自己的配置：

```bash
sudo cp configs/llama-server.service.example /etc/systemd/system/llama-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
```

示例默认仅监听 `127.0.0.1`。确需局域网访问时再显式修改，并自行配置访问控制。

### 4. 跑基准矩阵

不要在生产服务仍占用目标 GPU 时启动测试进程。单项测试：

```bash
LLAMA_SERVER=/opt/llama.cpp/build/bin/llama-server \
MODEL=/models/Qwen3.8-27B-Q4_K_M.gguf \
CUDA_VISIBLE_DEVICES=0,1 \
./scripts/benchmark_llama.sh two-gpu-mtp3 -- \
  -sm tensor -ts 1,1 -fa on --jinja -c 262144 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

核心参数扫描：

```bash
LLAMA_SERVER=/opt/llama.cpp/build/bin/llama-server \
MODEL=/models/Qwen3.8-27B-Q4_K_M.gguf \
./scripts/reproduce_core_matrix.sh
```

脚本不会停止或修改系统服务，也不包含密码。原始响应和服务日志写入 `results/raw/`（默认不纳入 Git）。

vLLM 或其他 OpenAI 兼容服务可用：

```bash
python3 scripts/benchmark_openai.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model your-served-model \
  --prompt-file prompts/english-token-stream.txt \
  --tokens 256 --repeat 3
```

## 推荐生产参数

```text
CUDA_VISIBLE_DEVICES=0,1
GGML_CUDA_ALLREDUCE=internal
-ngl 999 -sm tensor -ts 1,1 -fa on --jinja -c 262144
--spec-type draft-mtp --spec-draft-n-max 3
```

这组参数只适用于带内嵌 MTP 头的相应 GGUF。没有 MTP 头的模型应去掉投机解码参数。

## 仓库结构

```text
configs/                 安全默认的 systemd 示例
docs/                    完整技术报告与复现方法
patches/llama.cpp/       P2P all-reduce 格式化补丁及说明
prompts/                 固定英文/中文测试提示词
results/                 已整理的测试结果
scripts/                 拓扑检查和基准脚本
```

## vLLM 对照说明

本项目测试的是第三方 [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) `v0.1.17`（基于 vLLM `0.21.0`），没有复制或重新发布其源码。它在现代 CPU 平台上有很强的 2080 Ti 路线，但本机的老 Xeon 控制面开销使单请求解码明显落后；其长提示词预填充仍然更快。

## 许可与上游

本仓库原创脚本和文档采用 [MIT License](LICENSE)。llama.cpp 补丁基于 MIT 许可的 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，补丁文件包含必要的上游上下文。vLLM 及其衍生项目仅作为独立依赖和对照对象引用，各自遵循其原许可证。
