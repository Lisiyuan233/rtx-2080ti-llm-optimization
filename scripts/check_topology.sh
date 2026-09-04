#!/usr/bin/env bash
set -euo pipefail

section() {
  printf '\n## %s\n' "$1"
}

section "Timestamp"
date --iso-8601=seconds

section "Kernel"
uname -a

section "CPU"
lscpu | grep -E '^(Architecture|CPU\(s\)|Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)):' || true

section "Memory"
free -h

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

section "Driver and GPUs"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,temperature.gpu,power.limit --format=csv

section "GPU topology"
nvidia-smi topo -m

section "NVLink status"
nvidia-smi nvlink --status || true

section "CUDA peer-access reminder"
echo "nvidia-smi topology describes the physical route; validate cudaDeviceCanAccessPeer in the runtime as well."
