# Measuring GPU busy time under CUDA Graphs: Nsight Systems' kernel table alone will mislead you

A methodology note from a llama.cpp tensor-parallel decode study. The numbers are specific to our setup, but the measurement pitfall is general.

## TL;DR

When CUDA Graph replay is active, kernels executed inside a graph replay do not appear in Nsight Systems' ordinary CUDA kernel table (`CUPTI_ACTIVITY_KIND_KERNEL` in the SQLite export). With graph tracing enabled, replay executions are recorded in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`.

If GPU busy time or gap analysis uses the kernel table alone, it can greatly overestimate idle time and point optimization work in the wrong direction.

| Steady-state decode metric | Kernel table only | Kernel + graph trace merged |
|---|---:|---:|
| GPU0 busy | 4.6% | **76.4%** |
| GPU0 idle | 95.4% | **23.6%** |
| Implied diagnosis | Host sync or launch overhead dominates | The GPU is doing real work |

The naive reading suggested building multi-GPU mega-graphs to eliminate host synchronization. The corrected reading showed that synchronization overlapped real GPU execution and that the remaining idle time was concentrated around speculative-draft orchestration.

## What happened

Inference engines such as llama.cpp capture per-iteration kernel sequences into CUDA Graphs and replay them. During replay, CUPTI does not necessarily emit an ordinary kernel-activity row for each kernel. Instead, the trace contains graph execution intervals with start/end timestamps and graph IDs.

Our workload replayed roughly 130 small graphs per decode step and used ordinary kernels for the all-reduce path. The kernel table therefore contained the non-graph kernels but omitted most compute. A kernel-only gap analysis faithfully reported that missing compute as “idle.”

## Correct busy-time calculation

Collect intervals from both activity tables:

```sql
SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL
UNION ALL
SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
ORDER BY deviceId, start;
```

Then group by device, merge overlapping intervals, and sum their union:

```python
def busy_fraction(intervals, wall):
    intervals = sorted(intervals)
    total = 0
    current_start = current_end = None

    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end

    if current_start is not None:
        total += current_end - current_start

    return total / wall
```

## Practical checks

- Enable execution-level graph tracing, for example `--cuda-graph-trace=node` on Nsight Systems versions that support it.
- Inspect the exported database schema rather than assuming table availability or column names across tool versions.
- Merge intervals separately for every `deviceId`; executions on different GPUs legitimately overlap.
- Calculate true idle as wall time minus the merged busy-time union.
- Attribute idle by inspecting the operations bordering each merged gap, not by counting missing ordinary kernel rows.
- Validate the diagnosis with an end-to-end graph on/off A/B. Profiler counters do not establish user-visible speedup by themselves.

## Takeaway

If a CUDA-Graph-enabled profile appears to show a mostly idle GPU based only on the kernel table, merge graph execution intervals before changing the program. In this study that correction stopped an unjustified synchronization rewrite and redirected the investigation to the actual draft-side host window.
