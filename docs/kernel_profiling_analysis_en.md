# GEMM Kernel Profiling Analysis

## 1. Executive Summary

This report consolidates the profiling results for the GEMM optimization path from `kernel1` to the autotuned kernel. The main result is:

- At matrix size 4096, performance improves from **941 GFLOPS** (`kernel1`, naive) to **54,228 GFLOPS** (warp autotuned), a **57.6x speedup** based on the displayed throughput values.
- The optimization path is not a linear reduction of one bottleneck. The bottleneck moves through several parts of the GPU:

```text
uncoalesced global access
  -> excessive global-memory instructions
  -> shared-memory pipeline pressure
  -> insufficient data reuse
  -> register pressure / low occupancy
  -> shared-memory bank conflicts
  -> global-load latency inside an otherwise compute-heavy kernel
  -> parameter balance found by autotuning
```

- The final 17% improvement from hand-tuned `kernel8` to the autotuned kernel does **not** come from higher occupancy. Theoretical occupancy remains 33.33%. It comes primarily from raising FMA utilization and reducing Long Scoreboard stalls.

The key engineering lesson is that optimization decisions should follow the current limiting resource. Coalescing, shared-memory tiling, register tiling, vectorization, warp tiling, and autotuning solve different problems; applying them without checking profiler evidence can produce little or no benefit.

---

## 2. Scope and Interpretation

The benchmark table covers square GEMM sizes from 128 to 4096. Detailed Nsight Compute comparisons are used to explain the transitions between kernels and are most representative of the large-matrix regime.

Two cautions apply when reading the numbers:

1. Benchmark GFLOPS and profiler duration can come from different measurement runs and incur different profiling overhead. They should be used for trend analysis rather than recombined into exact timing identities.
2. A high `Memory Throughput` or `L1/TEX` percentage does not necessarily mean DRAM bandwidth saturation. These aggregate metrics also reflect on-chip paths. Stall reasons such as LG Throttle, MIO Throttle, Long Scoreboard, and Short Scoreboard are more useful for locating the actual wait source.

---

## 3. End-to-End Performance

### 3.1 Throughput by matrix size

| Kernel | Optimization | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Naive | 82 | 348 | 708 | 813 | 912 | 941 |
| 2 | Coalesced access | 503 | 2,077 | 5,017 | 6,026 | 6,774 | 6,027 |
| 3 | Shared-memory tiling | 668 | 2,729 | 6,890 | 8,459 | 9,579 | 9,266 |
| 4 | 1D block/register tiling | 406 | 1,812 | 7,701 | 20,702 | 26,454 | 28,506 |
| 5 | 2D block/register tiling | 157 | 655 | 2,912 | 12,345 | 33,557 | 36,675 |
| 6 | Vectorized access | 185 | 780 | 3,276 | 13,444 | 34,525 | 38,497 |
| 8 | Hand-tuned warp tiling | 203 | 862 | 3,878 | 16,915 | 46,224 | 46,336 |
| 7 | Tunable 2D, autotuned | 508 | 2,339 | 10,878 | 31,500 | 48,036 | 52,509 |
| 9 | Warp tiling, autotuned | 410 | 2,048 | 10,057 | 32,520 | 49,949 | **54,228** |

All values are GFLOPS.

### 3.2 Speedup at size 4096

| Kernel | GFLOPS | Speedup vs. naive | Increment vs. previous listed stage |
|---|---:|---:|---:|
| 1 naive | 941 | 1.0x | - |
| 2 coalesced | 6,027 | 6.4x | 6.40x |
| 3 shared memory | 9,266 | 9.9x | 1.54x |
| 4 1D tiling | 28,506 | 30.3x | 3.08x |
| 5 2D tiling | 36,675 | 39.0x | 1.29x |
| 6 vectorized | 38,497 | 40.9x | 1.05x |
| 8 warp tiling | 46,336 | 49.2x | 1.20x |
| 9 warp autotuned | 54,228 | **57.6x** | 1.17x |

The two autotuned families are alternative search paths, so `kernel7` is retained in the complete benchmark table but is not treated as a sequential step between kernels 6 and 8.

### 3.3 Size sensitivity

The more advanced tiled kernels are slower at small sizes because launch, synchronization, indexing, and tile setup costs are not sufficiently amortized. Their advantages become clear from size 1024 onward. A production implementation should therefore dispatch different kernels by shape instead of assuming the largest-matrix winner is universal.

---

## 4. Bottleneck Evolution

| Transition | Main change | Primary Metrics | Result | New limit / cost |
|---|---|---|---|---|
| 1 -> 2 | Make warp accesses coalesced | LG Throttle 119.57 -> 8.90; memory throughput 1.54 -> 10.11 GB/s | 6.4x at 4096 | Global/L1 load pipeline still costly |
| 2 -> 3 | Reuse global data through shared memory | Instructions 16.67B -> 5.60B; LG Throttle 8.90 -> 0.40 | 1.54x | MIO Throttle rises to 23.79 |
| 3 -> 4 | Each thread computes `TM=8` outputs | MIO 23.79 -> 6.53; FMA 7.83% -> 25.79% | 3.08x | Register dependency and remaining LSU pressure |
| 4 -> 5 | 2D `TM x TN = 8 x 8` register tile | LSU 82.13% -> 46.16%; FMA Heavy 19.94% -> 39.91% | 1.29x | 95 registers/thread; occupancy 33.33% |
| 5 -> 6 | `float4` vectorized loads/stores | Instructions 3.22B -> 2.85B | 1.05x | Occupancy and bank conflicts unchanged |
| 6 -> 8 | Warp tiling and transposed shared-memory layout | MIO nearly zero; Compute 49.81% > Memory 43.01% | 1.20x | 106 registers/thread; Long Scoreboard 1.64 |
| 8 -> 9 | Autotune `BK`, lane width, warp/thread tiles | FMA 46.19% -> 53.86%; Long Scoreboard 1.64 -> 0.48 | 1.17x | Occupancy remains register-limited |

---

## 5. Detailed Transition Analysis

### 5.1 Kernel 1 -> Kernel 2: global-memory coalescing

`kernel1` maps threads so that lanes in a warp access separated addresses. The problem is not simply “too much DRAM traffic”: the load/store path is unable to serve the fragmented requests efficiently.

| Metric | Kernel 1 | Kernel 2 | Interpretation |
|---|---:|---:|---|
| Memory utilization | 98.91% | 80.13% | Busy does not imply useful bandwidth |
| Compute utilization | 11.99% | 80.13% | Coalescing makes data arrive fast enough to feed compute |
| Memory throughput | 1.54 GB/s | 10.11 GB/s | 6.6x more useful transfer rate |
| IPC | 0.23 | 1.56 | Instruction issue recovers |
| SM Busy | 6.95% | 48.48% | More cycles perform useful work |
| LG Throttle | 119.57 | 8.90 cycles/issue | Global/local load queue pressure collapses |
| Warp cycles/issued instruction | 134.00 | 20.48 | The scheduler spends far less time waiting |

The same total instruction count is reported for these two kernels, so the gain comes from changing access efficiency rather than reducing work. Kernel 2 is still not fully optimized: LG Throttle remains 8.90 cycles/issue and accounts for a large fraction of warp stalls.

**Diagnosis:** coalescing removes the pathological access pattern and delivers the largest single early-stage gain, but every output still repeatedly fetches operands through the global/L1 path.

### 5.2 Kernel 2 -> Kernel 3: shared-memory tiling

Kernel 3 loads tiles of A and B into shared memory, then reuses them within a thread block.

| Metric | Kernel 2 | Kernel 3 | Change |
|---|---:|---:|---:|
| Duration | 25.70 ms | 17.07 ms | 1.50x faster |
| Instructions executed | 16.67B | 5.60B | -66% |
| LG Throttle | 8.90 | 0.40 | Global queue pressure removed |
| MIO Throttle | 0.03 | 23.79 | Shared-memory path becomes the bottleneck |
| Long Scoreboard | 3.36 | 6.68 | More visible wait on tile loads |
| Barrier | 0.00 | 2.96 | Synchronization cost appears |
| IPC | 1.56 | 0.79 | Lower per-cycle issue efficiency |
| FMA utilization | 17.64% | 7.83% | Lower percentage, despite lower total time |

The counterintuitive part is that IPC and FMA utilization fall while runtime improves. The instruction count drops to roughly one third, which more than compensates for the higher average cost per instruction.

**Diagnosis:** shared memory successfully reduces global-memory work, but each thread still computes only one C element. Shared-memory reads remain nearly proportional to FMA operations, moving the bottleneck from LG Throttle to MIO Throttle.

### 5.3 Kernel 3 -> Kernel 4: 1D register tiling

Kernel 4 makes each thread compute `TM=8` output elements. A value loaded from shared memory into a register can now feed eight FMAs.

| Metric | Kernel 3 | Kernel 4 | Change |
|---|---:|---:|---:|
| Duration | 17.07 ms | 5.51 ms | 3.10x faster |
| Instructions executed | 5.60B | 3.33B | -40% |
| MIO Throttle | 23.79 | 6.53 | Approximately one quarter |
| Warp cycles/issue | 40.46 | 21.11 | Approximately halved |
| IPC | 0.79 | 1.49 | Approximately doubled |
| FMA utilization | 7.83% | 25.79% | 3.3x |
| Registers/thread | 38 | 48 | Cost of register reuse |
| Theoretical occupancy | 66.67% | 66.67% | No occupancy loss yet |

Short Scoreboard rises from 0.43 to 2.47 cycles/issue because multiple FMAs depend on the same recently loaded register value. This is a much healthier trade than flooding the MIO path with repeated shared-memory loads.

**Diagnosis:** this is the first step that increases arithmetic intensity through register reuse instead of merely relocating traffic.

### 5.4 Kernel 4 -> Kernel 5: 2D register tiling

Kernel 5 expands the per-thread output from a 1D strip to an `8 x 8` tile. Both A and B operands are cached in register arrays, so `TM + TN` shared-memory reads feed `TM x TN` FMAs.

| Metric | Kernel 4 | Kernel 5 | Change |
|---|---:|---:|---:|
| Duration | 5.51 ms | 4.28 ms | 1.29x faster |
| LSU utilization | 82.13% | 46.16% | Shared/load-store pressure sharply reduced |
| FMA Heavy | 19.94% | 39.91% | Compute becomes dominant |
| IPC | 1.49 | 2.25 | Higher issue efficiency |
| Warp cycles/issue | 21.11 | 7.02 | Approximately one third |
| MIO Throttle | 6.53 | 0.89 | Mostly removed |
| Long Scoreboard | 5.49 | 1.31 | Mostly removed |
| Registers/thread | 48 | 95 | Nearly doubled |
| Theoretical occupancy | 66.67% | 33.33% | Halved |
| Active warps/scheduler | 7.88 | 3.95 | Halved |

This is the transition where compute work first overtakes load/store work, but it also creates the next major constraint. Sixty-four accumulators plus operand registers push register usage to 95 per thread, allowing only two blocks per SM in the profiled configuration.

**Diagnosis:** 2D tiling solves most shared-memory traffic stalls and improves each warp, but register pressure limits the number of resident warps available to hide latency.

### 5.5 Kernel 5 -> Kernel 6: vectorization

Kernel 6 replaces scalar transfers with `float4` operations for global-to-shared loads and C stores.

| Metric | Kernel 5 | Kernel 6 | Change |
|---|---:|---:|---:|
| Duration | 4.28 ms | 4.04 ms | 1.06x faster |
| Instructions executed | 3.22B | 2.85B | -11% |
| LSU utilization | 46.16% | 42.67% | Slight reduction |
| IPC | 2.25 | 2.02 | Slightly lower |
| Warp cycles/issue | 7.02 | 7.60 | Slightly higher |
| Registers/thread | 95 | 95 | Unchanged |
| Occupancy | 33.33% | 33.33% | Unchanged |
| Active warps/scheduler | 3.95 | 3.84 | Essentially unchanged |

Vectorization reduces instruction count, but it does not address the two important limits: register-constrained occupancy and an observed shared-memory bank-conflict pattern. Long Scoreboard, MIO Throttle, and Barrier stalls rise slightly, offsetting much of the instruction-count benefit.

**Diagnosis:** vectorization is valid but only incremental because the kernel is no longer primarily limited by the number of scalar memory instructions.

### 5.6 Kernel 6 -> Kernel 8: warp tiling and shared-memory layout

The hand-tuned warp kernel uses:

```text
BM=128, BN=128, BK=16
WM=64, WN=32
LaneSizeN=4
TM=8, TN=8
threads/block=128
```

Warp tiling assigns explicit warp-level output tiles and transposes the shared-memory layout of A. This changes the inner-loop A access from a strided column pattern with severe bank conflicts to a substantially friendlier row-oriented pattern.

| Metric | Kernel 6 | Kernel 8 | Change |
|---|---:|---:|---:|
| Duration | 4.04 ms | 3.20 ms | 1.26x faster |
| Compute throughput | 37.71% | 49.81% | Higher |
| Memory throughput | 46.98% | 43.01% | Lower |
| DRAM utilization | 6.70% | 6.05% | Essentially unchanged and low |
| LSU utilization | 42.67% | 26.14% | Much lower |
| FMA utilization | - | 46.19% | FMA now dominates |
| MIO Throttle | - | 0.03 | Effectively zero |
| LG Throttle | - | 0.00 | Zero |
| Registers/thread | 95 | 106 | Higher |
| Theoretical occupancy | 33.33% | 33.33% | Unchanged |
| Achieved occupancy | 33.55% | 32.58% | Slightly lower |

`Not Selected` becomes the largest sampled stall at 2.09 cycles/issue, while Long Scoreboard is 1.64. A high `Not Selected` value is not automatically a defect: it often means eligible warps exist and are waiting for scheduler selection rather than blocked on a resource dependency.

The important correction to the earlier expectation is that warp tiling does **not** reduce register pressure or raise occupancy here. Registers increase from 95 to 106 per thread, while occupancy remains 33.33%. Its benefit comes from eliminating shared-memory contention and performing more useful arithmetic per issued load.

**Diagnosis:** kernel 8 is compute-dominant and no longer limited by DRAM bandwidth or MIO congestion. Its remaining actionable stall is global-load latency, visible as Long Scoreboard, under a fixed register-limited occupancy ceiling.

### 5.7 Kernel 8 -> Kernel 9: autotuned warp tiling

The best autotuned configuration changes several coupled parameters:

| Parameter | Kernel 8 | Autotuned best | Effect |
|---|---:|---:|---|
| `BK` | 16 | 32 | More work per K tile; fewer tile-load rounds |
| `LaneSizeN` | 4 | 8 | Wider lane-level transfer/reuse |
| `TM x TN` | 8 x 8 | 4 x 4 | Smaller per-thread logical tile |
| `WM x WN` | 64 x 32 | 32 x 64 | Different warp-level mapping |
| Threads/block | 128 | 256 | Twice as many warps per block |
| GFLOPS | 46,336 | 54,228 | +17.0% |

| Metric | Kernel 8 | Autotuned best | Change |
|---|---:|---:|---:|
| Duration | 3.20 ms | 2.71 ms | 1.18x faster |
| Compute throughput | 49.81% | 57.51% | Higher |
| FMA utilization | 46.19% | 53.86% | +7.67 percentage points |
| IPC active | 2.39 | 2.65 | Higher |
| Long Scoreboard | 1.64 | 0.48 | -71% |
| Not Selected | 2.09 | 2.29 | More scheduler-ready work |
| MIO Throttle | 0.03 | 0.44 | Higher but still small |
| Registers/thread | 106 | 112 | Higher |
| Shared memory/block | 17.41 KB | 33.79 KB | Approximately doubled |
| Theoretical occupancy | 33.33% | 33.33% | Unchanged |
| Achieved occupancy | 32.58% | 31.64% | Slightly lower |

Increasing `BK` from 16 to 32 halves the number of K-tile iterations and moves more computation behind each global-load phase. Together with `LaneSizeN=8`, this raises useful FMA issue and hides global-load latency much more effectively. Long Scoreboard falls by about 70%.

The occupancy result is important: the autotuned kernel does not win by increasing occupancy. Registers per thread and shared memory per block both rise, while occupancy stays in the same register-limited tier. The winning configuration extracts more useful work from the resident warps.

**Diagnosis:** the autotuner finds a better balance among tile depth, vector width, warp geometry, and per-thread work. The gain is higher FMA utilization at the same occupancy tier, not a larger resident-warp count.

---

## 6. Cross-Kernel Findings

### 6.1 “Memory-bound” must be qualified

Kernel 1 has high memory-side activity but only 1.54 GB/s of useful throughput. Its problem is fragmented requests and LG queue pressure, not saturation of peak DRAM bandwidth. Later kernels show only about 6% DRAM utilization even when aggregate `Memory` or `L1/TEX` metrics are much higher. The relevant bottleneck progressively moves into on-chip queues and dependency latency.

### 6.2 Lower instruction efficiency can still reduce runtime

Kernel 3 has lower IPC and FMA utilization than kernel 2, but it executes roughly one third as many instructions. Total work reduction dominates per-instruction efficiency. No single percentage metric should be used as a proxy for end-to-end performance.

### 6.3 Occupancy is a constraint, not the objective

Kernel 5 cuts occupancy from 66.67% to 33.33% and still improves performance because register reuse drastically increases per-warp efficiency. The final autotuned kernel is faster than kernel 8 even though achieved occupancy is slightly lower. The useful objective is latency hiding plus execution-pipeline utilization, not maximum occupancy by itself.

### 6.4 `Not Selected` can be healthy

Once dependency and throttle stalls are small, `Not Selected` becomes prominent. This generally means runnable warps are available and the scheduler is choosing among them. It should not be optimized in isolation or interpreted as a 40% removable loss.

### 6.5 Vectorization is conditional

`float4` reduces instruction count, but its benefit is small when register pressure, bank conflicts, or dependency stalls dominate. Alignment and shape handling must also be preserved. Vectorization should follow, not replace, reuse and layout optimization.

---

## 7. Recommended Next Experiments

The current best kernel reaches 54.2 TFLOPS but still uses only about 54% FMA utilization. The next work should target the remaining dependency and scheduling gaps without assuming more occupancy is always better.

1. **Overlap global loads with computation.** Test software pipelining, ping-pong shared-memory buffers, or architecture-appropriate asynchronous copy mechanisms. Validate with Long Scoreboard, Barrier stalls, FMA utilization, and total runtime.
2. **Explore the current occupancy tier deliberately.** Since 106 and 112 registers/thread both remain at 33.33% theoretical occupancy, search for configurations that add reuse without crossing the next residency threshold. Also test whether a material register reduction can actually move to a higher occupancy tier.
3. **Profile shared-memory bank conflicts directly.** Confirm that the transposed layout minimizes `l1tex__data_bank_conflicts...` counters for all candidate tile shapes; do not infer this only from MIO Throttle.
4. **Separate edge handling from the fast path.** Keep aligned vectorized kernels for dimensions divisible by tile/vector sizes and use a guarded fallback for irregular shapes.
5. **Use shape-aware dispatch.** Kernel 2 is stronger for some small sizes, while autotuned tiled kernels dominate at large sizes. Build a lookup or small runtime heuristic keyed by M/N/K, alignment, and possibly batch size.
6. **Report stability.** Record median and tail latency across repeated runs, along with GPU clock, power, and thermal state. Autotuning decisions should not depend on a single noisy timing sample.

---

## 8. Final Assessment

The optimization sequence is technically coherent and the profiler evidence explains nearly every performance step:

- Coalescing fixes catastrophic global-load inefficiency.
- Shared-memory tiling reduces global instruction volume but exposes MIO pressure.
- 1D and 2D register tiling increase arithmetic intensity and shift execution toward FMA.
- Vectorization reduces instruction count but gives only a small gain because it does not address the active bottleneck.
- Warp tiling plus a transposed shared-memory layout removes bank-conflict/MIO pressure and makes the kernel compute-dominant.
- Autotuning reduces global-load dependency stalls and raises FMA utilization while remaining in the same occupancy tier.

The final result is **54,228 GFLOPS at size 4096, approximately 57.6x faster than the naive kernel**, with the remaining optimization opportunity concentrated in load/compute overlap, register-tier tradeoffs, and shape-specific tuning.
