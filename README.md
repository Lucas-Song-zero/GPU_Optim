# SGEMM CUDA Optimization

A step-by-step single-precision GEMM (`C = α·A·B + β·C`) optimization project for
NVIDIA GPUs, going from a naive kernel to warp tiling and parameter autotuning.

**English** | [简体中文](README.zh-CN.md)

---

## Overview

This repository implements a progressive series of SGEMM kernels, each building on
the previous one, and uses Nsight Compute to profile the micro-architectural
bottlenecks of every stage. The final warp-tiling kernel, combined with an
autotuner, reaches **54.2 TFLOPS** at `M = N = K = 4096` — a **57.7×** speedup over
the naive kernel and **66.9%** of cuBLAS on the same device.

| Stage | GFLOPS @ 4096 | Speedup vs. naive |
|---|---|---|
| 1. naive | 941 | 1.0× |
| 2. global memory coalescing | 6027 | 6.4× |
| 3. shared memory | 9266 | 9.9× |
| 4. 1D block tiling | 28506 | 30.3× |
| 5. 2D block tiling | 36675 | 39.0× |
| 6. vectorize (`float4`) | 38497 | 40.9× |
| 8. warp tiling (hand-tuned) | 46336 | 49.3× |
| 7. 2D block tiling (autotuned) | 52509 | 55.8× |
| **9. warp tiling (autotuned)** | **54228** | **57.7×** |

A detailed per-step ablation report is in [`docs/ablation.md`](docs/ablation.md).

## Environment

| Item | Value |
|---|---|
| GPU | sm_120 (Blackwell) |
| CUDA Toolkit | 13.3 (`/usr/local/cuda-13.3`) |
| Compiler | `nvcc`, `-arch=sm_120 -std=c++20 -O3` |
| cuBLAS | linked via `-lcublas` (fp32 reference) |
| Profiler | Nsight Compute (`ncu`) |

## Kernels

| # | Kernel | Source | Description |
|---|---|---|---|
| 0 | cuBLAS | — | Reference (fp32, tensor-op) |
| 1 | naive | `src/GEMM/1_naive.cuh` | One thread per C element, no reuse |
| 2 | coalescing | `src/GEMM/2_global_mem_coalescing.cuh` | Coalesced global access |
| 3 | shared memory | `src/GEMM/3_shared_mem.cuh` | Explicit A/B tiles in SMEM |
| 4 | 1D block tiling | `src/GEMM/4_1D_block_tiling.cuh` | Register tiling along one dimension |
| 5 | 2D block tiling | `src/GEMM/5_2D_block_tiling.cuh` | `TM×TN` register tile + SMEM |
| 6 | vectorize | `src/GEMM/6_vectorize.cuh` | `float4` loads/stores |
| 7 | autotune (2D) | `src/GEMM/7_autotune.cuh` | 2D block tiling, autotuned via Python |
| 8 | warp tiling | `src/GEMM/8_warp_tiling.cuh` | Warp-level tiling + A transpose |
| 9 | warp autotune | `src/GEMM/9_autotune_more.cuh` | Warp tiling, autotuned via Python |

> Kernels 7 and 9 are templated on compile-time tile sizes, so they are enumerated
> by the Python autotuner (`scripts/autotune/GEMM/search.py`), not by the main
> binary. The main binary's `--kernel` accepts `0–8` (7 throws, as noted in
> `src/runner.cu`).

## Project Structure

```
GPU_Optim/
├── src/
│   ├── GEMM/                     # GEMM kernels (one file per stage)
│   │   ├── 1_naive.cuh
│   │   ├── 2_global_mem_coalescing.cuh
│   │   ├── 3_shared_mem.cuh
│   │   ├── 4_1D_block_tiling.cuh
│   │   ├── 5_2D_block_tiling.cuh
│   │   ├── 6_vectorize.cuh
│   │   ├── 7_autotune.cuh
│   │   ├── 8_warp_tiling.cuh
│   │   ├── 9_autotune_more.cuh
│   │   └── 9_double_buffer.cuh   
│   ├── Attention/                # more Attention part optimization
│   ├── kernels.cuh               # includes all kernels
│   ├── runner.cu                 # kernel dispatch + helpers
│   └── runner.cuh
├── scripts/
│   ├── gemm.cu                   # CLI entrypoint (benchmark / profile)
│   └── autotune/GEMM/
│       ├── search.py             # autotune search driver (+ --profile-best)
│       ├── constraints.py        # tile-size constraint enumeration
│       ├── analyze.py            # results analysis
│       └── ablation_plot.py      # ablation figures
├── docs/                         # ablation report + figures
├── build/                        # build artifacts (gitignored)
├── results/                      # profiling output (gitignored)
├── Makefile
└── README.md
```

## Build & Run

```bash
make                     # build build/gemm
make run KERNEL=4        # benchmark kernel 4 over sizes {128..4096}
make run KERNEL=1,2,3,4  # run several kernels in sequence

make debug               # build build/gemm_dbg (with -G)
make run-debug KERNEL=4

make clean               # remove build/
```

The binary uses flag-only arguments:

```bash
./build/gemm --kernel 4                          # benchmark (default mode)
./build/gemm --kernel 4 --mode profile --M 4096 --N 4096 --K 4096 --warmup 10
```

| Flag | Default | Description |
|---|---|---|
| `--kernel N` | — | Kernel number (required, `0–12` accepted; `0–8` implemented) |
| `--M/--N/--K N` | `4096` | Problem size |
| `--warmup N` | `10` | Warm-up iterations |
| `--mode benchmark\|profile` | `benchmark` | `benchmark` scans sizes, `profile` runs a fixed size |

`benchmark` mode verifies correctness against cuBLAS once per size, then times 50
iterations. `profile` mode verifies once, warms up, then profiles exactly one
launch inside `cudaProfilerStart()/cudaProfilerStop()`.

## Profiling

```bash
make profile KERNEL=4                    # ncu profile kernel 4 @ 4096
make profile KERNEL=4 PROFILE_M=2048     # override problem size
make profile KERNEL=4 PROFILE_FORMAT=csv # export CSV instead of txt

make sanitize KERNEL=4                   # compute-sanitizer memcheck
```

`make profile` runs Nsight Compute over one target-kernel launch and writes
`.ncu-rep`, `.txt`, and (optionally) `.csv` files to `results/profiles/`.

> **Note:** profiling performance counters requires root by default
> (`ERR_NVGPUCTRPERM`). Run with `sudo make profile ...`, or allow non-root
> profiling by setting `NVreg_RestrictProfilingToAdminUsers=0` (via
> `/etc/modprobe.d/` + reloading the driver) and rebooting.

## Autotune

```bash
python scripts/autotune/GEMM/search.py --kernel tunable   # 2D block tiling
python scripts/autotune/GEMM/search.py --kernel warp      # warp tiling

# profile the best config found in results.json (needs root for ncu)
sudo python scripts/autotune/GEMM/search.py --kernel warp --profile-best

python scripts/autotune/GEMM/analyze.py --top 5           # analyze results
python scripts/autotune/GEMM/ablation_plot.py             # regenerate figures
```

The autotuner generates a single self-contained `.cu` that switches over every
valid tile-size configuration at runtime, compiles it once, and parses per-config
GFLOPS from stdout into `results.json`.

## Profiling Metrics Reference

Since GFLOPS saturates once `M = N = K` reaches 4096, profiling is done at that
fixed size. The metrics below are the ones we watch, grouped by category.

### 1. Compute Throughput

| Metric | Meaning | What we watch |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM throughput utilization | Is the SM busy enough? |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA pipeline activation | FP32 GEMM compute-pipeline utilization |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | Thread-level FFMA instruction count | Actual FMA compute workload |
| `gpu__time_duration.sum` | Kernel execution time | Compute throughput from FLOPS |

### 2. Memory Throughput

| Metric | Meaning | What we watch |
|---|---|---|
| `dram__bytes_read.sum` | DRAM read bytes | Real global-memory read streams |
| `dram__bytes_write.sum` | DRAM write bytes | Real global-memory write streams |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth utilization | How close to the bandwidth limit |
| `lts__t_sector_hit_rate.pct` | L2 sector hit rate | L2 data-reuse efficiency |
| `l1tex__t_sector_hit_rate.pct` | L1/TEX cache hit rate | L1 data-reuse efficiency |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 sectors from global loads | Memory-access evaluation |

### 3. Shared Memory

| Metric | Meaning |
|---|---|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared-load bank conflicts |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | Shared-store bank conflicts |
| Shared Load/Store Wavefronts | Shared-memory execution cost |
| Shared Memory Throughput | Shared-memory data throughput |

### 4. Occupancy & Register Pressure

| Metric | Meaning | What we watch |
|---|---|---|
| `launch__registers_per_thread` | Registers allocated per thread | Is the thread tile too large? |
| `launch__shared_mem_per_block` | Shared memory per block | Does it limit CTA residency? |
| Achieved Occupancy | Active warps vs. hardware maximum | Are there enough warps in flight? |
| Theoretical Occupancy | Occupancy upper bound from resources + launch config | Which resource limits parallelism? |
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | Average active-warp occupancy | Actual warp residency |

### 5. Warp Stall

From `WarpStateStats` and `SchedulerStats`.

| Stall type | Meaning | Likely cause |
|---|---|---|
| Long Scoreboard | Waiting on long-latency data dependencies | Global/local loads, cache miss |
| Short Scoreboard | Waiting on short-latency data dependencies | Shared-memory and other short-latency ops |
| Barrier | Waiting on CTA synchronization | `__syncthreads()` etc. |
| Wait | Waiting on operation/instruction dependencies | MIO-pipeline-related waits |
| Not Selected | Warp eligible but not picked this cycle | Enough schedulable warps available |
| Math Pipe Throttle | Compute-pipeline structural resource limit | Instruction issue to the math pipe is throttled |
| LG Throttle | Local/global memory instruction queue pressure | Dense global/local memory traffic |

## Results

See [`docs/ablation.md`](docs/ablation.md) for the full step-by-step ablation,
per-size optimal configurations, and reproduction commands.
