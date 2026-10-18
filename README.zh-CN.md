# SGEMM CUDA 优化

针对 NVIDIA GPU 的单精度 GEMM（`C = α·A·B + β·C`）逐级优化项目，从最朴素的
naive 内核一路做到 warp tiling 与参数自动调优。

[English](README.md) | **简体中文**

---

## 项目简介

本仓库实现了一组逐步递进的 SGEMM 内核，每一级都在上一级基础上做一项优化，并用
Nsight Compute 分析每个阶段的微架构瓶颈。最终的 warp-tiling 内核配合自动调优，在
`M = N = K = 4096` 时达到 **54.2 TFLOPS** —— 相对 naive 内核 **57.7×** 加速，达到同设备
cuBLAS 的 **66.9%**。

| 阶段 | GFLOPS @ 4096 | 相对 naive 加速比 |
|---|---|---|
| 1. naive | 941 | 1.0× |
| 2. 全局访存合并（coalescing） | 6027 | 6.4× |
| 3. 共享内存（shared memory） | 9266 | 9.9× |
| 4. 1D block tiling | 28506 | 30.3× |
| 5. 2D block tiling | 36675 | 39.0× |
| 6. 向量化（`float4`） | 38497 | 40.9× |
| 8. warp tiling（手调） | 46336 | 49.3× |
| 7. 2D block tiling（自动调优） | 52509 | 55.8× |
| **9. warp tiling（自动调优）** | **54228** | **57.7×** |

逐步优化的详细 ablation 报告见 [`docs/ablation.md`](docs/ablation.md)。

## 环境

| 项目 | 值 |
|---|---|
| GPU | sm_120（Blackwell） |
| CUDA Toolkit | 13.3（`/usr/local/cuda-13.3`） |
| 编译器 | `nvcc`，`-arch=sm_120 -std=c++20 -O3` |
| cuBLAS | 通过 `-lcublas` 链接（fp32 参考实现） |
| 性能分析器 | Nsight Compute（`ncu`） |

## Kernel 列表

| # | 内核 | 源文件 | 说明 |
|---|---|---|---|
| 0 | cuBLAS | — | 参考实现（fp32，tensor-op） |
| 1 | naive | `src/GEMM/1_naive.cuh` | 每线程算一个 C 元素，无数据复用 |
| 2 | coalescing | `src/GEMM/2_global_mem_coalescing.cuh` | 全局访存合并 |
| 3 | shared memory | `src/GEMM/3_shared_mem.cuh` | A/B 分块显式存入共享内存 |
| 4 | 1D block tiling | `src/GEMM/4_1D_block_tiling.cuh` | 单维寄存器分块 |
| 5 | 2D block tiling | `src/GEMM/5_2D_block_tiling.cuh` | `TM×TN` 寄存器 tile + SMEM |
| 6 | vectorize | `src/GEMM/6_vectorize.cuh` | `float4` 读写 |
| 7 | autotune（2D） | `src/GEMM/7_autotune.cuh` | 2D block tiling，Python 自动调优 |
| 8 | warp tiling | `src/GEMM/8_warp_tiling.cuh` | warp 级分块 + A 转置 |
| 9 | warp autotune | `src/GEMM/9_autotune_more.cuh` | warp tiling，Python 自动调优 |

> 内核 7、9 以编译期 tile 尺寸为模板参数，由 Python 自动调优器
> （`scripts/autotune/GEMM/search.py`）逐一枚举，而非由主程序调用。主程序的
> `--kernel` 接受 `0–8`（其中 7 会抛异常，见 `src/runner.cu`）。

## 目录结构

```
GPU_Optim/
├── src/
│   ├── GEMM/                     # GEMM 内核（每级一个文件）
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
│   ├── Attention/                # 注意力部分的加速
│   ├── kernels.cuh               # 包含所有内核
│   ├── runner.cu                 # 内核分发 + 工具函数
│   └── runner.cuh
├── scripts/
│   ├── gemm.cu                   # 命令行入口（benchmark / profile）
│   └── autotune/GEMM/
│       ├── search.py             # 自动调优搜索驱动（+ --profile-best）
│       ├── constraints.py        # tile 尺寸约束枚举
│       ├── analyze.py            # 结果分析
│       └── ablation_plot.py      # ablation 绘图
├── docs/                         # ablation 报告与图
├── build/                        # 构建产物（已 gitignore）
├── results/                      # 性能分析输出（已 gitignore）
├── Makefile
└── README.md
```

## 构建与运行

```bash
make                     # 构建 build/gemm
make run KERNEL=4        # 对 size {128..4096} 基准测试内核 4
make run KERNEL=1,2,3,4  # 依次运行多个内核

make debug               # 构建 build/gemm_dbg（带 -G）
make run-debug KERNEL=4

make clean               # 删除 build/
```

二进制只接受 flag 形式的参数：

```bash
./build/gemm --kernel 4                          # 基准测试（默认 mode）
./build/gemm --kernel 4 --mode profile --M 4096 --N 4096 --K 4096 --warmup 10
```

| Flag | 默认值 | 说明 |
|---|---|---|
| `--kernel N` | — | 内核编号（必填，接受 `0–12`；已实现 `0–8`） |
| `--M/--N/--K N` | `4096` | 问题尺寸 |
| `--warmup N` | `10` | warm-up 次数 |
| `--mode benchmark\|profile` | `benchmark` | `benchmark` 扫描各尺寸，`profile` 跑固定尺寸 |

`benchmark` 模式每个尺寸先对照 cuBLAS 做一次正确性验证，再计时 50 次迭代。
`profile` 模式先验证一次、再 warm-up，最后在
`cudaProfilerStart()/cudaProfilerStop()` 区间内精确采集一次 kernel launch。

## 性能分析

```bash
make profile KERNEL=4                    # ncu 分析内核 4 @ 4096
make profile KERNEL=4 PROFILE_M=2048     # 覆盖问题尺寸
make profile KERNEL=4 PROFILE_FORMAT=csv # 导出 CSV 而非 txt

make sanitize KERNEL=4                   # compute-sanitizer 内存检查
```

`make profile` 用 Nsight Compute 采集一次目标内核 launch，把 `.ncu-rep`、`.txt`、
（可选）`.csv` 写到 `results/profiles/`。

> **注意：** 采集硬件性能计数器默认需要 root 权限（`ERR_NVGPUCTRPERM`）。请用
> `sudo make profile ...`，或通过 `/etc/modprobe.d/` 设置
> `NVreg_RestrictProfilingToAdminUsers=0` 并重载驱动后重启，以允许非 root 分析。

## 自动调优

```bash
python scripts/autotune/GEMM/search.py --kernel tunable   # 2D block tiling
python scripts/autotune/GEMM/search.py --kernel warp      # warp tiling

# 分析 results.json 里的最优配置（ncu 需要 root）
sudo python scripts/autotune/GEMM/search.py --kernel warp --profile-best

python scripts/autotune/GEMM/analyze.py --top 5           # 分析结果
python scripts/autotune/GEMM/ablation_plot.py             # 重新生成图
```

自动调优器会生成一个自包含的 `.cu` 文件，在运行时 switch 所有合法的 tile 尺寸配置，
一次性编译后逐个 size 运行，再把每个配置的 GFLOPS 从 stdout 解析进 `results.json`。

## Profiling 指标参考

由于 `M = N = K` 到 4096 后 GFLOPS 已趋于饱和，profiling 固定在该尺寸进行。下面按
类别列出我们关注的指标。

### 1. 计算吞吐（Compute Throughput）

| 指标 | 含义 | 分析重点 |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐整体利用率 | SM 是否足够繁忙 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA 管线激活度 | FP32 GEMM 计算管线利用率 |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | 线程级 FFMA 指令数 | 实际 FMA 计算工作量 |
| `gpu__time_duration.sum` | 内核执行时间 | 结合 FLOPS 计算吞吐 |

### 2. 访存吞吐（Memory Throughput）

| 指标 | 含义 | 分析重点 |
|---|---|---|
| `dram__bytes_read.sum` | DRAM 读字节数 | 真实的全局内存读流 |
| `dram__bytes_write.sum` | DRAM 写字节数 | 真实的全局内存写流 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率 | 离带宽上限有多近 |
| `lts__t_sector_hit_rate.pct` | L2 扇区命中率 | L2 数据复用效率 |
| `l1tex__t_sector_hit_rate.pct` | L1/TEX 缓存命中率 | L1 数据复用效率 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 全局加载进入 L1 的扇区数 | 访存开销评估 |

### 3. 共享内存（Shared Memory）

| 指标 | 含义 |
|---|---|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 共享内存加载 bank conflict |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | 共享内存存储 bank conflict |
| Shared Load/Store Wavefronts | 共享内存执行开销 |
| Shared Memory Throughput | 共享内存数据吞吐 |

### 4. Occupancy 与寄存器压力

| 指标 | 含义 | 分析重点 |
|---|---|---|
| `launch__registers_per_thread` | 每线程分配的寄存器数 | Thread tile 是否过大 |
| `launch__shared_mem_per_block` | 每 block 的共享内存分配 | 是否限制 CTA residency |
| Achieved Occupancy | 实际活跃 warp 占硬件上限的比例 | 是否有足够的并行 warp |
| Theoretical Occupancy | 基于资源与 launch 配置计算的 occupancy 上限 | 哪些资源限制并行度 |
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | 活跃 warp 的平均占用水平 | 实际 warp residency |

### 5. Warp Stall

取自 `WarpStateStats` 与 `SchedulerStats`。

| Stall 类型 | 意义 | 可能原因 |
|---|---|---|
| Long Scoreboard | 等待较长延迟的数据依赖 | Global/Local Memory load、cache miss |
| Short Scoreboard | 等待较短延迟的数据依赖 | Shared memory 等短延迟操作 |
| Barrier | 等待 CTA 同步 | `__syncthreads()` 等 |
| Wait | 等待操作或指令依赖完成 | MIO 等管线相关等待 |
| Not Selected | Warp eligible，但本周期未被 scheduler 选中 | 可能有充分的可调度 warp |
| Math Pipe Throttle | 计算管线相关结构资源受限 | 指令发射到数学管线受到限制 |
| LG Throttle | Local/Global memory 指令队列压力 | 高密度的 global/local 访存 |

## 结果

完整的分步 ablation、各尺寸最优配置与复现命令见
[`docs/profiling_analysis.md`](docs/kernel_profiling_analysis.zh-CN.md)。
