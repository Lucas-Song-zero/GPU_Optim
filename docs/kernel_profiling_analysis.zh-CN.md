# GEMM Kernel Profiling Analysis

## 1. 结论摘要

本文整合了 `kernel1` 到自动调优版本的 GEMM 性能数据与 Nsight Compute 指标。

- 在 4096 方阵上，性能从 naive kernel 的 **941 GFLOPS** 提升到 warp autotuned kernel 的 **54,228 GFLOPS**；按表中吞吐量计算，加速约 **57.6 倍**。
- 优化过程中，瓶颈并非被一次消除，而是在 GPU 的不同层级间迁移：

```text
非合并全局访存
  -> global load 指令与队列压力
  -> shared-memory 管线压力
  -> 数据复用不足
  -> 寄存器压力与低 occupancy
  -> shared-memory bank conflict
  -> compute-heavy kernel 中的 global-load 延迟
  -> autotune 寻找参数平衡点
```
---

## 2. 数据范围与解读原则

性能表覆盖 128 到 4096 的方阵 GEMM；详细 Nsight Compute 数据主要解释大矩阵场景下的 kernel 演进。

解读时需要注意：

1. GFLOPS benchmark 与 profiler duration 可能来自不同采样过程，profiling 本身也会引入开销。因此应比较趋势，不应强行用两组数据反算完全一致的时间。
2. `Memory Throughput` 或 `L1/TEX` 百分比高，不等价于 DRAM 带宽饱和。聚合指标还包含片上访存路径。定位等待来源时，应结合 LG Throttle、MIO Throttle、Long Scoreboard 和 Short Scoreboard。

---

## 3. 整体性能

### 3.1 各矩阵规模吞吐量

| Kernel | 优化方式 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
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

单位均为 GFLOPS。

### 3.2 4096 方阵上的加速比

| Kernel | GFLOPS | 相对 naive | 相对上一阶段 |
|---|---:|---:|---:|
| 1 naive | 941 | 1.0x | - |
| 2 coalesced | 6,027 | 6.4x | 6.40x |
| 3 shared memory | 9,266 | 9.9x | 1.54x |
| 4 1D tiling | 28,506 | 30.3x | 3.08x |
| 5 2D tiling | 36,675 | 39.0x | 1.29x |
| 6 vectorized | 38,497 | 40.9x | 1.05x |
| 8 warp tiling | 46,336 | 49.2x | 1.20x |
| 9 warp autotuned | 54,228 | **57.6x** | 1.17x |

`kernel7` 与 `kernel9` 是两条不同的 autotune 路径，因此完整 benchmark 保留 `kernel7`，但详细演进链不把它视为 kernel 6 与 kernel 8 之间的顺序步骤。

### 3.3 矩阵规模的影响

高级 tiling kernel 在小矩阵上并不占优。128 和 256 规模下，launch、同步、索引和 tile 初始化成本无法充分摊薄；优势主要从 1024 以后显现。生产实现应根据 M/N/K、对齐条件和 batch size 选择 kernel，而不是让大矩阵最优配置处理所有输入。

---

## 4. 瓶颈演进总览

| 迁移 | 核心改动 | 关键指标 | 收益 | 新瓶颈或代价 |
|---|---|---|---|---|
| 1 -> 2 | Warp 内访存合并 | LG Throttle 119.57 -> 8.90；1.54 -> 10.11 GB/s | 4096 上 6.4x | Global/L1 load 管线仍有压力 |
| 2 -> 3 | Shared memory 复用 global 数据 | 指令 16.67B -> 5.60B；LG 8.90 -> 0.40 | 1.54x | MIO Throttle 升至 23.79 |
| 3 -> 4 | 每线程计算 `TM=8` 个输出 | MIO 23.79 -> 6.53；FMA 7.83% -> 25.79% | 3.08x | 寄存器依赖及剩余 LSU 压力 |
| 4 -> 5 | `8 x 8` 二维寄存器 tile | LSU 82.13% -> 46.16%；FMA Heavy 19.94% -> 39.91% | 1.29x | 95 registers/thread；occupancy 33.33% |
| 5 -> 6 | `float4` 向量化访存 | 指令 3.22B -> 2.85B | 1.05x | Occupancy 与 bank conflict 未改善 |
| 6 -> 8 | Warp tiling + 转置 SMEM 布局 | MIO 接近零；Compute 49.81% > Memory 43.01% | 1.20x | 106 registers/thread；Long SB 1.64 |
| 8 -> 9 | 调优 `BK`、lane 宽度和 warp/thread tile | FMA 46.19% -> 53.86%；Long SB 1.64 -> 0.48 | 1.17x | Occupancy 仍受寄存器限制 |

---

## 5. 分阶段 Profiling Analysis

### 5.1 Kernel 1 -> Kernel 2：全局访存合并

`kernel1` 中，同一 warp 的 lane 访问分散地址。问题不是简单的 DRAM 流量过大，而是碎片化请求使 load/store 路径无法有效服务。

| 指标 | Kernel 1 | Kernel 2 | 含义 |
|---|---:|---:|---|
| Memory utilization | 98.91% | 80.13% | Busy 高不等于有效带宽高 |
| Compute utilization | 11.99% | 80.13% | Coalescing 后数据供给改善 |
| Memory throughput | 1.54 GB/s | 10.11 GB/s | 有效传输速率提高 6.6x |
| IPC | 0.23 | 1.56 | 指令发射明显恢复 |
| SM Busy | 6.95% | 48.48% | 更多周期用于有效计算 |
| LG Throttle | 119.57 | 8.90 cycles/issue | Load 队列压力大幅下降 |
| Warp cycles/instruction | 134.00 | 20.48 | Warp 等待显著减少 |

两个 kernel 的总指令数相同，因此收益来自访问效率，而非工作量减少。Kernel 2 的 LG Throttle 仍有 8.90 cycles/issue，说明 global/L1 load 路径仍然昂贵。

**分析：** coalescing 修复了最严重的访存模式，但每个输出元素仍反复通过 global/L1 路径取数。

### 5.2 Kernel 2 -> Kernel 3：Shared-memory tiling

Kernel 3 将 A/B tile 搬入 shared memory，再由 block 内线程复用。

| 指标 | Kernel 2 | Kernel 3 | 变化 |
|---|---:|---:|---:|
| Duration | 25.70 ms | 17.07 ms | 1.50x |
| Instructions executed | 16.67B | 5.60B | -66% |
| LG Throttle | 8.90 | 0.40 | Global 队列压力基本消失 |
| MIO Throttle | 0.03 | 23.79 | SMEM 路径成为新瓶颈 |
| Long Scoreboard | 3.36 | 6.68 | 搬运 tile 的等待更明显 |
| Barrier | 0.00 | 2.96 | 出现同步成本 |
| IPC | 1.56 | 0.79 | 单周期发射效率下降 |
| FMA utilization | 17.64% | 7.83% | 比例下降但总时间更短 |

IPC 和 FMA utilization 虽然下降，但总指令数降到约三分之一，工作量减少足以抵消每条指令平均成本上升。

**分析：** shared memory 减少了 global load，但每线程仍只计算一个 C 元素，SMEM read 次数接近 FMA 次数。瓶颈由 LG Throttle 转移为 MIO Throttle。

### 5.3 Kernel 3 -> Kernel 4：1D register tiling

Kernel 4 让每个线程计算 `TM=8` 个输出，一个从 SMEM 读入寄存器的值可供 8 次 FMA 复用。

| 指标 | Kernel 3 | Kernel 4 | 变化 |
|---|---:|---:|---:|
| Duration | 17.07 ms | 5.51 ms | 3.10x |
| Instructions executed | 5.60B | 3.33B | -40% |
| MIO Throttle | 23.79 | 6.53 | 降至约四分之一 |
| Warp cycles/issue | 40.46 | 21.11 | 约减半 |
| IPC | 0.79 | 1.49 | 约翻倍 |
| FMA utilization | 7.83% | 25.79% | 3.3x |
| Registers/thread | 38 | 48 | 寄存器复用的代价 |
| Theoretical occupancy | 66.67% | 66.67% | 尚无 occupancy 损失 |

Short Scoreboard 从 0.43 升到 2.47 cycles/issue，因为多个 FMA 依赖同一个刚加载的寄存器值。相较于反复发出 SMEM load 并堵塞 MIO，这是一项有利交换。

**分析：** 这是第一次通过寄存器复用真正提高算术强度，而不是把访存压力转移到另一层。

### 5.4 Kernel 4 -> Kernel 5：2D register tiling

Kernel 5 将每线程输出扩展为 `8 x 8` tile。A、B 两侧均缓存到寄存器，`TM + TN` 次 SMEM read 可供 `TM x TN` 次 FMA 使用。

| 指标 | Kernel 4 | Kernel 5 | 变化 |
|---|---:|---:|---:|
| Duration | 5.51 ms | 4.28 ms | 1.29x |
| LSU utilization | 82.13% | 46.16% | 访存压力大幅下降 |
| FMA Heavy | 19.94% | 39.91% | 计算开始占主导 |
| IPC | 1.49 | 2.25 | 发射效率提高 |
| Warp cycles/issue | 21.11 | 7.02 | 降至约三分之一 |
| MIO Throttle | 6.53 | 0.89 | 基本消除 |
| Long Scoreboard | 5.49 | 1.31 | 大幅下降 |
| Registers/thread | 48 | 95 | 几乎翻倍 |
| Theoretical occupancy | 66.67% | 33.33% | 减半 |
| Active warps/scheduler | 7.88 | 3.95 | 减半 |

这一步使计算管线首次超过 load/store 管线，但 64 个累加器和操作数寄存器将用量推到 95 registers/thread，活跃 warp 数减半。

**分析：** 2D tiling 消除了大部分 SMEM traffic stall，提高了单 warp 效率；代价是寄存器压力限制 resident warp 数。

### 5.5 Kernel 5 -> Kernel 6：向量化

Kernel 6 使用 `float4` 完成 global-to-shared load 和 C store。

| 指标 | Kernel 5 | Kernel 6 | 变化 |
|---|---:|---:|---:|
| Duration | 4.28 ms | 4.04 ms | 1.06x |
| Instructions executed | 3.22B | 2.85B | -11% |
| LSU utilization | 46.16% | 42.67% | 小幅下降 |
| IPC | 2.25 | 2.02 | 略降 |
| Warp cycles/issue | 7.02 | 7.60 | 略升 |
| Registers/thread | 95 | 95 | 不变 |
| Occupancy | 33.33% | 33.33% | 不变 |
| Active warps/scheduler | 3.95 | 3.84 | 基本不变 |

Vectorization 减少了指令数，却没有改善寄存器限制的 occupancy 和 SMEM bank conflict。Long Scoreboard、MIO Throttle 与 Barrier stall 略有增加，抵消了大部分收益。

**分析：** `float4` 本身有效，但当前 kernel 已不再受标量访存指令数主导，因此只能提供增量收益。

### 5.6 Kernel 6 -> Kernel 8：Warp tiling 与 SMEM 布局

手调配置为：

```text
BM=128, BN=128, BK=16
WM=64, WN=32
LaneSizeN=4
TM=8, TN=8
threads/block=128
```

Warp tiling 显式划分 warp 级输出 tile，并转置 A 在 shared memory 中的布局，使内层 A 访问由高冲突的跨列读取变为更友好的行向读取。

| 指标 | Kernel 6 | Kernel 8 | 变化 |
|---|---:|---:|---:|
| Duration | 4.04 ms | 3.20 ms | 1.26x |
| Compute throughput | 37.71% | 49.81% | 上升 |
| Memory throughput | 46.98% | 43.01% | 下降 |
| DRAM utilization | 6.70% | 6.05% | 低且基本不变 |
| LSU utilization | 42.67% | 26.14% | 大幅下降 |
| FMA utilization | - | 46.19% | FMA 成为主导 |
| MIO Throttle | - | 0.03 | 接近零 |
| LG Throttle | - | 0.00 | 为零 |
| Registers/thread | 95 | 106 | 上升 |
| Theoretical occupancy | 33.33% | 33.33% | 不变 |
| Achieved occupancy | 33.55% | 32.58% | 略降 |

`Not Selected` 成为最大 sampled stall，达到 2.09 cycles/issue；Long Scoreboard 为 1.64。较高的 `Not Selected` 通常意味着已有 eligible warp 等待 scheduler 选择，而不是所有 warp 都被资源依赖阻塞。

需要修正早期预期：warp tiling 在这里**没有**降低寄存器压力或提高 occupancy。寄存器由 95 增至 106，occupancy 仍为 33.33%。收益来自消除 SMEM contention，并让每次 load 对应更多有效计算。

**分析：** kernel 8 已转为 compute-dominant，不再受 DRAM bandwidth 或 MIO congestion 限制。主要剩余 stall 是 Long Scoreboard 暴露出的 global-load latency。

### 5.7 Kernel 8 -> Kernel 9：Warp autotuning

| 参数 | Kernel 8 | Autotuned best | 作用 |
|---|---:|---:|---|
| `BK` | 16 | 32 | 增加每个 K tile 的计算量，减少加载轮数 |
| `LaneSizeN` | 4 | 8 | 提高 lane 级传输与复用宽度 |
| `TM x TN` | 8 x 8 | 4 x 4 | 减小逻辑 per-thread tile |
| `WM x WN` | 64 x 32 | 32 x 64 | 调整 warp 映射方向 |
| Threads/block | 128 | 256 | 每 block warp 数翻倍 |
| GFLOPS | 46,336 | 54,228 | +17.0% |

| 指标 | Kernel 8 | Autotuned best | 变化 |
|---|---:|---:|---:|
| Duration | 3.20 ms | 2.71 ms | 1.18x |
| Compute throughput | 49.81% | 57.51% | 上升 |
| FMA utilization | 46.19% | 53.86% | +7.67 个百分点 |
| IPC active | 2.39 | 2.65 | 上升 |
| Long Scoreboard | 1.64 | 0.48 | -71% |
| Not Selected | 2.09 | 2.29 | 更多 scheduler-ready work |
| MIO Throttle | 0.03 | 0.44 | 上升但仍低 |
| Registers/thread | 106 | 112 | 上升 |
| Shared memory/block | 17.41 KB | 33.79 KB | 约翻倍 |
| Theoretical occupancy | 33.33% | 33.33% | 不变 |
| Achieved occupancy | 32.58% | 31.64% | 略降 |

`BK` 从 16 增至 32，使 K-tile 迭代次数减半，并让每轮 global load 后执行更多计算；配合 `LaneSizeN=8`，global-load latency 被更好地隐藏，Long Scoreboard 下降约 70%。

最重要的结论是：autotuned kernel 并非通过提高 occupancy 获胜。Registers/thread 与 SMEM/block 都上升，occupancy 仍处于同一个寄存器限制档位。最优配置是在有限 resident warp 上做了更多有效工作。

**分析：** autotuner 找到了 tile depth、vector width、warp geometry 与 per-thread work 之间更好的平衡。收益来自相同 occupancy 档位下更高的 FMA 利用率。

---

## 6. 跨 Kernel 的关键认识

### 6.1 “Memory-bound” 必须说明具体层级

Kernel 1 的 memory-side activity 很高，但有效吞吐量只有 1.54 GB/s，问题是请求碎片化与 LG queue pressure，而不是峰值 DRAM bandwidth 饱和。后续 kernel 中 DRAM utilization 仅约 6%，主要活动已经位于片上路径。

### 6.2 单项利用率下降不代表性能退化

Kernel 3 的 IPC 与 FMA utilization 低于 kernel 2，但只执行约三分之一的指令，最终 runtime 更短。不能把任意单一指标直接当作端到端性能。

### 6.3 Occupancy 是约束，不是目标本身

Kernel 5 将 occupancy 从 66.67% 降到 33.33% 后仍然更快，因为 register reuse 提升了单 warp 效率。最终 autotuned kernel 的 achieved occupancy 略低于 kernel 8，但性能更高。目标应是足够的 latency hiding 与管线利用率，而非单纯最大化 occupancy。

### 6.4 `Not Selected` 通常是健康信号

当 dependency 和 throttle stall 较低时，`Not Selected` 会变得突出。这说明存在可运行 warp 等待 scheduler 选择，不应把该比例直接视为可消除损失。

### 6.5 Vectorization 的收益取决于当前瓶颈

`float4` 可以减少指令数，但在 register pressure、bank conflict 或 dependency stall 占主导时收益有限。向量化应建立在正确的数据复用和布局之上。

---

## 7. 后续优化建议

当前最优版本达到 54.2 TFLOPS，但 FMA utilization 约为 54%，仍有提升空间。

1. **重叠 global load 与计算。** 测试 software pipelining、ping-pong shared-memory buffer 或架构支持的异步 copy；关注 Long Scoreboard、Barrier、FMA utilization 与实际 runtime。
2. **围绕 occupancy 档位搜索。** 106 与 112 registers/thread 都处于 33.33% 理论 occupancy 档位，可在不跨越下一资源阈值时继续增加复用；也应测试寄存器显著下降后能否真正进入更高档位。
3. **直接采集 bank-conflict counter。** 对候选 tile 检查 `l1tex__data_bank_conflicts...`，不要仅通过 MIO Throttle 间接推断。
4. **分离边界处理与 fast path。** 对满足 tile/vector 对齐的 shape 使用快速 kernel，对不规则 shape 使用 guarded fallback。
5. **建立 shape-aware dispatch。** 小矩阵可能由 kernel 2 或轻量 kernel 占优，大矩阵由 autotuned tiling kernel 主导。
6. **加入 cuBLAS/cuBLASLt 基线。** 保证 datatype、accumulation mode、时钟、warmup 和计时方式一致。
7. **报告稳定性。** 多次运行并记录 median、tail latency、GPU clock、power 与 thermal state，避免 autotune 被单次噪声误导。

---

## 8. 最终评价

- Coalescing 修复灾难性的全局访存模式。
- Shared-memory tiling 降低 global 指令量，但暴露 MIO 压力。
- 1D/2D register tiling 提高算术强度，使执行重心转向 FMA。
- Vectorization 减少指令数，但因未触及主瓶颈，只产生小幅收益。
- Warp tiling 与转置 SMEM 布局清除 bank-conflict/MIO 压力，使 kernel 转为 compute-dominant。
- Autotuning 在相同 occupancy 档位下降低 global-load dependency stall，并进一步提高 FMA utilization。

最终结果为：**4096 方阵上 54,228 GFLOPS，相对 naive kernel 提升约 57.6 倍**。剩余优化空间主要集中在 load/compute overlap、寄存器资源档位和 shape-specific tuning。
