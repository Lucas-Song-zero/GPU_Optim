"""Constraint model + search-space generation for the GEMM autotuner.

The tunable kernel (src/GEMM/7_autotune.cuh) has signature::

    template<const int THREADS_NUM, const int BM, const int BN,
             const int BK, const int TM, const int TN>
    __global__ void gemm_tunable(int M, int N, int K, float alpha,
                                 float *A, float *B, float beta, float *C);

plus a ``VEC_CONFIG`` preprocessor macro that toggles float4 loads.

This module is deliberately free of CUDA: it only derives, from the kernel
source, which parameter combinations are legal.  ``search.py`` consumes it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    threads: int  # THREADS_NUM (threads per block)
    BM: int       # block-tile rows (M)
    BN: int       # block-tile cols (N)
    BK: int       # block-tile depth (K)
    TM: int       # per-thread sub-tile rows
    TN: int       # per-thread sub-tile cols


@dataclass(frozen=True)
class Device:
    """Hardware limits.  Defaults target sm_120 with *static* shared memory."""

    max_threads_per_block: int = 1024
    # Static __shared__ budget.  Default 48 KB (no opt-in).  The kernel uses
    # static __shared__ arrays, so the dynamic opt-in path does not apply.
    shared_mem_per_block: int = 48 * 1024
    # Heuristic: accumulator tmp[TM*TN] + regM[TM] + regN[TN] + addressing.
    # Raised to 256 to admit cuBLAS-style budgets: cuBLAS's SIMT SGEMM runs
    # ~200 registers/thread (128 accumulators + prefetch/address regs) at only
    # 16.67% occupancy and still hits 68.8% FMA utilization.  Our warp kernel
    # reaches the same 128 accumulators via WM=WN=64 (needs ~152 regs).
    max_regs_per_thread: int = 256
    warp_size: int = 32


def valid(cfg: Config, device: Device, vec: bool = True) -> bool:
    """Return True iff *cfg* is legal given the kernel's indexing scheme."""
    # The block is tiled (BM/TM) x (BN/TN); every thread owns one TM x TN tile.
    if cfg.BM * cfg.BN % (cfg.TM * cfg.TN) != 0:
        return False
    if cfg.threads != cfg.BM * cfg.BN // (cfg.TM * cfg.TN):
        return False
    # cRow = tid / (BN/TN), cCol = tid % (BN/TN) require exact division.
    if cfg.BM % cfg.TM != 0 or cfg.BN % cfg.TN != 0:
        return False

    if not (device.warp_size <= cfg.threads <= device.max_threads_per_block):
        return False
    if cfg.threads % device.warp_size != 0:
        return False

    # float4 loads: A tile is BM x BK (vectorized along BK), B tile is BK x BN
    # (vectorized along BN), so both BK and BN must be multiples of 4.
    if vec and (cfg.BK % 4 != 0 or cfg.BN % 4 != 0):
        return False

    # Static shared memory: A_bk[BM*BK] + B_bk[BK*BN] floats.
    smem = (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * 4
    if smem > device.shared_mem_per_block:
        return False

    # Register budget (accumulator + two cache vectors).
    if cfg.TM * cfg.TN + cfg.TM + cfg.TN > device.max_regs_per_thread:
        return False

    return True


def gen_configs(
    BM=(64, 128, 256),
    BN=(64, 128, 256),
    BK=(8, 16, 32),
    TM=(4, 8),
    TN=(4, 8),
    vec: bool = True,
    device: Device = Device(),
):
    """Yield every valid :class:`Config` from the Cartesian product."""
    for bm, bn, bk, tm, tn in itertools.product(BM, BN, BK, TM, TN):
        cfg = Config(bm * bn // (tm * tn), bm, bn, bk, tm, tn)
        if valid(cfg, device, vec):
            yield cfg


# ---------------------------------------------------------------------------
# Warp-tiling kernel (src/GEMM/9_autotune_more.cuh)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarpConfig:
    threads: int  # threads per block = (BM*BN / (WM*WN)) * 32
    BM: int       # block-tile rows (M)
    BN: int       # block-tile cols (N)
    BK: int       # block-tile depth (K)
    WM: int       # warp-tile rows (M)
    WN: int       # warp-tile cols (N)
    LaneSizeN: int  # lanes per row of the warp sub-tile (N direction)
    TM: int       # per-thread sub-tile rows
    TN: int       # per-thread sub-tile cols


def valid_warp(cfg: WarpConfig, device: Device) -> bool:
    """Return True iff *cfg* is legal for gemm_warp_tunable's indexing."""
    if cfg.BM % cfg.WM != 0 or cfg.BN % cfg.WN != 0:
        return False
    # warps are laid out (BM/WM) x (BN/WN); need exact division.
    if cfg.BM * cfg.BN % (cfg.WM * cfg.WN) != 0:
        return False
    warps = cfg.BM * cfg.BN // (cfg.WM * cfg.WN)
    if cfg.threads != warps * device.warp_size:
        return False
    if not (device.warp_size <= cfg.threads <= device.max_threads_per_block):
        return False
    if cfg.threads % device.warp_size != 0:
        return False

    # LaneSizeN must divide the warp; warp sub-tile must divide the warp tile.
    if device.warp_size % cfg.LaneSizeN != 0:
        return False
    sub_m = (device.warp_size // cfg.LaneSizeN) * cfg.TM
    sub_n = cfg.LaneSizeN * cfg.TN
    if cfg.WM % sub_m != 0 or cfg.WN % sub_n != 0:
        return False

    # float4 loads: A vectorized along BK, B along BN (both must be mult of 4).
    if cfg.BK % 4 != 0 or cfg.BN % 4 != 0:
        return False

    # Static shared memory: A_bk[BM*BK] (transposed, same size) + B_bk[BK*BN].
    smem = (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * 4
    if smem > device.shared_mem_per_block:
        return False

    # Register budget: accumulator tmp + regM + regN.
    wm_iter = cfg.WM // sub_m
    wn_iter = cfg.WN // sub_n
    if (wm_iter * cfg.TM * wn_iter * cfg.TN + wm_iter * cfg.TM +
            wn_iter * cfg.TN) > device.max_regs_per_thread:
        return False

    return True


def gen_warp_configs(
    BM=(64, 128, 256),
    BN=(64, 128),
    BK=(16, 32),
    WM=(32, 64),
    WN=(32, 64),
    LaneSizeN=(4, 8),
    TM=(4, 8),
    TN=(4, 8),
    device: Device = Device(),
):
    """Yield every valid :class:`WarpConfig` from the Cartesian product."""
    for bm, bn, bk, wm, wn, lane, tm, tn in itertools.product(
            BM, BN, BK, WM, WN, LaneSizeN, TM, TN):
        warps = bm * bn // (wm * wn)
        cfg = WarpConfig(warps * device.warp_size, bm, bn, bk, wm, wn, lane,
                         tm, tn)
        if valid_warp(cfg, device):
            yield cfg


if __name__ == "__main__":
    dev = Device()
    for vec in (True, False):
        cfgs = list(gen_configs(vec=vec, device=dev))
        print(f"tunable vec={int(vec)}: {len(cfgs)} valid configs")
        for c in cfgs[:5]:
            print("  ", c)
    warps = list(gen_warp_configs(device=dev))
    print(f"warp: {len(warps)} valid configs")
    for c in warps[:5]:
        print("  ", c)
