#!/usr/bin/env python3
"""Compare autotuned GEMM against cuBLAS and plot the recovery ratio.

Benchmarks cuBLAS with the same warmup + cudaEvent methodology as the
autotuner (the stock ``gemm 0`` path skips warmup, which poisons the small
sizes), reads the best per-size autotune result from ``results.json``, and
writes ``recovery_ratio.png``.

Usage::

    python plot.py                 # uses results.json
    python plot.py --results results.json --out recovery_ratio.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # scripts/autotune/GEMM -> repo root

CUBLAS_CU = HERE / "bench_cublas.cu"
CUBLAS_BIN = HERE / "bench_cublas"
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_OUT = HERE / "recovery_ratio.png"

NVCC = "/usr/local/cuda-13.3/bin/nvcc"
ARCH = "sm_120"

# Problem sizes benchmarked; must match those in results.json.
SIZES = (128, 256, 512, 1024, 2048, 4096)
REPEAT = 50

CUBLAS_SRC = f"""\
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static const int SIZES[] = {{{", ".join(map(str, SIZES))}}};
static const int NSIZES = {len(SIZES)};
static const int REPEAT = {REPEAT};

int main() {{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 0.5f, beta = 3.0f;

    for (int s = 0; s < NSIZES; ++s) {{
        int M = SIZES[s], N = SIZES[s], K = SIZES[s];
        float *A, *B, *C;
        cudaMalloc(&A, (size_t)M * K * sizeof(float));
        cudaMalloc(&B, (size_t)K * N * sizeof(float));
        cudaMalloc(&C, (size_t)M * N * sizeof(float));
        cudaMemset(A, 0, (size_t)M * K * sizeof(float));
        cudaMemset(B, 0, (size_t)K * N * sizeof(float));
        cudaMemset(C, 0, (size_t)M * N * sizeof(float));

        // warmup: amortize handle/workspace setup before timing
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta,
                     C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();

        cudaEvent_t beg, end;
        cudaEventCreate(&beg); cudaEventCreate(&end);
        cudaEventRecord(beg);
        for (int j = 0; j < REPEAT; ++j)
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta,
                         C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, beg, end);
        ms /= REPEAT;
        double gflops = 2.0 * (double)M * N * K / (ms * 1e-3) / 1e9;
        printf("CUBLAS %d %.6f %.1f\\n", M, ms, gflops);

        cudaEventDestroy(beg); cudaEventDestroy(end);
        cudaFree(A); cudaFree(B); cudaFree(C);
    }}
    cublasDestroy(handle);
    return 0;
}}
"""


def bench_cublas() -> dict[int, float]:
    """Compile and run the cuBLAS micro-benchmark, return {size: gflops}."""
    CUBLAS_CU.write_text(CUBLAS_SRC)
    cmd = [NVCC, f"-arch={ARCH}", "-std=c++20", "-O3", "-lcublas",
           "-o", str(CUBLAS_BIN), str(CUBLAS_CU)]
    subprocess.run(cmd, cwd=str(REPO), check=True)

    proc = subprocess.run([str(CUBLAS_BIN)], cwd=str(REPO),
                          capture_output=True, text=True, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "CUBLAS":
            out[int(parts[1])] = float(parts[3])
    return out


def best_autotune(results: Path) -> dict[int, dict]:
    records = json.loads(results.read_text())
    ok = [r for r in records if r["ok"]]
    best = {}
    for r in ok:
        if r["size"] not in best or r["gflops"] > best[r["size"]]["gflops"]:
            best[r["size"]] = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot GEMM vs cuBLAS recovery ratio")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    cublas = bench_cublas()
    best = best_autotune(Path(args.results))

    sizes = sorted(s for s in best if s in cublas)
    auto = [best[s]["gflops"] for s in sizes]
    cubl = [cublas[s] for s in sizes]
    ratio = [a / c for a, c in zip(auto, cubl)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # left: raw GFLOPS (log scale)
    x = list(range(len(sizes)))
    w = 0.38
    b1 = ax1.bar([i - w / 2 for i in x], cubl, w, label="cuBLAS",
                 color="#8da0cb", zorder=3)
    b2 = ax1.bar([i + w / 2 for i in x], auto, w, label="autotune (kernel 7)",
                 color="#fc8d62", zorder=3)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_xlabel("size (M=N=K)")
    ax1.set_ylabel("GFLOPS (log)")
    ax1.set_title("Throughput")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    for bars in (b1, b2):
        for r in bars:
            ax1.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2,
                                                   r.get_height()),
                         ha="center", va="bottom", fontsize=7)

    # right: recovery ratio
    ax2.axhline(1.0, color="#4daf4a", linestyle="--", linewidth=1.5,
                label="cuBLAS (=1.0)")
    ax2.plot(x, ratio, marker="o", color="#e41a1c", zorder=3)
    for i, r in enumerate(ratio):
        ax2.annotate(f"{r:.2f}", (x[i], r), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("size (M=N=K)")
    ax2.set_ylabel("autotune / cuBLAS")
    ax2.set_title("Performance recovery ratio")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)
    for s, a, c, r in zip(sizes, auto, cubl, ratio):
        print(f"  size={s:5d}  autotune={a:8.1f}  cuBLAS={c:8.1f}  ratio={r:.2f}")


if __name__ == "__main__":
    main()
