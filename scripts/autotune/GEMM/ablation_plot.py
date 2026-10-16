#!/usr/bin/env python3
"""Plot the GEMM ablation study grouped by optimization principle.

Writes two figures into ``docs/``:

  * ``ablation.png``         -- 2x2 grid, one panel per principle category
  * ``ablation_speedup.png`` -- cumulative speedup-vs-naive @4096 bar chart

Usage::

    python scripts/autotune/GEMM/ablation_plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册一个支持中文的字体，避免标题里的汉字渲染成方框。
for _name in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
              "AR PL UMing CN", "Droid Sans Fallback"):
    _path = None
    for _f in font_manager.fontManager.ttflist:
        if _f.name == _name:
            _path = _f.fname
            break
    if _path:
        font_manager.fontManager.addfont(_path)
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DOCS = REPO / "docs"

SIZES = [128, 256, 512, 1024, 2048, 4096]
CUBLAS = [650.2, 5186.3, 21698.4, 51873.6, 73607.5, 81042.8]

# (label, gflops) — the full progression
SERIES = {
    "naive":        [81.8, 348.3, 708.0, 813.3, 911.5, 940.5],
    "coalescing":   [503.2, 2077.1, 5017.4, 6025.6, 6774.1, 6026.9],
    "shared mem":   [667.6, 2728.5, 6889.8, 8459.1, 9578.5, 9266.2],
    "1D block":     [405.5, 1812.3, 7700.8, 20701.8, 26453.9, 28505.8],
    "2D block":     [156.9, 655.1, 2912.2, 12345.0, 33556.8, 36674.9],
    "vectorize":    [185.0, 779.9, 3275.5, 13444.0, 34524.7, 38496.5],
    "warp tiling":  [202.6, 861.8, 3877.5, 16915.0, 46224.3, 46335.7],
    "tunable 2D":   [507.8, 2338.7, 10878.0, 31499.9, 48036.3, 52509.2],
    "warp autotune":[409.7, 2048.2, 10057.1, 32520.3, 49949.1, 54228.3],
}

# Each panel: (title, [(label, color, marker, linestyle, lw, annotate?)])
GROUPS = [
    ("① 访存优化：合并 + 共享内存", [
        ("naive",        "#9ecae1", "o", "-", 1.8, False),
        ("coalescing",   "#4292c6", "s", "-", 1.8, False),
        ("shared mem",   "#08519c", "^", "-", 2.4, True),
    ]),
    ("② 寄存器分块：1D → 2D", [
        ("shared mem",   "#a1d99b", "o", "-", 1.8, False),
        ("1D block",     "#41ab5d", "s", "-", 1.8, False),
        ("2D block",     "#006d2c", "^", "-", 2.4, True),
    ]),
    ("③ 向量化 & warp tiling", [
        ("2D block",     "#bcbddc", "o", "-", 1.8, False),
        ("vectorize",    "#807dba", "s", "-", 1.8, False),
        ("warp tiling",  "#54278f", "^", "-", 2.4, True),
    ]),
    ("④ 参数自动调优", [
        ("2D block",     "#bdbdbd", "o", "-", 1.6, False),
        ("tunable 2D",   "#fb6a4a", "s", "-", 2.6, True),
        ("warp tiling",  "#bdbdbd", "D", "--", 1.6, False),
        ("warp autotune","#cb181d", "^", "-", 2.6, True),
        ("cuBLAS",       "#000000", "", "--", 1.4, False),
    ]),
]

GROUP_COLOR = {
    "naive": "#9ecae1", "coalescing": "#4292c6", "shared mem": "#08519c",
    "1D block": "#41ab5d", "2D block": "#006d2c",
    "vectorize": "#807dba", "warp tiling": "#54278f",
    "tunable 2D": "#fb6a4a", "warp autotune": "#cb181d",
}
GROUP_TAG = {
    "naive": "访存", "coalescing": "访存", "shared mem": "访存",
    "1D block": "寄存器分块", "2D block": "寄存器分块",
    "vectorize": "向量化/warp", "warp tiling": "向量化/warp",
    "tunable 2D": "自动调优", "warp autotune": "自动调优",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def style_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_ylim(50, 120000)
    ax.grid(which="major", axis="y", alpha=0.25, zorder=0)
    ax.set_xlabel("size (M=N=K)")
    ax.set_ylabel("GFLOPS")


def make_grid() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for ax, (title, series) in zip(axes.ravel(), GROUPS):
        for label, color, marker, ls, lw, annotate in series:
            g = SERIES[label] if label != "cuBLAS" else CUBLAS
            ax.plot(SIZES, g, ls, color=color, lw=lw, marker=marker or None,
                    ms=5 if marker else 0, mfc="white", mec=color, mew=1.3,
                    zorder=4 if annotate else 3, label=label)
            if annotate:
                ax.annotate(f"{g[-1]/1000:.1f} TF", (SIZES[-1], g[-1]),
                            textcoords="offset points", xytext=(-4, 8),
                            ha="right", fontsize=8, fontweight="bold",
                            color=color)
        style_axis(ax)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", DOCS / "ablation.png")


def make_speedup() -> None:
    base = SERIES["naive"][-1]
    order = ["naive", "coalescing", "shared mem", "1D block", "2D block",
             "vectorize", "warp tiling", "tunable 2D", "warp autotune"]
    labels = [l for l in order]
    sp = [SERIES[l][-1] / base for l in order]
    colors = [GROUP_COLOR[l] for l in order]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, sp, color=colors, height=0.62, zorder=3)
    for yi, (lab, v) in zip(y, zip(labels, sp)):
        ax.text(v + 1.4, yi, f"{v:.1f}×", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(1.0, color="#999999", lw=1, ls=":", zorder=2)
    ax.set_xlabel("speedup vs naive @4096 (×)")
    ax.set_title("GEMM 优化累计加速比 @4096", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_xlim(0, max(sp) * 1.14)
    fig.tight_layout()
    fig.savefig(DOCS / "ablation_speedup.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", DOCS / "ablation_speedup.png")


def make_recovery() -> None:
    cu = CUBLAS[-1]
    order = ["naive", "coalescing", "shared mem", "1D block", "2D block",
             "vectorize", "warp tiling", "tunable 2D", "warp autotune"]
    labels = list(order)
    pct = [SERIES[l][-1] / cu * 100 for l in order]
    colors = [GROUP_COLOR[l] for l in order]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, pct, color=colors, height=0.62, zorder=3)
    for yi, (lab, v) in zip(y, zip(labels, pct)):
        ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(100.0, color="#000000", lw=1.4, ls="--", zorder=2)
    ax.text(100.0, len(labels) - 0.35, "cuBLAS = 100%", va="bottom",
            ha="right", fontsize=9, color="#000000")
    ax.set_xlabel("相对 cuBLAS 的恢复程度 @4096 (%)")
    ax.set_title("GEMM 各版本相对 cuBLAS 的恢复程度 @4096",
                 loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_xlim(0, 115)
    fig.tight_layout()
    fig.savefig(DOCS / "ablation_cublas.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", DOCS / "ablation_cublas.png")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    make_grid()
    make_speedup()
    make_recovery()
