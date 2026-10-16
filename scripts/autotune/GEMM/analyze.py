#!/usr/bin/env python3
"""Analyze autotune results collected by ``search.py``.

Reads ``results.json`` (a flat list of per-(size, config) records), ranks
configs by GFLOPS, and prints:
  * the best config per problem size, and
  * an optional full ranking table.

Usage::

    python analyze.py                      # best per size
    python analyze.py --top 5              # top-5 per size
    python analyze.py --all                # full ranking table
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "results.json"


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def cfg_str(r: dict) -> str:
    parts = [f"threads={r['threads']:4d}"]
    if "WM" in r:  # warp-tiling record
        for p in ("BM", "BN", "BK", "WM", "WN", "LaneSizeN", "TM", "TN"):
            parts.append(f"{p}={r[p]:3d}")
    else:
        for p in ("BM", "BN", "BK", "TM", "TN"):
            parts.append(f"{p}={r[p]:3d}")
    return "  ".join(parts)


def group_by_size(records: list[dict]) -> dict[int, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        groups[r["size"]].append(r)
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze GEMM autotune results")
    ap.add_argument("--results", default=str(RESULTS_JSON))
    ap.add_argument("--top", type=int, default=1, help="configs to show per size")
    ap.add_argument("--all", action="store_true", help="print full ranking")
    ap.add_argument("--only-ok", action="store_true",
                    help="drop configs that failed correctness")
    args = ap.parse_args()

    records = load(Path(args.results))
    if args.only_ok:
        records = [r for r in records if r["ok"]]

    print(f"{'size':>6}  {'rank':>4}  {'gflops':>9}  {'ms':>9}  config")
    for size in sorted(group_by_size(records)):
        ranked = sorted(group_by_size(records)[size],
                        key=lambda r: r["gflops"], reverse=True)
        top = ranked if args.all else ranked[: args.top]
        for i, r in enumerate(top, 1):
            flag = "" if r["ok"] else "  [BAD]"
            print(f"{size:>6}  {i:>4}  {r['gflops']:>9.1f}  {r['ms']:>9.4f}"
                  f"  {cfg_str(r)}{flag}")


if __name__ == "__main__":
    main()
