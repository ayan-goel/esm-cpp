"""Render benchmarks/results/*.json into docs/benchmarks.md table.

Idempotent: re-running with new results.json files regenerates the
table block; commit history records every measurement.

Usage:
  python tools/render_benchmark_table.py \\
      --results benchmarks/results \\
      --out docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_TABLE_BEGIN = "<!-- BENCH_TABLE_BEGIN -->"
_TABLE_END = "<!-- BENCH_TABLE_END -->"


def _load_results(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"skipping {p}: {exc}")
    return out


def _row(name: str, host: str, isa: str, num_seqs: int, mean_ms: float,
          throughput: float, speedup: float | None) -> str:
    speedup_str = f"{speedup:.2f}x" if speedup else "—"
    return (
        f"| {name} | {host} | {isa} | {num_seqs} | {mean_ms:.1f} | "
        f"{throughput:.2f} | {speedup_str} |"
    )


def _render_table(results: list[dict[str, Any]]) -> str:
    if not results:
        return "_(no results.json files found)_"
    lines = [
        "| Run | Host | ISA | Seqs | Mean ms | Throughput (seqs/s) | Speedup vs HF |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        cfg = r.get("config", {})
        hw = r.get("hardware", {})
        runs = r.get("results", {})
        host = hw.get("cpu_model", hw.get("platform", "unknown"))
        name = cfg.get("model", "unknown")
        for mode, m in runs.items():
            speedup = None
            if mode == "esm-cpp-fp32":
                speedup = r.get("speedup_fp32_vs_hf")
            elif mode == "esm-cpp-int8":
                speedup = r.get("speedup_int8_vs_hf")
            lines.append(_row(
                f"{name} / {mode}", host,
                m.get("isa", "n/a"),
                cfg.get("num_sequences", 0),
                m.get("mean_ms", 0.0),
                m.get("throughput_seqs_per_s", 0.0),
                speedup,
            ))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Directory containing results.json files.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Markdown file to update.")
    args = parser.parse_args()

    results = _load_results(args.results)
    table = _render_table(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        body = args.out.read_text()
    else:
        body = (
            "# Benchmarks\n\n"
            "Phase 3 public benchmark results. Reproduction commands are\n"
            "in this file; raw JSON results live in `benchmarks/results/`.\n\n"
            f"{_TABLE_BEGIN}\n\n{_TABLE_END}\n"
        )
    if _TABLE_BEGIN in body and _TABLE_END in body:
        before, _, rest = body.partition(_TABLE_BEGIN)
        _, _, after = rest.partition(_TABLE_END)
        new = (f"{before}{_TABLE_BEGIN}\n\n{table}\n\n{_TABLE_END}{after}")
    else:
        new = body + f"\n\n{_TABLE_BEGIN}\n\n{table}\n\n{_TABLE_END}\n"
    args.out.write_text(new)
    print(f"updated {args.out} with {len(results)} result rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
