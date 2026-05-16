"""Phase 3 Slice 6: smoke test for the public benchmark harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _hf_cache_has_8m() -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / \
        "models--facebook--esm2_t6_8M_UR50D" / "snapshots"
    return cache.is_dir() and any(cache.glob("*/model.safetensors"))


@pytest.mark.skipif(not _hf_cache_has_8m(), reason="HF 8M cache missing")
def test_compare_synthetic_smoke(tmp_path: Path) -> None:
    """Synthetic batch, esm-cpp-fp32 mode only — no HF required."""
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, "-m", "esm_cpp.bench.compare",
         "--model", "esm2_t6_8M",
         "--modes", "esm-cpp-fp32",
         "--batch", "4", "--len", "32",
         "--warmup", "1", "--iters", "2",
         "--output", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"compare.py failed: stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}")
    assert out.exists()
    data = json.loads(out.read_text())
    assert "config" in data
    assert "hardware" in data
    assert "results" in data
    assert "esm-cpp-fp32" in data["results"]
    fp = data["results"]["esm-cpp-fp32"]
    assert fp["throughput_seqs_per_s"] > 0
    assert fp["mean_ms"] > 0


def test_render_benchmark_table(tmp_path: Path) -> None:
    """Take 1 result.json, render a markdown table into a fresh out file."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out = tmp_path / "docs" / "benchmarks.md"
    sample = {
        "config": {"model": "esm2_t6_8M", "num_sequences": 4},
        "hardware": {"cpu_model": "Test CPU", "platform": "test"},
        "results": {
            "esm-cpp-fp32": {
                "throughput_seqs_per_s": 100.0,
                "mean_ms": 40.0,
                "isa": "neon",
            },
        },
        "speedup_fp32_vs_hf": None,
    }
    (results_dir / "row.json").write_text(json.dumps(sample))
    proc = subprocess.run(
        [sys.executable, "tools/render_benchmark_table.py",
         "--results", str(results_dir),
         "--out", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    body = out.read_text()
    assert "Test CPU" in body
    assert "esm2_t6_8M / esm-cpp-fp32" in body
    assert "100.00" in body
