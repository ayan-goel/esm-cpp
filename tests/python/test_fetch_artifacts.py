"""Phase 14 T6: smoke tests for esm-cpp-fetch-artifacts CLI.

Stages a fake release tree under a temp dir (manifest.json + a single
tarball with a known sha256), points the CLI at `--repo file:///tmp/...`,
and confirms the cache dir is populated correctly + the SHA check fires
on tampering.

No network access. Doesn't depend on Apple-only artifacts — the tarball
content is just a marker text file inside the expected layout.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_fake_release(release_root: Path, model_short: str = "esm2_t6_8M",
                       hf_id: str = "facebook/esm2_t6_8M_UR50D") -> dict:
    """Create a tarball + manifest under release_root.

    Layout (mirrors a real GitHub Releases dump):
        release_root/
            manifest.json
            esm2-8m-whole-graph-B1-L67.tar.gz
              -> whole-graph/B-1_L-67/whole_graph.mlmodelc/marker.txt
    """
    release_root.mkdir(parents=True, exist_ok=True)

    # Build a tarball whose contents extract to ~/.cache/esm_cpp/<key>/
    # following the auto-load layout the C++ ArtifactCache expects.
    staging = release_root / "_staging"
    bundle = staging / "whole-graph" / "B-1_L-67" / "whole_graph.mlmodelc"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "marker.txt").write_text("fake-mlmodelc-content\n")

    tarball = release_root / "esm2-8m-whole-graph-B1-L67.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        # Add the whole-graph/ subtree at the tarball root.
        tar.add(staging / "whole-graph", arcname="whole-graph")

    sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
    size_mb = tarball.stat().st_size / 1e6

    manifest = {
        "engine_version": "0.0.0-test",
        "release_tag": "vTEST",
        "trace_sha": "test-trace-sha",
        "models": {
            model_short: {
                "hf_id": hf_id,
                "cache_key": hf_id.replace("/", "--"),
                "whole_graph": {
                    "B-1_L-67": {
                        "file": tarball.name,
                        "size_mb": round(size_mb, 2),
                        "sha256": sha,
                        "corr_vs_hf": 0.999999,
                    }
                },
            }
        },
    }
    (release_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Cleanup staging — the test only needs the tarball + manifest.
    import shutil
    shutil.rmtree(staging)
    return manifest


def _run_cli(args: list[str], env_overrides: dict[str, str] | None = None,
              cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the CLI as a subprocess (matches the installed-script path)."""
    import os
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "esm_cpp.fetch_artifacts", *args],
        env=env,
        capture_output=True, text=True,
        cwd=cwd or REPO_ROOT,
    )


def test_fetch_local_repo_round_trip(tmp_path: Path) -> None:
    """Fake release tree -> fetch_artifacts CLI -> populated cache."""
    release_root = tmp_path / "fake_release"
    _make_fake_release(release_root)
    cache_root = tmp_path / "cache"

    p = _run_cli(
        ["--model", "esm2_t6_8M",
         "--repo", f"file://{release_root}",
         "--tag", "vTEST",
         "--out", str(cache_root)],
    )
    assert p.returncode == 0, f"stderr:\n{p.stderr}\nstdout:\n{p.stdout}"

    # The expected extracted layout: <cache>/<key>/whole-graph/B-1_L-67/...
    extracted = cache_root / "facebook--esm2_t6_8M_UR50D" / "whole-graph" / \
        "B-1_L-67" / "whole_graph.mlmodelc" / "marker.txt"
    assert extracted.is_file(), \
        f"missing extracted file at {extracted}\nstdout:\n{p.stdout}"
    assert extracted.read_text() == "fake-mlmodelc-content\n"

    # Tarball should be deleted by default (no --keep-tarballs).
    leftover = list((cache_root / "facebook--esm2_t6_8M_UR50D").glob("*.tar.gz"))
    assert not leftover, f"unexpected leftover tarball(s): {leftover}"


def test_list_prints_manifest(tmp_path: Path) -> None:
    release_root = tmp_path / "fake_release"
    _make_fake_release(release_root)
    p = _run_cli(
        ["--model", "esm2_t6_8M",
         "--repo", f"file://{release_root}",
         "--tag", "vTEST",
         "--list",
         "--out", str(tmp_path / "cache")],
    )
    assert p.returncode == 0, f"stderr:\n{p.stderr}"
    assert "esm2-8m-whole-graph-B1-L67.tar.gz" in p.stdout
    # --list does NOT extract.
    assert not (tmp_path / "cache").is_dir() or not any(
        (tmp_path / "cache").rglob("marker.txt"))


def test_shapes_filter(tmp_path: Path) -> None:
    """--shapes filter excludes non-matching whole-graph shapes."""
    release_root = tmp_path / "fake_release"
    _make_fake_release(release_root)
    # Filter to a shape that doesn't exist -> should error helpfully.
    p = _run_cli(
        ["--model", "esm2_t6_8M",
         "--repo", f"file://{release_root}",
         "--tag", "vTEST",
         "--shapes", "1x256",
         "--out", str(tmp_path / "cache")],
    )
    assert p.returncode != 0
    assert "no matching artifacts" in (p.stderr + p.stdout)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    release_root = tmp_path / "fake_release"
    _make_fake_release(release_root)
    cache_root = tmp_path / "cache"
    p = _run_cli(
        ["--model", "esm2_t6_8M",
         "--repo", f"file://{release_root}",
         "--tag", "vTEST",
         "--dry-run",
         "--out", str(cache_root)],
    )
    assert p.returncode == 0, f"stderr:\n{p.stderr}"
    assert "dry-run" in p.stdout
    # Cache dir may exist (we mkdir before iterating) but should be empty.
    if cache_root.is_dir():
        residues = list(cache_root.rglob("*"))
        # Allow the empty <key>/ subdir to exist (mkdir before extract loop).
        assert all(p.is_dir() for p in residues), \
            f"dry-run wrote files: {residues}"


def test_sha_mismatch_aborts(tmp_path: Path) -> None:
    """If the manifest's sha256 doesn't match the tarball, abort + cleanup."""
    release_root = tmp_path / "fake_release"
    _make_fake_release(release_root)
    # Tamper with the manifest: claim a wrong SHA.
    manifest_path = release_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"]["esm2_t6_8M"]["whole_graph"]["B-1_L-67"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2))

    cache_root = tmp_path / "cache"
    p = _run_cli(
        ["--model", "esm2_t6_8M",
         "--repo", f"file://{release_root}",
         "--tag", "vTEST",
         "--out", str(cache_root)],
    )
    assert p.returncode != 0
    assert "SHA mismatch" in (p.stderr + p.stdout)
    # Tarball should NOT be left lying around after a failed SHA check.
    leftover = list(cache_root.rglob("*.tar.gz"))
    assert not leftover, f"leftover tarball after SHA failure: {leftover}"
