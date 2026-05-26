"""Phase 14 T7: maintainer-side publisher for Apple artifacts.

Builds the full artifact set for a given ESM-2 size, tarballs each
(model, kind, shape), writes the top-level manifest.json, and (only with
--push) uploads each asset via `gh release upload` to a specified
release tag.

Not run in CI. Used once per release by a maintainer with `gh auth login`
already set up. Default is --dry-run: build + tarball locally, no push.

Usage:
    # Dry-run: stage everything under /tmp/esm-cpp-publish-650m/
    /tmp/ct312/bin/python tools/publish_apple_artifacts.py \\
        --model esm2_t33_650M \\
        --out /tmp/esm-cpp-publish-650m

    # Actually push to v0.2.0 release on the project repo
    /tmp/ct312/bin/python tools/publish_apple_artifacts.py \\
        --model esm2_t33_650M \\
        --out /tmp/esm-cpp-publish-650m \\
        --push --tag v0.2.0

The fetch CLI (T6) is the consumer-side mirror — `gh release upload`
puts the tarballs + manifest at exactly the URLs the fetch CLI fetches.

Requires the convert-time env (Python 3.12 + coremltools + torch +
transformers). The runtime fetch path (T6) needs none of those.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

# Convert-time deps imported lazily so --help doesn't require them.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _artifact_manifest import compute_trace_sha, write_manifest  # noqa: E402


# Mapping table -- keep in sync with python/esm_cpp/fetch_artifacts.py.
_HF_ID_FOR_SHORTHAND: dict[str, str] = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
}

# Per-model size-short -> the standard shape sweep we ship at release time.
# 1xL covers single-sequence inference; 8xL covers OAS-style batches.
_DEFAULT_SHAPES = {
    "esm2_t6_8M":   ["1x256", "8x256", "1x512", "1x1024"],
    "esm2_t12_35M": ["1x256", "8x256", "1x512", "1x1024"],
    "esm2_t30_150M": ["1x256", "8x256", "1x512", "1x1024"],
    # 650M whole-graph @ 1x1024 is ~1.3 GB; 8x1024 same. Per-shape decision.
    "esm2_t33_650M": ["1x256", "8x256", "1x512", "1x1024"],
}


def _short_size_tag(model_short: str) -> str:
    """`esm2_t33_650M` -> `650m`, used in tarball names."""
    # Take the trailing alphanum chunk and lowercase.
    last = model_short.rsplit("_", 1)[-1]
    return last.lower()


def _safetensors_path_for(hf_id: str) -> Path:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / \
        f"models--{hf_id.replace('/', '--')}" / "snapshots"
    if not cache.is_dir():
        raise SystemExit(
            f"HF cache miss for {hf_id}; run `huggingface-cli download {hf_id}`")
    cands = list(cache.glob("*/model.safetensors"))
    if not cands:
        raise SystemExit(f"no model.safetensors under {cache}")
    return cands[0]


def _run(cmd: list[str], *, check: bool = True, capture: bool = False
         ) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def _build_amx(model_short: str, hf_id: str, work_dir: Path) -> Path:
    """Build the AMX artifact set under work_dir/amx/."""
    sft = _safetensors_path_for(hf_id)
    amx_dir = work_dir / "amx-fp16"
    if amx_dir.is_dir():
        shutil.rmtree(amx_dir)
    amx_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "tools/build_amx_artifacts.py",
          "--safetensors", str(sft),
          "--precision", "fp16",
          "--out", str(amx_dir)])
    # build_amx_artifacts writes a manifest at amx_dir/esm_cpp_artifact.json
    # but its model_id is the safetensors path. Overwrite with the HF id.
    write_manifest(
        amx_dir,
        kind="amx-fp16",
        model_id=hf_id,
        precision="fp16",
        compute_units="CPU_ONLY",  # AMX path uses CPU_ONLY (BNNSGraph)
        tool_version=f"esm-cpp publish_apple_artifacts",
    )
    return amx_dir


def _build_whole_graph(model_short: str, hf_id: str, shapes: list[str],
                        work_dir: Path) -> Path:
    """Build the whole-graph artifacts for the requested shapes."""
    wg_dir = work_dir / "whole-graph"
    if wg_dir.is_dir():
        shutil.rmtree(wg_dir)
    wg_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "tools/build_whole_graph_artifacts.py",
          "--model", hf_id,
          "--shapes", ",".join(shapes),
          "--out", str(wg_dir),
          "--precision", "fp16",
          "--compute-units", "CPU_AND_NE"])
    return wg_dir


def _tarball(src_dir: Path, tar_name: str, out_dir: Path,
              arc_prefix: str) -> Path:
    """tar -czf out_dir/tar_name <src_dir's contents under arc_prefix/>.

    The arc_prefix matters: the fetch CLI extracts the tarball at the
    cache key root (~/.cache/esm_cpp/<key>/), and the auto-loader
    expects amx-fp16/... or whole-graph/B-X_L-Y/... under that root.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / tar_name
    if dst.exists():
        dst.unlink()
    with tarfile.open(dst, "w:gz") as tar:
        for entry in sorted(src_dir.iterdir()):
            arcname = f"{arc_prefix}/{entry.name}" if arc_prefix else entry.name
            tar.add(entry, arcname=arcname)
    return dst


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gh_release_upload(repo: str, tag: str, tarball: Path) -> None:
    """Upload one asset via `gh release upload --clobber`."""
    _run(["gh", "release", "upload", tag, str(tarball),
          "--repo", repo, "--clobber"])


def _write_top_manifest(out_dir: Path, *, repo: str, tag: str,
                        model_short: str, hf_id: str,
                        amx_tar: Path | None, amx_corr: float | None,
                        wg_assets: list[dict[str, Any]]) -> Path:
    """Write the top-level manifest.json the fetch CLI consumes."""
    # Mirror ArtifactCache::CacheKeyFor in src/artifact_cache.cpp: HF
    # cache pattern .../models--<org>--<name>/ -> <org>--<name>.
    cache_key = hf_id.replace("/", "--") if "/" in hf_id else hf_id
    model_entry: dict[str, Any] = {
        "hf_id": hf_id,
        "cache_key": cache_key,
        "whole_graph": {
            e["shape_key"]: {
                "file": e["file"],
                "size_mb": e["size_mb"],
                "sha256": e["sha256"],
                "corr_vs_hf": e["corr_vs_hf"],
            } for e in wg_assets
        },
    }
    if amx_tar is not None:
        model_entry["amx_fp16"] = {
            "file": amx_tar.name,
            "size_mb": round(amx_tar.stat().st_size / 1e6, 2),
            "sha256": _sha256(amx_tar),
            "corr_vs_hf": amx_corr,
        }
    manifest = {
        "engine_version": "publish-dev",  # T8 hookup: read esm_cpp.__version__
        "release_tag": tag,
        "trace_sha": compute_trace_sha(),
        "repo": repo,
        "models": {model_short: model_entry},
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=list(_HF_ID_FOR_SHORTHAND),
                    help="ESM-2 shorthand to build artifacts for.")
    ap.add_argument("--shapes", default="",
                    help="Comma-separated BxL whole-graph shapes "
                          "(default: model-specific recommended set).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Local staging dir for tarballs + manifest.")
    ap.add_argument("--push", action="store_true",
                    help="Actually upload via `gh release upload`. "
                          "Requires --tag and an existing GitHub release.")
    ap.add_argument("--tag", default=None,
                    help="Release tag to upload to (e.g., v0.2.0). "
                          "Required with --push.")
    ap.add_argument("--repo", default="ayan-goel/esm-cpp",
                    help="Target <owner>/<repo> for --push. Default: "
                          "ayan-goel/esm-cpp")
    ap.add_argument("--skip-amx", action="store_true",
                    help="Skip the per-Linear AMX build (whole-graph only).")
    ap.add_argument("--skip-whole-graph", action="store_true",
                    help="Skip the whole-graph build (AMX only).")
    args = ap.parse_args()

    hf_id = _HF_ID_FOR_SHORTHAND[args.model]
    shapes = (args.shapes.split(",") if args.shapes
              else _DEFAULT_SHAPES[args.model])
    size_tag = _short_size_tag(args.model)
    work_dir = args.out / "work"
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.push and not args.tag:
        raise SystemExit("--push requires --tag")
    if args.push:
        # Verify gh is installed + authenticated up front so we fail fast.
        try:
            _run(["gh", "auth", "status"], capture=True)
        except FileNotFoundError as e:
            raise SystemExit(
                "gh CLI not found. Install via `brew install gh` (or your "
                "platform's equivalent), then `gh auth login`.") from e
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                "gh CLI not authenticated. Run `gh auth login` first.") from e

    wg_assets: list[dict[str, Any]] = []
    amx_tar: Path | None = None

    if not args.skip_amx:
        print(f"\n=== AMX-fp16 build ({args.model}) ===")
        amx_dir = _build_amx(args.model, hf_id, work_dir)
        amx_tar = _tarball(
            amx_dir, f"esm2-{size_tag}-amx-fp16.tar.gz", out_dir,
            arc_prefix="amx-fp16")
        print(f"  -> {amx_tar} ({amx_tar.stat().st_size/1e6:.1f} MB)")
        if args.push:
            _gh_release_upload(args.repo, args.tag, amx_tar)

    if not args.skip_whole_graph:
        print(f"\n=== whole-graph build ({args.model}, shapes={shapes}) ===")
        wg_dir = _build_whole_graph(args.model, hf_id, shapes, work_dir)
        # Tarball each shape dir individually so users with --shapes filter
        # don't have to download the whole graph for every shape.
        for shape_dir in sorted(wg_dir.iterdir()):
            if not shape_dir.is_dir():
                continue
            name = shape_dir.name  # e.g., B-1_L-256
            # Re-format B-1_L-256 -> B1-L256 in the tarball name for
            # readability + URL safety.
            try:
                b_part, l_part = name.split("_")
                tar_short = f"{b_part.replace('B-', 'B')}-{l_part.replace('L-', 'L')}"
            except ValueError:
                tar_short = name
            tar_name = f"esm2-{size_tag}-whole-graph-{tar_short}.tar.gz"
            tar_path = _tarball(
                shape_dir, tar_name, out_dir,
                arc_prefix=f"whole-graph/{shape_dir.name}")
            sha = _sha256(tar_path)
            wg_assets.append({
                "shape_key": name,
                "file": tar_name,
                "size_mb": round(tar_path.stat().st_size / 1e6, 2),
                "sha256": sha,
                "corr_vs_hf": None,  # T6+T7 leave this null until the
                                     # per-shape spot-check is wired in
            })
            print(f"  -> {tar_path} ({wg_assets[-1]['size_mb']} MB)")
            if args.push:
                _gh_release_upload(args.repo, args.tag, tar_path)

    manifest_path = _write_top_manifest(
        out_dir, repo=args.repo, tag=args.tag or "(dry-run)",
        model_short=args.model, hf_id=hf_id,
        amx_tar=amx_tar, amx_corr=None,
        wg_assets=wg_assets,
    )
    print(f"\nmanifest -> {manifest_path}")

    if args.push:
        _gh_release_upload(args.repo, args.tag, manifest_path)
        print(f"\n[done] uploaded to https://github.com/{args.repo}/releases/tag/{args.tag}")
    else:
        print(f"\n[dry-run] staged at {out_dir}. Re-run with --push --tag <vX.Y.Z> to publish.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
