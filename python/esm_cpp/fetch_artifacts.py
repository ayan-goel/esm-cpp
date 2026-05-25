"""esm-cpp-fetch-artifacts: download pre-built Apple artifacts from a
GitHub Release into the local cache that Model::Load* auto-discovers.

Usage:
    esm-cpp-fetch-artifacts --model esm2_t33_650M
    esm-cpp-fetch-artifacts --model esm2_t6_8M --shapes 1x256,8x256
    esm-cpp-fetch-artifacts --model esm2_t6_8M --list
    esm-cpp-fetch-artifacts --model esm2_t6_8M --repo file:///tmp/my-mirror

The release tag defaults to `v<esm_cpp.__version__>` — every release pins
its own artifact set so the SHA-256 trace check in Model::Load* always
agrees with the engine. Artifacts extract into `~/.cache/esm_cpp/<key>/`
where `<key>` matches the HF model id (e.g., `esm2_t33_650M_UR50D`).

Pure stdlib implementation: `urllib.request` for downloads (with a
small progress hook), `hashlib` for SHA-256 verification against the
release's `manifest.json`, `tarfile` for extraction, `json` for the
manifest. No new Python dependencies — `[project.dependencies]` in
pyproject.toml stays at numpy only.

`--repo file:///some/path` is honored everywhere a URL would be — used
by the local round-trip test in tests/python/test_fetch_artifacts.py to
exercise the CLI without network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import esm_cpp


# Same mapping table as esm_cpp.convert / esm_cpp.bench.compare — keeping
# them in sync is a maintenance task that lives in the publisher tool
# (T7) which validates the table against the release manifest.
_HF_ID_FOR_SHORTHAND: dict[str, str] = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
}

# `esm-cpp` GitHub repo for fetching artifacts. Override with --repo at
# install time / per-invocation when mirroring. The placeholder value is
# clearly not a real org — by the time we publish, T8 / a maintainer
# updates this to the real "owner/name".
_DEFAULT_REPO = "esmcpp/esm-cpp"

CACHE_KEY_FROM_HF: dict[str, str] = {
    hf_id: hf_id.split("/", 1)[-1]
    for hf_id in _HF_ID_FOR_SHORTHAND.values()
}


def _platform_cache_dir() -> Path:
    """Mirrors ArtifactCache::EffectiveCacheDir() in src/artifact_cache.cpp."""
    env = os.environ.get("ESM_CPP_CACHE_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        lap = os.environ.get("LOCALAPPDATA")
        if lap:
            return Path(lap) / "esm_cpp"
        up = os.environ.get("USERPROFILE")
        if up:
            return Path(up) / ".cache" / "esm_cpp"
    home = os.environ.get("HOME")
    if home:
        return Path(home) / ".cache" / "esm_cpp"
    return Path(tempfile.gettempdir()) / "esm_cpp"


def _url_for(repo: str, tag: str, asset: str) -> str:
    """Build the download URL for a release asset.

    `repo` is either `owner/name` (a GitHub Releases URL is constructed)
    or a `file:///path` (asset is read locally — used by tests).
    """
    if repo.startswith("file://"):
        # Strip the scheme and join — file:// URLs don't get a tag
        # subdirectory; the local mirror is laid out flat.
        base = repo[len("file://"):]
        return f"file://{base.rstrip('/')}/{asset}"
    # GitHub Releases asset URL.
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _download(url: str, dest: Path, *, dry_run: bool = False) -> None:
    """Stream a URL to `dest`, printing a coarse progress line.

    Atomic: writes to a `.tmp`, renames on success. Resumable in the
    sense that we don't redownload if the dest is already present (the
    SHA check at the caller catches truncated old downloads).
    """
    if dry_run:
        print(f"  [dry-run] would fetch {url}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  fetching {url}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        shutil.copyfile(parsed.path, tmp)
    else:
        # Stream with a small progress hook every 16 MB.
        def _reporthook(blocknum: int, blocksize: int, total: int) -> None:
            if total <= 0:
                return
            downloaded = blocknum * blocksize
            if downloaded == 0 or (downloaded // (16 * 1024 * 1024)) != \
                    ((downloaded - blocksize) // (16 * 1024 * 1024)):
                pct = 100.0 * downloaded / total
                print(f"    {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB "
                      f"({pct:.0f}%)")
        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
    os.replace(tmp, dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_manifest(repo: str, tag: str) -> dict[str, Any]:
    """Pull the release's manifest.json (small, no caching needed)."""
    url = _url_for(repo, tag, "manifest.json")
    print(f"  fetching {url}")
    try:
        with urllib.request.urlopen(url) as resp:
            return json.load(resp)
    except Exception as e:
        raise SystemExit(
            f"failed to fetch manifest from {url}: {e}\n"
            f"(make sure --repo / --tag point at a published esm-cpp "
            f"release with the artifacts uploaded)") from e


def _parse_shapes(s: str) -> list[str] | None:
    """`--shapes 1x256,8x256` -> ['B1-L256', 'B8-L256']. None on empty."""
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            b, l = tok.lower().split("x")
            out.append(f"B{int(b)}-L{int(l)}")
        except (ValueError, IndexError):
            raise SystemExit(f"bad --shapes token {tok!r} (want NxN)")
    return out


def _select_assets(
    manifest: dict[str, Any], model_short: str,
    shape_filter: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return [(asset_name, asset_info), ...] for the requested model+shapes.

    asset_info is the per-asset metadata block (file, size_mb, sha256, ...).
    """
    models = manifest.get("models", {})
    if model_short not in models:
        raise SystemExit(
            f"model {model_short!r} not in release manifest. Available: "
            f"{sorted(models.keys())}")
    m = models[model_short]
    out: list[tuple[str, dict[str, Any]]] = []
    if "amx_fp16" in m:
        out.append((m["amx_fp16"]["file"], m["amx_fp16"]))
    for shape_key, info in m.get("whole_graph", {}).items():
        if shape_filter is not None and shape_key not in shape_filter:
            continue
        out.append((info["file"], info))
    return out


def _extract(tarball: Path, dest: Path, *, dry_run: bool = False) -> None:
    """Extract `tarball` into `dest`, creating dirs as needed."""
    if dry_run:
        print(f"  [dry-run] would extract {tarball.name} -> {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, mode="r:gz") as tar:
        # Python 3.12+: pass filter to silence a deprecation warning;
        # 'data' is the safe default (rejects absolute paths + traversal).
        tar.extractall(dest, filter="data")


def _resolve_model(arg: str) -> tuple[str, str]:
    """`--model esm2_t33_650M` -> (short, cache_key).

    Accepts a shorthand (esm2_t33_650M) or an HF id
    (facebook/esm2_t33_650M_UR50D). The cache_key is what
    ArtifactCache::CacheKeyFor produces from the HF cache path.
    """
    if arg in _HF_ID_FOR_SHORTHAND:
        hf_id = _HF_ID_FOR_SHORTHAND[arg]
    else:
        hf_id = arg
        if hf_id not in CACHE_KEY_FROM_HF and "/" not in hf_id:
            raise SystemExit(
                f"unknown --model {arg!r}; pass either a shorthand "
                f"({sorted(_HF_ID_FOR_SHORTHAND.keys())}) or the full HF id.")
    if "/" not in hf_id:
        cache_key = hf_id
    else:
        cache_key = hf_id.split("/", 1)[-1]
    return arg, cache_key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="ESM-2 shorthand (esm2_t33_650M) or HF id "
                          "(facebook/esm2_t33_650M_UR50D).")
    ap.add_argument("--shapes", default="",
                    help="Comma-separated BxL filter for the whole-graph "
                          "downloads (e.g., '1x256,8x256'). Default: all.")
    ap.add_argument("--list", action="store_true",
                    help="Print the release manifest's artifact list "
                          "for --model and exit (no download).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be downloaded + extracted and exit.")
    ap.add_argument("--repo", default=_DEFAULT_REPO,
                    help=f"<owner>/<repo> for GitHub Releases, OR "
                          f"file:///path to a local mirror (for tests). "
                          f"Default: {_DEFAULT_REPO}")
    ap.add_argument("--tag", default=None,
                    help="Release tag (default: v<esm_cpp.__version__>).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Cache root (default: $ESM_CPP_CACHE_DIR or platform "
                          "default).")
    ap.add_argument("--keep-tarballs", action="store_true",
                    help="Keep the downloaded .tar.gz files in the cache "
                          "dir (default: delete after extraction).")
    args = ap.parse_args()

    model_arg, cache_key = _resolve_model(args.model)
    tag = args.tag or f"v{esm_cpp.__version__}"
    out_root = args.out or _platform_cache_dir()
    target_dir = out_root / cache_key
    print(f"[fetch] model={model_arg} (cache_key={cache_key})")
    print(f"        repo={args.repo}  tag={tag}")
    print(f"        target={target_dir}")

    manifest = _fetch_manifest(args.repo, tag)
    shape_filter = _parse_shapes(args.shapes)
    assets = _select_assets(manifest, model_arg, shape_filter)
    if not assets:
        raise SystemExit(
            f"no matching artifacts in release {tag} for model={model_arg} "
            f"shapes={args.shapes or '(all)'}")

    if args.list:
        print(f"[list] {len(assets)} asset(s) available:")
        for name, info in assets:
            print(f"  {name}  ({info.get('size_mb', '?')} MB, "
                  f"sha256={info.get('sha256', '?')[:12]}…)")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    n_fetched = 0
    for asset_name, info in assets:
        url = _url_for(args.repo, tag, asset_name)
        tarball = target_dir / asset_name
        if tarball.is_file():
            sha = _sha256(tarball)
            if sha == info.get("sha256"):
                print(f"  skip: {asset_name} (already up to date)")
            else:
                print(f"  re-fetching {asset_name} (SHA mismatch)")
                tarball.unlink()
                _download(url, tarball, dry_run=args.dry_run)
        else:
            _download(url, tarball, dry_run=args.dry_run)

        if not args.dry_run:
            actual = _sha256(tarball)
            expected = info.get("sha256")
            if expected and actual != expected:
                tarball.unlink()
                raise SystemExit(
                    f"SHA mismatch on {asset_name}: got {actual}, "
                    f"expected {expected}. Tarball deleted; re-run to retry.")
        _extract(tarball, target_dir, dry_run=args.dry_run)
        if not args.keep_tarballs and not args.dry_run:
            tarball.unlink()
        n_fetched += 1

    print(f"[done] {n_fetched} artifact(s) extracted into {target_dir}")
    print(f"  ESM_CPP_CACHE_DIR={out_root}  ->  Model::Load* will auto-engage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
