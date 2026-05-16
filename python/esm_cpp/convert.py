"""Convert an HF ESM-2 checkpoint to esm.cpp's GGUF v3 format.

Usage:
  python -m esm_cpp.convert --hf facebook/esm2_t6_8M_UR50D \\
      --out weights/esm2_8m.gguf

Loads the source via Model.load_from_safetensors and writes via
Model.save_to_gguf. No HF transformers dependency at run time — only
the safetensors file is read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import esm_cpp


_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_source(src: str) -> Path:
    p = Path(src)
    if p.is_file():
        return p
    snapshots = _HF_CACHE / f"models--{src.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        raise SystemExit(
            f"could not find a safetensors file for '{src}'. Pass a path "
            "to a model.safetensors file, or run "
            "`huggingface-cli download <hf-id>` first."
        )
    candidates = list(snapshots.glob("*/model.safetensors"))
    if not candidates:
        raise SystemExit(f"no model.safetensors under {snapshots}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True,
                        help="HF model ID (e.g. facebook/esm2_t6_8M_UR50D) "
                              "or path to a model.safetensors file.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Destination GGUF file path.")
    args = parser.parse_args()

    src = _resolve_source(args.hf)
    print(f"Loading {src}")
    model = esm_cpp.Model.load_from_safetensors(str(src))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.out}")
    model.save_to_gguf(str(args.out))
    size_mb = args.out.stat().st_size / (1024 * 1024)
    cfg = model.config
    print(
        f"Wrote {args.out} ({size_mb:.1f} MB) — "
        f"{cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}, "
        f"ffn={cfg.intermediate_size}, heads={cfg.num_attention_heads}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
