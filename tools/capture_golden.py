"""Capture HuggingFace ESM-2 forward-pass tensors as ground truth for esm.cpp.

Run once per model. Each invocation produces:
  - tests/golden/<model_short>/seq_<i>.npz  (i = 0..N-1)
  - tests/golden/<model_short>/manifest.json

Each npz contains:
  - sequence            : str
  - input_ids           : int32 [L]
  - attention_mask      : int32 [L]
  - hidden_state_<i>    : float32 [L, d] for i in [0, num_layers]
                          (i = 0 is post-embed; i = num_layers is post-final-LN)
  - logits              : float32 [L, vocab_size]
  - layer0_*            : layer-0 debug intermediates for bisecting bugs

Phase 0, Slice 3 of the plan.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer

CANONICAL_AA = "LAGVSERTIDPKQNFYMHWC"

MODEL_REGISTRY = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
}


def random_sequence(rng: random.Random, lo: int, hi: int) -> str:
    length = rng.randint(lo, hi)
    return "".join(rng.choices(CANONICAL_AA, k=length))


def _to_np(t: Any) -> Any:
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    if isinstance(t, tuple):
        return tuple(_to_np(x) for x in t)
    return t


def capture(model_short: str, num_seqs: int, seed: int, out_dir: Path,
            lo: int, hi: int) -> dict[str, Any]:
    hf_id = MODEL_REGISTRY[model_short]
    print(f"[{model_short}] loading {hf_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForMaskedLM.from_pretrained(hf_id)
    model.eval()
    cfg = model.config

    rng = random.Random(seed)
    sequences = [random_sequence(rng, lo, hi) for _ in range(num_seqs)]

    out_dir.mkdir(parents=True, exist_ok=True)

    layer0 = model.esm.encoder.layer[0]
    intermediates: dict[str, Any] = {}
    handles = []

    def make_hook(name: str):
        def fn(_module, _inputs, output):
            intermediates[name] = _to_np(output)
        return fn

    handles.append(layer0.attention.LayerNorm.register_forward_hook(
        make_hook("pre_attn_ln_output")))
    handles.append(layer0.attention.self.query.register_forward_hook(
        make_hook("q_proj_out")))
    handles.append(layer0.attention.self.key.register_forward_hook(
        make_hook("k_proj_out")))
    handles.append(layer0.attention.self.value.register_forward_hook(
        make_hook("v_proj_out")))
    handles.append(layer0.attention.self.rotary_embeddings.register_forward_hook(
        make_hook("rope_out")))
    handles.append(layer0.attention.self.register_forward_hook(
        make_hook("self_attn_out")))
    handles.append(layer0.attention.register_forward_hook(
        make_hook("attn_residual")))
    handles.append(layer0.LayerNorm.register_forward_hook(
        make_hook("pre_ffn_ln_output")))
    handles.append(layer0.intermediate.register_forward_hook(
        make_hook("intermediate_post_gelu")))

    try:
        for i, seq in enumerate(sequences):
            intermediates.clear()
            enc = tok(seq, add_special_tokens=True, return_tensors="pt")
            with torch.no_grad():
                output = model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"],
                               output_hidden_states=True)

            payload: dict[str, Any] = {
                "sequence": np.array(seq),
                "input_ids": enc["input_ids"].numpy().astype(np.int32)[0],
                "attention_mask": enc["attention_mask"].numpy().astype(np.int32)[0],
                "logits": output.logits.detach().numpy()[0],
            }
            for j, hs in enumerate(output.hidden_states):
                payload[f"hidden_state_{j}"] = hs.detach().numpy()[0]

            # Strip the leading batch dim (we capture single-seq inference).
            pre_ln_in = intermediates["pre_attn_ln_output"][0]      # [L, d]
            q_proj = intermediates["q_proj_out"][0]                  # [L, d]
            k_proj = intermediates["k_proj_out"][0]                  # [L, d]
            v_proj = intermediates["v_proj_out"][0]                  # [L, d]
            rope_q, rope_k = intermediates["rope_out"]               # ([B,H,L,d_h],)*2
            rope_q = rope_q[0]                                       # [H, L, d_h]
            rope_k = rope_k[0]
            self_attn = intermediates["self_attn_out"][0][0]         # tuple -> [B,L,d] -> [L,d]
            attn_res = intermediates["attn_residual"][0]             # [L, d]
            pre_ffn_ln = intermediates["pre_ffn_ln_output"][0]       # [L, d]
            inter = intermediates["intermediate_post_gelu"][0]       # [L, 4d]

            payload.update({
                "layer0_pre_attn_ln_output": pre_ln_in,
                "layer0_q_proj_out": q_proj,
                "layer0_k_proj_out": k_proj,
                "layer0_v_proj_out": v_proj,
                "layer0_q_after_rope": rope_q,
                "layer0_k_after_rope": rope_k,
                "layer0_self_attn_out": self_attn,
                "layer0_attn_residual": attn_res,
                "layer0_pre_ffn_ln_output": pre_ffn_ln,
                "layer0_intermediate_post_gelu": inter,
            })
            np.savez_compressed(out_dir / f"seq_{i:04d}.npz", **payload)
            if (i + 1) % 25 == 0 or i + 1 == num_seqs:
                print(f"  [{model_short}] {i + 1}/{num_seqs}", flush=True)
    finally:
        for h in handles:
            h.remove()

    manifest = {
        "model_short": model_short,
        "model_hf_id": hf_id,
        "num_sequences": num_seqs,
        "seed": seed,
        "sequence_length_range": [lo, hi],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "captured_at_utc": datetime.utcnow().isoformat() + "Z",
        "config": {
            "num_hidden_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "layer_norm_eps": cfg.layer_norm_eps,
            "token_dropout": cfg.token_dropout,
            "mask_token_id": cfg.mask_token_id,
            "emb_layer_norm_before": cfg.emb_layer_norm_before,
            "position_embedding_type": cfg.position_embedding_type,
        },
        "fields": {
            "sequence": "str",
            "input_ids": "int32 [L]",
            "attention_mask": "int32 [L]",
            "hidden_state_<i>": (
                f"float32 [L, {cfg.hidden_size}] for i in [0, {cfg.num_hidden_layers}]; "
                "0=post-embed, num_layers=post-final-LN"),
            "logits": f"float32 [L, {cfg.vocab_size}]",
            "layer0_*": "layer-0 debug intermediates for bisecting forward-graph bugs",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--len-lo", type=int, default=50)
    parser.add_argument("--len-hi", type=int, default=300)
    args = parser.parse_args()

    capture(args.model, args.num_seqs, args.seed, args.out,
            args.len_lo, args.len_hi)
    print(f"done: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
