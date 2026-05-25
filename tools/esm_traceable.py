"""Phase 13: clean, trace-friendly ESM-2 masked-LM forward.

HF transformers 5.9 wraps ESM-2's forward in a vmap-based mask-construction
layer that emits ops (`new_ones`, `new_zeros` in mask-functions) the CoreML
PyTorch frontend does not implement. We can't trace HF directly.

This module *reimplements* the ESM-2 forward bit-for-bit, loads weights from
the HF `EsmForMaskedLM` checkpoint, and uses only ops coremltools can convert:

  - `nn.Embedding`              → `mb.gather`
  - `nn.LayerNorm`              → `mb.layer_norm`
  - `nn.Linear`                 → `mb.linear`
  - `torch.cat / slice`         → `mb.concat / slice_by_index`
  - `torch.matmul`              → `mb.matmul`
  - `torch.softmax`             → `mb.softmax`
  - `torch.erf`, `torch.cos/sin` → unary elementwise

Three load-bearing ESM-2 quirks (see CLAUDE.md):
  1. Half-then-half RoPE (NOT interleaved).
  2. token_dropout 0.88 rescale at inference (compensates for masked-token
     training distribution), gated on `attention_mask.sum`.
  3. Q is scaled by 1/sqrt(d_head) BEFORE RoPE, not after.

Parity vs HF is asserted by the spike script. If you change anything in this
file, re-run `tools/spike_whole_graph_smoke.py` against HF eager.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EsmCfg:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_eps: float
    max_position_embeddings: int
    pad_token_id: int
    mask_token_id: int
    token_dropout: bool
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, hf_config) -> "EsmCfg":
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            layer_norm_eps=hf_config.layer_norm_eps,
            max_position_embeddings=hf_config.max_position_embeddings,
            pad_token_id=hf_config.pad_token_id,
            mask_token_id=hf_config.mask_token_id,
            token_dropout=bool(hf_config.token_dropout),
        )


def gelu_erf(x: torch.Tensor) -> torch.Tensor:
    """ESM's exact GELU (the one HF documents as 'subtly different from F.gelu')."""
    return x * 0.5 * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


def rotate_half(x: torch.Tensor, half_dim: int) -> torch.Tensor:
    # NOTE: half-then-half (Llama/GPT-NeoX), NOT interleaved (RoFormer/GPT-J).
    # See esm/rotary_embedding.py in facebookresearch/esm. `half_dim` is
    # head_dim // 2, passed as a Python int so the trace doesn't emit an
    # aten::Int on a dynamic shape (coremltools chokes on those).
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               half_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin shapes: (1, L, head_dim).  q/k shapes: (B, H, L, head_dim).
    cos = cos.unsqueeze(1)  # (1, 1, L, head_dim)
    sin = sin.unsqueeze(1)
    q_rope = (q * cos) + (rotate_half(q, half_dim) * sin)
    k_rope = (k * cos) + (rotate_half(k, half_dim) * sin)
    return q_rope, k_rope


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for a fixed sequence length.

    Traced shape is fixed (L is known at convert time), so we just register
    the (1, L, head_dim) cos/sin buffers — no per-step computation.
    """

    def __init__(self, head_dim: int, seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = positions[:, None] * inv_freq[None, :]              # (L, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)                     # (L, head_dim)
        self.register_buffer("cos", emb.cos()[None, :, :], persistent=False)  # (1, L, head_dim)
        self.register_buffer("sin", emb.sin()[None, :, :], persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos, self.sin


class SelfAttention(nn.Module):
    def __init__(self, cfg: EsmCfg, batch: int, seq_len: int):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.half_dim = cfg.head_dim // 2
        # Fixed shape constants — bake into the graph so the trace emits no
        # dynamic-shape ops (which coremltools rejects with aten::Int errors).
        self.batch = batch
        self.seq_len = seq_len
        self.hidden = cfg.hidden_size
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.scale = self.head_dim ** -0.5  # NOTE: applied to Q before RoPE

    def forward(self, x: torch.Tensor, mask_add: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D).  mask_add: (B, 1, 1, L) additive (0 or -inf).
        B, L, H, Dh = self.batch, self.seq_len, self.num_heads, self.head_dim
        # NOTE: scale Q before RoPE, not after. RoPE is norm-preserving only in
        # exact arithmetic; order matters for bit-equivalence to the checkpoint.
        q = (self.query(x) * self.scale).view(B, L, H, Dh).transpose(1, 2)
        k = self.key(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.value(x).view(B, L, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, self.half_dim)
        # attention: (B, H, L, L)
        attn = torch.matmul(q, k.transpose(-1, -2)) + mask_add
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                  # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EsmCfg, batch: int, seq_len: int):
        super().__init__()
        self.attn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = SelfAttention(cfg, batch, seq_len)
        self.attn_out = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.ffn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor, mask_add: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        a = self.attn_ln(x)
        a = self.attn(a, mask_add, cos, sin)
        a = self.attn_out(a)
        x = x + a
        f = self.ffn_ln(x)
        f = self.fc1(f)
        f = gelu_erf(f)
        f = self.fc2(f)
        return x + f


class EsmMaskedLMTraceable(nn.Module):
    """Traceable ESM-2 MLM.

    Forward expects two int32 tensors of fixed shape (B, L):
      - input_ids
      - attention_mask  (1 for real tokens, 0 for pad)
    Returns logits (B, L, vocab_size) as fp32.
    """

    def __init__(self, cfg: EsmCfg, batch: int, seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.batch = batch
        self.seq_len = seq_len

        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([EncoderLayer(cfg, batch, seq_len) for _ in range(cfg.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.rope = RotaryEmbedding(cfg.head_dim, seq_len, cfg.rope_theta)

        # LM head
        self.lm_dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.lm_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.lm_decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        # NOTE: ESM-2 hardcodes the train-time mask ratio (0.15 * 0.8). At
        # inference with no masked tokens (mask_token_id absent from input_ids),
        # this rescales all embeddings by 1 / (1 - mask_ratio_train) = 1/0.88.
        # Skipping this gives qualitatively correct but quantitatively wrong logits.
        self._mask_ratio_train = 0.15 * 0.8

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids, attention_mask: int32 of shape (B, L)
        mask_f = attention_mask.to(torch.float32)                     # (B, L)
        embeds = self.word_embeddings(input_ids)                      # (B, L, D)

        if self.cfg.token_dropout:
            # WHY: ESM-2 rescales all embeddings by 0.88 at inference even when no
            # tokens are masked. Skipping this gives qualitatively correct but
            # quantitatively wrong logits. See esm/model/esm2.py.
            is_mask = (input_ids == self.cfg.mask_token_id).to(torch.float32)         # (B, L)
            embeds = embeds * (1.0 - is_mask.unsqueeze(-1))
            src_lengths = mask_f.sum(dim=-1, keepdim=True)                            # (B, 1)
            # mask_ratio_observed per sample (mask_token_id count / src_length)
            mask_ratio_observed = is_mask.sum(dim=-1, keepdim=True) / src_lengths     # (B, 1)
            scale = (1.0 - self._mask_ratio_train) / (1.0 - mask_ratio_observed)      # (B, 1)
            embeds = embeds * scale.unsqueeze(-1)                                     # (B, L, D)

        # Apply attention_mask to embeddings (HF does this after LayerNorm in
        # EsmEmbeddings, but ESM-2's `emb_layer_norm_before=False` means there's
        # no pre-encoder LayerNorm here. We multiply directly to zero-out pad rows.
        embeds = embeds * mask_f.unsqueeze(-1)

        # Build additive attention mask: 0 where mask=1, -inf where mask=0.
        # Shape: (B, 1, 1, L).  Use -1e4 not -inf so fp16 softmax stays sane.
        mask_add = (1.0 - mask_f) * (-1.0e4)
        mask_add = mask_add.unsqueeze(1).unsqueeze(1)                # (B, 1, 1, L)

        cos, sin = self.rope()                                       # (1, L, head_dim), (1, L, head_dim)

        x = embeds
        for layer in self.layers:
            x = layer(x, mask_add, cos, sin)

        x = self.final_ln(x)

        # LM head
        h = self.lm_dense(x)
        h = gelu_erf(h)
        h = self.lm_ln(h)
        logits = self.lm_decoder(h) + self.lm_bias                   # (B, L, vocab_size)
        return logits


# ---------- weight loading ----------


def _ln_state(prefix: str, hf_sd: dict) -> dict:
    return {"weight": hf_sd[prefix + ".weight"], "bias": hf_sd[prefix + ".bias"]}


def _linear_state(prefix: str, hf_sd: dict) -> dict:
    return {"weight": hf_sd[prefix + ".weight"], "bias": hf_sd[prefix + ".bias"]}


def load_from_hf(model: EsmMaskedLMTraceable, hf_model) -> EsmMaskedLMTraceable:
    """Copy weights from a Huggingface EsmForMaskedLM into the traceable model."""
    sd = hf_model.state_dict()

    # word_embeddings
    model.word_embeddings.weight.data.copy_(sd["esm.embeddings.word_embeddings.weight"])

    cfg = model.cfg
    for i in range(cfg.num_hidden_layers):
        lp = f"esm.encoder.layer.{i}"
        L = model.layers[i]

        # attention pre-LN (HF stores at .attention.LayerNorm)
        L.attn_ln.weight.data.copy_(sd[f"{lp}.attention.LayerNorm.weight"])
        L.attn_ln.bias.data.copy_(sd[f"{lp}.attention.LayerNorm.bias"])

        # q/k/v
        L.attn.query.weight.data.copy_(sd[f"{lp}.attention.self.query.weight"])
        L.attn.query.bias.data.copy_(sd[f"{lp}.attention.self.query.bias"])
        L.attn.key.weight.data.copy_(sd[f"{lp}.attention.self.key.weight"])
        L.attn.key.bias.data.copy_(sd[f"{lp}.attention.self.key.bias"])
        L.attn.value.weight.data.copy_(sd[f"{lp}.attention.self.value.weight"])
        L.attn.value.bias.data.copy_(sd[f"{lp}.attention.self.value.bias"])

        # attention output projection (HF: attention.output.dense)
        L.attn_out.weight.data.copy_(sd[f"{lp}.attention.output.dense.weight"])
        L.attn_out.bias.data.copy_(sd[f"{lp}.attention.output.dense.bias"])

        # ffn pre-LN (HF: layer.LayerNorm)
        L.ffn_ln.weight.data.copy_(sd[f"{lp}.LayerNorm.weight"])
        L.ffn_ln.bias.data.copy_(sd[f"{lp}.LayerNorm.bias"])

        # fc1 / fc2
        L.fc1.weight.data.copy_(sd[f"{lp}.intermediate.dense.weight"])
        L.fc1.bias.data.copy_(sd[f"{lp}.intermediate.dense.bias"])
        L.fc2.weight.data.copy_(sd[f"{lp}.output.dense.weight"])
        L.fc2.bias.data.copy_(sd[f"{lp}.output.dense.bias"])

    # final encoder LN (HF: esm.encoder.emb_layer_norm_after)
    model.final_ln.weight.data.copy_(sd["esm.encoder.emb_layer_norm_after.weight"])
    model.final_ln.bias.data.copy_(sd["esm.encoder.emb_layer_norm_after.bias"])

    # LM head
    model.lm_dense.weight.data.copy_(sd["lm_head.dense.weight"])
    model.lm_dense.bias.data.copy_(sd["lm_head.dense.bias"])
    model.lm_ln.weight.data.copy_(sd["lm_head.layer_norm.weight"])
    model.lm_ln.bias.data.copy_(sd["lm_head.layer_norm.bias"])
    # decoder weight (tied to word_embeddings in HF; we don't tie here)
    model.lm_decoder.weight.data.copy_(sd["lm_head.decoder.weight"])
    model.lm_bias.data.copy_(sd["lm_head.bias"])

    return model.eval()
