"""Type stubs for the esm.cpp pybind11 module."""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

__version__: str

def current_isa() -> str: ...
def host_isa() -> str: ...

class ActivationObserver:
    def __init__(self) -> None: ...
    def percentile(self, pctile: float) -> dict[str, float]: ...
    def clear(self) -> None: ...

class Tokenizer:
    vocab_size: int
    cls_id: int
    pad_id: int
    eos_id: int
    unk_id: int
    mask_id: int
    model_max_length: int

    def __init__(self) -> None: ...
    def encode(self, text: str, add_special: bool = True,
               truncate: bool = True) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...
    def token_to_id(self, token: str) -> int: ...
    def id_to_token(self, id: int) -> str: ...

class Config:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    layer_norm_eps: float
    token_dropout: bool
    mask_token_id: int
    weights_quantized: bool

class Model:
    num_threads: int
    @property
    def config(self) -> Config: ...
    @property
    def workspace_capacity_bytes(self) -> int: ...
    @staticmethod
    def load_from_safetensors(path: str) -> Model: ...
    def quantize_weights(self) -> None: ...
    def apply_smoothquant(self, act_stats: dict[str, float], alpha: float) -> None: ...
    def forward_with_observer(
        self,
        input_ids: npt.NDArray[np.int32],
        attention_mask: npt.NDArray[np.int32] | None,
        observer: ActivationObserver,
    ) -> npt.NDArray[np.float32]: ...
    def forward(
        self,
        input_ids: npt.NDArray[np.int32],
        attention_mask: npt.NDArray[np.int32] | None = None,
    ) -> npt.NDArray[np.float32]: ...
    def forward_with_hidden_states(
        self,
        input_ids: npt.NDArray[np.int32],
        attention_mask: npt.NDArray[np.int32] | None = None,
    ) -> tuple[npt.NDArray[np.float32], list[npt.NDArray[np.float32]]]: ...
    def forward_batch(
        self,
        input_ids: list[npt.NDArray[np.int32]],
        attention_masks: list[npt.NDArray[np.int32]] | None = None,
    ) -> list[npt.NDArray[np.float32]]: ...
