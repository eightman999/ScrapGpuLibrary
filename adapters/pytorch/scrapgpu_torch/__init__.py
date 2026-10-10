"""Inference-only ScrapGpu custom ops with CPU tensors and explicit CPU/OpenCL selection."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

__all__ = ["load", "ScrapLinear", "has_device", "scaled_dot_product_attention"]


def _default_ext_path() -> Path:
    return Path(__file__).resolve().parent / "_C.so"


def load(library_path: str | os.PathLike | None = None) -> None:
    """Load the extension and initialize SCRAPGPU_TORCH_BACKEND / SCRAPGPU_TORCH_DEVICE.

    Defaults are opencl11 and 0. Set environment variables before the first load,
    device query, or custom op: configuration is fixed for the process afterwards.
    """
    path = Path(library_path) if library_path is not None else _default_ext_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"scrapgpu_torch extension not found at {path}. "
            "Build with -DSCRAPGPU_BUILD_PYTORCH=ON; OpenCL is optional."
        )
    torch.ops.load_library(str(path))
    torch.ops.scrapgpu.load()


def has_device() -> bool:
    """Return whether the selected backend has devices; this fixes backend configuration."""
    try:
        torch.ops.load_library(str(_default_ext_path()))
        return int(torch.ops.scrapgpu.device_count()) > 0
    except (OSError, RuntimeError):
        return False


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
    query_position_offset: int = 0,
    max_workspace_bytes: int = 64 * 1024 * 1024,
) -> torch.Tensor:
    """Run explicit inference SDPA on contiguous CPU float32 [B,H,L,D] tensors.

    Mask is float32 additive or bool (True means visible), broadcastable to
    [B,H,L,S]. Offset supports KV-cache decoding: causal visibility is
    key_index <= query_index + query_position_offset. Workspace bounds backend
    scratch allocations, excluding input/output transfers. Dropout and autograd
    are rejected. This does not replace torch.nn.functional attention globally.
    """
    return torch.ops.scrapgpu.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
        query_position_offset, max_workspace_bytes,
    )


class ScrapLinear(nn.Module):
    """Linear layer using scrapgpu.linear on CPU float activations (no bias)."""

    def __init__(self, in_features: int, out_features: int, quant_type: str = "f32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=5**0.5)
        self.register_buffer("weight", w)
        self._scrap_weight = None
        self._weight_version = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference tensors have no version counter, so conservatively repack them.
        version = None if torch.is_inference(self.weight) else (self.weight.data_ptr(), self.weight._version)
        if self._scrap_weight is None or version is None or self._weight_version != version:
            self._scrap_weight = torch.classes.scrapgpu.Weight(self.weight, self.quant_type)
            self._weight_version = version
        return torch.ops.scrapgpu.linear(x, self._scrap_weight)
