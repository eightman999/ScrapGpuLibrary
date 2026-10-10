"""Stage A PyTorch adapter for ScrapGpuLibrary (CPU tensors in/out, GPU compute via custom ops)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

__all__ = ["load", "ScrapLinear", "has_device"]


def _default_ext_path() -> Path:
    return Path(__file__).resolve().parent / "_C.so"


def load(library_path: str | os.PathLike | None = None) -> None:
    """Load the C++ extension and initialize the default OpenCL device (device 0)."""
    path = Path(library_path) if library_path is not None else _default_ext_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"scrapgpu_torch extension not found at {path}. "
            "Build with -DSCRAPGPU_BUILD_PYTORCH=ON -DSCRAPGPU_BUILD_OPENCL11=ON."
        )
    torch.ops.load_library(str(path))
    torch.ops.scrapgpu.load()


def has_device() -> bool:
    """Return True if at least one ScrapGpu OpenCL device is visible (no GPU init)."""
    try:
        torch.ops.load_library(str(_default_ext_path()))
    except OSError:
        return False
    return int(torch.ops.scrapgpu.device_count()) > 0


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._scrap_weight is None:
            self._scrap_weight = torch.classes.scrapgpu.Weight(self.weight, self.quant_type)
        return torch.ops.scrapgpu.linear(x, self._scrap_weight)
