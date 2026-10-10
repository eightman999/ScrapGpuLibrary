#!/usr/bin/env python3
"""Two-layer MLP: ScrapLinear (selected ScrapGpu backend) vs nn.Linear on CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scrapgpu_torch as sgt  # noqa: E402


class TorchMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class ScrapMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, quant: str = "f32") -> None:
        super().__init__()
        self.fc1 = sgt.ScrapLinear(d_in, d_hidden, quant_type=quant)
        self.fc2 = sgt.ScrapLinear(d_hidden, d_out, quant_type=quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.ops.scrapgpu.silu(self.fc1(x))
        return self.fc2(h)


@torch.no_grad()
def main() -> int:
    if not sgt.has_device():
        print("No selected ScrapGpu device; nothing to run.")
        return 0

    sgt.load()
    torch.manual_seed(0)
    d_in, d_hidden, d_out = 64, 128, 32
    batch = 4

    ref = TorchMLP(d_in, d_hidden, d_out)
    scrap = ScrapMLP(d_in, d_hidden, d_out, quant="q4_0")
    scrap.fc1.weight.copy_(ref.fc1.weight)
    scrap.fc2.weight.copy_(ref.fc2.weight)

    x = torch.randn(batch, d_in)
    y_ref = ref(x)
    y_scrap = scrap(x)

    diff = (y_ref - y_scrap).abs().max().item()
    print(f"max abs diff (q4_0 MLP vs nn.Linear): {diff:.6e}")
    print("OK" if diff < 0.05 else "large diff (expected for q4_0 vs f32 reference)")

    scrap_f32 = ScrapMLP(d_in, d_hidden, d_out, quant="f32")
    scrap_f32.fc1.weight.copy_(ref.fc1.weight)
    scrap_f32.fc2.weight.copy_(ref.fc2.weight)
    y_f32 = scrap_f32(x)
    diff_f32 = (y_ref - y_f32).abs().max().item()
    print(f"max abs diff (f32 ScrapMLP vs nn.Linear): {diff_f32:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
