#!/usr/bin/env python3
"""Compare scrapgpu custom ops with torch on CPU (float32). Skip when no OpenCL device."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Package root: adapters/pytorch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scrapgpu_torch as sgt  # noqa: E402


def nmse(got: torch.Tensor, ref: torch.Tensor) -> float:
    d = (got.double() - ref.double()).reshape(-1)
    r = ref.double().reshape(-1)
    num = (d * d).sum().item()
    den = (r * r).sum().item()
    return num / den if den > 0 else num


def main() -> int:
    if not sgt.has_device():
        print("SKIP: no ScrapGpu OpenCL device")
        return 0

    sgt.load()
    print(f"device: {torch.ops.scrapgpu.device_name(0)}")

    results: dict[str, float] = {}
    torch.manual_seed(42)

    # --- linear f32 ---
    k, m, n = 96, 24, 5
    w = torch.randn(m, k)
    x = torch.randn(n, k)
    wt = torch.classes.scrapgpu.Weight(w, "f32")
    y = torch.ops.scrapgpu.linear(x, wt)
    y_ref = F.linear(x, w)
    results["linear_f32"] = nmse(y, y_ref)

    # --- linear vector ---
    xv = torch.randn(k)
    yv = torch.ops.scrapgpu.linear(xv, wt)
    yv_ref = F.linear(xv.unsqueeze(0), w).squeeze(0)
    results["linear_vec_f32"] = nmse(yv, yv_ref)

    # --- quantized linear ---
    for q in ("q4_0", "q8_0", "q4_k", "q6_k"):
        kk = 256 if q in ("q4_k", "q6_k") else 96
        wq = torch.randn(m, kk)
        xq = torch.randn(3, kk)
        wobj = torch.classes.scrapgpu.Weight(wq, q)
        yq = torch.ops.scrapgpu.linear(xq, wobj)
        w_deq = torch.ops.scrapgpu.ref_dequant_linear_weight(wq, q)
        yq_ref = F.linear(xq, w_deq)
        results[f"linear_{q}"] = nmse(yq, yq_ref)

    # --- add ---
    a = torch.randn(11, 7)
    b = torch.randn(11, 7)
    results["add"] = nmse(torch.ops.scrapgpu.add(a, b), a + b)

    # --- silu ---
    t = torch.randn(13, 9)
    results["silu"] = nmse(torch.ops.scrapgpu.silu(t), F.silu(t))

    # --- rms_norm (LLaMA-style: norm then multiply by weight) ---
    rows, cols = 9, 37
    xn = torch.randn(rows, cols)
    wn = torch.randn(cols)
    eps = 1e-5
    y_rn = torch.ops.scrapgpu.rms_norm(xn, wn, eps)
    rms = xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + eps)
    y_rn_ref = rms * wn
    results["rms_norm"] = nmse(y_rn, y_rn_ref)

    # --- rope (ScrapGpu layout [head_dim, heads, tokens]) ---
    hd, nh, nt = 32, 3, 5
    xr = torch.randn(hd, nh, nt)
    pos = torch.randint(0, 100, (nt,), dtype=torch.int32)
    y_rope = torch.ops.scrapgpu.rope(xr, pos, hd, 128, 10000.0, 1.0, 0.0, 1.0, 32.0, 1.0, 0)
    y_rope_ref = torch.ops.scrapgpu.ref_rope(xr, pos, hd, 128, 10000.0, 1.0, 0.0, 1.0, 32.0, 1.0, 0)
    results["rope"] = nmse(y_rope, y_rope_ref)

    # --- zero-sized tensors must not crash (audit M-3: numel() / size(-1) divided by zero) ---
    e = torch.empty(0)
    results["add_empty"] = float(torch.ops.scrapgpu.add(e, e).numel() != 0)
    results["silu_empty"] = float(torch.ops.scrapgpu.silu(e).numel() != 0)
    results["rms_norm_empty"] = float(torch.ops.scrapgpu.rms_norm(e, e, 1e-5).numel() != 0)
    e3 = torch.empty(3, 0)
    results["add_empty_inner"] = float(torch.ops.scrapgpu.add(e3, e3).numel() != 0)
    xe = torch.empty(0, 0, 0)
    pe = torch.empty(0, dtype=torch.int32)
    results["rope_empty"] = float(
        torch.ops.scrapgpu.rope(xe, pe, 0, 128, 10000.0, 1.0, 0.0, 1.0, 32.0, 1.0, 0).numel() != 0
    )

    limit = 1e-6
    failed = []
    for name, v in results.items():
        ok = v < limit
        print(f"  {name}: NMSE {v:.3e} {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)

    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("all ops OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
