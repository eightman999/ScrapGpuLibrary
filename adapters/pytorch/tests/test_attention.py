#!/usr/bin/env python3
"""Independent PyTorch SDPA oracles and boundary tests (CPU or selected OpenCL)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scrapgpu_torch as sgt


def rejects(fn, message: str) -> None:
    try:
        fn()
    except (RuntimeError, TypeError, ValueError) as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError(f"expected rejection containing {message!r}")


def main() -> int:
    torch.set_num_threads(1)
    # Failure to load is a failure, not a silent skip. Set backend before starting.
    sgt.load()
    torch.manual_seed(2026)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 7, 8)
    v = torch.randn(2, 4, 7, 6)
    cases = 0

    def compare(name, query=q, key=k, value=v, **kwargs):
        nonlocal cases
        got = sgt.scaled_dot_product_attention(query, key, value, **kwargs)
        oracle_kwargs = dict(kwargs)
        if oracle_kwargs.get("attn_mask") is not None and oracle_kwargs["attn_mask"].dim() < 2:
            # PyTorch's fused CPU entrypoint requires at least two mask axes.
            oracle_kwargs["attn_mask"] = oracle_kwargs["attn_mask"].expand(query.size(2), key.size(2))
        ref = F.scaled_dot_product_attention(query, key, value, **oracle_kwargs)
        torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-5)
        assert got.is_contiguous() and not got.requires_grad
        cases += 1
        print(f"PASS: {name}")

    compare("dense")
    compare("causal rectangular", is_causal=True)
    compare("custom scale", scale=0.17)
    compare("zero scale", scale=0.0)
    compare("negative scale", scale=-0.17)
    compare("GQA", key=k[:, :2].contiguous(), value=v[:, :2].contiguous(), enable_gqa=True)
    compare("MQA", key=k[:, :1].contiguous(), value=v[:, :1].contiguous(), enable_gqa=True)
    compare("additive 2D mask", attn_mask=torch.randn(5, 7) * 0.3)
    compare("additive broadcast mask", attn_mask=torch.randn(2, 1, 1, 7) * 0.3)
    compare("per-head mask", attn_mask=torch.randn(4, 5, 7) * 0.3)
    visible = torch.rand(5, 7) > 0.4
    visible[1] = False  # Fully masked rows must be exactly zero, not NaN or uniform.
    compare("bool mask including fully masked row", attn_mask=visible)
    compare("fully masked additive row", attn_mask=torch.where(visible, 0.0, -float("inf")))
    compare("large finite negative mask", attn_mask=torch.full((5, 7), -1e38))
    compare("bool scalar mask", attn_mask=torch.tensor(True))
    compare("all-masked scalar", attn_mask=torch.tensor(False))

    # Decode offset has no direct F.sdpa keyword; create its equivalent bool mask.
    for offset in (2, 20):
        causal_mask = torch.arange(k.size(2))[None, :] <= torch.arange(q.size(2))[:, None] + offset
        got = sgt.scaled_dot_product_attention(q, k, v, is_causal=True, query_position_offset=offset)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-5)
        cases += 1
        print(f"PASS: causal KV offset {offset}")

    # Force one-query tiles; compare to a separate PyTorch implementation.
    budget = 4 * k.size(2) * (v.size(-1) + 2)
    got = sgt.scaled_dot_product_attention(q, k, v, attn_mask=visible, max_workspace_bytes=budget)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=visible)
    torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-5)
    cases += 1
    print("PASS: one-query workspace tiling")

    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, max_workspace_bytes=budget - 1, attn_mask=visible), "workspace")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, max_workspace_bytes=-1), "nonnegative")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, query_position_offset=-1), "nonnegative")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, dropout_p=0.1), "dropout_p")
    rejects(lambda: sgt.scaled_dot_product_attention(q.clone().requires_grad_(), k, v), "inference only")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, attn_mask=torch.zeros(5, 7, requires_grad=True)), "inference only")
    rejects(lambda: sgt.scaled_dot_product_attention(q.half(), k, v), "float32")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(5, 7, dtype=torch.int32)), "float32 or bool")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(2, 3)), "broadcast")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k[:, :2].contiguous(), v[:, :2].contiguous()), "enable_gqa")
    rejects(lambda: sgt.scaled_dot_product_attention(q[:, :3].contiguous(), k[:, :2].contiguous(), v[:, :2].contiguous(), enable_gqa=True), "divisible")
    rejects(lambda: sgt.scaled_dot_product_attention(q[0], k, v), "4-D")
    rejects(lambda: sgt.scaled_dot_product_attention(q.transpose(-1, -2), k, v), "contiguous")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, scale=float("nan")), "finite")
    rejects(lambda: sgt.scaled_dot_product_attention(q, k, v, scale=float("inf")), "finite")
    rejects(lambda: torch.ops.scrapgpu.scaled_dot_product_attention(q, k, v, enable_gqa="yes"), "bool")
    print("PASS: 16 schema/semantic rejections")

    # Native calls release the GIL; shared upload/execute/download must stay serialized.
    def concurrent_call(i):
        qi = q + i / 10
        got = sgt.scaled_dot_product_attention(qi, k, v, is_causal=bool(i % 2))
        expected = F.scaled_dot_product_attention(qi, k, v, is_causal=bool(i % 2))
        torch.testing.assert_close(got, expected, atol=3e-5, rtol=3e-5)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(concurrent_call, range(12)))
    print(f"PASS: {cases} attention oracles, 16 rejections, 12 concurrent calls; torch={torch.__version__}, backend={os.getenv('SCRAPGPU_TORCH_BACKEND', 'opencl11')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
