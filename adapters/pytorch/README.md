# PyTorch adapter

Explicit `torch.ops.scrapgpu.*` inference operations and `torch.classes.scrapgpu.Weight` use either the CPU/RAM backend or a selected OpenCL 1.1 device. Inputs and outputs remain contiguous CPU `float32` tensors. GPU calls upload inputs and download outputs; packed weights persist across calls.

## Build

Use the same Python/PyTorch environment for compilation and execution. Rebuild the extension after changing PyTorch; its C++ ABI is version dependent.

```bash
cmake -S . -B build-pytorch \
  -DSCRAPGPU_BUILD_CORE=ON \
  -DSCRAPGPU_BUILD_OPENCL11=OFF \
  -DSCRAPGPU_BUILD_PYTORCH=ON \
  -DPython3_EXECUTABLE=/path/to/venv/bin/python
cmake --build build-pytorch -j
```

Set `SCRAPGPU_BUILD_OPENCL11=ON` to additionally support GPUs. CPU-only builds do not require OpenCL, CUDA, or an NVIDIA driver. The extension is written to `adapters/pytorch/scrapgpu_torch/_C.so`; add `adapters/pytorch` to `PYTHONPATH` when importing from elsewhere.

## Backend selection

Set environment variables **before the first load, device query, or operation**. Settings are captured once per process so an uploaded weight cannot silently move to another device. A new process is required to change devices. The selected backend is registered only once; querying or repeatedly loading does not duplicate devices.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCRAPGPU_TORCH_BACKEND` | `opencl11` | `cpu` or `opencl11` |
| `SCRAPGPU_TORCH_DEVICE` | `0` | Nonnegative index within the selected backend |
| `SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK` | `0` | `1` explicitly allows reference execution for unsupported primitive operations |
| `SCRAPGPU_TORCH_MAX_STAGING_BYTES` | `268435456` | Maximum host tensor staging for an opted-in fallback; zero disables staging |

Fallback is attempted only when native support reports `UNSUPPORTED`, never after a native execution failure. BF16 weight matmul on legacy OpenCL hardware requires opt-in fallback. SDPA uses the selected backend's F32 operations and its own workspace budget; the primitive fallback flag does not switch an entire attention call to CPU.

```python
import os
os.environ["SCRAPGPU_TORCH_BACKEND"] = "cpu"  # Or opencl11, with the desired device index.

import torch
import scrapgpu_torch as sgt

sgt.load()
print(torch.ops.scrapgpu.device_count(), torch.ops.scrapgpu.device_name(0))
w = torch.classes.scrapgpu.Weight(torch.randn(32, 64), "q4_0")
y = torch.ops.scrapgpu.linear(torch.randn(4, 64), w)
```

This adapter does not emulate CUDA APIs, replace NVIDIA drivers, or turn an old GPU into a supported CUDA device for a current PyTorch wheel. A CPU PyTorch wheel plus a compatible vendor OpenCL driver can call these explicit custom operations. It does not automatically offload arbitrary PyTorch models.

## Operations

| Operation | Inputs | Notes |
| --- | --- | --- |
| `Weight` | `[out,in]` F32 and quantization string | CPU uses natural quantized layout; GPU uses R32 |
| `linear` | `[N,K]` or `[K]`, `Weight` | No bias; output F32 |
| `add` | Two equal-shaped tensors | No broadcasting |
| `silu` | Tensor | Inference only |
| `rms_norm` | Tensor, last-dimension weight, epsilon | LLaMA-style RMS normalization |
| `rope` | Existing ScrapGpu layout and int32 positions | See the custom-op schema for RoPE parameters |
| `scaled_dot_product_attention` | Q/K/V and optional mask | Described below |

Weights support `f32`, `f16`, `bf16`, `q4_0`, `q8_0`, `q4_k`, and `q6_k`. Quantized rows must be multiples of the format's block size. F16/BF16 conversion uses PyTorch rounding. `ScrapLinear` caches the packed weight and refreshes it after normal in-place updates or `load_state_dict` changes. Mutation through `.data` bypasses PyTorch version tracking and is not supported.

All operations reject tensors that require gradients. Detach model weights/activations or prepare inference buffers under `torch.no_grad()` before calling the adapter. Calls, weight upload, and release serialize access to the shared stream, including calls from multiple Python threads.

## Attention

```python
q = torch.randn(2, 8, 3, 64)       # [B,Hq,L,D]
k = torch.randn(2, 2, 128, 64)    # [B,Hkv,S,D]
v = torch.randn(2, 2, 128, 48)    # [B,Hkv,S,Dv]
y = sgt.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    enable_gqa=True,
    query_position_offset=125,
    max_workspace_bytes=8 * 1024 * 1024,
)                                # [B,Hq,L,Dv]
```

`attn_mask` may be an additive F32 mask or a boolean mask (`True` means visible), broadcastable to `[B,Hq,L,S]`. Fully masked rows produce zero output. `scale=None` uses `1/sqrt(D)`; finite explicit scales, including zero, are accepted. GQA requires `Hq` to be divisible by `Hkv` and `enable_gqa=True` when the head counts differ. For causal KV-cache decoding, key `s` is visible exactly when `s <= query_position_offset + l`.

The workspace budget bounds temporary **device** buffers (transposed V, tiled scores, and a mask tile), excluding input/output buffers and host staging. A budget too small for one query tile fails explicitly. Tensor dimensions must be positive, and Q/K/V must be contiguous F32 CPU tensors. Batch broadcasting, dropout, autograd, and non-F32 activations are not supported. This function does not patch `torch.nn.functional.scaled_dot_product_attention`; call the wrapper or `torch.ops.scrapgpu.scaled_dot_product_attention` explicitly.

## Verification

```bash
SCRAPGPU_TORCH_BACKEND=cpu /path/to/venv/bin/python adapters/pytorch/tests/test_ops.py
SCRAPGPU_TORCH_BACKEND=cpu /path/to/venv/bin/python adapters/pytorch/tests/test_attention.py
/path/to/venv/bin/python adapters/pytorch/tests/test_backend_config.py
SCRAPGPU_TORCH_BACKEND=cpu /path/to/venv/bin/python adapters/pytorch/examples/mlp.py

# Repeat with each backend-local GPU index after an OpenCL-enabled build.
SCRAPGPU_TORCH_BACKEND=opencl11 SCRAPGPU_TORCH_DEVICE=0 \
  /path/to/venv/bin/python adapters/pytorch/tests/test_attention.py
SCRAPGPU_TORCH_BACKEND=opencl11 SCRAPGPU_TORCH_DEVICE=0 \
SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK=1 \
  /path/to/venv/bin/python adapters/pytorch/tests/test_ops.py
```

Attention tests compare against PyTorch's independent SDPA implementation, including masks, GQA/MQA, KV offsets, custom scales, one-query tiles, invalid inputs, and concurrent callers. The primitive test compares quantized matmul against independently executed PyTorch `linear` on dequantized weights. Device tests are process isolated because backend configuration is fixed after first use.
