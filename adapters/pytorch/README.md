# PyTorch adapter (Stage A)

Custom `torch.ops.scrapgpu.*` ops and `torch.classes.scrapgpu.Weight` run compute on the ScrapGpu OpenCL backend. **Activations and results are CPU `float32` tensors**; each call uploads inputs and downloads outputs (host staging). Stage B may add PrivateUse1 device tensors.

## Build

From the repo root (needs OpenCL 1.1 backend):

```bash
cmake -S . -B build-pytorch \
  -DSCRAPGPU_BUILD_CORE=ON \
  -DSCRAPGPU_BUILD_OPENCL11=ON \
  -DSCRAPGPU_BUILD_PYTORCH=ON \
  -DPython3_EXECUTABLE=/path/to/venv/bin/python
cmake --build build-pytorch -j
```

The extension is written to `adapters/pytorch/scrapgpu_torch/_C.so`.

## Use

```python
import scrapgpu_torch as sgt

sgt.load()  # loads _C.so and opens device 0
print(sgt.has_device(), torch.ops.scrapgpu.device_count())

w = torch.classes.scrapgpu.Weight(torch.randn(32, 64), "q4_0")
y = torch.ops.scrapgpu.linear(x, w)  # x: [N, 64] or [64] CPU float32
```

`ScrapLinear` wraps weight upload and `linear` for a minimal `nn.Module`.

## Ops (CPU float in/out)

| Op | Signature | Notes |
| --- | --- | --- |
| `load` | `()` | Register backend, create stream on device 0 |
| `device_count` | `() -> int` | |
| `device_name` | `(i: int) -> str` | |
| `Weight` | `(tensor [out,in] f32, quant_type: str)` | GPU buffer; quant types use R32 upload |
| `linear` | `(x, weight) -> Tensor` | `MUL_MAT`; weight is `[out,in]` PyTorch layout |
| `add` | `(a, b)` | Same shape |
| `silu` | `(x)` | |
| `rms_norm` | `(x, w, eps)` | `RMS_NORM_MUL`; `w` length = last dim of `x` |
| `rope` | `(x, pos, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, mode)` | `x` shape `[head_dim, n_heads, n_tokens]`; `pos` int32 |

Quant types for `Weight`: `f32`, `f16`, `q4_0`, `q8_0`, `q4_k`, `q6_k`.

## Tests

```bash
/path/to/venv/bin/python adapters/pytorch/tests/test_ops.py
/path/to/venv/bin/python adapters/pytorch/examples/mlp.py
```

## Limits (Stage A)

- One global stream on device 0; not thread-safe.
- Every op copies full tensor payloads host ↔ device.
- No `nn.Parameter` GPU residency; `Weight` holds the packed GPU buffer only.
- No autograd through custom ops.

## Stage B open questions

- PrivateUse1 backend and required ATen registrations (`empty`, `_copy_from`, etc.).
- Torch version–locked ABI builds and wheel packaging.
- Pinned host memory / async copies overlapping with CPU.
- CUDA / Vulkan backends vs OpenCL-only Stage A.
