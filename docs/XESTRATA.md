# XeStrata (XS) decode primitives — CPU reference baseline

This first compatibility milestone adds four XeStrata/Qwen3.8 decode
primitives to the ordinary ScrapGpu C ABI. It is **not** an XeStrata adapter,
SYCL backend, or end-to-end model runner. The CPU reference fixes mathematical
semantics before individual GPU backends are developed.

Source of the *contracts*, not implementation code:
- https://github.com/MistVVK/XeStrata/blob/main/include/strata/kernels/gdn.hpp
- https://github.com/MistVVK/XeStrata/blob/main/include/strata/kernels/qsa.hpp

## Operator interface

Shapes are in ScrapGpu/ggml order: dimension 0 is the fastest dimension.
All GDN tensors here are F32, with natural layout.

| Operation | Input tensors | Output tensors | Parameters |
| --- | --- | --- | --- |
| SCRAPGPU_OP_GDN_STEP | state[S,Hv,S], q[S,Hk], k[S,Hk], v[S,Hv], gate[Hv], beta[Hv] | next_state[S,Hv,S], out[S,Hv] | none |
| SCRAPGPU_OP_GDN_CONV_STEP | state[d_conv-1,C], x[C], weights[d_conv,C] | next_state[d_conv-1,C], out[C] | none |
| SCRAPGPU_OP_GDN_L2_NORM | x[S,rows] (higher-dimensional row batches allowed) | y (same shape) | scrapgpu_rms_norm_params (eps) |
| SCRAPGPU_OP_QSA_STEP | pos[1] (I32) | step[4] (I32) | scrapgpu_qsa_step_params (idx_block, idx_top_k) |

### GDN recurrence

State is [S,Hv,S] in (j,h,i) index order, with j fastest. For
each value head h and column j, the source query/key head is
source = h % Hk (**modulo**, not interleave).

    decay      = exp(gate[h])
    sk         = sum_i (state[j,h,i] * decay) * k[i,source]
    delta      = (v[j,h] - sk) * beta[h]
    next[j,h,i]= state[j,h,i] * decay + k[i,source] * delta
    out[j,h]   = sum_i next[j,h,i] * q[i,source]

- q and k are already L2-normalized; q is already scaled by 1/sqrt(S).
- gate is **before** exp; beta is **after** sigmoid.
- Hv is a positive multiple of Hk. The GPU and CPU paths must agree on the
  state layout; the XeStrata host reference uses a different ordering.
- CPU accumulation is F32 in GDN_STEP. Use numerical tolerances when
  comparing GPU implementations, not token-exact inference output.
- An exact in-place state update (outputs[0] identical to inputs[0]) is safe.
  All partial overlaps and all other input/output aliases are rejected by
  the CPU backend.

### Causal GDN convolution

Input state contains d_conv-1 samples per channel, **oldest first**;
weights[d_conv-1,c] multiplies the incoming x[c]. The new state discards
the oldest sample and appends x[c]. d_conv must be at least 2. An exact
state input/output alias is allowed (shifting from low to high indices).

GDN_L2_NORM divides each row by sqrt(sum(x*x) + eps), **not**
sqrt(mean(x*x) + eps), as regular RMS_NORM does. eps must be finite and >0.
Exact input/output alias is supported.

### QSA captured-step metadata

The four I32 entries have a single writer and the following meanings:

| index | value |
| --- | --- |
| 0 | pos |
| 1 | n_kv = pos + 1 |
| 2 | n_bid = n_kv / idx_block (completed blocks) |
| 3 | width = min(n_kv, idx_top_k + idx_block - 1) |

Input position must be between 0 and INT32_MAX-1; idx_block and idx_top_k
are positive, with their sum fitting the I32 output. The XeStrata artifact
uses idx_block=4 and idx_top_k=2048: the identity selection holds through
2,051 cells (including the unfinished three-cell tail).

For graph capture, a GPU implementation must read this **device-resident
per-token step**, not freeze one host scalar position when the graph is
first captured. Doing the latter can silently make later tokens attend
to an unchanged first cached cell.

## Build and check

    cmake -S . -B build -DSCRAPGPU_BUILD_TESTS=ON
    cmake --build build -j
    ctest --test-dir build --output-on-failure -R scrapgpu_xs_ops_tests

The fixture covers independent analytic zero-state GDN, modulo head
mapping, second-step decay, in-place state, rejection of unsafe aliases,
convolution tap order, L2-vs-RMS normalization, QSA boundary cases,
invalid positions and invalid geometry.

The **CPU** backend and standalone CPU reference support these ops.
Existing OpenCL 1.1 / Vulkan backends must return UNSUPPORTED until an
actual implementation is added; there is no implicit GPU acceleration.

## Next integration stages

1. Add an XS adapter translating actual XeStrata tensor descriptors
   and state layout to the ScrapGpu C ABI.
2. Implement GDN and QSA metadata natively in CL11, then Vulkan, with
   CPU-reference parity tests and measured kernel-launch costs.
3. Add an optional SYCL backend through the existing backend vtable;
   do not replace CL11, especially on Fermi and Kepler.
4. Extend contracts for QSA paged KV gather, selection, PLE and
   MoE caching, profiling slow devices before assigning them layers.

No claims of full Qwen3.8-Flash-Next compatibility or 125B inference are
made by this milestone.
