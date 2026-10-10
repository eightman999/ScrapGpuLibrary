# ScrapGpuLibrary

**A general-purpose tensor / runtime library that runs current ML software at usable speed on used and old-generation GPUs (the "scrap GPU" ones).**

GPUs whose driver and API are frozen (NVIDIA Fermi / Kepler, old Radeons) either cannot be used by current inference engines or are slow when they are. ScrapGpuLibrary provides the runtime, tensors, ops, quantization and backends for those GPUs without depending on a particular engine. ggml (llama.cpp), PyTorch and ONNX Runtime connect as adapters on top of it.

```
ScrapGpuLibrary
├─ runtime    device / buffer / stream, event / allocator / multi-GPU transport
├─ tensor     dtype / layout / view, stride
├─ ops        matmul, matvec / softmax / norm / rope / attention / elementwise
├─ quant      q4, q8 / k-quant / device-specific packing
├─ backends   opencl11 / opencl12 / vulkan / cuda
└─ adapters   ggml / PyTorch extension / ONNX Runtime / standalone C API
```

The responsibilities of each layer, the direction of the dependencies, the design assumptions and the migration plan are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). A Japanese version of this page is [README.md](README.md); other pages are Japanese only.

## Status

The first result is CL11, a ggml backend that runs on OpenCL 1.1 alone. Version 0.2 splits the single file that held every layer into runtime / backends / core / adapter (see [docs/plans/V0.2_SPLIT_BRIEF.md](docs/plans/V0.2_SPLIT_BRIEF.md)). `main` still holds the implementation from before the split.

| Stage | Content | State |
| --- | --- | --- |
| P0 | The CL11 ggml backend, per-GPU tuning, the out-of-tree plugin | Done. Gate 0 fixed the speed and correctness baselines ([docs/BASELINES.md](docs/BASELINES.md)) |
| P1 | Integration of the per-GPU tuning, fixed performance baselines and correctness tolerances | Done (Gate 1). The contract is [docs/API_CONTRACT.md](docs/API_CONTRACT.md). KL divergence tolerances are calibrated per model and text against the official backends |
| P2 | Splitting out the runtime and `backends/opencl11` | Done (Gate 2, 2026-10-09). device / context / kernels / buffer / executor moved into `src/backends/opencl11/` with no ggml includes. All six TB250 configurations are within the baseline (NVIDIA between -0.9% and +0.3%), and the launches per token, `test-backend-ops` and the greedy agreement are the same as before the split |
| P3 | Splitting out tensor / ops / quant and a C API callable without ggml | Done (Gate 3, 2026-10-10). GPU tensor ops run through the public C API alone (`examples/standalone_matmul.c`); the ggml-free op tests pass on all six TB250 configurations and the regression stays within the baseline |
| P4 | multi-GPU transport | Not started |
| P5 | More backends (opencl12 / vulkan / cuda) and adapters (PyTorch / ONNX Runtime) | Not started |

[ROADMAP.md](ROADMAP.md) collects what comes next.

### Where the code lives

| Path | Content |
| --- | --- |
| `src/backends/opencl11/` | The OpenCL 1.1 backend: device (capabilities, generation, tuning), context (lazy queue and program creation), kernels (kernel table, program build, `.cl` embedding), buffer (allocation, limits, lazy-allocation probe), executor (launch and profiling), device_state (the runtime state of a device), tensor_abi (the descriptors passed to the kernels). No ggml includes |
| `src/tensor/` | The tensor module, free of ggml and OpenCL: dtype, TensorDesc, views / transposition / broadcasting, division-free indexing, the R32 layout |
| `src/core/` | The common layer that depends on neither ggml nor OpenCL: the ABI version, op contract validation, backend registration, and the CPU reference implementation (`src/core/ref/`: every op, Q4_0 / Q8_0 / Q4_K / Q6_K quantization and dequantization) |
| `include/scrapgpu/` | The public C API (`scrapgpu_` prefix) and the contract headers |
| `adapters/ggml/` | The ggml adapter: the CL11 backend itself (`ggml-cl11.cpp`) and the verification tools (`mvm_check`, `attn_check`, `kepler-sweep`) |
| `tests/core/`, `tests/tensor/`, `tests/ref/` | The contract tests, the tensor module tests and the CPU reference tests (all without llama.cpp) |

### Building

The core and the contract tests build without llama.cpp.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

| CMake option | Default | Content |
| --- | --- | --- |
| `SCRAPGPU_BUILD_CORE` | `ON` | The common layer that uses neither ggml nor OpenCL (`scrapgpu_core`) |
| `SCRAPGPU_BUILD_OPENCL11` | `OFF` | The OpenCL 1.1 backend (`scrapgpu_opencl11`). Defines `CL_TARGET_OPENCL_VERSION=110` |
| `SCRAPGPU_BUILD_GGML_ADAPTER` | `OFF` | The ggml adapter. `ON` adds `adapters/ggml` |
| `SCRAPGPU_BUILD_TESTS` | `ON` | The contract tests and the dependency check |

- `ctest` runs the contract tests and the check that `src/core` and the public headers include neither ggml nor OpenCL, and `src/backends/opencl11` includes no ggml (`cmake/check_core_dependencies.cmake`).
- The ggml adapter is written for llama.cpp b11384. Both the out-of-tree plugin and the in-tree patch build are described in [adapters/ggml/](adapters/ggml/).
- The checks a change has to pass are in [CONTRIBUTING.md](CONTRIBUTING.md).

### CL11

- It is written with the OpenCL 1.1 host API and OpenCL C 1.1 kernels only. No fp16 arithmetic, no sub-groups, no API newer than OpenCL 1.1.
- The verified GPUs are the GeForce GT 430 (Fermi), GT 710 and GT 730 (Kepler) with the NVIDIA 390 driver, and the Radeon Pro WX 2100 (GCN, Clover and rusticl) and Radeon RX 6400 (RDNA2, rusticl) with Mesa. The tuning values are collected per GPU generation in `cl11_tuning_for()` (measured for Fermi, Kepler and AMD GCN, only the AMD-wide settings for AMD RDNA2).
- Llama / Qwen2 / Qwen3 models run with every layer offloaded (Flash Attention is not supported). LFM2 / LFM2-MoE (short convolutions, `MUL_MAT_ID`) run every op too (confirmed on a Mac; not measured on the TB250 GPUs). Q4_0 / Q8_0 / Q4_K / Q6_K weights are kept in a GPU-friendly repacked layout. Ops that are not supported fall back to the CPU.
- The supported ops, the performance, the environment variables, the limitations and the approaches that were tried and not kept are in [docs/CL11.md](docs/CL11.md).
- How to attach it to llama.cpp (out-of-tree plugin, in-tree patch) is in [adapters/ggml/](adapters/ggml/).

llama-bench (`-ngl 99 -ub 128 -b 128 -p 128 -n 64`, CPU: two threads of the Celeron G3930 of the same machine; pp128 / tg64 in t/s):

| Model | GPU | CL11 pp128 | CL11 tg64 | CPU pp128 | CPU tg64 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q8_0 | GT 430 | 89.4 | 22.3 | 25.9 | 15.6 |
| | GT 730 | 241.9 | 34.9 | 25.9 | 15.9 |
| | GT 710 | 120.0 | 14.7 | 26.0 | 15.7 |
| Qwen3-0.6B-Base Q4_K_M | GT 430 | 89.9 | 23.7 | 6.1 | 4.6 |
| | GT 730 | 231.6 | 39.2 | 6.2 | 4.6 |
| | GT 710 | 115.6 | 17.8 | 6.2 | 4.6 |
| Qwen2-0.5B-ft Q4_0 | GT 430 | 118.5 | 36.5 | 47.5 | 25.8 |

Qwen3-1.7B Q4_K_M (1.2GB) fits with all 29 layers offloaded on a GT 710 (2GB) (pp128 39.7 / tg64 7.5, CPU 2.0 / 1.6). The GT 710 has only 11.7 GB/s of bandwidth, so generating with Q8_0 is slower than the CPU (Q4_K_M is faster). On the Radeons the official Vulkan backend is faster, so CL11 is not used there ([docs/HARDWARE.md](docs/HARDWARE.md)).

## Test machine

Development happens on a BIOSTAR TB250-BTC PRO with a GeForce GT 430 / GT 710 / GT 730, a Radeon Pro WX 2100 and a Radeon RX 6400 on PCIe x1. The bandwidth of each GPU and the known driver quirks are in [docs/HARDWARE.md](docs/HARDWARE.md).

## Origin

CL11 was developed on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) b11384 (commit 0faee50). The first version in `adapters/ggml/` was taken from that development branch (`fermi-cl11`, commit `9ab3f557c`).

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later, [LICENSE](LICENSE)). The parts derived from ggml / llama.cpp (MIT) and their copyright notice are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
