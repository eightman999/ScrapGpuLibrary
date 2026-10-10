# Third-party notices

ScrapGpuLibrary is licensed under the GNU General Public License v3.0 or later (see [LICENSE](LICENSE)).
It contains or is derived from the following third-party work, which keeps its own license.

## ggml / llama.cpp

- Source: https://github.com/ggml-org/llama.cpp (developed against b11384, commit `0faee50`)
- License: MIT (full text below)
- What is derived from it:
  - `adapters/ggml/`: the ggml backend implements the ggml backend interface (`ggml-backend-impl.h`) and follows the structure of the ggml backends; the first version was developed in a branch of llama.cpp.
  - `adapters/ggml/llama.cpp/in-tree-registration.patch`: a patch to llama.cpp source files (the context lines are llama.cpp code).
  - `src/backends/opencl11/kernels/*.cl`: the quantized block formats (Q4_0, Q8_0, Q4_K, Q6_K) and their dequantization follow `ggml-common.h` / `ggml-quants.c`, so the weights are bit-compatible with GGUF.
  - `src/backends/vulkan/shaders/*.comp`: the elementwise-binary, RMS-norm and mat-vec/mat-mat shader logic (stride/index handling, the Q4_0/Q8_0 block layouts and their dequantization) is ported from the ggml-vulkan backend's `vulkan-shaders` (`generic_binary_head.glsl`, `rms_norm.comp`, `mul_mat_vec_base.glsl`, `dequant_funcs.glsl`, `types.glsl`). The Vulkan host code of `src/backends/vulkan/` is new; only the shader logic is ported.
  - `src/backends/opencl11/api.cpp`: the rope yarn corr-dims helper (`api_rope_yarn_corr_dims`) follows `ggml.c`.
  - `src/tensor/`: the descriptor conventions (ne/nb order, block-quantized dtypes, the definition of contiguity) follow the ggml tensor layout, so the ggml adapter converts without reshaping.
  - CPU reference implementations of the quantized formats, where they are ported from `ggml-quants.c` (`src/core/ref/quant.cpp`).

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Adding third-party code

Follow [docs/REUSE.md](docs/REUSE.md): MIT / BSD / BSL-1.0 / CC0 and Apache-2.0 code may be ported with its notice (Apache-2.0: add its full text and mark changed files, carry any NOTICE). Add an entry to this file for every port. GPL tools such as clpeak and mixbench are only run as external programs.

## Vulkan-Headers

- Source: https://github.com/KhronosGroup/Vulkan-Headers, tag `v1.4.365` (only `include/`, `README.md` and the license files are kept, in `third_party/Vulkan-Headers/`)
- License: Apache-2.0 or MIT (per file, see `third_party/Vulkan-Headers/LICENSE.md` and `LICENSES/`)
- Used by the Vulkan backend (`src/backends/vulkan/`) to build without the system Vulkan SDK.

## volk

- Source: https://github.com/zeux/volk, tag `vulkan-sdk-1.4.363.0` (`volk.c`, `volk.h`, `LICENSE.md`, `README.md`, in `third_party/volk/`; the upstream CI files were dropped)
- License: MIT (`third_party/volk/LICENSE.md`)
- The Vulkan meta-loader of the Vulkan backend: loads `libvulkan` at run time.
