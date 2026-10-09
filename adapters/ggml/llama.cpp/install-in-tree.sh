#!/usr/bin/env bash
# Install the CL11 backend into a llama.cpp tree (in-tree build, option GGML_CL11).
#
#   adapters/ggml/llama.cpp/install-in-tree.sh <llama.cpp dir>
#
# - copies CMakeLists.txt, ggml-cl11.cpp, OpenCL11 backend sources and kernels/ to <llama.cpp>/ggml/src/ggml-cl11/
#   and include/ggml-cl11.h to <llama.cpp>/ggml/include/
# - applies in-tree-registration.patch (CMake option GGML_CL11 + backend registry)
# Idempotent: files are only copied when they differ, an already applied patch is skipped.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
adapter="$(cd "$here/.." && pwd)"
repo="$(cd "$adapter/../.." && pwd)"
patch_file="$here/in-tree-registration.patch"
expected_commit="0faee50"   # llama.cpp b11384

if [ $# -ne 1 ] || [ ! -d "$1/ggml/src" ]; then
    echo "usage: $0 <llama.cpp dir>" >&2
    exit 2
fi
tree="$(cd "$1" && pwd)"

if commit="$(git -C "$tree" rev-parse --short=7 HEAD 2>/dev/null)"; then
    if [ "$commit" != "$expected_commit" ]; then
        echo "warning: $tree is at $commit, CL11 is written for llama.cpp b11384 ($expected_commit);" >&2
        echo "         the ggml backend API is not stable, the patch or the build may fail." >&2
    fi
fi

changed=0
copy_if_different() { # src dst
    if [ ! -f "$2" ] || ! cmp -s "$1" "$2"; then
        mkdir -p "$(dirname "$2")"
        cp "$1" "$2"
        echo "copied   ${2#"$tree"/}"
        changed=1
    fi
}

dest="$tree/ggml/src/ggml-cl11"
copy_if_different "$adapter/CMakeLists.txt" "$dest/CMakeLists.txt"
copy_if_different "$adapter/ggml-cl11.cpp"  "$dest/ggml-cl11.cpp"
for f in "$repo"/src/backends/opencl11/*.{h,cpp}; do
    copy_if_different "$f" "$dest/opencl11/$(basename "$f")"
done
for f in "$repo"/src/backends/opencl11/kernels/*.cl; do
    copy_if_different "$f" "$dest/opencl11/kernels/$(basename "$f")"
done
copy_if_different "$adapter/include/ggml-cl11.h" "$tree/ggml/include/ggml-cl11.h"
# kernels that no longer exist in the adapter
if [ -d "$dest/opencl11/kernels" ]; then
    for f in "$dest"/opencl11/kernels/*.cl; do
        if [ ! -f "$repo/src/backends/opencl11/kernels/$(basename "$f")" ]; then
            rm "$f"; echo "removed  ${f#"$tree"/}"; changed=1
        fi
    done
fi

cd "$tree"
if git apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "patch already applied"
elif git apply --check "$patch_file" >/dev/null 2>&1; then
    git apply "$patch_file"
    echo "patch applied"
    changed=1
else
    echo "error: in-tree-registration.patch neither applies nor is already applied in $tree" >&2
    echo "       (partially applied or a different llama.cpp version?)" >&2
    exit 1
fi

if [ "$changed" = 0 ]; then
    echo "nothing to do (already installed)"
else
    echo "done. build with: cmake -B build -DGGML_CL11=ON -DCMAKE_BUILD_TYPE=Release"
fi
