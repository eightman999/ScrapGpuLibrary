# llama.cpp への組み込み（CL11）

CL11 は llama.cpp **b11384（commit `0faee50`、ggml 0.25.3、backend API 3）** に対して書かれています。ggml の backend API は安定していないため、別のバージョンではコンパイルエラーになるか、プラグインが読み込まれないことがあります（ツリー外ビルドは configure 時にバージョンを確認し、違えば警告を出します）。

組み込み方は 2 通りです。

| | ツリー外プラグイン | ツリー内パッチ |
| --- | --- | --- |
| llama.cpp の改造 | なし | 4 ファイル・15 行 + ファイル追加 |
| llama.cpp のビルド | `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` が必須 | `-DGGML_CL11=ON`（静的リンクも可） |
| CL11 の更新 | `libggml-cl11.so` だけ作り直す | llama.cpp を再ビルド |
| 読み込み | 環境変数 `GGML_BACKEND_PATH` | 常に組み込み済み |

## 1. ツリー外プラグイン

llama.cpp は改造しません。llama.cpp を DL モードでビルドし、CL11 を別の共有ライブラリとしてビルドします。

```bash
# pristine な llama.cpp b11384
git -C <llama.cpp> checkout 0faee50

# (a) llama.cpp 本体（GGML_BACKEND_DL=ON と BUILD_SHARED_LIBS=ON が必須）
cmake -S <llama.cpp> -B <llama.cpp>/build-dl \
  -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build <llama.cpp>/build-dl -j --target llama-cli llama-bench llama-completion test-backend-ops

# (b) CL11 プラグイン
cmake -S adapters/ggml -B build-cl11 \
  -DLLAMA_CPP_SOURCE_DIR=<llama.cpp> -DLLAMA_CPP_BUILD_DIR=<llama.cpp>/build-dl
cmake --build build-cl11 -j
#   -> build-cl11/libggml-cl11.so（macOS でも拡張子は .so）

# (c) 実行
export GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so
<llama.cpp>/build-dl/bin/llama-bench --list-devices          # CL11_0 が見える
<llama.cpp>/build-dl/bin/test-backend-ops -b CL11_0
<llama.cpp>/build-dl/bin/llama-bench -m model.gguf -dev CL11_0 -ngl 99
```

- ルートの `CMakeLists.txt` からも作れます：`cmake -S . -B build-cl11 -DSCRAPGPU_BUILD_GGML_ADAPTER=ON -DLLAMA_CPP_SOURCE_DIR=... -DLLAMA_CPP_BUILD_DIR=...`。
- `LLAMA_CPP_SOURCE_DIR` からは `ggml/include` と `ggml/src`（内部ヘッダ）を、`LLAMA_CPP_BUILD_DIR` からは `libggml-base` を使います。`libggml-base` が別の場所にあるときは `-DGGML_BASE_LIBRARY=<path>` を渡します。
- プラグインは `libggml-base` の場所を RPATH に埋め込むので、llama.cpp のビルドを動かすと動かなくなります。移したときは作り直してください。
- `GGML_BACKEND_PATH` は 1 つのパスです。他のプラグインと併用するときは、`ggml_backend_load_all_from_path` の検索対象（実行ファイルと同じディレクトリ）に `libggml-cl11.so` を置く方法もあります。

### Linux（TB250: Debian 13、NVIDIA 390 の OpenCL）

```bash
sudo apt install build-essential git ocl-icd-opencl-dev opencl-headers   # cmake は ~/bench/cmake-venv
export PATH=~/bench/cmake-venv/bin:$PATH

git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 0faee50
# GGML_NATIVE=ON と GGML_BACKEND_DL=ON は併用できない（cmake のエラー）。ホストの命令セットを明示する（TB250 の Celeron G3930 は SSE4.2、AVX なし）
cmake -S llama.cpp -B llama.cpp/build-dl -G Ninja -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-dl -j2 --target llama-cli llama-bench llama-completion llama-perplexity test-backend-ops

cmake -S adapters/ggml -B build-cl11 -G Ninja \
  -DLLAMA_CPP_SOURCE_DIR=$PWD/llama.cpp -DLLAMA_CPP_BUILD_DIR=$PWD/llama.cpp/build-dl
cmake --build build-cl11 -j2

export GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so
llama.cpp/build-dl/bin/llama-bench --list-devices
llama.cpp/build-dl/bin/test-backend-ops -b CL11_0
```

## 2. ツリー内パッチ

```bash
adapters/ggml/llama.cpp/install-in-tree.sh <llama.cpp>
cd <llama.cpp>
cmake -B build-cl11 -DGGML_CL11=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release   # Linux は -DGGML_NATIVE=ON も
cmake --build build-cl11 -j --target llama-cli llama-bench test-backend-ops
./build-cl11/bin/test-backend-ops -b CL11_0
```

- `install-in-tree.sh` は、`adapters/ggml/` の `CMakeLists.txt`、`ggml-cl11.cpp`、`src/backends/opencl11/` の backend sources / kernels を `<llama.cpp>/ggml/src/ggml-cl11/` に、`include/ggml-cl11.h` を `<llama.cpp>/ggml/include/` にコピーし、`in-tree-registration.patch` を当てます。冪等で、コピー済み・適用済みなら何もしません。実行時に llama.cpp の commit が `0faee50` でなければ警告します。
- パッチの内容は、CMake オプション `GGML_CL11`、`ggml_add_backend(CL11)`、`ggml-backend-reg.cpp` への登録（`GGML_USE_CL11` と `ggml_backend_load_best("cl11", ...)`）、`ggml-config.cmake.in` の依存です。
- `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` と組み合わせると、`libggml-cl11.so` が `build/bin` に作られ、`GGML_BACKEND_PATH` なしで（実行ファイルと同じディレクトリから）読み込まれます。

## 制約

- 対象は llama.cpp b11384（commit `0faee50`）のみです。ggml-backend-impl.h / ggml-impl.h の内部 API を使うので、上げるときは追従が必要です。
- ツリー外プラグインは `GGML_BACKEND_DL=ON`（`BUILD_SHARED_LIBS=ON`）でビルドした llama.cpp でだけ使えます。静的ビルドの llama.cpp には読み込ませられません。
- OpenCL（ICD ローダとヘッダ）が必要です。macOS はシステムの OpenCL.framework を使います。
