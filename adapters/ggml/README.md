# adapters/ggml

OpenCL 1.1 だけで動く ggml バックエンド「CL11」です（llama.cpp / ggml 用）。いまは実装が 1 つのまとまり（`ggml-cl11.cpp`、`kernels/*.cl`）になっていて、将来は runtime / tensor / ops / quant / `backends/opencl11` に分け、ここには ggml との接続部分だけを残す予定です（[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)）。

| パス | 内容 |
| --- | --- |
| `ggml-cl11.cpp` | バックエンド本体（`GGML_BACKEND_DL_IMPL(ggml_backend_cl11_reg)` を含む） |
| `kernels/*.cl` | OpenCL C 1.1 のカーネル（CMake が文字列として埋め込む） |
| `include/ggml-cl11.h` | 公開ヘッダ |
| `CMakeLists.txt` | ツリー内（`ggml_add_backend_library`）とツリー外プラグインの 2 モード |
| `llama.cpp/` | llama.cpp への組み込み（ツリー内パッチ、`install-in-tree.sh`、手順 README） |

## ビルド（ツリー外プラグイン）

llama.cpp b11384（commit `0faee50`）を `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` でビルドしたうえで:

```bash
cmake -S adapters/ggml -B build-cl11 \
  -DLLAMA_CPP_SOURCE_DIR=<llama.cpp> -DLLAMA_CPP_BUILD_DIR=<llama.cpp>/build-dl
cmake --build build-cl11 -j
GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so <llama.cpp>/build-dl/bin/test-backend-ops -b CL11_0
```

ツリー内ビルド、Linux（TB250）の手順、制約は [llama.cpp/README.md](llama.cpp/README.md)、設計・性能・環境変数は [docs/CL11.md](../../docs/CL11.md) を参照してください。

## CMake の 2 モード

- `ggml_add_backend_library` が定義されていれば（llama.cpp の `ggml/src/ggml-cl11/` として読み込まれた場合）ツリー内モードです。ggml の関数がターゲット、定義、インクルードパスを設定します。
- 定義されていなければ単独のプロジェクトとして動き、`libggml-cl11.{so,dylib}` を MODULE として作ります。`GGML_BACKEND_DL` / `GGML_BACKEND_BUILD` / `GGML_BACKEND_SHARED` を付け、`LLAMA_CPP_SOURCE_DIR` の `ggml/include`・`ggml/src` を参照し、`LLAMA_CPP_BUILD_DIR` の `libggml-base` にリンクします。configure 時に `LLAMA_CPP_SOURCE_DIR` の ggml バージョン、backend API 番号、commit を確認し、b11384 と違えば警告します。
