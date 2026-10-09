# adapters/ggml

OpenCL 1.1 だけで動く ggml バックエンド「CL11」です（llama.cpp / ggml 用）。runtime / tensor / ops / quant / `backends/opencl11` への段階的な切り出し中で、ここには ggml との接続部分だけを残していきます（[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)）。

| パス | 内容 |
| --- | --- |
| `ggml-cl11.cpp` | バックエンド本体（`GGML_BACKEND_DL_IMPL(ggml_backend_cl11_reg)` を含む） |
| `../../src/backends/opencl11/kernels/*.cl` | OpenCL C 1.1 のカーネル（CMake が文字列として埋め込む） |
| `include/ggml-cl11.h` | 公開ヘッダ |
| `tools/attn_check.cpp` | attention（K / V キャッシュ）のグラフ単位の確認プログラム。`test-backend-ops` は融合したグラフを作らないため別に持つ（先頭のコメントにビルド方法） |
| `tools/mvm_check.cpp` | 起動回数を減らす融合（複数の `MUL_MAT` を 1 回で、K / V キャッシュの `SET_ROWS` の組）のグラフ単位の確認プログラム（二重精度の参照との比較、先頭のコメントにビルド方法） |
| `tools/mm_bench.cpp` | 1 つの `MUL_MAT`（型・m・n・k を指定）の GFLOPS を CSV で出す。`test-backend-ops perf` に無い n=128 を測る |
| `tools/kepler-sweep/` | TB250 で GT 730 / GT 710 / GT 430 の test・greedy・mat-mat 候補・llama-bench を一括で測るスクリプト。GT 430 の denormal product（`dn_*` ステージ。test、mat-vec の GB/s、greedy、KL ダイバージェンス、llama-bench）と全 GPU の回帰確認（`regress` ステージ）も含む（`sweep.sh`、使い方は先頭のコメントと [docs/CL11.md](../../docs/CL11.md) の "Kepler round 3" / "Fermi round"。`DRY_RUN=1` でコマンドだけ出力） |
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
