# Phase 3：ggml なしでテンソル演算を呼べるようにする

## 目標

ggml を使わずに、ScrapGpuLibrary の C API だけでテンソル演算を GPU で実行できる状態にする。ggml adapter は、同じ演算の実装を呼ぶ翻訳層になる。

```c
scrapgpu_opencl11_register();                        // OpenCL 1.1 バックエンドを登録する
scrapgpu_get_device(0, &dev);
scrapgpu_stream_create(dev, &stream);                // in-order のキュー
scrapgpu_buffer_alloc(dev, bytes, &buf);
scrapgpu_tensor_upload(stream, &w, host_q4_0, size); // layout が R32 なら詰め替えて書く
scrapgpu_op_execute(stream, &mul_mat);               // 1 演算（複数なら plan_build / plan_execute）
scrapgpu_tensor_download(stream, &y, host_out, size);
scrapgpu_stream_sync(stream);
```

## 方針

- 演算の実装は 1 つだけにする。`cl11_op_*` をバックエンドへ移し、ggml adapter と C API の両方から呼ぶ。
- バックエンド内部のテンソルは `cl11_tensor`（dtype、ne、nb、`cl_mem`、オフセット、layout）。adapter はノードごとに `ggml_tensor` からスタック上に作る（ヒープ確保なし）。ggml の `op_params` の解釈は adapter に残し、バックエンドは演算ごとの引数の構造体を受け取る。
- グラフの文脈が要る融合（RMS_NORM+MUL、SET_ROWS の対、multi mat-vec、CONT の省略）は、ggml adapter では今の判定をそのまま使う。C API では融合した演算を明示的な op（`SCRAPGPU_OP_RMS_NORM_MUL` など）として呼ぶ。plan での自動の融合は後回し。
- 速度の条件は Gate 2 と同じ（NVIDIA ±3%、AMD は許容幅、1 トークンあたりの起動回数が増えない、`clBuildProgram` のソースとオプションが変わらない）。

## 段階（1 段階 1 コミット）

| 順 | 内容 | 主なファイル |
| --- | --- | --- |
| S1 | デバイスの実行時の状態（context、queue、プログラム、カーネル、調整値、定数サブバッファ、作業用バッファ、実行の状態）をバックエンドの `cl11_device` に移す。ggml のデバイスは `cl11_device` を指すだけにする | `src/backends/opencl11/device_state.*`、`ggml-cl11.cpp` |
| S2 | `cl11_tensor` と、記述子（`cl11_td`）・除算なしの添字（`cl11_fastdiv_init`）・ブロードキャスト（`cl11_bcast`）をバックエンドへ。adapter は変換だけ | `src/backends/opencl11/tensor.*` |
| S3 | R32 の詰め替え（pack / unpack / 部分転送）をバックエンドへ | `src/backends/opencl11/r32.*` |
| S4 | 演算（`cl11_op_*`）と対応判定の ggml に依存しない部分をバックエンドへ | `src/backends/opencl11/ops.*` |
| S5 | 公開 API：stream、`scrapgpu_opencl11_register`（vtable の実装）、`scrapgpu_op_execute`、`scrapgpu_tensor_upload / download`、`op_supported`、1 演算 1 ノードの plan | `include/scrapgpu/*.h`、`src/core/*`、`src/backends/opencl11/api.cpp` |
| S6 | CPU 参照実装（全演算と Q4_0 / Q8_0 / Q4_K / Q6_K の逆量子化）と、演算ごとの単体試験（GPU と参照を NMSE で比べる）。ggml を使わない例（`examples/standalone_matmul.c`） | `src/core/ref/*`、`tests/ops/*`、`examples/*` |

S6 の CPU 参照実装は S1〜S5 と独立しているので並行して作る（公開ヘッダーの `scrapgpu_tensor_desc` とホストのメモリだけを使う）。

## 各段階の確認（Mac）

- プラグインが `-Wall -Wextra` で警告なしにビルドできる。in-tree ビルドも通る（S4 と S5 の後）。
- `test-backend-ops -b CL11_0` が既定・`GGML_CL11_FORCE_GEN=kepler`・`fermi` で 3406 / 3406（1 構成 約 4 分）。
- `mvm_check` 18 / 18、`attn_check` 8 / 8、`ctest`。
- `clBuildProgram` のソースとオプションが変わらない（`clBuildProgram` を記録して比べる）。
- `src/backends/opencl11`、`src/core`、`include/scrapgpu`、`tests/ops`、`examples` に ggml の include がない（`cmake/check_core_dependencies.cmake`）。

## Gate 3

- Gate 2 と同じ TB250 の回帰（6 構成、`BASELINES.md` と比べる）。
- ggml を使わない単体試験と例が、TB250 の 6 構成と Mac で通る。
