# ScrapGpuLibrary

**中古・旧世代の GPU（いわゆる「スクラップ GPU」）と旧式 CPU・RAM で、現行の ML ソフトウェアを実用速度で動かすための、汎用 tensor / runtime ライブラリです。**

ドライバも API も止まった GPU（NVIDIA Fermi / Kepler、古い Radeon など）は、現行の推論エンジンでは使えないか、使えても遅いことが多くあります。ScrapGpuLibrary は、そうした GPU のための runtime、tensor、op、量子化、バックエンドを、特定のエンジンに依存しない形で提供します。ggml（llama.cpp）、PyTorch、ONNX Runtime などは、その上の adapter として接続します。

```
ScrapGpuLibrary
├─ runtime    device / buffer / stream・event / allocator / multi-GPU transport
├─ tensor     dtype / layout / view・stride
├─ ops        matmul・matvec / softmax / norm / rope / attention / elementwise
├─ quant      q4・q8 / k-quant / device-specific packing
├─ backends   cpu / opencl11 / vulkan / remote
└─ adapters   ggml / PyTorch extension / ONNX Runtime / standalone C API
```

各層の責務、依存の向き、設計の前提、移行計画は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) にまとめています。英語版は [README.en.md](README.en.md) です。

## 現状

最初の成果は、OpenCL 1.1 だけで動く ggml backend「CL11」です。v0.2 では、1 つのファイルに全層が同居していた実装を、runtime / backends / core / adapter に切り出しています（[docs/plans/V0.2_SPLIT_BRIEF.md](docs/plans/V0.2_SPLIT_BRIEF.md)）。現在の `main` には P3 までの分離済み実装と、P4（複数 GPU）・P5（Vulkan バックエンド、PyTorch / ONNX Runtime adapter）の最初の段階があります。ggml adapter は引き続き互換性のための翻訳層を持ちます。

| 段階 | 内容 | 状態 |
| --- | --- | --- |
| P0 | CL11（ggml backend）の実装と GPU 別の最適化、ツリー外プラグイン化 | 済み。Gate 0 で速度と正しさの基準表を固定（[docs/BASELINES.md](docs/BASELINES.md)） |
| P1 | GPU 別の最適化の統合、性能の基準表と正しさの許容値の固定 | 済み（Gate 1）。契約は [docs/API_CONTRACT.md](docs/API_CONTRACT.md)。KL ダイバージェンスの許容値は、モデルと文章ごとに公式バックエンドで校正する |
| P2 | runtime と backends/opencl11 の切り出し | 済み（Gate 2、2026-10-09）。device / context / kernels / buffer / executor を移し、ggml の include を外した。TB250 の 6 構成はすべて基準表の範囲内（NVIDIA は -0.9%〜+0.3%）、1 トークンあたりの起動回数・`test-backend-ops`・greedy の一致は分割前と同じ |
| P3 | tensor / ops / quant の切り出し、ggml なしで呼べる C API | 済み（Gate 3、2026-10-10）。公開 C API だけで GPU の演算を呼べる（`examples/standalone_matmul.c`）。TB250 の 6 構成で ggml なしの演算の試験が全件通り、回帰は基準以内（[docs/plans/PHASE3_STANDALONE.md](docs/plans/PHASE3_STANDALONE.md)、判定は [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md)） |
| P4 | multi-GPU transport | 実施中。1 プロセスで GT 730・GT 710・RX 6400・WX 2100 を使い、Qwen3-8B Q4_K_M を全層 GPU で動かした（CL11 4 枚で生成 4.63 t/s、CPU の約 13 倍）。C API の GPU 間転送とパイプラインの例を追加（[docs/MULTI_GPU.md](docs/MULTI_GPU.md)） |
| P5 | バックエンド（opencl12 / vulkan / cuda）と adapter（PyTorch / ONNX Runtime）の追加 | 実施中。Vulkan バックエンドの最初の段階（AMD、加算・乗算・RMS_NORM・行列積）、PyTorch の custom op（Stage A）、ONNX Runtime のプラグイン EP（MatMul・Add・Mul）。CUDA はドライバの切り替えが要るので対象外 |

2026-10-11 に CPU/RAM backend、BF16、予算付きの明示 CPU fallback、F32 Attention（GQA・causal・KV offset）を追加しました。PyTorch は CPU/OpenCL を選択できます。opencl12・CUDA backend は未実装です。今後の予定は [ROADMAP.md](ROADMAP.md) にまとめています。

**XeStrata 向け演算（試験実装）**：GDN の状態更新、depthwise 畳み込み、L2 正規化、QSA のグラフ再生向けステップ情報を公開 op / CPU 参照実装に追加しています。形状・エイリアス・数値契約とテストは [docs/XESTRATA.md](docs/XESTRATA.md)。**現時点では CPU 実装のみで、SYCL バックエンドや XeStrata のモデル全体の実行には未対応です。**

### 検証済みの構成（2026-10-11）

- TB250 の GT 430 / GT 710 / GT 730 / WX 2100 / RX 6400：PyTorch 2.14.1 の演算・Attention、llama.cpp `23b0202` の演算・生成を実機検証。NVIDIA 390.157 を維持し、AMD は rusticl を使用。
- 6f（i5-3230M）：CPU CTest 12/12、PyTorch の4試験が PASS。OpenCL・CUDA は不要。
- Mac：OpenCL CTest 15/15、ASan / UBSan 12/12 PASS。

[利用手順と対応範囲](docs/HETEROGENEOUS_RUNTIME.md)・[実測ログと未検証範囲](docs/benchmarks/heterogeneous-20261011/README.md)。PyTorch は明示的な custom op で、任意CUDA呼び出しや全モデルの自動置換には対応しません。

### クイックスタート

**1. コアと CPU 参照実装を試す（llama.cpp / OpenCL 不要）**

CMake 3.14 以降と C/C++ コンパイラ（C++17）が必要です。

```bash
git clone https://github.com/jimoto-no-llm/ScrapGpuLibrary.git
cd ScrapGpuLibrary
cmake -S . -B build -DSCRAPGPU_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

旧式 CPU と RAM 向けに、CPU 参照実装の行列演算を小さな panel と SSE2 で高速化しています。`-DSCRAPGPU_BUILD_BENCHMARKS=ON` で比較用 benchmark をビルドできます。6f / TB250 の実測と再現手順は [CPU と RAM の最適化](docs/CPU_RAM_OPTIMIZATION.md) を参照してください。

`scrapgpu_cpu_register()` で CPU を通常の device / buffer / stream / plan API から利用できます。BF16 重み、予算付きの明示 CPU fallback、タイル化した Attention（GQA・causal・KV offset）を追加しました。[TB250・旧 CPU と現行フレームワーク](docs/HETEROGENEOUS_RUNTIME.md) に GPU ごとの実行先、NVIDIA ドライバの制約、現行 llama.cpp / PyTorch の検証条件をまとめています。[PyTorch adapter](adapters/pytorch/) は CPU と OpenCL を選択できます。

**2. OpenCL 1.1 バックエンドと独立 C API のサンプルを試す**

OpenCL ヘッダー、ICD ローダー、対象 GPU に対応する OpenCL 実装が別途必要です。

```bash
cmake -S . -B build-opencl \
  -DSCRAPGPU_BUILD_OPENCL11=ON \
  -DSCRAPGPU_BUILD_EXAMPLES=ON \
  -DSCRAPGPU_BUILD_TESTS=ON
cmake --build build-opencl -j
ctest --test-dir build-opencl --output-on-failure
./build-opencl/scrapgpu_standalone_matmul
```

この例は **ggml / llama.cpp に依存せず**、公開 C API で Q4_0 重みを R32 配置で GPU に転送し、`MUL_MAT` の結果を CPU 参照実装と NMSE で比較します。OpenCL デバイスが見つからない場合、サンプルは演算せず終了します。GPU テストもデバイスがなければ skip されるため、**テストが終了コード 0 でも GPU 実行が検証されたとは限りません**。

llama.cpp から使う場合は [ggml adapter の手順](adapters/ggml/) を参照してください（開発基準は b11384、追加検証済みの版は `23b0202`）。性能と対応演算は [docs/BASELINES.md](docs/BASELINES.md) と [docs/CL11.md](docs/CL11.md) に記録しています。

### コードの置き場所

| パス | 内容 |
| --- | --- |
| `src/backends/cpu/` | システム RAM 上の同期 CPU backend。既存の CPU 参照カーネルを公開 device API に接続 |
| `src/backends/opencl11/` | OpenCL 1.1 バックエンド。device（能力・世代・調整値）、context（queue・program の遅延生成）、kernels（カーネル表・program build・`.cl` の埋め込み）、buffer（確保・上限・遅延確保の probe）、executor（起動とプロファイル）、device_state（デバイスの実行時の状態）、tensor_abi（カーネルに渡す記述子）。ggml の include はありません |
| `src/tensor/` | ggml と OpenCL に依存しない tensor モジュール（dtype、TensorDesc、ビュー・転置・ブロードキャスト、除算なしの添字、R32 の配置） |
| `src/core/` | ggml と OpenCL に依存しない共通層。ABI 版、op の契約検証、backend の登録、CPU 参照実装（`src/core/ref/`：全演算と Q4_0 / Q8_0 / Q4_K / Q6_K の量子化・逆量子化） |
| `include/scrapgpu/` | 公開の C API（`scrapgpu_` 接頭辞）と契約のヘッダ |
| `adapters/ggml/` | ggml adapter。CL11 のバックエンド本体（`ggml-cl11.cpp`）と検証ツール（`mvm_check`、`attn_check`、`kepler-sweep`） |
| `tests/core/`、`tests/tensor/`、`tests/ref/` | 契約テスト、tensor モジュールの試験、CPU 参照実装の試験（いずれも llama.cpp なしで動く） |

### ビルド

core と契約テストは llama.cpp なしでビルドできます。

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

| CMake の選択肢 | 既定 | 内容 |
| --- | --- | --- |
| `SCRAPGPU_BUILD_CORE` | `ON` | ggml と OpenCL を使わない共通層（`scrapgpu_core`） |
| `SCRAPGPU_BUILD_OPENCL11` | `OFF` | OpenCL 1.1 バックエンド（`scrapgpu_opencl11`）。`CL_TARGET_OPENCL_VERSION=110` を定義します |
| `SCRAPGPU_BUILD_PYTORCH` | `OFF` | CPU/OpenCL の PyTorch custom op。CORE と PyTorch のある Python 環境が必要。拡張は C++20 |
| `SCRAPGPU_BUILD_GGML_ADAPTER` | `OFF` | ggml adapter。`ON` で `adapters/ggml` を追加します |
| `SCRAPGPU_BUILD_EXAMPLES` | OPENCL11 と同じ | ggml なしの C API サンプル。OpenCL11 が必要 |
| `SCRAPGPU_BUILD_TESTS` | `ON` | 契約テストと依存の検査 |
| `SCRAPGPU_BUILD_BENCHMARKS` | `OFF` | CPU 参照行列演算と CL11 のホスト側記述子準備の benchmark。CORE が必要 |

- `ctest` は、契約テストと、`src/core` と公開ヘッダーに ggml・OpenCL の include がなく、`src/backends/opencl11` に ggml の include がないことを確かめる検査（`cmake/check_core_dependencies.cmake`）を実行します。
- ggml adapter の開発基準は llama.cpp b11384 です。`23b0202`（ggml 0.26.0 / backend API 3）も実機検証済みです。ツリー外プラグインとツリー内パッチの両方のビルド方法は [adapters/ggml/](adapters/ggml/) を参照してください。
- 変更ごとに実行する検査は [CONTRIBUTING.md](CONTRIBUTING.md) にまとめています。

### CL11

- OpenCL 1.1 のホスト API と OpenCL C 1.1 のカーネルだけで書いています。fp16 演算、サブグループ、OpenCL 1.2 以降の API は使いません。
- 検証した GPU は、NVIDIA 390 ドライバの GeForce GT 430（Fermi）・GT 710 / GT 730（Kepler）と、Mesa の Radeon Pro WX 2100（GCN、Clover と rusticl）・Radeon RX 6400（RDNA2、rusticl）です。調整値は GPU の世代ごとに `cl11_tuning_for()` に集めています（Fermi、Kepler、AMD GCN は測定した値、AMD RDNA2 は AMD 共通の設定だけ）。
- Llama / Qwen2 / Qwen3 系のモデルを全層オフロードできます（Flash Attention は非対応）。LFM2 / LFM2-MoE（短い畳み込み、`MUL_MAT_ID`）も全 op が載ります（Mac での確認まで。TB250 の GPU では未測定）。Q4_0 / Q8_0 / Q4_K / Q6_K の重みは GPU 向けの配置に並べ替えて持ちます。対応していない op は自動で CPU に回ります。
- 対応 op、性能、環境変数、制約、試して採用しなかった手法は [docs/CL11.md](docs/CL11.md) を参照してください。
- llama.cpp への組み込み方（ツリー外プラグイン、ツリー内パッチ）は [adapters/ggml/](adapters/ggml/) を参照してください。

llama-bench（`-ngl 99 -ub 128 -b 128 -p 128 -n 64`、CPU は同じマシンの Celeron G3930 の 2 スレッド。pp128 / tg64、単位 t/s）:

| モデル | GPU | CL11 pp128 | CL11 tg64 | CPU pp128 | CPU tg64 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q8_0 | GT 430 | 89.4 | 22.3 | 25.9 | 15.6 |
| | GT 730 | 241.9 | 34.9 | 25.9 | 15.9 |
| | GT 710 | 120.0 | 14.7 | 26.0 | 15.7 |
| Qwen3-0.6B-Base Q4_K_M | GT 430 | 89.9 | 23.7 | 6.1 | 4.6 |
| | GT 730 | 231.6 | 39.2 | 6.2 | 4.6 |
| | GT 710 | 115.6 | 17.8 | 6.2 | 4.6 |
| Qwen2-0.5B-ft Q4_0 | GT 430 | 118.5 | 36.5 | 47.5 | 25.8 |

Qwen3-1.7B Q4_K_M（1.2GB）は GT 710（2GB）で 29 層すべてをオフロードできます（pp128 39.7 / tg64 7.5、CPU は 2.0 / 1.6）。GT 710 は帯域が 11.7 GB/s しかなく、Q8_0 の生成は CPU より遅くなります（Q4_K_M なら速い）。Radeon の llama.cpp 推論では、既存の実測で速い公式 Vulkan を推奨します。C API / PyTorch の CL11 経路は rusticl で検証しています（[docs/HARDWARE.md](docs/HARDWARE.md)）。

## 検証機

BIOSTAR TB250-BTC PRO に、GeForce GT 430 / GT 710 / GT 730、Radeon Pro WX 2100、Radeon RX 6400 を PCIe x1 で載せた環境で開発しています。GPU ごとの帯域とドライバの既知の癖は [docs/HARDWARE.md](docs/HARDWARE.md) にあります。

## 出自

CL11 は [llama.cpp](https://github.com/ggml-org/llama.cpp) b11384（commit 0faee50）の上で開発しました。`adapters/ggml/` の初版は、その開発ブランチ（`fermi-cl11`、commit `9ab3f557c`）から取り込んだものです。

## ライセンス

GNU General Public License v3.0 以降（GPL-3.0-or-later、[LICENSE](LICENSE)）。ggml / llama.cpp（MIT）に由来する部分とその著作権表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にまとめています。
