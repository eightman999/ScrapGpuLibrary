# 再利用できる実装の調査

状態: 調査済み（2026-10-08）。方針は「引けるものは引く。全部を自前で作らない」です。

## 調査の方法と限界

- ライセンスは各リポジトリの LICENSE 本文、最終 push 日とリリースは `gh api`、OpenCL の API 使用はソースの grep で確認しました。clone は作業用ディレクトリ（リポジトリの外）にだけ置いています。
- **TB250 の実機では何も動かしていません。** 「動く」と書いたものは、ソース上の要件が 1.1 と矛盾しないという意味で、実機の確認ではありません。実機で確かめる項目は末尾の「未検証」にまとめました。
- 記号: **取り込む**＝依存として使う、またはそのまま同梱する。**移植する**＝ソースの一部を自分のコードに移す。**参考にする**＝設計や値だけ借りる。**使わない**。

### ライセンスの扱い

| ライセンス | 扱い |
| --- | --- |
| MIT / BSD-3 / BSL-1.0 / CC0 | 著作権表示を残せば、取り込み・移植とも可。 |
| Apache-2.0（CLBlast、MNN、OpenCL-CLHPP など） | MIT と両立する。同梱・移植するときは Apache-2.0 の全文を添え、改変したファイルには変更を明記する（§4）。NOTICE があれば引き継ぐ。`THIRD_PARTY` に併記する。 |
| GPL（clpeak: GPL-3.0、mixbench: GPL-2.0、Tasmota、PowerJoular、likwid） | ソースを取り込まない。外部プロセスとして実行し、出力（JSON / CSV）だけを使う。 |
| BabelStream の独自ライセンス | 改変物の結果を公表するときは「改変物」と明記する条件がある。外部ツールとしてのみ使う。 |
| LICENSE ファイルがない（**Relic**） | README に「MIT」とあるだけで、GitHub のライセンス判定も null。法的な根拠が弱いので、取り込まない（後述）。 |

## 層ごとの推奨（要約）

| 層 | 推奨 | 対象 |
| --- | --- | --- |
| runtime: OpenCL ラッパ | **自作**（参考: CLBlast の `clpp11.hpp`） | `CL_TARGET_OPENCL_VERSION=110` 固定の薄い RAII。ヘッダは OpenCL-Headers、ローダは OpenCL-ICD-Loader を**取り込む**。Boost.Compute、ViennaCL、clBLAS、ArrayFire、clew は**使わない** |
| runtime: デバイスデータベース | **参考にする** | CLBlast の検索階層（vendor → type → arch → name → default）とチューナ JSON。ggml-vulkan の調整表を初期値の根拠にする |
| runtime: multi-GPU transport | **参考にする** | llama.cpp の ggml-rpc（`socket_t` 抽象、1 プロセス 1 デバイスのサーバー）と `--split-mode layer`。exo、Petals、distributed-llama は使わない |
| runtime: 計測・電力 | **自作 + 外部ツール** | 帯域・ピーク FLOPS は mixbench / SHOC / clpeak を実行して測る。電力は amdgpu hwmon を自前で読む |
| tensor | **自作** | 流用できる実装は見つからなかった。ggml の形・ストライド規約と互換にして adapter の翻訳を単純にする |
| ops: CPU 参照 | **移植する** | ggml-quants.c / ggml-common.h の Q4_0・Q8_0・k-quant の参照実装（MIT） |
| ops: F32 / F16 GEMM | **取り込む**（比較対象にもなる） | CLBlast `Xgemm` / `Xgemv`（Apache-2.0、OpenCL 1.1 対応）。自前の mat-mm（8×4）との比較ベースラインにもなる |
| quant: GPU 別パッキング | **参考にする** | ggml-opencl の SOA 再配置（`cvt.cl`、`*_flat.cl`）。カーネルは使わない |
| backends/opencl11 | **CL11 を出発点にする** + **参考にする**（旧 ggml-opencl、MNN） | カーネルは CL11（`src/backends/opencl11/kernels/`）から切り出す。旧 CLBlast 版 ggml-opencl は GT 430 で CPU より遅かった実装なので、正しさの比較用の素朴な参照としてだけ使う。MNN の `*_buf.cl`（int4 / int8 gemv・gemm）は fp16 を `#ifdef` で外せるので移植候補。現行の ggml-opencl は OpenCL 2.0 必須で**使わない** |
| backends/opencl12 | **取り込む / 参考にする** | opencl11 のカーネルを共有。CLBlast チューナで Kepler の調整値を作る。dlprimitives の `program_cache` を参考にする |
| backends/vulkan | **移植する**（シェーダ）+ **取り込む**（volk / vk-bootstrap / VMA） | ggml-vulkan の `*.comp`（mul_mat_vec、dequant）。ホスト側は自作。Vulkan 1.2 と 16bit storage が前提 |
| backends/cuda | **参考にする** | ggml-cuda はコードを取り込まず、cc 別の選択表と dp4a fallback だけ見る。Fermi は不可。Kepler は CUDA 11.x + ドライバ 470 の実験扱い |
| adapters/ggml | **そのまま使う** | `ggml_backend_sched`、`test-backend-ops`（検証）。ggml 本体は取り込まない |
| adapters/PyTorch | **参考にする + 一部移植する** | 公式 PrivateUse1 ドキュメントと torch_openreg、pytorch_dlprim（MIT）の登録コード |
| adapters/ONNX Runtime | **取り込む**（テンプレート） | プラグイン EP の `example_plugin_ep_kernel_registry`（MIT）。ORT 1.24 以上 |
| 他の推論系（MLC-LLM / TVM、tinygrad、exo、Petals、mllm など） | **使わない** | OpenCL 2.0 以上の前提、旧 GPU 非対応、保守停止のいずれか |
| **Relic** | **使わない**（設計を参考にする + 比較実験のベースライン） | 後述 |

## Relic（anamnesic/relic）

https://github.com/anamnesic/relic。調べた版は 2026-10-06 の commit `a6754dc`。

### 事実

| 項目 | 内容 |
| --- | --- |
| ライセンス | **LICENSE ファイルなし**。README に「MIT」とあるのみで、GitHub のライセンス判定は null |
| 保守 | 作成 2026-05-20、58 コミット、最終 push 2026-10-06、リリースなし、star 0 / fork 0。`AGENTS.md` があり、エージェント支援で書かれている形跡がある（推測） |
| 規模 | C++ 約 1.6 万行（うち `embedded_kernels.h` 3840 行は `kernels/kernels.cl` 1697 行の生成物）。C++23 必須。CUDA は任意 |
| 対応モデル | Llama 系（SmolLM2 を含む）と Qwen 3.5（Gated DeltaNet）。**Qwen2 / Qwen3 は未対応**。FP16 / Q8_0 / Q4_0（Q4_K / Q6_K の GEMV カーネルもある） |
| 検証した GPU | GTX 1650（Turing）と Intel UHD、PoCL（CPU）。ADR 0022 は GTX 1650 を PoCL の CUDA デバイス経由、またはネイティブ CUDA で測定。ADR 0023 は、1 GB の Caicos XT（TeraScale 2）が**ラボから撤去され検証に使えない**と記録している。旧 GPU（Fermi / Kepler / TeraScale）での実測は、見た範囲では**ない** |
| 公表している数値 | README にはない。ADR に GTX 1650 での tok/s（例: Qwen 3.5 2B Q4_0 で 16.8 tok/s、ADR 0021 / 0022）と、ネイティブ CUDA の Q4_0 GEMV 約 47〜50 GB/s（ADR 0022、理論帯域に対する比は書かれていない）がある |

### OpenCL 1.1 で動くか

- ホスト側で使う OpenCL API は、`clCreateCommandQueue`、`clCreateBuffer`、`clEnqueueRead/WriteBuffer`、`clEnqueueCopyBuffer`、`clCreateProgramWithSource`、`clBuildProgram` など **1.0 / 1.1 にある関数だけ**（grep で確認。`clEnqueueFillBuffer`、`clCreateImage`、`clCompileProgram`、サブバッファも使っていない）。
- ただし `CL_TARGET_OPENCL_VERSION=120` を定義している。README も「OpenCL 1.2 & 3.0」と書き、1.1 は謳っていない。
- カーネルは `#pragma` 拡張なし、サブグループなし、fp16 演算なしの OpenCL C。fp16 は保存形式の手動変換（`fp16_to_fp32`）。
- したがって **ソース上は OpenCL 1.1 でも動く可能性が高いが、実機（GT 430）では未確認**。

### 実装の中身（読んだ範囲）

| 主張 | 実体 |
| --- | --- |
| auto-tuned kernel registry | `kernel_registry.cpp` はデバイス種別（NVIDIA / AMD / Intel）× カーネル名の**静的な表**（workgroup 幅、行数など）。`autotune()` は固定値を登録し直すだけで、**計測はしていない** |
| hardware micro-profiler | `hardware_profile.cpp` は `clGetDeviceInfo` とベンダー文字列から、帯域・演算性能を**定数**で埋める（NVIDIA は 128 GB/s / 2.98 TFLOPS など、GTX 1650 の値）。帯域の実測はしない。`load_from_file()` は未実装（`probe_system()` を返す） |
| benefit_per_byte | `adaptive_planner.cpp`。テンソルごとに「CPU の推定時間 − GPU の推定時間」をバイト数で割る。FLOPS は定数。norm と attn は優先固定。VRAM に収まる順に載せる。1 GPU + iGPU + ホスト RAM のための配置であり、本研究の問題（全層 GPU、PCIe x1）とは別 |
| Q4_0 / Q8_0 の repack | **Q8_0 を Q4_0 へ量子化し直す**（`upload_auto_q4`）処理。レイアウトの再配置（R32 など）ではない。精度が落ちるので、比較するときは Q4_0 の GGUF を与えて repack を避ける必要がある |
| 分散投機的デコード | 同一プロセス内の draft と target の 2 デコーダ、greedy の一致検証。複数デバイス間・複数ホスト間の分散は、この実装には見当たらない |
| Q8_0 の GEMV | 34 バイトのブロックをバイト単位で読み、`ushort` を組み立てる。1 ワークグループ 32 スレッドで 4 行、`barrier` による削減。Fermi 向けの除算除去、バイト → float 変換、物理再配置はない（CL11 の R32 に相当するものはない） |
| 単一確保の上限 | `clCreateBuffer` 1 か所。上限を超えた確保の扱いや、公称上限を超えて通るドライバの癖への対処はない |

### 判断

| 観点 | 判断 |
| --- | --- |
| 取り込める部分 | **コードは取り込まない。** LICENSE がなく、そのまま取り込むとライセンスの根拠が弱い。作者に LICENSE の追加を依頼できるなら、その後に再検討する。カーネルの内容（Q4_0 の GEMV、Qwen 3.5 の recurrent op など）は ScrapGpuLibrary の対象外か、既存の CL11 のほうが旧 GPU に合わせてある |
| 参考にする点 | 「静的な表 + 実測で上書き」というデバイス DB の発想（本研究では CLBlast の階層のほうを主に参考にする）。ベンチマークの JSON / CSV 出力の項目。unified memory / pinned host pool の扱い（ScrapGpuLibrary の対象外だが、iGPU 対応時に見る） |
| ベースラインとして使えるか | **使える可能性が高い（要実機確認）**。GT 730 / GT 710（Kepler、OpenCL 1.2）、WX 2100 / RX 6400（rusticl の OpenCL 3.0）は要件を満たす。GT 430（OpenCL 1.1）も、API が 1.1 の範囲なので動く可能性がある。条件: (1) モデルは Llama 系の **Q4_0**（Qwen3 は不可）、(2) 公称上限（GT 730 で 245MiB）を超えるテンソルがないこと、(3) 同じ GGUF・同じ量子化で CL11 と比べる、(4) C++23 のコンパイラ（Debian 13 の gcc 14 で可、推測） |
| 重なりの評価 | **中〜大。ただし差は保てる。** 「旧 GPU + OpenCL + GGUF 風ランタイム」という構図は同じ。一方、Relic は (a) 旧 GPU での実測がなく、(b) 帯域に対する達成率を報告せず、(c) OpenCL 1.1 を対象と明言していない。本研究の差分（性能工学、1.1 での検証、ggml プラグイン、正しさの評価、ドライバの不具合）は維持できる。ただし「最適化の寄与を測る」前に Relic の新しい版が旧 GPU の数値を出す可能性があるので、投稿前に再確認する（PAPER_PLAN.md 参照） |

## 層ごとの詳細

### runtime / device（デバイスデータベース）

**CLBlast**（CNugteren/CLBlast、Apache-2.0、最終 push 2026-09-28、v1.7.0）

- README が「OpenCL 1.1 ハードウェアで使うとき」を明記。`clpp11.hpp` は `CL_TARGET_OPENCL_VERSION 110` が既定で、ホスト側は 1.2 API を使わない。カーネルはサブグループ・atomic を使わない（`cl_intel_subgroups` を Intel 向けに条件付きで使うのみ）。C++11、CMake 3.14。
- デバイスデータベース（`src/database/`）: カーネル名と精度 → ベンダー + デバイス種別 → アーキ → デバイス名 → 各階層の `default` と引く。NVIDIA のアーキは `CL_DEVICE_COMPUTE_CAPABILITY_NV`（SM 2.0 など）、AMD は `cl_amd_device_attribute_query`。値は `scripts/database/database.py` が JSON から C++ ヘッダに変換する。チューナは `src/tuning/`。
- 収録: Xgemm F32 の NVIDIA は SM 2.0（GTX 480 / 580）、SM 3.0、SM 3.5（**GT 730** を含む）。AMD は Ellesmere（RX 480 / 580）、gfx1034（RX 6500 XT）、**RX 6400** がある。GT 430 相当（SM 2.1）と WX 2100 の専用エントリはなく、default に落ちる（推測）。
- 注意: rusticl / Clover が `cl_amd_device_attribute_query` を出すか未確認。出さないとベンダーの default に落ちる（推測）。
- **使い方**: DB の**階層設計を参考にする**（ARCHITECTURE.md の「調整値をコードではなくデータに置く」と同じ考え方）。生成ヘッダの値は GEMM 用なので mat-vec には使えない。チューナの JSON 出力という流れは真似る。

**OpenCL ホスト側ラッパ**

| 候補 | ライセンス | 判断 |
| --- | --- | --- |
| OpenCL-Headers / OpenCL-ICD-Loader | Apache-2.0 | **取り込む**。`CL_TARGET_OPENCL_VERSION=110` で宣言を絞り、1.2 API の混入をコンパイル時に防ぐ |
| OpenCL-CLHPP | Apache-2.0 | 使うなら `CL_HPP_TARGET_OPENCL_VERSION=110` と `CL_HPP_MINIMUM_OPENCL_VERSION=110` を固定。C API を公開する本ライブラリでは内部限定にする |
| CLBlast `clpp11.hpp`（906 行） | Apache-2.0 | RAII ラッパの**実装例として参考**。1.1 前提で、ちょうどよい薄さ |
| Boost.Compute | BSL-1.0 | 使わない。1.2 API は `BOOST_COMPUTE_CL_VERSION_1_2` 等のガード付きで 1.1 動作は推測。依存が重く、runtime 層に不要 |
| ViennaCL | MIT 系 | 使わない。最終コミット 2019-07 |
| clBLAS | Apache-2.0 | 使わない。最終コミット 2017-03。CLBlast が後継 |
| ArrayFire | BSD-3 | 使わない。OpenCL バックエンドは 1.2 以上が必須（release notes v3.1.2） |
| clew | BSL-1.0 | 使わない。最終 2016-09、1.2 までのヘッダ |
| pocl | MIT | 開発機（Mac など GPU がない環境）での**カーネル検証用**に使う。`-cl-std=CL1.1` を付けて古い言語要件を強制する。組み込まない |
| clinfo | CC0 | デバイス情報の採取（デバイス DB のキー）に使う |
| MNN の `OpenCLWrapper` | Apache-2.0 | libOpenCL を dlopen して関数ポインタを解決する方式（`CL_HPP_TARGET_OPENCL_VERSION 110`）。ICD がない環境の扱いの参考 |
| TFLite GPU（LiteRT）の `cl_device.cc` | Apache-2.0 | OpenCL のバージョンを解析して機能を `>= kCl1_2` のように切り替える設計の参考 |

### runtime / multi-GPU transport

- **llama.cpp ggml-rpc**（MIT）: README が「proof-of-concept、fragile and insecure」と明記。ただし `socket_t` を抽象化した transport（TCP、RDMA）があり、ホストの各デバイスをサーバーとして公開する。**1 プロセス 1 デバイスのサーバーを複数立てる**構成は、NVIDIA 390 で 1 プロセスに 2 枚目の GPU を開けない問題の回避になり得る。transport の抽象を**参考にする**。共有メモリ transport は自前で足す。
- **llama.cpp `--split-mode layer`**: `docs/multi-gpu.md` が、パイプライン並列で遅い接続に耐えると書いている。`tensor` モードは 1 層に複数回の GPU 間 reduction が要り、PCIe x1 には向かない。
- 使わない: distributed-llama（MIT、テンソル並列でノード数が 2 のべき乗に制約、Vulkan 1.2 要求）、exo（Apache-2.0、MLX が主体）、Petals（MIT、最終 push 2024-09、事実上停止）。

### runtime / 計測・電力・ルーフライン

| 候補 | ライセンス | 判断 |
| --- | --- | --- |
| mixbench（OpenCL 版） | GPL-2.0 | **外部ツールとして実行**。ピーク FLOPS と帯域。OpenCL C API 直叩きで、使う関数は 1.1 にあるもの（実機は未確認） |
| SHOC（OpenCL 版） | BSD-3 系 | **実行する / 一部を移植できる**。BusSpeedDownload / Readback（PCIe x1 の約 0.40 GB/s の再測定に使える）、DeviceMemory、MaxFlops、QueueDelay。`CL_USE_DEPRECATED_OPENCL_1_1_APIS` を使い 1.1 向け。最終コミット 2020-04 |
| clpeak 3.x | GPL-3.0 | 外部ツールとして実行。**JSON 出力（format v3）の形式を、デバイス DB の入力の参考にする**。OpenCL バックエンドの基準が 1.2（`CL_HPP_MINIMUM_OPENCL_VERSION 120`）なので、**Fermi では動かない可能性がある** |
| clamchowder/Microbenchmarks | Apache-2.0 | ポインタチェイスの遅延カーネルを**移植できる**。バッファ版は 1.1 で動く可能性があり、テクスチャ版は `clCreateImage`（1.2）を使う |
| BabelStream | 独自 | 外部ツールのみ。OpenCL 実装は `CL_HPP_MINIMUM_OPENCL_VERSION 120`。論文では測定方法の引用先（Deakin ほか 2018）になる |
| Empirical Roofline Toolkit | BSD-3 系 | 使わない。CPU / OpenMP / MPI 向けで OpenCL の記述がない。元の cs-roofline-toolkit は Bitbucket、確認できたのはミラー（2019 年）。プロットの考え方だけ参考 |
| Nsight Compute / rocprofiler-compute | 独自 / MIT | 使わない。前者は Turing 以降、後者は MI100 以降（Polaris / RDNA2 は対象外） |

- **ルーフラインは自作**する。mixbench や自前の streaming カーネルで実測した帯域とピーク FLOPS を matplotlib で描く。
- **電力**: AMD は amdgpu の hwmon（`/sys/class/drm/card*/device/hwmon/hwmon*/`）を読む。カーネルのソースでは、**Polaris（smu7）は `power1_input` だけ、RX 6400（Navi24）は `power1_average` だけ**が公開される。存在するほうを読む実装にする。値は SoC の電力で、ボード全体ではない可能性がある（推測）。NVIDIA 390 の Fermi / Kepler の GeForce では NVML で取れない（HARDWARE.md の前提）ので、外部電力計（コンセント側）を使う。`runtime` に `power_probe`（hwmon / NVML / 外部 CSV の 3 方式）を置く。
- 既存の電力ツール（CodeCarbon: MIT、Zeus: Apache-2.0、PowerJoular: GPL-3.0）は NVML や amdsmi 前提で、Polaris / RDNA2 の動作は未確認。使わない。

### ops と quant

**ggml / llama.cpp**（MIT、最終 push 2026-10-07）

- **ggml-quants.c / ggml-common.h**: Q4_0（18 バイト / 32 重み）、Q8_0（34 バイト）、Q4_K、Q6_K の `quantize_row_*_ref` と `dequantize_row_*`。GGUF とビット互換。`ggml-impl.h` の `GGML_ASSERT` と FP16 変換、`ggml_row_size`、`ggml_type_size` への依存を、小さな shim に置き換えれば切り出せる（推測。実際には試していない）。IQ 系は `ggml_quantize_init` のテーブルに依存するので落とす。**移植する**。MIT 表記を残す。
- **ggml-opencl**（現行）: **OpenCL 2.0 以上が必須**（`opencl_c_version.major < 2` で拒否）。さらに Adreno / Intel 以外、`cl_khr_fp16` がない、OpenCL 3.0 でサブグループがない、のいずれも拒否。カーネルは約 200 本でサブグループ・fp16・`cl_qcom_*` を多用する。**カーネルは使わない**。参考にするのは次の 2 点。
  - SOA 再配置（`GGML_OPENCL_SOA_Q`）。scale 列と量子値列を分けて置く発想で、`quant/` の GPU 別パッキングの参考。
  - プログラムバイナリのキャッシュ（`cl-program-cache.cpp`、キーはカーネルソース + オプション + デバイス + ドライバ + プラットフォーム）。`clGetProgramInfo` / `clCreateProgramWithBinary` は 1.1 にもあるので、起動時間対策として使える。
- **旧 CLBlast バックエンド**（ルートの `ggml-opencl.cpp`）: PR #7735（2024-06-04、"ggml : remove OpenCL"）で削除。削除直前は commit `e84b71c2c`（2024-05-23）。MIT。OpenCL C 1.x の dequantize と dequantize_mul_mat_vec（Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0、k-quant）、F32 / F16 の GEMM は CLBlast に委譲。ホスト側は 1.1 の API だけ（`clCreateSubBuffer` を含む）、カーネルにサブグループ・atomic なし、半精度は `vload_half`。復活の提案（#8986）は未マージで閉じている。**`backends/opencl11` のカーネルの移植元**にする。`ggml_tensor` との結合は、ホスト側を捨てて runtime 層に載せ替える前提。Fermi 向けの調整（R32、除算除去など）は CL11 のほうが進んでいるので、移植するのは旧版の**参照用の素朴なカーネル**（正しさの比較対象）に限るのが現実的。
- **ggml-vulkan**: `vulkan-shaders/*.comp`（`mul_mat_vec.comp`、`mul_mm.comp`、`dequant_*.comp`、Q4_0 などは `#define DATA_A_Q4_0` で切り替え）。必須: Vulkan 1.2（`api_version < VK_API_VERSION_1_2` で終了）、16bit storage、8bit storage、explicit arithmetic types。サブグループは `USE_SUBGROUP_ADD` のときだけ使い、shared memory 経路もある。ビルドは `glslc`。ホスト側（`ggml-vulkan.cpp` 約 1.7 万行）は `ggml_tensor` と密結合で**使わない**。シェーダのうち mat-vec と dequant を**移植する**。デバイス別の調整（`get_device_architecture` が AMD_GCN / RDNA1 / RDNA2 / RDNA3 を判定し、GCN かつ RADV では warptile や行数を切り替える）は、デバイス DB の**初期値の根拠にする**。WX 2100 は GCN、RX 6400 は RDNA2 に分類されるはず（推測）。
- **ggml-cuda**: 現行の CMake の既定は sm_50 以上（コメントは「CUDA 12 の最低」）。コード中に Kepler / Fermi の記述はなく、Kepler は自己ビルドの運用（#12140: K80 を CUDA 11.4 + ドライバ 470 + `-DCMAKE_CUDA_ARCHITECTURES=...;37` で動かした報告。#18743: K40c はビルドできるが実行時にクラッシュ）。「Fermi」を含む issue は 0 件。cc 別の mmvq / mmq の選択表と `ggml_cuda_dp4a` の fallback は参考になる。Fermi（sm_21）は CUDA 8 までで、現行のビルドは不可能（一般知識で未検証）。**参考にする**。
- **`ggml_backend_sched` と `tests/test-backend-ops.cpp`**: adapter の検証にそのまま使う（ARCHITECTURE.md の「正しさの基準」と同じ）。

**CLBlast Xgemm / Xgemv**: 上記のとおり OpenCL 1.1 で動く F32 / F16 の GEMM と GEMV。**取り込む**（`find_package` またはベンダリング）。用途は (1) prefill 用の dequantize + GEMM の素朴な経路、(2) 自前の mat-mm の比較ベースライン。チューナを実機で走らせた JSON を差し込む overlay の API はあるが、動作は未検証。

**MNN**（alibaba/MNN、Apache-2.0、最終 push 2026-09-30、v3.6.1）

- OpenCL のホストが `CL_HPP_TARGET_OPENCL_VERSION 110` / `MINIMUM 110`。LLM 向けのバッファ版カーネル（`source/backend/opencl/execution/cl/*_buf.cl`）に、`gemm_conv1x1_int4_b4_buf.cl`、`gemm_conv1x1_int4_b8_buf.cl`、`gemm_conv1x1_int8_b4_buf.cl`、`gemv_conv1x1_buf.cl`、`attention_flash_decode_*`、`attention_flash_prefill_buf.cl`、`fused_proj_gemv_buf.cl` がある。
- `cl_khr_fp16` は `#ifdef MNN_SUPPORT_FP16` の中だけで、無効にすれば fp16 なしで読める構成。ただし一部のカーネルに image（`read_imagef`）とサンプラの宣言がある。チューニングの仕組みは `OpenCLTuneHeuristic.hpp` と `OpenCLGemmTune.cpp`。
- **1.1 を保ったまま量子化 mat-vec / GEMM を持つ数少ない実装**。ファイル単位で**移植する候補**にする（Apache-2.0 の表示を残す）。Fermi 等での動作と性能は未検証で、CL11 の R32 との比較対象にもなる。

### backends

- **opencl11 / opencl12**: 上記の旧 ggml-opencl、MNN、CLBlast。**dlprimitives**（artyom-beilis、MIT、最終 push 2026-10-02）は OpenCL 1.2 前提（`CL_TARGET_OPENCL_VERSION 120`、1.2 専用 API は grep で見つからない）で、`program_cache.cpp` の SQLite バイナリキャッシュ（NVIDIA は PTX が返るため無効にするコメントあり）と `sgemm.cl`、Winograd が参考になる。Fermi / Kepler の記載はなく、LLM 用の量子化カーネルや embedding はない。
- **vulkan**: ホスト側の土台は **volk**（MIT）+ **vk-bootstrap**（MIT、Vulkan ヘッダ 1.1 以上）+ **VulkanMemoryAllocator**（MIT、1.0〜1.4）を取り込み、薄く自作する。**Kompute**（Apache-2.0、最新リリース v0.9.0 は 2024-01）は 16bit storage の feature chain とサブグループサイズ指定を扱うコードが見つからず、土台としては使わない。**ncnn**（BSD-3 + 同梱の第三者ライセンス）の `gpu.cpp`（fp16 / int8 storage の可否とベンダー別の回避）、`allocator.cpp`、`pipelinecache.cpp` は設計の参考。**torch-vulkan**（peterc3-dev、MIT、個人製、star 2）は Q4_K の `matmul_q4` などのシェーダが参考になるが、成熟していない。NVIDIA 390 の Vulkan ICD は使えないので、対象は AMD（RADV）に限る。
- **cuda**: 上記のとおり参考のみ。優先度は最後（ARCHITECTURE.md と同じ）。

### adapters

- **PyTorch**: 公式チュートリアル "Facilitating New Backend Integration by PrivateUse1"、公式サンプル `test/cpp_extensions/open_registration_extension/torch_openreg`（README は機構検証用の STUB と明記）。必須オペレータは `empty.memory_format`、`empty_strided`、`as_strided`、`view`、`_reshape_alias`、`resize_`、`_copy_from`、`_copy_from_and_resize`、`_local_scalar_dense`、`set_` と CPU fallback。**pytorch_dlprim**（MIT、最終 push 2026-10-02）に、OpenCL デバイスを PrivateUse1 に載せた登録コード（`DeviceGuardImpl`、`PrivateUse1HooksInterface`、142 のオペレータ登録）があり、**一部を移植する**（MIT 表記を残す）。LLM 向けの embedding、`scaled_dot_product_attention`、量子化は未登録。PrivateUse1 拡張は libtorch の C++ ABI に依存するため、torch の版ごとにビルドが要る（推測）。
- **ONNX Runtime**: プラグイン EP の API は ORT 1.23 で導入、カーネル型のプラグイン EP の API は 1.24.1 以降（最新 v1.30.0、MIT）。共有ライブラリが `CreateEpFactories()` と `ReleaseEpFactory()` を export する C ABI で、バージョン付き構造体（`ort_version_supported`）。テンプレートは `onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry`。**取り込む**（ORT 1.24 以上）。EP から ScrapGpuLibrary の C API を呼ぶ薄い層にする。OpenCL EP はコアにも、見つけた範囲のコミュニティにもない。ドキュメントに「experimental」の注記が残る関数があり、現行版で再確認が要る。
- **standalone C API**: 流用元なし。自前で設計する（ONNX Runtime のバージョン付き構造体の作法は参考になる）。

### 他の旧 GPU / OpenCL 推論系

| 名前 | ライセンス | 判断 |
| --- | --- | --- |
| MNN | Apache-2.0 | 上記。最も有力な移植元 |
| llama.cpp ggml-opencl（現行） | MIT | OpenCL 2.0 必須。使わない（設計のみ） |
| MLC-LLM / TVM | Apache-2.0 | 使わない。TVM の OpenCL は既定で 1.2 を想定。コンパイラ基盤で重い |
| tinygrad | MIT | 使わない。公式ドキュメントは「OpenCL 2.0 compatible device」 |
| mllm | MIT | 使わない。OpenCL は初期実装 |
| TFLite GPU / LiteRT | Apache-2.0 | `cl_device.cc` の機能分岐の設計だけ参考 |
| distributed-llama / exo / Petals | MIT / Apache-2.0 / MIT | 使わない（上記） |

## 取り込むときの手順

1. 取り込んだ・移植したファイルは `third_party/` またはファイル冒頭の表記で出典・コミット・ライセンスを残し、リポジトリ直下の `THIRD_PARTY`（新規）に一覧する。
2. Apache-2.0 のコードを同梱する場合は、ライセンス全文を添付し、改変したファイルに変更の旨を書く。
3. GPL のツール（clpeak、mixbench）はリポジトリに入れず、実行手順と出力の取り込み方だけ文書にする。
4. 移植したカーネルは、必ず CPU 参照との NMSE（ARCHITECTURE.md の「正しさの基準」）と、CL11 のカーネルとの性能比較を取ってから採用する。

## 未検証（実機または追加調査が必要）

- Relic が GT 430 / GT 710 / GT 730 / WX 2100 / RX 6400 で動くか、動いたときの速度。
- MNN のバッファ版 int4 / int8 カーネルが Fermi / Kepler / Polaris / RDNA2 で通るか。
- clpeak と dlprimitives が OpenCL 1.1（Fermi）で動くか。`CL_HPP_TARGET_OPENCL_VERSION=110` で dlprimitives がビルドできるか。
- CLBlast のチューナ結果を overlay で差せるか。rusticl / Clover で CLBlast のアーキ判定が働くか。
- RADV が ggml-vulkan の必須拡張（16bit / 8bit storage、subgroup size control など）を出すか。
- Polaris の `power1_input` が示す範囲（SoC だけかボード全体か）。古いカーネルでの hwmon の挙動。
- Mesa の Clover が OpenCL 1.1 と 1.2 のどちらを報告するか（Mesa 25.2 のリリースノートに Clover の削除の記載があり、Mesa 25.0 の検証機は Clover を使える）。
- 旧ソース `e84b71c2c` のカーネルを、そのままの形で CL11 の参照として動かせるか。
