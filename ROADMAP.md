# ロードマップ

段階 P0〜P5 の状態、Phase 3 の手順、既知のリスクと未決の項目を 1 か所にまとめたものです。設計と契約は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)・[docs/API_CONTRACT.md](docs/API_CONTRACT.md)、移行の記録は [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md)、変更時の検査は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## P0 基盤の固定 — 済み

モノリシックな CL11 を ggml backend として完成させ、ツリー外プラグイン化しました。GPU 別の最適化（GT 710 / GT 730、WX 2100、RX 6400）は llama.cpp 側の worktree で進め、P1 で `adapters/ggml/` に統合しています。

基準のコミット、llama.cpp の版、CL11 の組み込み方、計測環境、モデルの SHA256、速度（5 回の平均と標準偏差）、正しさ、再現手順は [docs/BASELINES.md](docs/BASELINES.md) に固定しました。Gate 0 の判定は **Go**（基準の速度は 2026-10-09 に TB250 で測定）。

## P1 GPU 別の最適化の統合と基準の固定 — 済み

GPU 別の最適化を統合し、`adapters/ggml/` を CL11 の正本にしました。性能の基準表と、正しさの許容値（公式バックエンド同士の差で校正）を [docs/BASELINES.md](docs/BASELINES.md) に固定しています。

Core / backend / adapter の契約は [docs/API_CONTRACT.md](docs/API_CONTRACT.md) の Phase 1 固定版です。ABI は `SCRAPGPU_ABI_VERSION_MAJOR=0` / `MINOR=2`、呼び出し側の構造体は `struct_size` による拡張規則、登録済み backend は 0 個、CPU 参照実装は Phase 3 / 4 に延期。CL11 の現行機能（MoE の op を含む）は表現できない項目なしという判定でした。Gate 1 の判定は **Go**。

## P2 runtime と backends/opencl11 の切り出し — 済み

OpenCL 1.1 のランタイムを、ggml の include がない `src/backends/opencl11/` に切り出しました。ggml adapter（`ggml-cl11.cpp`）は runtime を呼ぶ形で、互換のための shim を残しています。

| ステップ | 移したもの | 状態 |
| --- | --- | --- |
| 1 | `device.*`：`cl11_device_info`、device 文字列 / スカラー、`GGML_CL11_DEVICES` の絞り込み、AMD の世代判定、`cl11_tuning_for`、`cl11_config`、MMX の解析と検証、probe の補助 | 済み |
| 2 | `context.*`：OpenCL のエラー文字列、遅延する `cl_context` / in-order queue の生成、`GGML_CL11_PROFILE` と `GGML_CL11_LAUNCH_LOG` の設定読み取り | 済み |
| 3 | `kernels.*` と `opencl11_kernels.cmake`：カーネルの id / class / 表、program の build、`.cl` の埋め込み | 済み |
| 4 | `buffer.*`：確保の上限とアラインメント、公称上限を超える確保の扱い、末尾 write による probe、定数サブバッファの cache と新しい順の解放 | 済み |
| 5 | `executor.h`：`cl11_args`、`cl11_launch`、`cl11_div_up`、`cl11_fill`、起動回数の集計、`GGML_CL11_PROFILE` の集計と表示 | 済み |

Gate 2 の判定は **Go**（2026-10-09、`ccde9c3`）。TB250 の 6 構成すべてが [docs/BASELINES.md](docs/BASELINES.md) の基準を満たしました。NVIDIA の差は最大 -0.9%〜+0.3%（目標 ±3% 以内）で、1 トークンあたりの起動回数、`test-backend-ops`、greedy の一致は分割前と同じです。カーネルの program build に渡すソースとオプションは、既定・kepler・fermi の調整でバイト単位で同じことを Mac で確認しました。

Phase 3 に持ち越した点は 2 つあります。バックエンド側のエラー出力が ggml のログではなく `stderr` に出ること（ログのコールバックを入れる）、`install-in-tree.sh` が別の git リポジトリの中のディレクトリーに対してパッチを「適用済み」と誤判定すること（分割前からの挙動）です。

## P3 tensor / ops / quant の切り出しと C API — 済み（Gate 3、2026-10-10）

ggml を使わずに C API だけでテンソル演算を GPU で実行できる状態にします。ggml adapter は、同じ演算の実装を呼ぶ翻訳層になります（[docs/plans/PHASE3_STANDALONE.md](docs/plans/PHASE3_STANDALONE.md)）。

- 演算の実装は 1 つだけにします。`cl11_op_*` をバックエンドへ移し、ggml adapter と C API の両方から呼びます。
- バックエンド内部のテンソルは `cl11_tensor`。adapter はノードごとにスタック上に記述子を作り、ggml の `op_params` の解釈は adapter に残します。
- グラフの文脈が要る融合は、ggml adapter では今の判定をそのまま使い、C API では融合した演算を明示的な op として呼びます。
- 速度の条件は Gate 2 と同じです（NVIDIA ±3%、AMD は許容幅、1 トークンあたりの起動回数が増えない、`clBuildProgram` のソースとオプションが変わらない）。

| 順 | 内容 | 主なファイル |
| --- | --- | --- |
| S1 | デバイスの実行時の状態（context、queue、プログラム、カーネル、調整値、定数サブバッファ、作業用バッファ、実行の状態）をバックエンドの `cl11_device` に移す。ggml のデバイスは `cl11_device` を指すだけにする | `src/backends/opencl11/device_state.*`、`ggml-cl11.cpp`（済み） |
| S2 | ggml に依存しない tensor モジュール（Phase 3.1）。TensorDesc・dtype・layout・ビュー・ブロードキャスト・除算なしの添字・R32 の配置を `src/tensor/` に、カーネルに渡す記述子を `src/backends/opencl11/tensor_abi.h` に。adapter は変換だけ | `src/tensor/*`、`src/backends/opencl11/tensor_abi.h`（済み） |
| S3 | R32 の詰め替え（pack / unpack / 部分転送）をバックエンドへ | `src/backends/opencl11/r32.*`（済み） |
| S4 | 演算（`cl11_op_*`）と対応判定の ggml に依存しない部分をバックエンドへ | `src/backends/opencl11/ops*.*`（済み） |
| S5 | 公開 API：stream、`scrapgpu_opencl11_register`（vtable の実装）、`scrapgpu_op_execute`、`scrapgpu_tensor_upload / download`、`op_supported`、1 演算 1 ノードの plan | `include/scrapgpu/*.h`、`src/core/*`、`src/backends/opencl11/api.cpp`（済み） |
| S6 | CPU 参照実装（全演算と Q4_0 / Q8_0 / Q4_K / Q6_K の逆量子化）と、演算ごとの単体試験（GPU と参照を NMSE で比べる）。ggml を使わない例（`examples/standalone_matmul.c`） | `src/core/ref/*`（CPU 参照実装と試験は済み。ggml の CPU と 112 ケース一致、`adapters/ggml/tools/ref_check`）、`tests/ops/*`、`examples/*`（済み） |

1 段階 1 コミットです。S6 の CPU 参照実装は S1〜S5 と独立しているので並行して作れます。各段階で、`-Wall -Wextra` で警告のないビルド、`test-backend-ops` の 3 調整（既定・`FORCE_GEN=kepler`・`fermi`）、`mvm_check` 18 / 18、`attn_check` 8 / 8、`ctest`、`clBuildProgram` のソースとオプションの不変、`src/backends/opencl11`・`src/core`・`include/scrapgpu`・`tests/ops`・`examples` に ggml の include がないことを確認します。

**Gate 3**: Gate 2 と同じ TB250 の回帰（6 構成、BASELINES.md と比べる）と、ggml を使わない単体試験と例が TB250 の 6 構成と Mac で通ること。

## P4 multi-GPU transport — 未着手

層ごとに GPU を分けるパイプライン分割です。検証機の 5 枚の GPU はすべて PCIe x1 で、ホスト経由の転送は約 0.40 GB/s です。層ごとに分ければ GPU 間を流れるのは活性化だけで、隠れ層 4096 の FP32 なら 1 トークン・1 区間あたり 16KB、約 40µs で済みます。合計 VRAM は約 10GB あるので、1 枚には載らない 8B 級のモデルを全層 GPU で動かせる見込みがあります。

制約と未決:

- NVIDIA 390 では、1 プロセスで 2 枚目の GPU を開くと失敗する事例がある（pyopencl で確認、C API では未検証）。プロセスを分けて共有メモリで受け渡す transport も視野に入れます。
- NVIDIA の Vulkan ICD は使えない（クラッシュする）ため、NVIDIA は OpenCL、AMD は Vulkan または OpenCL という混在が前提になります。
- 複数の device をまたぐ転送、2 本目以降の stream と event の形、PyTorch の PrivateUse1 で必要になる確保・解放のフックとの関係は [docs/API_CONTRACT.md](docs/API_CONTRACT.md) 11 章の未決事項です。

## P5 バックエンドと adapter の追加 — 未着手

| 追加するもの | 前提・方針 |
| --- | --- |
| backends/opencl12 | opencl11 のカーネルを共有し、1.2 以降の機能は opencl12 側だけで使う。Kepler の調整値は CLBlast のチューナで作り、キャッシュは dlprimitives の `program_cache` を参考にする |
| backends/vulkan | ggml-vulkan の `mul_mat_vec` と dequant のシェーダーを移植し、ホスト側は自作（土台は volk / vk-bootstrap / VMA）。Vulkan 1.2 と 16bit storage が前提。NVIDIA 390 の Vulkan ICD は使えないので対象は AMD だけ |
| backends/cuda | コードは取り込まず、cc 別の選択表と dp4a の fallback だけ見る。Kepler は CUDA 11.x + ドライバ 470 の実験扱い、Fermi は CUDA 8 までで現行のビルドはできない。優先度は最後 |
| adapters/PyTorch | Stage A（`torch.ops.scrapgpu` のカスタム op）から始めて、Stage B で PrivateUse1。登録コードは pytorch_dlprim（MIT）を一部移植する |
| adapters/ONNX Runtime | プラグイン EP のテンプレート（MIT）を使い、EP から C API を呼ぶ薄い層にする（ORT 1.24 以上） |

adapter ごとに試験を定めます。リポジトリへの取り込み可否と条件は [docs/REUSE.md](docs/REUSE.md) のとおりです。

## 既知のリスクと未決

### 構造とビルド（[docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md) 6 章）

| 項目 | リスク | 未決 / 提案 |
| --- | --- | --- |
| out-of-tree plugin と in-tree patch | `adapters/ggml/CMakeLists.txt` は 2 経路を持つ。分離後 target 名や include path がずれると片方だけ壊れる | 共通 object / static lib を作り、最後の `ggml-cl11` target だけ経路差を持たせる提案 |
| symbol visibility | `libggml-cl11.so` 内に static library を link すると、内部 symbol が外へ漏れる / 消える可能性がある | public は `ggml_backend_cl11_*` と将来の `scrapgpu_*` C ABI だけに限定し、内部は hidden visibility にする提案 |
| core が OpenCL なしで build | `find_package(OpenCL REQUIRED)` は `SCRAPGPU_BUILD_OPENCL11=ON` のときだけ走り、`scrapgpu_core` は OpenCL の include / link を含まない（adapter 経路は独自に OpenCL を要求する） | `scrapgpu_core` は OpenCL なしを保つ |
| core が ggml なしで build | ggml の内部ヘッダを直接 include しているコードが adapter 側に残る | core と backend のヘッダ / ソースへの ggml include を禁止し、adapter target だけが ggml を見る（`ctest` の依存チェックで機械的に確認する） |
| kernel source embedding | CMake が `.cl` を header に埋め込む。backend 分離で生成 header の場所が変わる | 埋め込みは `backends/opencl11` target 所有に移す。`GGML_CL11_KERNEL_DIR` は backend の debug option として維持する |
| lifetime | context / program / cache が device context に混在し、ggml の dev / buft も同じ struct に入る | backend が持つ context と adapter が持つ vtable object の owner を分け、解放順を単体テスト化する提案 |
| fusion planner | ggml graph の adjacency と mutable reorder に依存する | core plan API ができるまで融合判定は adapter に残す。移行後も launch count を比較する |

### 性能

- 受け入れ条件は NVIDIA ±3%（AMD は `tol_pct`）、1 トークンあたりの起動回数が増えないこと、モデルの読み込み時間と VRAM の使用量が大きく変わらないことです。
- 抽象化で launch 前の overhead が増えると decode に直撃します。GT 430 の mat-vec はメモリ帯域の約 8 割を出しているので、起動ごとのわずかな上乗せも性能に響きます。per-node の plan cache、dense な kernel 配列、allocation-free な launch を受ける条件にします。

### 環境とドライバ（[docs/HARDWARE.md](docs/HARDWARE.md)）

- NVIDIA 390 の Vulkan ICD はクラッシュするため、Vulkan を使うプロセスには AMD の ICD だけを見せます。
- NVIDIA 390 は `CL_DEVICE_MAX_MEM_ALLOC_SIZE` を超える確保を受け付け、遅延して確保します（末尾への書き込みで失敗を検出する）。Mesa は超過を拒否するので、AMD では公称上限を厳守します。
- NVIDIA 390 と Kepler では、`__constant` に渡したサブバッファの開始オフセットが無視されます。Fermi では正しく動きます。
- NVIDIA 390 は 1 プロセスで 2 枚目の GPU を開けない事例があります。
- Fermi は 32bit の除算・剰余が遅く、グリッドの各次元が 65535 までです。
- Mesa rusticl + GCN では `fma()` がソフトウェアでエミュレートされ、`mad()` の約 120 倍遅いです。
- Clover は 1 回の起動コストが program の大きさに比例して増えるため、program を 3 つに分けます。RX 6400 の Clover は 2 回目の起動でハングするので使いません。
- Clover（WX 2100）では、値渡しの構造体の引数の後ろにローカルメモリの引数があると引数の位置がずれます。ローカルメモリの引数を使わず、固定長配列にします。
- rusticl は in-order queue で 1 起動あたり約 13.8µs かかるため（RX 6400 の実測）、生成はディスパッチ回数で頭打ちになります。

---

## English summary

**P0 (done).** The monolithic CL11 backend and the out-of-tree plugin are finished. [docs/BASELINES.md](docs/BASELINES.md) fixes the baseline commit, the llama.cpp version, the measurement environment, the model SHA256 sums, the speed (5 runs, mean and standard deviation), the correctness and the reproduction steps. Gate 0: **Go**.

**P1 (done).** The per-GPU tuning was integrated into `adapters/ggml/` and the tolerance rules were calibrated against the official backends. The contract in [docs/API_CONTRACT.md](docs/API_CONTRACT.md) expresses every current CL11 feature (including the MoE ops) and requires no ggml type. Gate 1: **Go**.

**P2 (done).** The OpenCL 1.1 runtime lives in `src/backends/opencl11/` (device, context, kernels, buffer, executor) with no ggml includes; the adapter still owns the ggml API, op dispatch, fusion planning and the R32 transfer until Phase 3. Gate 2 (2026-10-09, `ccde9c3`): **Go** — all six TB250 configurations are inside the baseline (NVIDIA between -0.9% and +0.3% against a ±3% target), with the same launches per token, `test-backend-ops` and greedy results as before the split. Carried into Phase 3: backend error output goes to `stderr` instead of the ggml log, and `install-in-tree.sh` still misjudges a patch as applied when it runs against a directory inside another git repository.

**P3 (done, Gate 3 on 2026-10-10).** Tensor / ops / quant and a C API that works without ggml, in six steps S1-S6 (device runtime state, `cl11_tensor` and descriptors, the R32 packing, the ops and their support logic, the public API, then the CPU reference implementation, the per-op unit tests and a ggml-free example). S6 is independent of S1-S5 and can run in parallel. Gate 3: the Gate 2 TB250 regression plus the ggml-free tests and examples on the six TB250 configurations and on a Mac.

**P4 (not started).** Multi-GPU transport as a pipeline split. All five GPUs of the test machine are on PCIe x1 with about 0.40 GB/s through the host; per-layer splitting moves only activations (about 16KB and 40µs per token and hop for a 4096-wide FP32 hidden layer), and the roughly 10GB of total VRAM could hold an 8B model on GPUs. Open: NVIDIA 390 failing to open a second GPU in one process (seen with pyopencl, not verified through the C API), shared-memory transport with one process per device, and the mixed NVIDIA-OpenCL / AMD-Vulkan setup the broken NVIDIA 390 Vulkan ICD forces.

**P5 (not started).** opencl12 (shared kernels, CLBlast tuner, dlprimitives-style program cache), vulkan (ported mat-vec and dequant shaders on volk / vk-bootstrap / VMA, AMD only because the NVIDIA 390 Vulkan ICD crashes), cuda (reference only, lowest priority), and the PyTorch (custom ops first, then PrivateUse1) and ONNX Runtime (plugin EP, ORT 1.24+) adapters. What may be taken in is decided by [docs/REUSE.md](docs/REUSE.md).

**Known risks.** The out-of-tree and in-tree build paths of `adapters/ggml` can drift apart; internal symbols can leak out of `libggml-cl11.so`; the core must stay free of ggml and OpenCL includes (checked by `ctest`); the fusion planner still depends on the ggml graph; and every abstraction has to keep the launch path allocation-free, with the kernel variant chosen once at plan build, because decode pays for each launch.
