# API Contract（Core / Backend / Adapter の契約）

状態: **Phase 1 固定版（API contract headers + minimal core + contract tests）**。`include/scrapgpu/` と `tests/core/` の契約テストで固定する。CL11 の現行機能（`adapters/ggml/ggml-cl11.cpp`、`fc27d10` 以降の MoE op を含む）をすべて表現できること、ggml の型を要求しないことが Gate 1 の条件。

この文書は「何を決めるか」と「CL11 で実際に問題になった点」を中心に書く。関数単位の対応は [MIGRATION_PLAN.md](MIGRATION_PLAN.md) を参照。

## 1. ABI とバージョン

- 公開 API は C（`scrapgpu_` 接頭辞）。内部は C++17。C++ の型・例外・STL を公開ヘッダに出さない。
- ハンドルは不透明なポインタ（`scrapgpu_device_t`、`scrapgpu_buffer_t`、`scrapgpu_stream_t`、`scrapgpu_plan_t`）。
- 拡張に備え、呼び出し側が渡す構造体は先頭に `size_t struct_size` を持つ。ライブラリは知らない末尾のフィールドを無視し、足りない場合は既定値を使う。
- `SCRAPGPU_ABI_VERSION_MAJOR / MINOR`。MAJOR が変わると互換性がない。`scrapgpu_get_abi_version()` で実行時にも確認できる。
- ggml プラグイン（`libggml-cl11.so`）は Core と CL11 バックエンドを静的にリンクし、ggml の API 以外のシンボルを外へ出さない（`-fvisibility=hidden`）。

## 2. 状態とエラー

- 戻り値は `scrapgpu_status`：`OK`、`UNSUPPORTED`、`INVALID_ARGUMENT`、`OUT_OF_MEMORY`、`BACKEND_ERROR`、`DEVICE_LOST`。
- 詳細な理由は、スレッドごとの最後のエラー文字列（`scrapgpu_last_error()`）。`UNSUPPORTED` には理由を必ず付ける（例：「R32 で再配置したテンソルのビュー」「ARGSORT の行がローカルメモリに収まらない」）。
- **遅延して現れる失敗**：NVIDIA 390 はメモリを最初に使うまで確保しない。確保の API は、末尾への書き込みで確保を確かめてから `OK` を返す（CL11 の現行の挙動）。カーネル実行中の失敗は、次の同期点（`stream_sync`）で `OUT_OF_MEMORY` / `BACKEND_ERROR` として返す。`abort` しない。

## 3. Device

`scrapgpu_device_info`（読み取り専用）：

| 区分 | 項目 |
| --- | --- |
| 識別 | 名前、ベンダー、プラットフォーム名、ドライバの版、バックエンド名（`opencl11`）、API の報告版と実際に使う版（CL11 は常に 1.1 として扱う） |
| 世代 | `generation`：`NV_FERMI`、`NV_KEPLER`、`AMD_GCN`、`AMD_RDNA`、`APPLE`、`UNKNOWN`。判定の根拠（compute capability、デバイス名、Mesa のコード名）も文字列で残す |
| 容量 | 全メモリ、単一確保の公称上限、実際に許す上限（NVIDIA 390 は公称を超えても可、Mesa は不可）、ワークグループの上限、ローカルメモリ、CU 数、warp / wave 幅 |
| ドライバの癖（ビット集合） | `LAZY_ALLOC`、`ALLOC_ABOVE_MAX_OK`、`CONST_SUBBUFFER_OFFSET_BROKEN`（Kepler + 390）、`SUBBUFFER_RELEASE_ORDER`（390）、`FMA_EMULATED`（rusticl + GCN）、`LAUNCH_COST_GROWS_WITH_PROGRAM`（Clover）、`DENORMALS_FLUSHED`（起動時のプローブの結果） |
| 調整 | 適用中のチューニングプロファイルの名前と、環境変数で上書きされた項目の一覧 |

ハードウェアの能力（容量、世代）とドライバの癖を分けて持つ。癖は、プローブで分かったものと既知の表によるものを区別する。

## 4. Buffer とアドレス指定

- 位置は「バッファのハンドル＋バイトオフセット」で表す。サブバッファは原則使わない（アライン制約と 390 の不具合のため）。例外は、定数メモリに渡す小さな入力（Fermi の mat-vec）で、バックエンドの内部だけで使う。
- アラインメントは 256 バイト以上。1 つのバッファは 2 GiB 未満（カーネルが 32bit のバイトオフセットを使うため）。
- 大きなモデルは、ggml-alloc が `max_size` に従って複数のバッファに分ける（現行どおり）。**1 つのテンソルが複数のバッファにまたがることは扱わない**（必要になったら、対応する op を限定して別途定義する）。
- 転送：`write(buffer, offset, host, size)`、`read(...)`、`copy(dst, dst_off, src, src_off, size)`、`fill(buffer, offset, size, byte)`（OpenCL 1.1 に `clEnqueueFillBuffer` はないのでカーネルで行う）。いずれも stream に積まれ、`blocking` を指定できる。
- 所有権：バッファは作成した device が所有する。解放は、そのバッファを使うコマンドの完了後でなければならない（呼び出し側の責任。デバッグビルドでは検査する）。

## 5. TensorDesc と PhysicalLayout

- `scrapgpu_tensor_desc`：`dtype`、`ne[4]`（0 番目が最も内側）、`nb[4]`（バイト単位のストライド）、`buffer`、`offset`、`layout`。ggml と同じ並びにして、adapter の変換を単純にする。
- `dtype`：`F32`、`F16`、`I32`、`I64`、`Q4_0`、`Q8_0`、`Q4_K`、`Q6_K`（ほかの ggml の型は値だけ予約し、未対応として扱う）。
- `layout`（物理配置）：`NATURAL`（論理と同じ並び。ストライドで連続か非連続かが分かる）、`R32`（Q4_0 / Q8_0 / Q4_K / Q6_K を 32 行ずつ再配置したもの。スライス `ne2*ne3` ごと）。
- **R32 はテンソル全体に付く属性**。R32 のテンソルのビューは作れない（`UNSUPPORTED`）。R32 が使えるのは、連続で `nb2 % 16 == 0` のときだけ。
- 論理と物理の変換（Pack / Unpack）は、ホストとの転送のときに Core が行う。部分的な読み書きも、量子化ブロックの境界で受け付ける。全量を逆量子化することはしない。

## 6. Stream と同期

- 1 つの device に、in-order の stream を 1 本（初期実装）。op の投入は非同期で、同期するのは `stream_sync` と、ブロッキングの読み出しだけ。**op ごとの `clFinish` はしない。**
- ホストに読み戻さないと進めない処理（`MUL_MAT_ID` のプロンプト処理で専門家の番号を読む）は、op の契約に「同期点を含む」と明記し、回数を計測できるようにする。
- 計測用の軽いカウンタを stream ごとに持つ：カーネルの起動回数、同期の回数、ホストとの転送バイト数。`GGML_CL11_PROFILE` 相当のカーネル別時間は、オプションで有効にする。
- スレッド：device と stream はスレッドセーフではない。1 本の stream は 1 スレッドから使う。

## 7. Op

- op の種類は Core 独自の列挙（`MUL_MAT`、`MUL_MAT_ID`、`GET_ROWS`、`SET_ROWS`、`CPY`、`ADD`、`MUL`、`SCALE`、`RMS_NORM`、`ROPE`、`SOFT_MAX`、`GLU`、`UNARY`、`SSM_CONV`、`CONCAT`、`PAD`、`SUM_ROWS`、`ARGSORT`、`TOP_K`、`ADD_ID` など）。ggml の列挙との対応表は adapter が持つ。
- 各 op は、入力と出力の `tensor_desc`、op 固有のパラメータ構造体（ROPE のモードと yarn の値、SOFT_MAX の scale と mask、GLU の種類など）を受け取る。
- `scrapgpu_op_supported(device, op, &reason)`：対応しない理由を返す。**暗黙の転送や CPU での実行はしない**。CPU に回すかどうかは adapter が決める（ggml ではスケジューラが行う）。
- 正しさの基準：op ごとに CPU の参照実装との NMSE の許容値を文書に持つ（llama.cpp の `test-backend-ops` と同じ値を出発点にする）。

## 8. KernelVariant（公開しない）

- バックエンド内部の概念。op・型・物理配置・形の区分（mat-vec か mat-mat か、`ne11` の大きさ）・世代の調整値から、実行するカーネルを決める。
- **選択はプランを作るときに 1 回だけ行い、結果（カーネルのハンドル、ワークグループの大きさ、引数の固定部分）をノードに保存する。** 1 回の起動ごとに表を引いたり、文字列を作ったり、メモリを確保したりしない。現行の CL11 は、1 トークンで約 450〜700 回カーネルを起動し、1 回の固定費が 10µs 前後の GPU があるので、起動ごとのわずかな上乗せも性能に直接響く。

## 9. Graph / ExecutionPlan（最小限）

融合の多くは、隣り合うノードの関係を見て判断している（RMS_NORM＋MUL、ADD＋RMS_NORM＋MUL、同じ src1 の複数の MUL_MAT、K / V の SET_ROWS の組、不要な CONT の省略、V キャッシュの CONT＋MUL_MAT）。

- adapter は、ノードの列（op、入出力の desc、パラメータ）と、各出力の使用先の数を Core に渡す。
- バックエンドは、その列の上で融合の可否を判断する。中間の結果がほかで使われていないことなど、条件はバックエンドが確かめる。
- ノードの並べ替え（multi mat-vec のために Q / K / V を隣に移す `graph_optimize`）は、adapter の側の処理として残す。Core には「並べ替えてよいか」の判断に要る依存関係だけを持たせる。
- プランは、グラフの形が同じなら再利用する（llama.cpp は毎トークンほぼ同じグラフを作る）。キャッシュのキーはノードの op・型・形・バッファの位置の列。
- 汎用のグラフコンパイラは作らない。現行の融合を表現できる最小限に留める。

## 10. CPU フォールバックと参照実装

- Core は「この op はこの device で実行できない」と理由つきで返すだけで、勝手に CPU で計算しない。
- 単体テストと standalone API のために、CPU の参照実装（F32、量子化の逆変換は ggml の `ggml-quants.c` から移植）を Core に持つ。性能は求めない。

## 11. 未決事項

- 2 本目以降の stream（コピー専用など）と event の形。
- 複数の device をまたぐ転送（P4）。
- PyTorch の PrivateUse1 で必要になる、メモリの確保・解放のフック（キャッシュアロケータ）との関係。
- 環境変数（`GGML_CL11_*`）の扱い。当面は CL11 バックエンドが読み、将来は `scrapgpu_device_config` で渡せるようにする。

## 12. Phase 1 で固定した決定

| 項目 | 決定 |
| --- | --- |
| ABI version | Phase 1 は `SCRAPGPU_ABI_VERSION_MAJOR=0`、`SCRAPGPU_ABI_VERSION_MINOR=2`。`scrapgpu_get_abi_version()` は `(major << 16) | minor` を返す。major 変更は ABI 非互換、minor 変更は後方互換の構造体末尾追加に限る。 |
| 構造体拡張 | 呼び出し側入力の構造体は `struct_size` を先頭に置く。Core は既知サイズより大きい末尾を無視し、古い小さい構造体は `struct_size` までだけ書く。Phase 1 では `scrapgpu_device_info_init()` と契約テストでこの規則を固定した。 |
| CPU 参照実装 | 10章のCPU参照実装は Phase 3/4 に延期する。Phase 1 の Core は型・形・契約検証だけを持ち、空のCPU実行プレースホルダは作らない。 |
| Backend registry | Phase 1 では登録済みbackendは0個でよい。`scrapgpu_op_supported(NULL, ...)` はop descを検証した後、理由付き `UNSUPPORTED` を返す。 |
| Backend interface | `include/scrapgpu/backend.h` は内部向けC ABIとして公開ヘッダと同じ依存禁止を受ける。backend vtable は device/buffer/stream/op query/plan build/execute/free を持つ。 |
| Plan | 最小planは `scrapgpu_plan_desc` のnode列と、各node出力のuse countだけを持つ。融合判定に必要な隣接関係と「中間出力が他で使われていない」条件を表せる最小形に留める。 |
| Tensor layout | R32 は `scrapgpu_layout` の値で表し、R32 eligibility helper は「contiguous かつ `nb[2] % 16 == 0`」だけを見る。R32 view禁止や量子化block境界の部分転送は Phase 3 のPack/Unpack契約で実装する。 |
| dtype reserved values | 基本dtypeは `F32/F16/I32/I64/Q4_0/Q8_0/Q4_K/Q6_K`。Gate 1 の `CONCAT` 表現のため、`I16` と `BF16` は予約済み2-byte dtypeとして名前だけ固定した。計算opの対応を意味しない。 |
| Op validation | Phase 1 の `scrapgpu_op_desc_validate()` はshape/type/parameter presenceの契約チェックだけを行う。GPU local memory limit、scratch容量、世代別kernel選択はbackendの `op_supported` が Phase 2 以降に判断する。 |
| CMake layout | `SCRAPGPU_BUILD_CORE=ON`、`SCRAPGPU_BUILD_OPENCL11=OFF`、`SCRAPGPU_BUILD_GGML_ADAPTER=OFF` を既定にした。root buildで `SCRAPGPU_BUILD_GGML_ADAPTER=ON` を渡すと従来どおり `adapters/ggml` を追加する。`cmake -S adapters/ggml ...` は独立したまま。 |

## 13. Gate 1 mapping

### CL11 Supported operations

| CL11 feature / op | Contract element | Gate 1 result |
| --- | --- | --- |
| `NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE` | `scrapgpu_op` values with zero-compute validation | Representable |
| `MUL_MAT` | `SCRAPGPU_OP_MUL_MAT`, `scrapgpu_mul_mat_params`, `scrapgpu_tensor_desc` dtype/layout/stride | Representable |
| `GET_ROWS` | `SCRAPGPU_OP_GET_ROWS`, `scrapgpu_get_rows_params`, index tensor dtype | Representable |
| `SET_ROWS` | `SCRAPGPU_OP_SET_ROWS`, `scrapgpu_set_rows_params`; paired form as `SCRAPGPU_OP_SET_ROWS_PAIR` in plans | Representable |
| `ADD`, `SUB`, `MUL`, `DIV` | Elementwise `scrapgpu_op` values, F32 tensor descs and broadcastable strides | Representable |
| `SCALE` | `SCRAPGPU_OP_SCALE`, `scrapgpu_scale_params { scale, bias }` | Representable |
| `CLAMP` | `SCRAPGPU_OP_CLAMP`, `scrapgpu_clamp_params` | Representable |
| `SQR`, `SQRT` | Dedicated unary-like `scrapgpu_op` values | Representable |
| `UNARY` (`SILU`, `RELU`, `NEG`, `ABS`, `SIGMOID`, `TANH`) | `SCRAPGPU_OP_UNARY`, `scrapgpu_unary_params.kind` | Representable |
| `GLU` (`SWIGLU`, `REGLU`, fused/split, swapped) | `SCRAPGPU_OP_GLU`, `scrapgpu_glu_params` | Representable |
| `RMS_NORM` | `SCRAPGPU_OP_RMS_NORM`; fusions as `SCRAPGPU_OP_RMS_NORM_MUL` and `SCRAPGPU_OP_ADD_RMS_NORM_MUL` in plans | Representable |
| `ROPE` normal / NEOX, yarn, `freq_factors`, `n_offs` | `SCRAPGPU_OP_ROPE`, `scrapgpu_rope_params` | Representable; mrope/vision remain unsupported exactly as CL11 documents |
| `SOFT_MAX` with optional mask and scale | `SCRAPGPU_OP_SOFT_MAX`, `scrapgpu_soft_max_params` | Representable; ALiBi/sinks remain unsupported |
| `CPY`, `DUP`, `CONT` | `SCRAPGPU_OP_CPY`, `SCRAPGPU_OP_DUP`, `SCRAPGPU_OP_CONT`, `scrapgpu_cpy_params` | Representable |
| `MUL_MAT_ID` | `SCRAPGPU_OP_MUL_MAT_ID`, `scrapgpu_mul_mat_id_params`, plan nodes for prompt/generation policy | Representable |
| `ADD_ID` | `SCRAPGPU_OP_ADD_ID` | Representable |
| `SSM_CONV` | `SCRAPGPU_OP_SSM_CONV`, `scrapgpu_ssm_conv_params` | Representable |
| `CONCAT` any dimension, 4-byte and 2-byte elements | `SCRAPGPU_OP_CONCAT`, `scrapgpu_concat_params.dim`, dtype type sizes including reserved `I16/BF16` | Representable |
| `PAD` / circular pad | `SCRAPGPU_OP_PAD`, `scrapgpu_pad_params.before/after/circular` | Representable |
| `SUM_ROWS` | `SCRAPGPU_OP_SUM_ROWS` | Representable |
| `ARGSORT`, `TOP_K` | `SCRAPGPU_OP_ARGSORT`, `SCRAPGPU_OP_TOP_K`, `scrapgpu_argsort_params.order/top_k` | Representable; local-memory row limit is backend support logic |

### CL11 optimization and runtime features from the brief

| CL11 feature | Contract element | Gate 1 result |
| --- | --- | --- |
| Q4_0 / Q8_0 / Q4_K / Q6_K R32 repack | `scrapgpu_dtype`, `SCRAPGPU_LAYOUT_R32`, R32 eligibility helper | Representable |
| Fermi / Kepler / AMD / Apple generation tuning | `scrapgpu_device_generation`, `tuning_profile_name`, backend plan build | Representable |
| Fermi denormal products | `SCRAPGPU_DEVICE_QUIRK_DENORMALS_FLUSHED`, backend-specific variant selection | Representable |
| Constant memory mat-vec and Kepler constant sub-buffer bug | `SCRAPGPU_DEVICE_QUIRK_CONST_SUBBUFFER_OFFSET_BROKEN`, backend-internal buffer/cache policy | Representable |
| 128-bit coalesced load variants | Backend `KernelVariant` hidden behind `plan_build`; tensor dtype/layout/shape carry the selection inputs | Representable |
| K / V cache kernels and attention path | `SCRAPGPU_OP_MUL_MAT`, `SCRAPGPU_OP_CONT_MUL_MAT`, tensor stride/layout and plan context | Representable |
| `RMS_NORM + MUL` and `ADD + RMS_NORM + MUL` fusion | `SCRAPGPU_OP_RMS_NORM_MUL`, `SCRAPGPU_OP_ADD_RMS_NORM_MUL`, use counts in `scrapgpu_plan_node` | Representable |
| Multiple `MUL_MAT` fusion | `SCRAPGPU_OP_MUL_MAT_MULTI`, plan node list and output use counts | Representable |
| Paired `SET_ROWS` fusion | `SCRAPGPU_OP_SET_ROWS_PAIR`, plan node list and output use counts | Representable |
| Unnecessary `CONT` omission / V cache `CONT + MUL_MAT` | `SCRAPGPU_OP_CONT_MUL_MAT`, plan node list and tensor stride/layout | Representable |
| Large tensor allocation split | Buffer contract keeps one tensor within one buffer; multiple buffers are adapter/runtime allocation policy, not tensor ABI | Representable for current CL11 behavior |
| `MUL_MAT_ID` generation kernels and prompt scratch path | `SCRAPGPU_OP_MUL_MAT_ID`, `scrapgpu_mul_mat_id_params`, backend plan policy | Representable |
| Mesa `split_programs` | Backend-internal program layout; no public ABI exposure required | Representable |
| AMD `mad()` tuning | `generation`, `tuning_profile_name`, backend plan/build options | Representable |
| Lazy allocation probe and allocations above reported max | `SCRAPGPU_DEVICE_QUIRK_LAZY_ALLOC`, `SCRAPGPU_DEVICE_QUIRK_ALLOC_ABOVE_MAX_OK`, reported vs used max allocation fields | Representable |
| Sub-buffer release order | `SCRAPGPU_DEVICE_QUIRK_SUBBUFFER_RELEASE_ORDER`; backend buffer lifetime policy | Representable |
| Launch cost grows with program | `SCRAPGPU_DEVICE_QUIRK_LAUNCH_COST_GROWS_WITH_PROGRAM`; backend program splitting policy | Representable |

Gate 1 non-representable items: **none**. Unsupported CL11 cases documented in `docs/CL11.md`（`FLASH_ATTN_EXT`、`NORM`、`GROUP_NORM`、`SSM_SCAN` / RWKV、未対応量子型、長すぎる `ARGSORT` / `TOP_K` など）は、現在のCL11機能ではなくCPU fallback対象なので、Core contractでは `op_supported` の理由付き `UNSUPPORTED` として表す。
