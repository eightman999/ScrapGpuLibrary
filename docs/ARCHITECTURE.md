# アーキテクチャ

ScrapGpuLibrary は、旧世代 GPU 向けの汎用 tensor / runtime ライブラリです。ggml（llama.cpp）は、その上に載る adapter の一つという位置付けにします。

```
ScrapGpuLibrary
├─ runtime
│  ├─ device              デバイスの列挙・能力・調整値・ドライバの癖（デバイスデータベース）
│  ├─ buffer              デバイスメモリ、ホストとの転送、公称上限を超える確保
│  ├─ stream / event      実行キュー、同期、プロファイル
│  ├─ allocator           バッファのプール、分割確保
│  └─ multi-GPU transport GPU 間のテンソル転送（PCIe x1 前提）
│
├─ tensor
│  ├─ dtype               F32 / F16（保存専用）/ 量子化型
│  ├─ layout              連続・パック済みなどの物理配置
│  └─ view / stride       ビュー、転置、ブロードキャスト、除算なしの添字計算
│
├─ ops                    op の定義、CPU 参照実装、バックエンドへのディスパッチ
│  ├─ matmul / matvec
│  ├─ softmax
│  ├─ norm
│  ├─ rope
│  ├─ attention
│  └─ elementwise
│
├─ quant
│  ├─ q4/q8               Q4_0 / Q8_0
│  ├─ k-quant             Q4_K / Q6_K など
│  └─ device-specific packing  GPU ごとの再配置（例: R32）とその逆変換
│
├─ backends               runtime と ops の実装（API ごと）
│  ├─ opencl11            OpenCL 1.1（Fermi など）
│  ├─ opencl12            OpenCL 1.2（Kepler など）
│  ├─ vulkan              AMD（Polaris / RDNA2）など
│  └─ cuda                Kepler 以降（ドライバの制約あり。後述）
│
└─ adapters
   ├─ ggml                ggml backend（llama.cpp からはツリー外プラグインとして読み込む）
   ├─ PyTorch extension
   ├─ ONNX Runtime        Execution Provider
   └─ standalone C API    安定 ABI。ほかの adapter の土台にもなる
```

## 依存の向き

```
adapters  ──►  ops / quant  ──►  tensor
    │              │
    └──────────────┴──►  runtime  ◄──  backends（runtime のインターフェースと op のカーネルを実装）
```

- `runtime`、`tensor`、`ops`、`quant` は、どのエンジン（ggml、PyTorch など）にも依存しません。
- `ops` は op の意味（入出力の形・型・数値の約束）と CPU 参照実装を持ちます。GPU のカーネルは `backends` 側が実装し、`ops` が実行時に選びます。
- adapter は、エンジンのテンソルや計算グラフを `tensor` のビューと `ops` の呼び出しに翻訳するだけにします。

## 設計の前提

| 項目 | 方針 |
| --- | --- |
| 言語 | 実装は C++17、公開 API は C（`scrapgpu_*`）。PyTorch / ONNX Runtime の adapter も C API の上に作る |
| 名前空間 | C++ は `scrapgpu`、C は `scrapgpu_` 接頭辞（`sgl` は SGLang と紛らわしいので使わない） |
| カーネル | OpenCL C 1.1 のカーネルを opencl11 / opencl12 で共有し、1.2 以降の機能は opencl12 側だけで使う。vulkan は GLSL → SPIR-V |
| 精度 | 演算は FP32。fp16 は保存形式として扱い、読み書きのときに変換する（Fermi / Kepler には fp16 演算がない） |
| デバイス知識 | ワークグループサイズ、カーネル選択、使ってはいけない機能などは、コードではなくデバイスデータベース（runtime/device）に置く。新しい GPU はデータの追加で対応する |
| 試験 | ops ごとに CPU 参照実装と比較する単体試験を持つ。正しさは誤差の許容範囲で判定する（下記「正しさの基準」） |

## 正しさの基準

生成したトークンが CPU と一致するかどうかは、正しさの基準にしません。量子化した重みとの内積で、CPU（ggml の CPU バックエンド）は活性化（src1）を 8bit に量子化してから計算し、CL11 は FP32 のまま計算します。そのため CL11 のほうが本来の積に近くても、Q4_K 系では 20 トークン前後で greedy の生成が分岐することがあります。トークンが違っても、誤りとは限りません。

| 階層 | 指標 | 位置付け |
| --- | --- | --- |
| op 単体 | CPU 参照との NMSE（llama.cpp の `test-backend-ops` と同じ考え方）。op ごとに許容値を持つ | 必須 |
| logits | 参照との KL ダイバージェンス（平均と 99 パーセンタイル）、top-1 の一致率、最大絶対誤差。llama.cpp の `llama-perplexity --kl-divergence-base / --kl-divergence` で測れる | 必須 |
| 品質 | perplexity の差 | 必須 |
| 健全性 | NaN / Inf がないこと、同じ語の繰り返しなどの崩壊がないこと | 必須 |
| greedy の一致 | 生成トークンの一致率（何トークン目まで一致したか） | 補助。回帰を素早く見つける用途 |

許容値は、良性の差を基準に校正します。具体的には、llama.cpp 公式のバックエンド同士の差（同じモデルを CPU と Vulkan（RX 6400）で動かしたときの KL ダイバージェンスなど）を測り、それを超えないことを条件にします。

許容値はモデルと文章で変わるので、固定値にはしません。同じモデル・同じ文章・同じ CPU の logits で公式のバックエンドとの差を測り、それを上限にします。たとえば Qwen3-0.6B-Base Q4_K_M では、公式 Vulkan（RX 6400）の CPU との差が平均 0.00776、99 パーセンタイル 0.0433、top-1 の一致率 95.10% で、CL11（GT 430）は 0.00681、0.0369、95.34% とすべて下回りました。別のモデルで決めた固定値（99 パーセンタイル 0.033）を当てはめると、同じ結果でも不合格と判定されます。

## バックエンド契約（P2 の前に固定する）

runtime / ops とバックエンドの間の約束を、次の概念だけで定めます。ggml 固有の概念（`ggml_tensor`、`ggml_cgraph`、ggml の型番号など）は入れません。

| 概念 | 決めること |
| --- | --- |
| Device | 列挙、能力（OpenCL のバージョン、warp / wave 幅、ローカルメモリ、単一確保の上限）、ドライバの癖のフラグ、調整値。デバイスデータベースから引く |
| Buffer | 確保と解放、「バッファ + バイトオフセット」によるアドレス指定（サブバッファのアライン制約を避けるため）、ホストとの転送、公称上限を超える確保の扱い |
| TensorDesc | 論理的な形・型・ストライド。ビュー、転置、ブロードキャスト。除算なしの添字計算に要る定数 |
| PhysicalLayout | デバイス上の物理配置（自然配置、R32 など）。論理形とは別の、テンソルの属性として持つ |
| Pack / Unpack | 論理配置と物理配置の相互変換。部分的な読み書きを含む。ホストとの転送時に runtime が呼ぶ |
| Command / Stream | in-order の実行キュー、同期点、イベント、エラーの伝え方 |
| Op | 入出力の形・型の約束、数値の許容誤差、CPU 参照実装 |
| KernelVariant | 1 つの op の実装候補。適用条件（型、物理配置、形）と調整値を持ち、デバイスデータベースの情報で選ぶ |

あわせて、次の点も決めます。

- **融合の表し方**: CL11 には、グラフの文脈がないとできない最適化があります（V の転置コピーと行列積の融合、不要な `CONT` の省略）。これを失わないよう、隣り合う op の組を表す最小限の Graph / Plan を任意の拡張として持つか、融合した op を Op として定義します。
- **エラーと同期**: 確保の遅延失敗（NVIDIA 390 で公称上限を超えたとき）など、非同期に現れる失敗をどこで報告するか。

## multi-GPU transport について

検証機（TB250）は 5 枚の GPU がすべて PCIe x1 で、ホスト経由の転送は約 0.40 GB/s です。

- 層ごとに GPU を分ける（パイプライン分割）なら、GPU 間を流れるのは活性化だけです。隠れ層 4096 の FP32 なら 1 トークン・1 区間あたり 16KB、約 40µs で済みます。
- 合計 VRAM（約 10GB）を使えば、1 枚には載らない 8B 級のモデルを全層 GPU で動かせる見込みがあります。
- 制約:
  - NVIDIA 390 では、1 プロセスで 2 枚目の GPU を開くと失敗する事例がある（pyopencl で確認、C API では未検証）。そのため、プロセスを分けて共有メモリで受け渡す transport も視野に入れる。
  - NVIDIA の Vulkan ICD は使えない（クラッシュする）。NVIDIA は OpenCL、AMD は Vulkan または OpenCL という混在が前提になる。

## cuda バックエンドについて

- Kepler（sm_35）の CUDA は 11.x までで、ドライバ 470 系が必要です。
- Fermi（sm_21）は CUDA 8 までで、古い gcc が必要です。
- TB250 は Fermi を使うため 390 ドライバに固定しています。cuda バックエンドの実機検証には、ドライバの切り替え（別環境や別ディスク）が要ります。優先度は最後にします。

## 現状と移行計画

現在の CL11 は、ggml backend として 1 ファイル（`ggml-cl11.cpp`、約 2300 行）に、ほぼすべての層が同居しています。動作と性能を保ったまま、段階的に切り出します。

### いまのコードと各層の対応（`ggml-cl11.cpp`）

| 現在の要素 | 移す先 |
| --- | --- |
| `cl11_probe_devices`、`cl11_device_info`、`cl11_tuning_for`、`GGML_CL11_DEVICES` | runtime/device（デバイスデータベース） |
| `cl11_ensure_context`、`cl11_build_program`、カーネル表、`cl11_launch`、プロファイル | backends/opencl11 |
| バッファ（ダミーのベースポインタ + オフセット）、公称上限を超える確保、`cl11_fill` | runtime/buffer、runtime/allocator |
| `cl11_td`（テンソル記述子）、`cl11_fastdiv_init`、`cl11_bcast` | tensor/view・stride |
| `cl11_rp_*`（R32 の pack / unpack / 転送） | quant/device-specific packing |
| `cl11_op_*`、`src/backends/opencl11/kernels/*.cl` | ops（意味と CPU 参照実装）+ backends/opencl11（カーネル） |
| `cl11_supports_op_impl`、`graph_compute`、`ggml_backend_*` の各関数、`cl11_can_skip_cont` | adapters/ggml |

### 段階

| 段階 | 内容 | 完了条件 |
| --- | --- | --- |
| P0 | モノリシックな CL11 を ggml backend として完成させる。ツリー外プラグイン化。GPU 別の最適化（GT 710/730、WX 2100、RX 6400）を llama.cpp 側の worktree で実施中 | 済み・実施中 |
| P1 | GPU 別の最適化を統合し、`adapters/ggml/` を CL11 の正本にする。性能の基準表と、正しさの許容値（公式バックエンド同士の差で校正）を固定する | 全 GPU で「正しさの基準」の必須項目を満たす |
| P1.5 | バックエンド契約（8 概念 + 融合 + エラーと同期）を文書とヘッダで固定する | 契約の文書と C/C++ のヘッダ。CL11 の現在の機能がすべて表現できることを確認 |
| P2 | runtime（device / buffer / stream / allocator、デバイスデータベース）と backends/opencl11 を、契約に沿って切り出す。ggml adapter は runtime を呼ぶ形にする | P1 と同じ試験 + 性能が基準表の ±3% 以内 |
| P3 | tensor、ops、quant を切り出し、C API と CPU 参照の単体試験を作る。ggml adapter は翻訳層だけにする | 単体試験 + P2 と同じ回帰試験 |
| P4 | multi-GPU transport（パイプライン分割）。異種 GPU にまたがって全層を GPU で動かす | 8B 級モデルを TB250 の複数 GPU で動かし、CPU との一致と速度を測る |
| P5 | backends の追加（opencl12、vulkan、cuda）と adapters の追加（PyTorch extension、ONNX Runtime） | adapter ごとに試験を定める |

切り出しは、性能に影響しないことを毎回確かめながら進めます。GT 430 の mat-vec はメモリ帯域の約 8 割まで出ているので、抽象化で 1 回の起動や添字計算が増えると、そのまま遅くなります。
