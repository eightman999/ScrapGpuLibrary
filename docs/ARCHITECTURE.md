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
| 試験 | ops ごとに CPU 参照実装と比較する単体試験を持つ。ggml adapter では llama.cpp の `test-backend-ops` と、生成結果の CPU との一致を回帰試験に使う |

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
| `cl11_op_*`、`kernels/*.cl` | ops（意味と CPU 参照実装）+ backends/opencl11（カーネル） |
| `cl11_supports_op_impl`、`graph_compute`、`ggml_backend_*` の各関数、`cl11_can_skip_cont` | adapters/ggml |

### 段階

| 段階 | 内容 | 完了条件 |
| --- | --- | --- |
| P0 | モノリシックな CL11 を ggml backend として完成させる。ツリー外プラグイン化。GPU 別の最適化（GT 710/730、WX 2100、RX 6400）を llama.cpp 側の worktree で実施中 | 済み・実施中 |
| P1 | GPU 別の最適化を統合し、`adapters/ggml/` を CL11 の正本にする。性能の基準表を固定する | 全 GPU で test-backend-ops が全 PASS、生成が CPU と一致 |
| P2 | runtime（device / buffer / stream / allocator、デバイスデータベース）と backends/opencl11 を切り出す。ggml adapter は runtime を呼ぶ形にする | P1 と同じ試験 + 性能が基準表の ±3% 以内 |
| P3 | tensor、ops、quant を切り出し、C API と CPU 参照の単体試験を作る。ggml adapter は翻訳層だけにする | 単体試験 + P2 と同じ回帰試験 |
| P4 | multi-GPU transport（パイプライン分割）。異種 GPU にまたがって全層を GPU で動かす | 8B 級モデルを TB250 の複数 GPU で動かし、CPU との一致と速度を測る |
| P5 | backends の追加（opencl12、vulkan、cuda）と adapters の追加（PyTorch extension、ONNX Runtime） | adapter ごとに試験を定める |

切り出しは、性能に影響しないことを毎回確かめながら進めます。GT 430 の mat-vec はメモリ帯域の約 8 割まで出ているので、抽象化で 1 回の起動や添字計算が増えると、そのまま遅くなります。
