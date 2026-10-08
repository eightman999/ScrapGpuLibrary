# AMD GPU の Vulkan（ggml-vulkan）をどこまで詰められるか

状態: **途中（下書き）**。TB250 の計測を、ユーザーの指示で中断した時点の結果をまとめています。未計測の項目は末尾の「未完了」にあります。表の数値は、特に断らない限り TB250 の実測です（llama.cpp b11384 の公式 Vulkan ビルド、`-ngl 99`、Mesa 25.0 の RADV、`GGML_VK_VISIBLE_DEVICES` で 1 枚だけ見せた単独計測）。計測のスクリプトとログは TB250 の `~/bench/scrapgpu/_w/vk/` にあります（`notes.md` に経過）。

## 要約

| | RX 6400（RDNA2） | WX 2100（Polaris 12 / GCN） |
| --- | --- | --- |
| 実効帯域（OpenCL の読み出しカーネル） | 126.6 GB/s | 43.6 GB/s |
| Qwen3-0.6B Q8_0 の tg の上限（帯域 ÷ 1 トークンの重み 633 MB） | 200 t/s | 69 t/s |
| 同・現状（`llama-bench -p 0 -n 256`） | 137.2 t/s（上限比 69%） | 44.5 t/s（65%） |
| 差（1 トークンあたり）の内訳 | 下の「1 トークンの内訳」 | 未計測（logits の読み戻しとサンプリングだけ計測） |

- 上限との差は、モデルが小さいほど大きいです（RX 6400 で 0.6B Q4_K_M は上限比 52%、8B Q2_K は 76%）。1 トークンあたり 2〜3 ms の、重みの量に比例しない固定費があるためです。
- 固定費の中身は、RX 6400 の Qwen3-0.6B Q8_0 で、mat-vec の帯域効率（111 GB/s、実効帯域の 88%）による 0.7 ms、小さい op（norm、rope、attention、KV 書き込み。約 200 回の dispatch）の 1.1 ms、ホスト側（logits の読み戻し、submit、同期）の約 0.4 ms でした。
- **サンプリングを GPU 側で行う `-bs`（`--backend-sampling`）は b11384 にあり、Vulkan で動きます。** CPU 側のサンプラー（top-k などで約 0.45〜0.9 ms/token）を消せるので、実用の tg が RX 6400 で +7%、WX 2100 で +10% 上がりました（`llama-bench` の tg は CPU のサンプリングを含まないため、この差は出ません）。
- 1 本の tg を大きく上げる手段は、見つかった範囲では (1) `-bs`、(2) 1 重みあたりのバイト数が少ない形式、(3) 同じ GPU 内のドラフトによる投機的デコード（既存の実測で 1.31 倍）、(4) 小さい op の融合（upstream の作業）です。合計スループットは、バッチ処理（同時シーケンス数）が最も効きます。

## 1. 上限と効率

1 トークンで GPU が読む重みは、GGUF のテンソルサイズの合計から計算しました（`gguf_bytes.py`）。`token_embd` は表引きで CPU に置かれるので、`output.weight` が別にあるモデルでは除き、tied embedding のモデルでは出力に GPU で使うので含めます。KV キャッシュと norm の重みは 1% 未満なので無視しました（`-n 64〜256` の平均コンテキストは短い）。

上限 = 実効帯域 ÷ 1 トークンの重み。現状は `llama-bench -p 0 -n 256 -r 3` の tg256（`-p 128 -n 64` の tg64 も別に計測し、差は 3% 以内）。

| モデル | 形式 | GPU で読む量 | 備考 | RX 6400 上限 | RX 6400 現状（上限比） | WX 2100 上限 | WX 2100 現状（上限比） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-q8_0 | Q8_0 | 633 MB | tied | 200 t/s | 137.2（69%）、tg64 140.1 | 69 | 44.5（65%） |
| qwen3-0.6b-base-q4_k_m | Q4_K_M | 391 MB | tied、出力は Q6_K | 324 | 169.2（52%）、tg64 181.5 | 112 | 56.3（50%） |
| Qwen2-0.5B-ft Q4_0 | Q4_0 + Q8_0 | 346 MB | tied | 366 | 200.4（55%）、tg64 200.5 | 126 | 未計測 |
| tinyllama-1.1b-chat.Q4_0 | Q4_0 | 599 MB | 出力 Q6_K | 211 | 149.4（71%）、tg64 151.1 | 73 | 未計測 |
| Qwen3-1.7B-Q4_K_M | Q4_K_M | 1101 MB | 出力 Q6_K | 115 | 87.4（76%）、tg64 88.3 | 40 | 未計測 |
| qwen3-8b-base-q2_k | Q2_K | 3072 MB | 出力 Q6_K | 41 | 31.3（76%） | 載らない | 載らない |
| qwen3-8b-base-q3_k_s | Q3_K_S | 3496 MB | | 36 | 23.8（66%、tg64） | 載らない | 載らない |
| 8B-ft Q4_K_M | Q4_K_M | 4672 MB | 8B Q4_K_M の微調整モデル。RX 6400 の 4 GB に載らない | | | | |

- 小さいモデルほど上限比が低いのは、固定費（約 2〜3 ms/token）が支配的になるためです。1 トークンの時間 − 重みの量 ÷ 実効帯域（固定費の概算）は、RX 6400 で 0.6B Q8_0 が 2.3 ms、0.6B Q4_K_M が 2.8 ms、Qwen2-0.5B-ft が 2.3 ms、tinyllama が 2.0 ms、1.7B が 2.7 ms、8B Q2_K が 7.6 ms（層数が多く、mat-vec の本数も多い）です。
- 形式別の傾向: 同じ 0.6B で Q8_0 → Q4_K_M にすると、重みは 0.62 倍になるのに、tg は 1.23 倍にしかなりません。Q4_K_M は出力が Q6_K で、K-quant の mat-vec が Q8_0 より帯域効率が低い可能性があります（Q4_K_M の op 別内訳は未計測）。
- 電力（`power1_average`。RX 6400 は 1 W 刻みで、約 5 秒の窓）: RX 6400 の tg 中は 0.6B Q8_0 で 32 W（0.234 J/token）、0.6B Q4_K_M で 30 W（0.175）、Qwen2-0.5B-ft Q4_0 で 31 W（0.156）、tinyllama Q4_0 で 34 W（0.225）、1.7B Q4_K_M で 35 W（0.40）、8B Q2_K で 42 W（1.34）。WX 2100（`power1_input`）は 0.6B Q8_0 で 23.9 W（0.54 J/token。HARDWARE.md の 0.46 とは窓の取り方が違う）、0.6B Q4_K_M で 24.1 W（0.43）。1 トークンあたりのエネルギーは、小さく速いモデルほど低く、RX 6400 は WX 2100 の約半分です。

## 2. 1 トークンの内訳（RX 6400、Qwen3-0.6B Q8_0）

`GGML_VK_PERF_LOGGER=1`（b11384 の `ggml-vulkan.cpp` が読む環境変数。ほかに `GGML_VK_PERF_LOGGER_FREQUENCY`、`GGML_VK_PERF_LOGGER_CONCURRENT`、`GGML_VK_SYNC_LOGGER`）で、デコード 127 グラフを集計しました（`perfparse.py`）。ロガーは op ごとに GPU のタイムスタンプを取りますが、ロガーを有効にすると壁時計が約 2 倍（14.3 ms/token）になるので、**GPU 時間の内訳としてだけ**使います。壁時計の 7.25 ms は、ロガー無しの `llama-completion` の eval time です。

| 区分 | 時間（µs/token） | 比率 | 備考 |
| --- | --- | --- | --- |
| mat-vec（Q8_0） | 5720 | 79% | 重み 633 MB ÷ 5.72 ms = 111 GB/s（実効帯域の 88%）。出力（m=151936）は 1343 µs で 123 GB/s |
| norm（RMS_NORM_MUL、QK-norm + rope + set_rows の融合） | 533 | 7% | 3.4〜6.5 µs の dispatch が 113 回 |
| attention（FLASH_ATTN_EXT、コンテキスト 256） | 394 | 5% | 11.4 µs × 28 |
| その他（SET_ROWS、GLU、ADD、GET_ROWS） | 172 | 3% | |
| GPU の op 合計 | 6819 | 94% | dispatch は 1 トークン約 397 回（mat-vec 197、その他 200） |
| ホスト側（残り） | 約 430 | 6% | 壁時計 7.25 ms − GPU 6.82 ms。logits の読み戻し、submit、フェンス待ち |

- 上限（126.6 GB/s で重みだけ）5.00 ms に対して、実測 7.25 ms の差 2.25 ms は、mat-vec の帯域効率 0.72 ms、小さい op 1.1 ms、ホスト側 0.43 ms です。
- mat-vec の個別の効率は、ffn の gate / up（m=3072）が 112 GB/s、qkv 相当（m=2048）が 107 GB/s、o 相当（m=1024、k=1024）が 101 GB/s、down（m=1024、k=3072、ADD 融合）が 104 GB/s でした。小さい行列ほど立ち上がりの影響が見えます。

### logits の読み戻しとサンプリング

n_vocab = 151936 なので、logits は 607,744 バイトです。

| | RX 6400 | WX 2100 |
| --- | --- | --- |
| リンク | Gen2 x4（理論 約 2 GB/s） | Gen2 x1（理論 約 0.5 GB/s） |
| OpenCL（rusticl）での転送、4 MiB 以上 | ホスト→GPU 1.42 GB/s、GPU→ホスト 1.0〜1.26 GB/s | 0.358 GB/s、0.28〜0.30 GB/s |
| OpenCL での logits 相当（607,744 B）の GPU→ホスト | 0.77〜1.13 ms | 2.7〜2.9 ms |
| 4 バイトの GPU→ホスト（レイテンシ） | 約 0.1〜0.16 ms | 約 0.11〜0.18 ms |

- Vulkan の実測では、読み戻しの影響は OpenCL の転送時間の見積もりより小さく見えます（下表）。ggml-vulkan は logits を固定（pinned）ホストメモリへ非同期にコピーするためと思われます（`ggml_backend_vk_get_tensor_async`。推測）。

## 3. 手段の評価

### a. GPU 側のサンプリング（`-bs`）

b11384 にあります。`--help` に `-bs, --backend-sampling`（experimental、既定は無効。環境変数 `LLAMA_ARG_BACKEND_SAMPLING`）があり、`llama-completion` と `llama-batched-bench` などの共通引数です。実装は `src/llama-sampler.cpp` の `backend_apply` で、greedy は `ggml_argmax`、top-k は `ggml_top_k`、top-p、min-p、temp、dist（softmax + cumsum + 乱数との比較）、penalties が GPU のグラフに入ります。ggml-vulkan は `ARGMAX`、`TOP_K`、`ARGSORT`、`SOFT_MAX` を持っています。制約は、grammar、reasoning budget と併用できないこと、`-sm tensor` では CPU に戻ること（`llama_context::set_sampler` の警告）、dry / xtc / typical / mirostat など `backend_apply` が無いサンプラーを含む構成では使えないこと（`llama_sampler_backend_support` で判定）です。

計測は `llama-completion -n 192 -c 512 --temp 0 -s 1 --ignore-eos`（ms/token。eval = デコード、sampling = CPU のサンプラー）。

| GPU / モデル | 設定 | eval | sampling | 合計 | t/s |
| --- | --- | --- | --- | --- | --- |
| RX 6400 / Qwen3-0.6B Q8_0 | 既定のサンプラー列（top-k 40 など） | 7.247 | 0.495 | 7.741 | 129.2 |
| | 同 + `-bs` | 7.200 | 0.012 | 7.212 | 138.7（+7.3%） |
| | top-k 1 のみ | 7.246 | 0.452 | 7.699 | 129.9 |
| | 同 + `-bs` | 7.130 | 0.012 | 7.142 | 140.0（+7.8%） |
| WX 2100 / Qwen3-0.6B Q8_0 | 既定のサンプラー列 | 22.003 | 0.871 | 22.874 | 43.7 |
| | 同 + `-bs` | 20.778 | 0.014 | 20.792 | 48.1（+10.0%） |

- RX 6400 は eval の差が 0.05〜0.12 ms しかなく、効果の大半は CPU のサンプラー（top-k を 152k 語彙に適用する約 0.45 ms）を消すことによるものです。logits の読み戻し自体（最大でも 0.1 ms 程度）は小さいです。eval の差は、GPU 側のサンプリング op の追加分を差し引いた値なので、読み戻しの下限です。
- WX 2100 は eval が 1.2 ms 下がり、CPU のサンプラーが 0.87 ms（Celeron G3930 の 2 スレッド）なので、合計で +10% です。
- `llama-bench` の tg は CPU のサンプリングを含まない（logits は読み戻す）ので、`-bs` の効果はベンチの数値には現れません。実際のアプリ（llama-server など）の tg は、ベンチより 6〜10% 低くなっているはずです。
- 1.7B Q4_K_M と 8B Q2_K（RX 6400）、1.7B（WX 2100）は未計測（キューにありました）。モデルが大きいと 1 トークンの時間が長いので、割合は小さくなります。
- ScrapGpuLibrary 側で独自に実現するなら、上の効果（RX 6400 で最大 0.5 ms/token、WX 2100 で最大 2 ms/token）が上限です。llama.cpp 側にあるので、ggml adapter では実装せず、`-bs` を使うことで足ります。

### b. 量子化形式

TB250 に F16 の元モデルがあります。`/mnt/hdd/specbench-models/qwen3-0.6b-base-f16.gguf`（1.2 GB）と `qwen3-8b-base-f16.gguf`（16 GB）で、Qwen3-1.7B の F16 / BF16 は見つかりませんでした。`llama-quantize` で Qwen3-0.6B-Base の F16 から、Q4_0、Q4_1、Q4_K_S、Q4_K_M、IQ4_XS、IQ4_NL、Q3_K_M、Q3_K_S、IQ3_S を `~/bench/scrapgpu/_w/models/q06b-*.gguf` に作成済みです（IQ3_XXS、Q2_K、Q5_K_M、Q6_K、Q8_0 は作成中に中断）。

- **各形式の tg、J/token、KL ダイバージェンス（CPU の F16 を基準）は未計測です**（`qbench.sh`、`kld.sh` を投入済みで、実行前に中断）。
- 先に分かっている範囲では、帯域律速の傾向は実測に出ています（Q8_0 → Q4_K_M で重み 0.62 倍、tg 1.23 倍。1 トークンの時間 7.29 → 5.91 ms で、重みの減少分 1.9 ms のうち 1.4 ms が時間に現れた）。形式ごとの mat-vec の帯域効率が違うため、バイト数だけでは決まりません。`GGML_VK_PERF_LOGGER` で形式ごとの mat-vec の GB/s を測るのが次の作業です。
- 0.6B のような小さいモデルは量子化に弱いので、KLD の結果が出るまで、Q4 より小さい形式は推奨できません。

### c. バッチ処理のスループット（暫定）

`llama-batched-bench -c 4096 -npp 16 -ntg 64 -npl 1,2,4,8,16,32`。**1 回目は初回のパイプラインコンパイルの影響で、B=8 と 16 に落ち込みが出ているため暫定値**です（同じ npl を 2 回繰り返す形に直して再計測を投入しましたが、中断しました）。RX 6400 の S_TG（合計 t/s）:

| モデル | B=1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q8_0 | 133 | 229 | 395 | 498（暫定） | 480（暫定） | 757 |
| Qwen3-0.6B Q4_K_M | 172 | 289 | 383 | 504（暫定） | 556（暫定） | 844 |
| Qwen3-1.7B Q4_K_M | 85 | 157 | 227 | 276（暫定） | 292（暫定） | 486 |

- B=4 までは 1 回の重みの読み込みで 4 トークン進むので、ほぼ線形（B=4 で 2.96 倍）です。B=32 で 5.7 倍（0.6B Q8_0）で、まだ演算律速には入っていないように見えます。確定値は再計測が必要です。
- WX 2100 のバッチ計測、8B（RX 6400）のバッチ計測、電力（バッチ時の J/token）は未計測です。

### d. 投機的デコード（既存の実測、`~/tb250-distill/runs/specbench/`）

ターゲット = Qwen3-8B-Base Q2_K（RX 6400 単独の tg 29.75 t/s）。

| 構成 | tg | 倍率 | 受理率 | J/token（合計） |
| --- | --- | --- | --- | --- |
| ドラフト 0.6B を **同じ RX 6400** に載せる（`d06b4rx_n4`、n_max 4） | 38.85 | **1.31 倍** | 0.76 | 1.18（基準 1.59） |
| ドラフトなしの n-gram（`ngram_mod`） | 34.8 | 1.17 倍 | 0.71 | 1.22 |
| ドラフト 0.6B を WX 2100 に載せる（n_max 2〜8） | 12.4〜19.4 | 0.42〜0.65 倍 | 0.64〜0.83 | 1.9〜3.0 |
| ドラフト 1.7B を WX 2100 に載せる | 7.3〜12.5 | 0.25〜0.42 倍 | 0.61〜0.81 | 2.8〜4.6 |
| ドラフト 1.7B を同じ RX 6400 に載せる | 1.24 | VRAM 超過 | | |
| ドラフト 0.6B を CPU | 4.16 | 0.14 倍 | | |

- **同じ GPU 内のドラフトだけが、単独の tg（帯域上限 41 t/s の 72%）を超え、上限の 95% まで届きました**。ただし受理長 2.9 トークン（n_max 4）に対して 1.31 倍にとどまり、ドラフトの実行とターゲットの検証（バッチ 5）のコストが効いています。Q2_K の 8B は検証も mat-vec ではなく K-quant の mat-mat 経路です（要確認）。
- ドラフトを別の GPU（WX 2100）に置くと、ドラフトの tg（約 46 t/s）と PCIe x1 の同期が足を引っ張り、単独より遅くなりました。
- 品質の面では `match_ref`（基準との出力一致）が 0.75（12/16）で、投機的デコードは本来、出力を変えないはずなので、数値差による分岐と見られます。

### e. 2 枚の協調

| 構成 | pp128 | tg64 | 備考 |
| --- | --- | --- | --- |
| 8B-ft Q4_K_M、RX 6400 のみ（`-dev Vulkan0`、4 GB を超えた分はシステムメモリ） | 55.2 | 3.2 | |
| 同、RX 6400 + WX 2100、`-sm layer`（既定の分割） | 95.9 | **10.75** | 1 枚に載らないモデルが動く |
| 同、`-sm row` | | | `error loading model: device Vulkan0 does not support split buffers`（ggml-vulkan にはスプリットバッファ型が無い） |
| 同、`-sm tensor` | 10.4 | **0.303** | 動くが実用外 |
| qwen3-8b-base-q3_k_s、RX 6400 のみ | 248.6 | 23.8 | |
| specbench の記録: qwen3-8b-base-q2_k の `-sm layer` | | 12.64（RX 6400 のみで 29.7） | 1 枚に載るなら分割は遅くなる |
| specbench の記録: qwen3-8b-base-q3_k_s の `-sm layer` | | 10.06（同 22.6） | |

- `-sm layer` は、1 枚に載らないモデルを動かす手段として有効です（3.2 → 10.75 t/s）。1 本のストリームでは層が直列に実行されるため、速くはなりません。分割比は未確認ですが、仮に RX 6400 が約 3.1 GB を 110 GB/s、WX 2100 が約 1.6 GB を 38 GB/s で読むとすると 28 + 41 = 69 ms（約 14.5 t/s）で、実測 93 ms との差は GPU 間の転送（ホスト経由）と同期です。
- 分割比の調整（`-ts 5,2` など）は未計測です。計算上の最適は帯域比（RX 6400 約 74%、WX 2100 約 26%）に近く、RX 6400 を重くするほど速くなるはずです（WX 2100 の 2 GB に入る範囲で）。
- `-sm tensor` が遅い理由: ggml-vulkan は GPU 間の直接コピーを持たず（`ggml_backend_vk_cpy_tensor_async` は同じデバイス内のみ）、`ggml-backend-meta.cpp` の allreduce はフォールバック（ホスト経由）になります。8B（36 層）では 1 トークンあたり層ごとに複数回の allreduce が要り、各回が 0.1〜0.2 ms 以上のレイテンシと、WX 2100 側の 0.3 GB/s の転送を払うため、0.3 t/s になりました。通信量の見積もりでは、1 回の allreduce で n_embd（4096）× 4 B = 16 KB を往復するだけで、帯域ではなくレイテンシ（1 回約 0.3〜0.5 ms × 72 回以上 ≒ 20〜40 ms）が支配的なはずですが、実測 3.3 秒/トークンはそれより 2 桁大きく、フォールバックの同期が大きく効いている可能性があります（原因は未調査）。
- 結論として、PCIe Gen2 x1 の WX 2100 を含む構成でテンソル分割は成り立ちません。RX 6400 同士を 2 枚載せる環境でも（Gen2 x4）、レイテンシ律速なので勧めません。

### f. ggml-vulkan のシェーダの改善余地（ソースの読み取り。一部は未計測）

- 機種の判定は `get_device_architecture`（`AMD_GCN` / `RDNA1` / `RDNA2` …）で、RADV では WX 2100 が GCN、RX 6400 が RDNA2 になります。調整の分岐は少ないです。
  - 1 ワークグループの行数 `rm_stdq`（Q4_0 など）は GCN で 2、ほかの AMD で 1。K-quant の `rm_kq` は GCN で 4、ほかで 2。整数ドット版（MMVQ）の `rm_stdq_int` は GCN で 4。
  - AMD は mat-vec のワークグループが常に 1 サブグループ（RDNA2 は wave32、GCN は wave64）です。NVIDIA / Intel にある「M が小さいとき大きいワークグループを使う」分岐（`dmmv_wg`）は AMD に無く、`GGML_VK_*` で切り替える手段もありません。
  - MMVQ（活性を Q8_1 に量子化して整数ドット積）は、AMD では k ≥ 2048 のときだけ、Q8_0 は GCN のみ、Q6_K は使いません。RX 6400 は `int dot: 1` ですが、Qwen3-0.6B（k=1024）では使われません。WX 2100 は `int dot: 0` で使えません。`GGML_VK_FORCE_MMVQ` / `GGML_VK_DISABLE_MMVQ` で切り替えられます。
  - RDNA2 の固有のサブグループ指定は `soft_max` と `im2col` の 64 だけです。
- RX 6400 の mat-vec は、実効帯域の 88% で、残りの伸びしろは約 0.7 ms/token（10%）です。小さい op（norm、rope、set_rows、attention、GLU）の 1.1 ms/token（15%）のほうが大きく、これらは 3〜11 µs の dispatch が約 200 回で、実行時間より dispatch の間の待ちが支配的です。
  - 手段: RMS_NORM を直前の ADD や直後の mat-vec と融合する、QK-norm + rope + set_rows のさらなる融合（すでに一部ある）、`GGML_VK_DISABLE_FUSION` との比較で効果の大きさを確認する。
- `GGML_VK_*` の試験（`FORCE_MMVQ`、`DISABLE_MMVQ`、`DISABLE_FUSION`、`DISABLE_ASYNC`、`MAX_NODES_PER_SUBMIT`、`DISABLE_MULTI_ADD`、`DISABLE_GRAPH_OPTIMIZE`、`-fa 0/1`）は実行を投入しましたが、結果は未確認です。HARDWARE.md には「設定を変えても既定を上回らなかった」とあるので、期待は低いです。
- 実装の置き場: 融合の追加や AMD 向けの分岐は**upstream（llama.cpp）への貢献**が筋です。ScrapGpuLibrary の `backends/vulkan` に ggml-vulkan のシェーダを移植して手を入れると、upstream の追従を失い、得られるのは数 % です。ScrapGpuLibrary の backends/vulkan は、ggml を使わない経路（standalone C API、PyTorch / ONNX Runtime adapter）で Vulkan が必要になったときに、mat-vec と dequant のシェーダを移植する用途に限るのが妥当です。

## 4. 電力効率

- 帯域律速なので、J/token は「1 トークンの時間 × 電力」です。電力はモデルの大きさにあまり依存せず（RX 6400 で 30〜35 W、WX 2100 で 24 W）、**速く終わるほど J/token が下がります**。
- 実測で最も良い構成は、RX 6400 の Qwen2-0.5B-ft Q4_0 で 0.156 J/token（200 t/s）、次が Qwen3-0.6B Q4_K_M の 0.175 J/token です。同じ Qwen3-0.6B 系で比べると、Q8_0（0.234）より Q4_K_M（0.175）が 25% 低くなります。WX 2100 は 0.43〜0.54 J/token で、RX 6400 の約 2.5 倍です。
- バッチ処理では、1 トークンあたりの電力がほぼ一定のまま tg が増えるので、J/token はほぼバッチ数に反比例して下がるはずです（未計測）。特定の電力設定やクロックの変更は、root が要るので試していません。

## 5. 推奨構成（暫定）

| 用途 | GPU | 構成 |
| --- | --- | --- |
| 1 本の tg を最大にする | RX 6400 | 公式 Vulkan の既定値、`-ngl 99`、アプリ側のサンプリングは `-bs`（+7〜8%）。形式は Q4_K_M か Q4_0（KLD の結果が出るまで暫定）。同じ GPU に 0.6B のドラフトを載せる投機的デコード（1.31 倍、8B Q2_K で確認）も有効 |
| 同上 | WX 2100 | 同様に `-bs`（+10%）。ドラフトの併用は試していない（単独で帯域が小さいため効果は限定的と推測） |
| 合計スループット | 両方 | バッチ処理（`-np`）。RX 6400 は B=4 までほぼ線形、B=32 で 5〜6 倍（暫定） |
| 1 枚に載らないモデル | RX 6400 + WX 2100 | `-sm layer`。RX 6400 を重くする分割比（`-ts`）の調整は未計測。`-sm row` は使えず、`-sm tensor` は 0.3 t/s で使えない |
| 電力効率 | RX 6400 | 小さいモデルの Q4 系、`-bs`、バッチ処理 |

## 6. ScrapGpuLibrary で取り組む価値（優先順位）

1. **何もしない（`-bs` を使う）**: GPU 側のサンプリングは llama.cpp にあり、Vulkan で動く。ScrapGpuLibrary 側で作る理由が無い。効果は RX 6400 で約 7%、WX 2100 で約 10%。使い方を HARDWARE.md / README に書くだけでよい。
2. **upstream への貢献（小さい op の融合、AMD 向けの mat-vec の調整）**: 伸びしろは約 1〜1.8 ms/token（RX 6400 の 0.6B Q8_0 で 15〜25%）。ただし変更の検証（`test-backend-ops`）と、RX 6400 / WX 2100 以外での無劣化の確認が必要で、時間がかかる。先に op 別の内訳（Q4_K_M、1.7B、WX 2100）を揃える。
3. **バッチ処理の利用側の設計**（tb250-distill の Teacher など）: 1 回の重みの読み込みで複数候補を採点する用途。確定値を出してから設計する。
4. **ScrapGpuLibrary の `backends/vulkan`**: ggml を使わない経路の要求が出てから。現時点の優先度は低い。
5. **GPU 間の協調（独自の転送、パイプライン並列）**: 1 枚に載らないモデルが必要なときだけ。`-sm layer` で足りる。テンソル分割は PCIe Gen2 では成立しない。

## 未完了（TB250 での作業。再開は指示を待つ）

- 基準の計測（`base.sh`）: WX 2100 の Qwen2-0.5B-ft / tinyllama / Qwen3-1.7B と、全モデルの WX 2100 の `-p 128 -n 64` の結果の取り込み。
- GPU 側のサンプリングの A/B（`bd3all.sh`）: WX 2100 の top-k 1、Qwen3-1.7B（RX / WX）、8B Q2_K（RX）。
- op 別の内訳（`bd.sh`）: Qwen3-0.6B Q4_K_M、1.7B Q4_K_M、WX 2100（Q8_0）。形式別の mat-vec の GB/s と、出力 Q6_K の効率を見る。
- 量子化形式: `quant.sh` の残り（IQ3_XXS、Q2_K、Q5_K_M、Q6_K、Q8_0）、`qbench.sh`（各形式の tg と電力、F16 を含む、RX / WX）、`kld.sh`（F16 の CPU 基準、`-c 512 --chunks 8`、各形式を RX 6400 で比較）。
- バッチ処理の再計測（`batch.sh`、npl を 2 回繰り返す形）: RX / WX の 3 モデルと 8B Q2_K。
- 2 枚の協調: `split.sh` の残り（q3_k_s の layer / tensor、0.6B の tensor / layer / row）と `split2.sh`（`-ts 5/2`、`3/1`、`7/3`）。
- `GGML_VK_*` の試験（`envsweep2.sh` ほか）の結果の取り込み。
- KLD の許容値との比較、docs/HARDWARE.md の推奨表への最終的な追記。
