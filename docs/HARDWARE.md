# 検証機とドライバの癖

## 検証機（TB250）

- マザーボード: BIOSTAR TB250-BTC PRO（マイニング用。GPU はライザー経由。リンクは下の「GPU」節）
- CPU: Intel Celeron G3930（2 コア、AVX なし）、メモリ 32GB
- OS: Debian 13（kernel 6.12）
- NVIDIA ドライバ: 390.157（Fermi を扱える最後の系列）。OpenCL のみ使用。
- AMD: Mesa 25.0（radeonsi）。Vulkan は RADV、OpenCL は rusticl（OpenCL 3.0、`RUSTICL_ENABLE=radeonsi`）または Clover（OpenCL 1.1）。

## GPU

帯域は OpenCL の float4 読み出しカーネルによる実測値です。

| GPU | 世代 | OpenCL | VRAM | 単一確保の公称上限 | 実測帯域 | 演算（理論 FP32） |
| --- | --- | --- | --- | --- | --- | --- |
| GeForce GT 430 | Fermi（sm_21） | 1.1 | 964MiB | 241MiB | 22.9 GB/s | 96 コア × 1.4GHz ≈ 269 GFLOPS |
| GeForce GT 730 | Kepler（sm_35, GK208, 2 SMX） | 1.2 | 981MiB | 245MiB | 37.9 GB/s | 384 コア × 954MHz ≈ 733 GFLOPS |
| GeForce GT 710 | Kepler（sm_35, GK208, 1 SMX） | 1.2 | 1.955GiB | 500MiB | 11.7 GB/s | 192 コア × 954MHz ≈ 366 GFLOPS |
| Radeon Pro WX 2100 | Polaris 12（GCN, 8 CU） | 1.1（Clover）/ 3.0（rusticl） | 2GB | 512MiB（Mesa は超過を拒否） | 43.6 GB/s | 512 シェーダ（wave64） |
| Radeon RX 6400 | RDNA2（navi24, 12 CU） | 1.1（Clover、使用不可）/ 3.0（rusticl） | 4GB | | 126.6 GB/s（rusticl） | |

- PCIe リンクは sysfs の `current_link_speed` / `current_link_width` で確認しました。RX 6400 は上流のルートポートからカード内部のブリッジまでが 5.0 GT/s x4（Gen2 x4、理論約 2 GB/s。カードは Gen4 x4 まで対応）で、WX 2100 / GT 730 / GT 710 / GT 430 は 5.0 GT/s x1（Gen2 x1、理論約 0.5 GB/s）です。CPU 側のルートポートの最大が 5.0 GT/s で、Celeron G3930 / B250 が本来対応する Gen3 になっていません（BIOS が Gen2 に固定している可能性。未確認で、変更は試していません）。RX 6400 の 03:00.0 に表示される 16 GT/s x16 はカード内部のブリッジとの間のリンクで、ホストとの帯域ではありません。
- ホストとデバイスの転送は、GT 730 / GT 710 で約 0.40 GB/s、WX 2100 で 0.36 GB/s（ホスト→GPU）/ 0.29 GB/s（GPU→ホスト）、RX 6400 で 1.4 GB/s（ホスト→GPU）/ 1.26 GB/s（GPU→ホスト）です（rusticl の `clEnqueueRead/WriteBuffer`、4 MiB 以上。詳細は [VULKAN_AMD.md](VULKAN_AMD.md)）。層ごとに CPU と GPU を往復する部分オフロードは遅くなるので、全層オフロードが前提です。
- 生成（tg）は帯域で、プロンプト処理（pp）は演算性能で決まります。そのため GT 710 は、演算性能では GT 430 を上回りますが、生成では GT 430 より遅くなります。

## ドライバの既知の癖

| 対象 | 症状 | 対処 |
| --- | --- | --- |
| NVIDIA 390 の Vulkan ICD | vulkaninfo や llama.cpp（Vulkan）がクラッシュする | Vulkan を使うプロセスには `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` を指定し、AMD だけを見せる |
| NVIDIA 390（GT 430 / GT 730） | `CL_DEVICE_MAX_MEM_ALLOC_SIZE` を超える `clCreateBuffer` も成功し、末尾まで正しく読み書きできる | CL11 は公称上限を超える確保も試し、末尾への書き込みで遅延確保の失敗を検出する（`GGML_CL11_STRICT_ALLOC=1` で無効化） |
| NVIDIA 390 + Kepler（GT 730） | `__constant` 引数に渡したサブバッファ（`clCreateSubBuffer`）の開始オフセットが無視され、親バッファの先頭が読まれる。Fermi（GT 430）では正しく動く | Kepler では、src1 を定数メモリで渡す経路を無効にする |
| NVIDIA 390（pyopencl で確認） | 1 プロセスで 2 枚目の GPU を開くと `INVALID_DEVICE` になる | 1 プロセス 1 GPU で使う |
| Fermi 全般 | 32bit 整数の除算・剰余が多数の命令に展開されて遅い（命令数は未実測。SASS で確認する）、整数→浮動小数の変換が低速 | 除算は事前計算した定数による乗算に置き換え、バイト→float は 2^23 の仮数に埋め込む方式で変換する（[CL11.md](CL11.md)） |
| Mesa rusticl + GCN（WX 2100） | `fma()` がソフトウェアでエミュレートされ、`mad()` の約 120 倍遅い（Clover では約 1.8 倍） | AMD では `fma` の代わりに `mad` を使う（CL11 の `CL11_MAD`）。行列積が 3.5 → 450 GFLOPS 超 |
| Mesa（Clover / rusticl） | `CL_DEVICE_MAX_MEM_ALLOC_SIZE` を超える `clCreateBuffer` を `CL_INVALID_BUFFER_SIZE` で拒否する（NVIDIA 390 と逆） | AMD では公称上限を厳守し、上限を超えるテンソルは CPU に回す |
| Mesa（WX 2100） | 空カーネルの連続起動に rusticl で約 100µs、Clover で約 15〜20µs かかる。CL11 の生成中はメモリクロックが 300 / 625 / 1500 MHz を行き来する（Vulkan では 1500 MHz に張り付く） | 生成はディスパッチ回数で頭打ちになる。速い既存バックエンド（Vulkan）を使う |
| Mesa Clover（WX 2100） | 1 回の起動コストが、そのカーネルを含む**プログラムの大きさに比例して**増える。空カーネルの `clEnqueueNDRangeKernel` は、同じプログラムに空のカーネルを +0 / +50 / +200 個足すと 1 回 5 / 24 / 104µs（rusticl は大きさによらず一定）。CL11 の全カーネルを 1 つのプログラムに入れると、1 トークン約 750 起動がこのコストを払う | Clover ではプログラムを decode 用 / 行列積用 / K-quant 用の 3 つに分ける（`split_programs`。`GGML_CL11_SPLIT_PROGRAMS=0/1` で切り替え）。Qwen3-0.6B Q8_0 の tg64 が 16.2 → 22.7 t/s（pp128 は 320 → 345 t/s）。rusticl と NVIDIA は分けない |
| Mesa Clover（25.0.7、kernel 6.12、RX 6400） | 2 回目のカーネル起動で GPU がハングする（`amdgpu: The CS has cancelled because the context is lost`）。自明なカーネルでも再現 | Clover は使わない。AMD の OpenCL は rusticl（`RUSTICL_ENABLE=radeonsi`）を使う |
| Mesa rusticl（RX 6400） | 空カーネルでも in-order キューで 1 起動あたり約 13.8µs（ホスト側の enqueue は約 4.6µs）。1 トークン約 900 起動なので、それだけで約 12ms の下限になる | 起動数を減らす（融合）。速い既存バックエンド（Vulkan）があればそちらを使う |
| Fermi 全般 | グリッドの各次元が 65535 まで | カーネルは 2 次元の NDRange で起動し、グループ番号を自分で線形化する |

## GPU ごとの推奨バックエンド（実測に基づく）

Qwen3-0.6B Q8_0、`-ngl 99 -ub 128 -b 128 -p 128 -n 64`。CPU はこのマシンの Celeron G3930（2 スレッド）で pp128 約 26 / tg64 約 15.7 t/s。

| GPU | 推奨 | 根拠 |
| --- | --- | --- |
| GeForce GT 430 | CL11 | 公式のバックエンドは動かない。CL11 は pp128 89.4 / tg64 22.3 t/s（CPU の tg は 15.6）。Q4_K_M は 89.9 / 23.7、Q4_0 のモデル（Qwen2-0.5B-ft）は 118.5 / 36.5（CPU は 47.5 / 25.8） |
| GeForce GT 730 | CL11 | 公式の CUDA / Vulkan は使えない。CL11 は pp128 241.9 / tg64 34.9 t/s（CPU の tg は 15.9）。Q4_K_M は 231.6 / 39.2 |
| GeForce GT 710 | CL11（Q4_K 系のモデルで） | CL11 は Q8_0 で pp128 120.0 / tg64 14.7 t/s。帯域が 11.7 GB/s なので、Q8_0 の生成は CPU（15.7）より遅い。Q4_K_M は 115.6 / 17.8（CPU の tg は 4.6）で、Qwen3-1.7B Q4_K_M も 29 層すべてをオフロードできる（7.5 t/s、CPU は 1.6） |
| Radeon Pro WX 2100 | 公式 Vulkan（既定の設定） | Qwen3-0.6B Q8_0 で Vulkan は pp128 559 / tg64 46.2 t/s、0.46 J/token。CL11 は Clover で 318 / 20.1 t/s、0.72 J/token（rusticl は pp 446 / tg 14.0）。Vulkan の設定を変えても差は 1% 未満 |
| Radeon RX 6400 | 公式 Vulkan（既定の設定） | Qwen3-0.6B Q8_0 で Vulkan は pp128 2455 / tg64 139.5 t/s、0.24 J/token。CL11（rusticl）は 854 / 20.3 t/s、0.45 J/token。Vulkan の設定を変えても既定を上回らなかった。CL11 は RDNA 向けの調整をしておらず、AMD 全般の設定（`mad`、公称上限の厳守、K-quant の mat-vec の形など）だけを有効にしている |

Vulkan の上限比・内訳・手段ごとの実測は [VULKAN_AMD.md](VULKAN_AMD.md) にあります。要点:

- tg は帯域の上限の 5〜7 割（RX 6400 で 55〜77%、WX 2100 で 49〜73%）。小さいモデルほど低く、重みの量に比例しない固定費（RX 6400 で 1 トークンあたり約 2〜2.6 ms、WX 2100 で 5〜8 ms）が効く。
- 実用の tg を上げる手段: サンプリングを GPU で行う `-bs`（`--backend-sampling`、b11384 にある）で RX 6400 は +4〜12%、WX 2100 は +5〜10%。形式は Q4_K_M で Q8_0 比 +28〜30%（Qwen3-0.6B-Base。品質重視なら Q6_K / Q5_K_M で +17〜20%）。Q3 以下は Q4 より遅く、品質も大きく落ちる。
- 合計スループットはバッチ処理（B=4 まで線形、RX 6400 は B=16、WX 2100 は B=8 が谷、`-c` は必要な分に絞る）。
- 1 枚に載らないモデルは `-dev Vulkan0/Vulkan1 -sm layer -ts N/1`（8B Q4_K_M の微調整モデルで 14.1 t/s）。`-sm row` は Vulkan で使えず、`-sm tensor` は PCIe のレイテンシで遅い。WX 2100 をドラフトにする投機的デコードは遅くなる。
- `GGML_VK_*` の設定は既定を上回らなかった（1.7B Q4_K_M の `GGML_VK_DISABLE_MMVQ=1` が +2.5% のみ）。

J/token は hwmon の `power1_average`（GPU 単体）による。RX 6400 のアイドル電力は約 1.04 W。WX 2100 は `power1_input` で測り、アイドル電力は約 3.22 W。Clover は WX 2100（Polaris）では動き、RX 6400（RDNA2）ではハングする。

## CL11 が AMD の世代を判定する方法

調整値は GCN と RDNA で分けています（`cl11_amd_generation()`、`adapters/ggml/ggml-cl11.cpp`）。Mesa は `cl_amd_device_attribute_query` を実装していないため、次の順に調べます。

1. `cl_amd_device_attribute_query` があれば `CL_DEVICE_GFXIP_MAJOR_AMD`（AMD 自身のドライバ、ROCm）。6〜9 が GCN、10 以上が RDNA。
2. デバイス名の `gfxNNN`（ROCm の名前。gfx803 は GCN、gfx1034 は RDNA）。
3. Mesa がデバイス名の括弧に入れるコード名。rusticl は `AMD Radeon Pro WX 2100 (radeonsi, polaris12, LLVM ..., DRM ...)`、Clover は `(polaris12, LLVM ...)`、RX 6400 は `(radeonsi, navi24, ...)`。`polarisNN`、`vegaNN`、`tonga`、`hawaii` などは GCN、`naviNN` と一部の APU は RDNA。
4. 商品名（`Pro WX` と 3 桁の `RX 5xx` は GCN、4 桁の `RX 6xxx` は RDNA）。

世代が分からない AMD の GPU は RDNA と同じ扱い（AMD 全般の設定だけ）にします。

| 世代 | 例 | 適用する設定 |
| --- | --- | --- |
| AMD 全般 | WX 2100、RX 6400 | `mad`（`fma` の代わり）、公称上限の厳守、V キャッシュの多グループ版カーネル（`mvt_groups`）、K-quant の mat-vec は 128 work-items × 1 行（WX 2100 と RX 6400 の両方で測定） |
| GCN | WX 2100（Polaris 12） | 上記に加えて、Q4_0 / Q8_0 の mat-vec は 256 work-items × 1 行、mat-mat は 64 x 64 タイル（WX 2100 で測定） |
| RDNA / 不明 | RX 6400（navi24） | AMD 全般の設定だけ。ほかは既定値（Vulkan を推奨するので、これ以上は調整しない） |
| Mesa Clover（任意の GPU） | WX 2100 | 上記に加えて `split_programs`（プログラムを 3 つに分ける。上の表の「Mesa Clover（WX 2100）」の行） |
