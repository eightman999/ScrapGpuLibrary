# 検証機とドライバの癖

## 検証機（TB250）

- マザーボード: BIOSTAR TB250-BTC PRO（マイニング用。GPU は PCIe x1 ライザー経由）
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
| Radeon Pro WX 2100 | Polaris 12（GCN, 8 CU） | 1.1（Clover）/ 3.0（rusticl） | 2GB | 512MiB（Mesa は超過を拒否） | 43.6 GB/s | |
| Radeon RX 6400 | RDNA2（navi24, 12 CU） | 1.1（Clover、使用不可）/ 3.0（rusticl） | 4GB | | 126.6 GB/s（rusticl） | |

- いずれも PCIe x1 です。ホストからデバイスへの転送は約 0.40 GB/s（GT 730 / GT 710 で実測）なので、層ごとに CPU と GPU を往復する部分オフロードは遅くなります。全層オフロードが前提です。
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
| Mesa Clover（25.0.7、kernel 6.12、RX 6400） | 2 回目のカーネル起動で GPU がハングする（`amdgpu: The CS has cancelled because the context is lost`）。自明なカーネルでも再現 | Clover は使わない。AMD の OpenCL は rusticl（`RUSTICL_ENABLE=radeonsi`）を使う |
| Mesa rusticl（RX 6400） | 空カーネルでも in-order キューで 1 起動あたり約 13.8µs（ホスト側の enqueue は約 4.6µs）。1 トークン約 900 起動なので、それだけで約 12ms の下限になる | 起動数を減らす（融合）。速い既存バックエンド（Vulkan）があればそちらを使う |
| Fermi 全般 | グリッドの各次元が 65535 まで | カーネルは 2 次元の NDRange で起動し、グループ番号を自分で線形化する |

## GPU ごとの推奨バックエンド（実測に基づく）

| GPU | 推奨 | 根拠 |
| --- | --- | --- |
| GeForce GT 430 | CL11 | 公式のバックエンドは動かない。Qwen3-0.6B Q8_0 で tg 21.3 t/s（CPU 15.9） |
| Radeon Pro WX 2100 | 公式 Vulkan（既定の設定） | Qwen3-0.6B Q8_0 で Vulkan は pp128 559 / tg64 46.2 t/s、0.46 J/token。CL11 は Clover で 318 / 20.1 t/s、0.72 J/token（rusticl は pp 446 / tg 14.0）。Vulkan の設定を変えても差は 1% 未満 |
| Radeon RX 6400 | 公式 Vulkan（既定の設定） | Qwen3-0.6B Q8_0 で Vulkan は pp128 2455 / tg64 139.5 t/s、0.24 J/token。CL11（rusticl）は 854 / 20.3 t/s、0.45 J/token。Vulkan の設定を変えても既定を上回らなかった |

J/token は hwmon の `power1_average`（GPU 単体）による。RX 6400 のアイドル電力は約 1.04 W。WX 2100 は `power1_input` で測り、アイドル電力は約 3.22 W。Clover は WX 2100（Polaris）では動き、RX 6400（RDNA2）ではハングする。
