# 複数 GPU（P4）

TB250 の 5 枚の GPU を、1 つのモデルや 1 つの計算に組み合わせて使った結果です。測定はすべて 2026-10-10 に TB250 で行いました。

## 1 プロセスで使える GPU の組み合わせ

| 組み合わせ | 1 プロセスで |
| --- | --- |
| Kepler 同士（GT 730 + GT 710） | 使える |
| NVIDIA と AMD | 使える |
| Fermi（GT 430）と Kepler | 使えない。後から開いた方の `clCreateContext` が `CL_INVALID_DEVICE`（NVIDIA 390、順番によらない） |
| AMD の rusticl と Clover | 同じ Radeon が両方に現れる。RX 6400 を Clover で開くと GPU がリセットされる |

- GT 730・GT 710・RX 6400・WX 2100 の 4 枚（VRAM 合計 約 9GB）を 1 プロセスで使える。`GGML_CL11_PLATFORMS=NVIDIA,rusticl RUSTICL_ENABLE=radeonsi` でプラットフォームを絞り、`GGML_CL11_DEVICES` で GT 430 を外す。
- GT 430 を加えるには、別のプロセスにする必要がある（ggml の RPC など。未実施）。

## llama.cpp：Qwen3-8B Q4_K_M（4.68 GiB）

`llama-bench -ngl 99 -ub 128 -b 128 -p 128 -n 32 -r 2 -t 2`、層ごとの分割（`-sm layer`）。モデルの SHA256 は `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`（Hugging Face の公開値と一致）。

| 構成 | 分割（`-ts`） | pp128（t/s） | tg32（t/s） |
| --- | --- | --- | --- |
| CPU だけ（Celeron G3930、2 スレッド） | | 0.40 | 0.36 |
| CL11：GT 730 + GT 710 + RX 6400 + WX 2100 | 1 / 2 / 4 / 2 | 23.09 | 4.63 |
| CL11（NVIDIA 2 枚）+ 公式 Vulkan（AMD 2 枚）、1 プロセス | 1 / 2 / 4 / 2 | 24.79 | 5.02 |
| 公式 Vulkan：RX 6400 + WX 2100 | 2 / 1 | **94.77** | **11.06** |

- 8B は CPU だけより、4 枚の CL11 で生成が約 13 倍、AMD 2 枚の公式 Vulkan で約 31 倍速い。
- 層ごとの分割では、各 GPU が受け持つ層の時間が足し合わされる。帯域の小さい GPU（GT 710 は 11.7 GB/s）に層を置くと、その分だけ全体が遅くなる。AMD 2 枚（6GB）に収まるモデルなら、NVIDIA を加えないほうが速い。NVIDIA の 2 枚は、AMD だけでは収まらないモデルの容量を足すために使う。
- CL11 と公式 Vulkan は、llama.cpp の同じプロセスに共存できる（`GGML_BACKEND_PATH` で CL11 を、公式ビルドの `libggml-vulkan.so` と一緒に読み込む。NVIDIA の Vulkan ICD は使えないので `VK_ICD_FILENAMES` で AMD だけにする）。

### 正しさ（CPU の logits との KL ダイバージェンス）

`llama-perplexity -c 512 --chunks 2 -b 128 -ub 128 -t 2`、文章は Gate 0 と同じ。CPU の PPL は 11.20。

| 構成 | 平均 KLD | 99% KLD | top-1 一致 | PPL / CPU の PPL |
| --- | --- | --- | --- | --- |
| CL11、4 枚 | 0.0058 ± 0.0005 | 0.0376 | 96.7% | 0.991 |
| 公式 Vulkan、AMD 2 枚（校正の基準） | 0.0045 ± 0.0003 | 0.0281 | 97.5% | 0.992 |

- CL11 の差は、公式 Vulkan の差の約 1.3 倍で、`docs/ARCHITECTURE.md` の基準（公式バックエンドの差を超えないこと）をわずかに超える。Qwen3-0.6B-Base Q4_K_M では逆に CL11 のほうが小さかった（`docs/BASELINES.md`）。
- 仮説：CPU と公式 Vulkan は、量子化した重みとの積で活性化を 8bit に量子化してから計算し、CL11 は FP32 のまま計算する。CPU を基準にすると、本来の積に近い CL11 のほうが「CPU との差」は大きく出ることがある。両方とも PPL は CPU より低い。
- 未決：CPU 以外の中立な基準（F16 / BF16 のモデルの logits）で比べる。8B の F16 は CPU で 16GB になり時間がかかるので、まず小さいモデルで仮説を確かめる。

## ScrapGpuLibrary の C API：GPU 間の転送

`scrapgpu_tensor_copy`（`docs/API_CONTRACT.md` 14 章）で、デバイスをまたいでテンソルをコピーする。違うデバイスの間はホストを経由する（OpenCL 1.1 にはコンテキストをまたぐイベントがない）。

| 試験（TB250、GT 730 + GT 710 + RX 6400 + WX 2100 を 1 プロセス） | 結果 |
| --- | --- |
| `tests/ops`：`tensor_copy` の同じデバイス・デバイス間（両方向） | バイト単位で一致、87 件すべて PASS |
| `examples/pipeline_mlp.c`：8 層の MLP（隠れ層 1024、Q4_0）を 4 枚に 2 層ずつ | NMSE 2.4e-10（1 枚と同じ） |

速度（64 トークン）：1 枚（GT 730）1.24 ms / トークン、4 枚 5.38 ms / トークン。AMD 2 枚では 1 枚 6.53 ms、2 枚 8.00 ms。1 枚に収まる小さな計算を分けると、転送ごとの同期（ホストへの読み出しを待つ）が足されて遅くなる。分割は、1 枚に収まらない大きさのときに容量を得るための手段で、速さのためではない。

## 次の課題

- GT 430 を別プロセスで加える（ggml の RPC、または ScrapGpuLibrary のプロセス間 transport）。
- 転送の待ちを減らす（ホスト経由のコピーを非同期にし、次の層の計算と重ねる）。
- 層の分割比を、VRAM ではなく各 GPU の層あたりの時間に合わせて決める。
