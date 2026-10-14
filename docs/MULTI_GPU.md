# 複数 GPU（P4）

TB250 の 5 枚の GPU を、1 つのモデルや 1 つの計算に組み合わせて使った結果です。測定はすべて 2026-10-10 に TB250 で行いました。

## 1 プロセスで使える GPU の組み合わせ

| 組み合わせ | 1 プロセスで |
| --- | --- |
| Kepler 同士（GT 730 + GT 710） | 使える |
| NVIDIA と AMD | 使える |
| Fermi（GT 430）と Kepler | 使えない。後から開いた方の `clCreateContext` が `CL_INVALID_DEVICE`（NVIDIA 390、順番によらない） |
| AMD の rusticl と Clover | 同じ Radeon が両方に現れる。RX 6400 を Clover で開くと GPU がリセットされる |

- 列挙の段階で Fermi と Kepler の衝突を検出する。同じプラットフォームの NVIDIA GPU に compute capability の major が複数あるときだけ、各デバイスのコンテキストを列挙順に同時に開いて確かめ、先に開けたものを残して、開けなかった方を警告付きで外す（警告に `GGML_CL11_DEVICES` での選び直しと、`remote` バックエンド / llama.cpp RPC で別プロセスにする案内を含む）。デバイスが 1 枚、世代が 1 つ、NVIDIA 以外ではコストは増えない。`GGML_CL11_NO_CONFLICT_CHECK=1` で無効にできる。外れたデバイスは、ggml プラグインでは自動で別プロセスに回す（下の「自動 RPC」）。
- GT 730・GT 710・RX 6400・WX 2100 の 4 枚（VRAM 合計 約 9GB）を 1 プロセスで使える。`GGML_CL11_PLATFORMS=NVIDIA,rusticl RUSTICL_ENABLE=radeonsi` でプラットフォームを絞り、`GGML_CL11_DEVICES` で GT 430 を外す。
- GT 430 は `remote` バックエンド（`SCRAPGPU_BUILD_REMOTE`、下の「別プロセスの device」）で加えられる。device サーバーが別プロセスで GT 430 だけを開き、呼び出しは Unix ソケット＋共有メモリで転送される。

## 自動 RPC（ggml プラグイン、Linux）

プラグインは、列挙で外れたデバイスごとに `ggml-rpc-server` を子プロセスとして起動し（`GGML_BACKEND_PATH`＝プラグイン自身、`GGML_CL11_DEVICES`＝そのデバイス名、`-H 127.0.0.1 -p <空きポート> -d CL11_0 -t 1`）、`ggml_backend_rpc_add_server` で RPC デバイス（`RPC0` など）として登録する。llama.cpp の RPC バックエンド（`libggml-rpc.so`）が先に読み込まれていることが条件。サーバーは `libggml-rpc.so` と同じディレクトリの `ggml-rpc-server`（`GGML_CL11_RPC_SERVER` で上書き）。親の終了で子も終了する（`PR_SET_PDEATHSIG` と `atexit`）。RPC が無い、サーバーが無い、起動に失敗したときは除外＋警告のまま。`GGML_CL11_AUTO_RPC=0` で無効。

- RPC デバイスは `CL11_n` ではなく `RPC0` として、CL11 のデバイスより前に並ぶ（`-ts` はこの順）。
- **`-fa off` が必須。** RPC デバイスは全 op を対応と答えるので、`FLASH_ATTN_EXT` がサーバーに回り、CL11 のサーバーが assert で落ちる。プラグイン側では避けられない。起動時の警告にも書いてある。
- 実測（深さ別）：LFM2.5-8B-A1B Q4_0、環境指定なし、5 枚の層分割、`-fa 0 -ub 128`。top-4 のエキスパートで tg32 は深さ 512 で 13.34 t/s、深さ 8192 で 10.92 t/s。実行後に `ps` で `ggml-rpc-server` は残っていない。
- 実測（TB250、環境指定なし、`RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=NVIDIA,rusticl`、LFM2.5-8B-A1B Q4_0、`-ts 8,8,18,40,18 -fa off`）：RPC0（GT 430）・GT 730・GT 710・RX 6400・WX 2100 の層分割で完走、pp 6.63 t/s、tg 13.15 t/s（63 トークン）。

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

### GT 430 を加えて 5 枚（llama.cpp の RPC）

GT 430 は Kepler と同じプロセスに入らないので、llama.cpp の `ggml-rpc-server`（公式ビルド b11384）を CL11 のプラグインと `GGML_CL11_DEVICES="GT 430"` で別プロセスに立て、メインプロセスから `--rpc 127.0.0.1:50052` で `RPC0` として使う。

| 構成（Qwen3-8B Q4_K_M、`-ts 1/2/4/2/1`） | pp128（t/s） | tg32（t/s） |
| --- | --- | --- |
| CL11：GT 730 + GT 710 + RX 6400 + WX 2100 + GT 430（RPC） | 20.71 | 3.94 |
| CL11（NVIDIA 2 枚）+ 公式 Vulkan（AMD 2 枚）+ GT 430（RPC） | 22.05 | 4.22 |

5 枚（VRAM 合計 約 10GB）を 1 つのモデルに使える。速さは 4 枚より少し落ちる（GT 430 の層の時間と RPC の往復が足される）ので、GT 430 は 4 枚に収まらないモデルの容量のために使う。

## ScrapGpuLibrary の C API：GPU 間の転送

`scrapgpu_tensor_copy`（`docs/API_CONTRACT.md` 14 章）で、デバイスをまたいでテンソルをコピーする。違うデバイスの間はホストを経由する（OpenCL 1.1 にはコンテキストをまたぐイベントがない）。

| 試験（TB250、GT 730 + GT 710 + RX 6400 + WX 2100 を 1 プロセス） | 結果 |
| --- | --- |
| `tests/ops`：`tensor_copy` の同じデバイス・デバイス間（両方向） | バイト単位で一致、87 件すべて PASS |
| `examples/pipeline_mlp.c`：8 層の MLP（隠れ層 1024、Q4_0）を 4 枚に 2 層ずつ | NMSE 2.4e-10（1 枚と同じ） |

速度（64 トークン）：1 枚（GT 730）1.24 ms / トークン、4 枚 5.38 ms / トークン。AMD 2 枚では 1 枚 6.53 ms、2 枚 8.00 ms。1 枚に収まる小さな計算を分けると、転送ごとの同期（ホストへの読み出しを待つ）が足されて遅くなる。分割は、1 枚に収まらない大きさのときに容量を得るための手段で、速さのためではない。

## 別プロセスの device（remote バックエンド）

`scrapgpu-device-server`（`--backend opencl11 --socket PATH`、環境は `GGML_CL11_DEVICES` などがそのまま通る）が 1 クライアントに vtable の呼び出しを全て提供する。クライアント側は `remote` バックエンド（`scrapgpu_remote_spawn` で fork+exec、または `scrapgpu_remote_connect` で既存のソケットに接続）。サーバーの device は `remote:` 接頭辞で `scrapgpu_get_device` の一覧にローカルと並んで現れ、`op_execute` / `tensor_upload` / `stream_sync` / plan など全部がそのまま動く。`scrapgpu_tensor_copy` のリモート↔ローカルは既存のホスト経由（`buffer_read` → 共有メモリ → `buffer_write`）。同じサーバーの device 間は BUFFER_COPY の 1 往復。

TB250 での使い方：メインプロセスは 4 枚（`GGML_CL11_PLATFORMS=NVIDIA,rusticl RUSTICL_ENABLE=radeonsi` + `GGML_CL11_DEVICES` で GT 430 を外す）をローカルに登録し、GT 430 は `scrapgpu_remote_spawn({"GGML_CL11_DEVICES=GT 430", "GGML_CL11_PLATFORMS=NVIDIA"})` で別プロセスにする。`examples/pipeline_mlp.c` では環境変数 `SCRAPGPU_PIPELINE_REMOTE="GGML_CL11_DEVICES=GT 430;GGML_CL11_PLATFORMS=NVIDIA"` で同じことをする。

コスト（M2 Max で実測、仕様は docs/API_CONTRACT.md 15 章）：呼び出し 1 回ごとにソケット 1 往復。アイドル `stream_sync` が約 11.5 µs、`op_execute` の投入だけで直接比 +27 µs、`op_execute` + `stream_sync` で +39 µs。bulk データ（buffer / tensor の転送）はソケットではなく共有メモリ（初期 1 MiB、需要で倍々拡大）を通る。`tests/ops` は `SCRAPGPU_TEST_BACKEND=remote` で全件（87 件のうち remote で 82 件実行：同じ NMSE、加えて remote↔local の `tensor_copy` も PASS）をサーバー越しに回せる。

## 次の課題

- ~~GT 430 を別プロセスで加える~~ → `remote` バックエンドとして実装し、TB250 で確かめた：GT 430 を remote、ほかの 4 枚をローカルにして、`tests/ops` が 87 件 PASS（remote↔local の `tensor_copy` もバイト一致）、`examples/pipeline_mlp.c` が 5 枚で NMSE 2.4e-10。終了後にサーバーのプロセスやソケットは残らない。ソケットは `mkdtemp` で作った本人だけのディレクトリに置き（0600）、サーバーは接続元が同じユーザーかを確かめる。
- 転送の待ちを減らす（ホスト経由のコピーを非同期にし、次の層の計算と重ねる）。
- 層の分割比を、VRAM ではなく各 GPU の層あたりの時間に合わせて決める。
- remote の呼び出しは 1 往復 / 呼び出しなので、小さい op を密に打つモデルでは往復が支配的になる。必要なら plan（1 往復でノード列を実行）に寄せる。
