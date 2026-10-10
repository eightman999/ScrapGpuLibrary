# TB250・旧式 CPU と現行フレームワーク

対象は TB250 の GT 430 / GT 730 / GT 710 / WX 2100 / RX 6400 と、6f の i5-3230M。既存の GPU 世代別カーネル調整と [CPU 行列演算の最適化](CPU_RAM_OPTIMIZATION.md) を使い、上位の演算要求を対応能力に合わせて実行する。

## NVIDIA ドライバと実行先

TB250 の NVIDIA 390.157 はそのまま使う。Fermi の公式最終ドライバ系列は R390、CUDA Toolkit は 8.0。新しい CUDA バイナリを旧ドライバで実行する互換層ではなく、OpenCL 1.1 のカーネルに演算を変換する。Kepler も CUDA 12 でサポートが削除されている。新しいドライバを入れれば全 GPU が使える、という前提は置かない。

根拠: [NVIDIA の互換性表](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)、[CUDA 12.0 release notes](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/)。

| 実行先 | 方針 |
| --- | --- |
| GT 430 (Fermi) | CL11。NVIDIA 390 の制約のため Kepler と別プロセスで使う |
| GT 730 / GT 710 (Kepler) | CL11。GT 710 の推論は既存の実測で Q4_K 系を推奨 |
| WX 2100 / RX 6400 | llama.cpp 推論は既存実測で公式 Vulkan を推奨。ScrapGpu C API / PyTorch の CL11 経路は rusticl を選ぶ |
| RX 6400 + Clover | 既知の GPU ハングを避けるため、デバイス列挙時に理由付きで除外 |
| 6f / TB250 の CPU | `scrapgpu_cpu_register()` で明示登録。OpenCL・CUDA 不要、AVX2/FMA 不要 |

推論速度の根拠とモデル・量子化ごとの差は [HARDWARE.md](HARDWARE.md)。汎用演算とモデル全体の推論速度は別々に評価する。未知の GPU に既存 GPU の速度比を当てはめない。

## C API

CPU は GPU と同じ buffer / stream / op / plan API を使う。登録は明示的なので、従来の空レジストリや GPU index を勝手に変えない。CPU は natural 配置、同期実行。量子化行列は重み全体を F32 に展開せず、既存の小さな panel と SSE2 / portable 実装を使う。

`scrapgpu_op_execute_compatible()` は実行前に指定デバイスの対応を確認する。対応していればそのまま実行する。対応判定が `UNSUPPORTED` の場合に限り、呼び出し側が `allow_cpu_fallback=1` と `max_staging_bytes` を指定していれば、CPU 参照実装へ切り替える。GPU 実行後の失敗や device lost はそのまま返す。

- report に実際の native / CPU fallback と転送 bytes を返す。
- fallback は連続 tensor に限定。R32 入力は転送時に natural に戻す。
- staging 上限は入力と出力の合計。CPU カーネル内部の一時領域は別枠。
- 部分書き込みを保全するため出力も読み込み、CPU 計算成功後に書き戻す。fallback は単一出力の対応演算に限定する。
- 無効な形状や未知の op を近似計算でごまかさず、status と理由を返す。
- PCIe x1 を往復する fallback は速度保証ではない。繰り返す処理は CPU に常駐させるか、対応する GPU backend を選ぶ。

BF16 は CPU で F32 に変換して蓄積する。`CPY` / `DUP` / `CONT` の F32・F16・BF16 相互変換、BF16 重み × F32 activation の `MUL_MAT` に対応する。GPU の BF16 命令を要求しない。F32→BF16 は round-to-nearest ties-to-even、NaN を保持する。

```c
#include <scrapgpu/cpu.h>
#include <scrapgpu/compat.h>

scrapgpu_cpu_register(); /* CPU を使うプロセスで明示的に登録 */
/* device / stream / tensor buffers を通常の API で作成 */
scrapgpu_compat_options options = {sizeof(options), 1, 64u * 1024u * 1024u};
scrapgpu_execution_report report = {0};
report.struct_size = sizeof(report);
/* scrapgpu_op_execute_compatible(device, stream, &op, &options, &report); */
```

## Attention

`scrapgpu_scaled_dot_product_attention()` は F32 の Q/K/V を、行列積 → Softmax → 行列積に分解する。Flash Attention / CUDA / subgroup / tensor core は不要。

- Q `[D,Q,Hq,B]`、K `[D,K,Hkv,B]`、V `[Dv,K,Hkv,B]`、出力 `[Dv,Q,Hq,B]`。
- GQA / MQA、causal、KV cache 用 query offset、加算 mask、任意の有限 scale に対応。
- 完全に `-inf` で覆われた行はゼロ。大きな負の有限 mask は加算値として扱う。
- query を tile に分け、ヘッド・バッチごとに workspace を使い回す。全ヘッドの Q×K を同時に確保しない。
- device workspace は概ね `4*K*(Dv + tileQ*(mask有無+1))` bytes。最低 1 query 分が上限内に入らなければ事前に失敗する。ドライバの最大 allocation にも合わせる。
- mask 作成・完全mask行の処理にはホスト転送がある。device workspace とは別に、最大 K×tileQ 程度の host mask を使う。
- F32 contiguous の単一デバイス実行。FP8、任意CUDA kernel、autograd、dropout、全モデルの透過的置換は提供しない。

## PyTorch

[PyTorch adapter](../adapters/pytorch/README.md) は明示的な custom ops。CPU ビルドと `SCRAPGPU_TORCH_BACKEND=cpu` で GPU のない 6f でも使える。OpenCL は `SCRAPGPU_TORCH_BACKEND=opencl11`。`SCRAPGPU_TORCH_DEVICE` で対象 index を選ぶ。環境設定は最初の読み込み前に行う。

`scrapgpu_torch.scaled_dot_product_attention()` は PyTorch の `[B,H,L,D]` 配置を受け、上の C API に接続する。PyTorch 本体の関数を自動置換しない。対応範囲と未対応入力は adapter の README を参照。

## llama.cpp

従来の b11384 に加え、2026-10-11 に確認した公開 HEAD `23b0202a189c44a54625aadcb37a946dd1d6278d`（ggml 0.26.0 / backend API 3）で TB250 の5 GPUのプラグイン・演算・生成を検証した。ggml の公開 backend ヘッダーは b11384 と同一で、追加の互換分岐は不要。API 2/3 の既存分岐を保つ。

未対応 op は llama.cpp の対応判定で CPU scheduler に渡る。新しい C API Attention を追加しても、llama.cpp の Flash Attention が自動で CL11 に載るわけではない。将来の llama.cpp HEAD は ABI や op が変わり得るため、その HEAD に対するビルドと演算・生成の確認が必要。

TB250 の Celeron G3930 では、`GGML_NATIVE=OFF` と AVX 無効だけでは不十分だった。現行 llama.cpp の既定で `GGML_BMI2=ON` が残り、CPU backend の feature score が 0 となってロードが拒否された。以下の設定で BMI2 も無効にする。既存の NVIDIA ドライバを更新する必要はない。

```sh
cmake -S /path/to/llama.cpp -B /path/to/llama-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
  -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build /path/to/llama-build --target ggml-cpu llama-bench llama-completion test-backend-ops -j2
cmake -S adapters/ggml -B build-ggml \
  -DLLAMA_CPP_SOURCE_DIR=/path/to/llama.cpp -DLLAMA_CPP_BUILD_DIR=/path/to/llama-build
cmake --build build-ggml -j2
```

この HEAD の生成 CLI は `llama-completion` を使う。CPU と CL11 の生成を比較するときは、両方で `-fa off` と同じ prompt・モデル・batch・context・greedy 設定を指定する。CPU の既定 `auto` と CL11 の `off` を比較すると、今回の Q4_K_M モデルでは生成が異なった。

## 再現

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DSCRAPGPU_BUILD_BENCHMARKS=ON
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
./build-cpu/scrapgpu_compat_probe cpu

# TB250 の単一 GPU。GT 430 と Kepler を同じプロセスで初期化しない。
GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES='GT 730' \
  ./build-opencl/scrapgpu_compat_probe opencl11
RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES='RX 6400' \
  SCRAPGPU_TEST_BACKEND=opencl11 ./build-opencl/scrapgpu_attention_tests
```

実測結果・source hash・PASS / FAIL / SKIP は [検証記録](benchmarks/heterogeneous-20261011/README.md) に保存した。TB250 5 GPU の PyTorch と最新 llama.cpp、6f の CPU / PyTorch、Mac の ASan / UBSan を検証済み。既存の CPU 高速化の数値は [CPU_RAM_OPTIMIZATION.md](CPU_RAM_OPTIMIZATION.md) を参照。
