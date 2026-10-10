# 異世代 GPU・旧 CPU・現行フレームワークの検証

2026-10-11 JST。実装の説明・再現コマンドは [HETEROGENEOUS_RUNTIME.md](../../HETEROGENEOUS_RUNTIME.md)。今回の結果を、既存の CPU 最適化や過去の GPU チューニングの測定値へ混ぜない。

## ソースと環境

- ScrapGpu 基準: `258e7085ee25b447ea107a40629f476b8802b17c` + 本変更の検証時点の差分。検証ソースの SHA256 は [source-sha256.txt](source-sha256.txt)。
- 転送ソースアーカイブ SHA256: `4fd06b5eaadda1525dfc7277adac25de441700b44c18f83fd08e668f50ebe044`。TB250・6f で照合済み。文書・測定ログの後追い追記はこのアーカイブに含まない。
- llama.cpp: `23b0202a189c44a54625aadcb37a946dd1d6278d`、ggml 0.26.0 / backend API 3。Git archiveで転送したためbenchの `build_commit` は `unknown`。アーカイブ SHA256 は `fb1c36d8173b4800e84a6e80ab96232868da9ae1ac816178bcf647108dbfb9c6`。
- TB250: Celeron G3930、GCC 14.2、NVIDIA 390.157、AMD rusticl。各GPUは別プロセス、既存 `gpu-bench.lock` で排他。
- 6f: i5-3230M、GCC 15.2。ScrapGpuはRelease、追加ISA指定なし。CPUのSSE2経路を実行。
- Mac: Apple M2 Max、AppleClang 21、OpenCL 1.2。
- 実装分担: Devin CLI `swe-2-high` がCPU backend、Cursor CLI `gpt-5.4-high` がAttentionを実装。統合時に転置・alias・短い記述子などを修正し、以下を実行。

## 結果

| 対象 | 結果 |
| --- | --- |
| Mac Release、OpenCL実機 | CTest 15/15 PASS、SKIPなし |
| Mac Debug、ASan + UBSan | CTest 12/12 PASS、報告なし |
| TB250、OpenCL + PyTorch統合構成 | CTest 18/18 PASS、SKIPなし（CTestのGPUはGT730） |
| 6f、CPU-only | CTest 12/12 PASS、SKIPなし |
| TB250各GPU、C API演算 | NVIDIA各81実行PASS/2未対応SKIP、AMD各82実行PASS/1未対応SKIP |
| TB250各GPU、F32/BF16 probe | 全PASS、NMSE=0。F32はnative、BF16は明示CPU fallback |
| TB250 CPU + 5 GPU、PyTorch 2.14.1+cpu | 13/13プロセスPASS。各SDPAは18数値比較、16拒否条件、12並行呼出し |
| 6f、PyTorch 2.14.1+cpu | CPU-only build、基本演算・SDPA・設定・MLPの4試験PASS |
| Mac、PyTorch 2.14.1 | CPU/OpenCL buildと試験10/10 PASS。BF16 fallback無効・予算ゼロの明示拒否も確認 |
| 最新llama.cpp、5 GPU | 各ADD 46実行PASS、MUL_MAT 72実行PASS/12未対応。未対応はpermutationを含むケース |
| 最新llama.cpp、greedy生成 | 5 GPUすべて、同条件のCPU生成と16 token一致 |

機械可読集計は [validation.json](validation.json)。各 `*-llama-op-ADD.log` / `*-llama-op-MUL_MAT.log` が実行件数付きのllama演算記録。初回の古いfilterは0件だったため集計から除外し、`run-llama-ops.sh` で再実行した。`status.tsv` の小文字 `llama-add` 行はその0件の診断実行であり、合格件数には含めない。

CPU/GPU生成は Qwen3-0.6B **base** Q4_K_M、prompt=`The capital of France is`、`-ngl 99`（CPUは0）、`-t 2 -b 64 -ub 64 -c 256 -n 16 --temp 0 -fa off`。CPUの初回 `auto` は異なる出力だったため、[cpu-generation-fa-off.txt](tb250/results-final/cpu-generation-fa-off.txt) を比較元とした。

## 短時間の推論測定

同じモデル、`llama-bench -b 64 -ub 64 -p 32 -n 16 -r 2 -fa off`。単位t/s。短時間の動作確認値であり、最適性能や長文・大モデルの速度保証ではない。AMDの推論推奨は既存の実測に従い公式Vulkanを維持する。

| GPU / CL11 platform | pp32 | tg16 |
| --- | ---: | ---: |
| GT730 / NVIDIA | 122.08 | 36.81 |
| GT430 / NVIDIA | 27.81 | 22.85 |
| GT710 / NVIDIA | 62.03 | 16.65 |
| WX2100 / rusticl | 143.19 | 17.11 |
| RX6400 / rusticl | 259.43 | 24.67 |

6fの既存CPU行列演算を再測定した [cpu-bench.csv](6f/cpu-bench.csv) では、K=1024/M=3072/N=16のQ4_0が6.848ms、Q8_0が6.733ms、Q4_Kが6.816ms。これはCPUカーネル単体の値。BF16対応や新しいdevice APIによる追加の速度向上を示す比較ではない。

6fのPyTorchは検証ディレクトリ専用のvenvに導入した。初回は`ensurepip`不足で失敗したが、[pip公式のget-pip方式](https://pip.pypa.io/en/stable/installation/)でvenv内だけにpipを導入して解消。システムPythonやaptは変更していない。[最終結果](6f/pytorch/summary.json) と [bootstrap記録](6f/pytorch/bootstrap.log) を参照。MLPのF32最大絶対誤差は`1.639128e-7`、基本演算の最大NMSEは`1.283e-13`。

## 検証範囲

NVIDIAドライバの更新、任意CUDAバイナリの互換化、PyTorch全モデルの自動offload、学習/autograd、FP8、llama.cpp Flash AttentionのCL11実行は行っていない。新規AttentionはF32の明示API。multi-GPU同時スケジュール・長時間負荷・大型モデル・MoEの現行HEADでの生成は今回未検証。RX6400+Cloverの実行は既知のハングを避けて行わず、拒否ポリシーを単体テストで検証した。
