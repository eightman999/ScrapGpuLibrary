# 回帰の基準（v0.2 分離の Gate 0）

v0.2 の分離作業（ブランチ `v0.2-split`）は、すべてこの基準と比べて判定する。

## 基準のコード

| 項目 | 値 |
| --- | --- |
| 基準のコミット | main `34817e7`（Mesa Clover での `ARGSORT` の修正）。速度は直前の `fc27d10` で測った値。両者の差は `k_argsort` だけで、下の速度の計測（Qwen3 の生成）が通る処理は同じ |
| llama.cpp | b11384（`0faee50`）、無改造。TB250 では `~/bench/scrapgpu/llama.cpp/build-dl`（`-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF`、CPU は SSE4.2。`GGML_NATIVE=ON` と `GGML_BACKEND_DL=ON` は併用できない） |
| CL11 の組み込み | ツリー外プラグイン（`GGML_BACKEND_PATH=.../libggml-cl11.so`） |

## 計測環境

| 項目 | TB250 | Mac |
| --- | --- | --- |
| OS | Debian 13、kernel 6.12.107+deb13-amd64 | macOS 26.6.2 |
| CPU / メモリ | Celeron G3930（2 コア、AVX なし）/ 32GB | Apple M2 Max |
| GPU とドライバ | GT 430 / GT 730 / GT 710：NVIDIA 390.157（OpenCL 1.1 / 1.2）。WX 2100 / RX 6400：Mesa 25.0.7（Clover：OpenCL 1.1、rusticl：OpenCL 3.0、`RUSTICL_ENABLE=radeonsi`）。RX 6400 の Clover はハングするので使わない | Apple の OpenCL 1.2 |
| PCIe | RX 6400 は Gen2 x4、ほかは Gen2 x1 | |
| 計測開始時の温度 | GT 430 38 ℃、GT 730 34 ℃、GT 710 36 ℃、AMD 37〜42 ℃ | |

## モデル

| 呼び名 | 量子化 | SHA256 |
| --- | --- | --- |
| Qwen3-0.6B | Q8_0 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| Qwen3-0.6B-Base | Q4_K_M | `e1c597652808f9db19a554b3b98d2b0fe3a3c5b00920452bc2a809317b83ae5a` |
| Qwen2-0.5B-ft（非公開の微調整モデル） | Q4_0 | `e3a189bce5443ba04ece04b69b4cc05a8ce03068a945d89785c6da84928a5ed4` |
| TinyLlama-1.1B-Chat | Q4_0 | `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556` |
| Qwen3-1.7B | Q4_K_M | `72c5c3cb38fa32d5256e2fe30d03e7a64c6c79e668ad84057e3bd66e250b24fb` |
| LFM2.5-8B-A1B | Q4_0 | `48ed1465d761311b2fd57b7fb46cf969a20b3a8281945b04e10f52bd1609e715` |

## 速度（TB250、2026-10-09）

`llama-bench -ngl 99 -ub 128 -b 128 -p 128 -n 64 -r 5 -t 2`、既定の調整値。値は平均 ± 標準偏差（t/s）。

| GPU | Qwen3-0.6B Q8_0 pp128 | tg64 | Qwen3-0.6B-Base Q4_K_M pp128 | tg64 |
| --- | --- | --- | --- | --- |
| GT 430 | 89.4 ± 0.00 | 22.3 ± 0.01 | 89.9 ± 0.00 | 23.7 ± 0.00 |
| GT 730 | 241.6 ± 0.52 | 34.8 ± 0.02 | 231.3 ± 0.65 | 39.1 ± 0.00 |
| GT 710 | 120.3 ± 0.90 | 14.7 ± 0.00 | 117.1 ± 0.60 | 17.8 ± 0.00 |
| WX 2100（Clover） | 334.2 ± 10.77 | 21.5 ± 0.45 | 333.7 ± 7.69 | 18.6 ± 0.14 |
| WX 2100（rusticl） | 451.9 ± 9.38 | 15.1 ± 0.22 | 435.9 ± 9.09 | 15.4 ± 0.23 |
| RX 6400（rusticl） | 917.2 ± 12.44 | 20.0 ± 1.17 | 937.2 ± 17.71 | 24.1 ± 2.15 |

- NVIDIA の 3 枚はばらつきが 1% 未満で、Gate 2 の ±3% はそのまま判定に使える。
- AMD は pp で 2〜3%、RX 6400 の tg で 5〜9% ばらつく（rusticl の起動コストが大きく、クロックも動く）。AMD は推奨が公式 Vulkan で、CL11 は動作の確認が目的なので、許容幅を広げて判定する（`baseline.csv` の `tol_pct`：RX 6400 は 20%）。
- この表は `adapters/ggml/tools/kepler-sweep/baseline.csv` と同じ値で、`sweep.sh` の `regress` ステージが自動で比較する。

ほかの参考値（同じ条件。基準の判定には使わないが、回帰に気づくために見る）：

| 構成 | 値（t/s） |
| --- | --- |
| GT 430、Qwen2-0.5B-ft Q4_0 | pp128 118.5 / tg64 36.5 |
| LFM2.5-8B-A1B Q4_0、専門家を CPU（`--cpu-moe`、`llama-completion -n 128 -c 1024`） | tg：GT 430 10.4、GT 710 8.8、GT 730 11.7（CPU だけ 6.3） |
| 公式 Vulkan（比較用） | `docs/VULKAN_AMD.md` を参照 |

## 正しさ

| 試験 | Mac（M2 Max） | GT 430 | GT 730 | GT 710 | WX 2100 Clover | WX 2100 rusticl | RX 6400 rusticl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `test-backend-ops -b CL11_0`（対応ケース） | 3406 / 3406（既定、`FORCE_GEN=kepler`、`FORCE_GEN=fermi`） | 3441 / 3442 | 3441 / 3442 | 3442 / 3442 | 3435 / 3435（`34817e7`） | 3435 / 3435 | 3442 / 3442 |
| greedy 64 トークン（Qwen3-0.6B Q8_0 と CPU） | 一致 | 一致 | 一致 | 一致 | 一致 | 一致 | 一致 |
| greedy（Q4_K_M、main のプラグインと） | | 一致 | 一致 | 一致 | 一致 | 一致 | 一致 |

- GT 430 / GT 730 で落ちる 1 件は、VRAM に入らない 1 GiB の `MUL_MAT_HADAMARD` で、想定どおり。
- 対応ケースの件数は、デバイスのローカルメモリや確保の上限で変わる（Mac 3406、AMD 3435、NVIDIA 3442）。件数の変化は回帰ではない。判定は「対応ケースがすべて通ること」で行う。
- `tools/mvm_check` 18 / 18、`tools/attn_check` 8 / 8（Mac と NVIDIA 3 枚）。
- KL ダイバージェンス（`llama-perplexity -c 512 --chunks 8 -b 128 -ub 128 -t 2`、CPU の logits と比較。許容値はモデルと文章ごとに公式バックエンドで校正する）：

| モデル | CL11 | 公式 Vulkan（RX 6400） |
| --- | --- | --- |
| Qwen3-0.6B-Base Q4_K_M（GT 430） | 平均 0.00681、99% 0.0369、Same top p 95.34% | 0.00776、0.0433、95.10% |
| LFM2.5-8B-A1B Q4_0、`--cpu-moe`、`-c 512 --chunks 4` | GT 430 0.0216、GT 730 / GT 710 0.0216〜0.0226 | 0.0196〜0.0224（CPU 同士の差 0.0185） |

## 再現手順

TB250（`flock ~/bench/gpu-bench.lock` で排他。GPU は `GGML_CL11_DEVICES` で 1 枚に絞る。85 ℃以上なら待つ）：

```sh
cd <リポジトリのコピー>
export PATH=$HOME/bench/cmake-venv/bin:$PATH
M=~/bench/models
OUT=$PWD/sweep-out/<名前> BASE_PLUGIN=<比較するプラグイン> \
  MODEL_Q8=$M/qwen3-0.6b-q8_0.gguf MODEL_Q4K=$M/qwen3-0.6b-base-q4_k_m.gguf MODEL_Q40=<Q4_0 のモデル> \
  BENCH_REPS=5 STAGES="build regress summary" adapters/ggml/tools/kepler-sweep/sweep.sh
```

結果は `$OUT/regress.csv` と `$OUT/summary.md`（`baseline.csv` との差を % で示し、許容幅を超えたら `REGRESSION` と書く）。

Mac：

```sh
cmake -S adapters/ggml -B <build> -DLLAMA_CPP_SOURCE_DIR=<llama.cpp b11384> -DLLAMA_CPP_BUILD_DIR=<その build-dl> -DCMAKE_BUILD_TYPE=Release
cmake --build <build>
for g in "" kepler fermi; do GGML_CL11_FORCE_GEN=$g GGML_BACKEND_PATH=<build>/libggml-cl11.so <build-dl>/bin/test-backend-ops test -b CL11_0; done
```

1 構成で 20〜50 分かかる。

## Gate 0 の判定

**Go**。基準のコミット、環境、モデルの SHA256、速度（5 回の平均と標準偏差）、正しさ、再現手順を固定した。基準を取る途中で見つかった Mesa Clover の `ARGSORT` の不具合（ローカルメモリの引数で引数の位置がずれる）は main で修正済み（`34817e7`、`docs/HARDWARE.md`）。

## Gate 2 で見る項目（Phase 2 の完了時）

- 上の速度表の全項目が、NVIDIA は ±3%、AMD は `tol_pct` 以内。
- 正しさの表の全項目。
- 1 トークンあたりのカーネル起動回数（`GGML_CL11_LAUNCH_LOG=1`）が増えていないこと（Qwen3-0.6B Q8_0：Kepler 452、Fermi 564）。
- モデルの読み込み時間と VRAM の使用量が大きく変わっていないこと。
