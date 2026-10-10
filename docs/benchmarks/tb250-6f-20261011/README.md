# TB250 / 6f 最適化の実測記録（2026-10-11）

対象: `jimoto-no-llm/ScrapGpuLibrary`、公開 main の基準 `83d82d8386507fccda6566a826afab26c8c1160d` と、測定時点の候補ソース。実装と利用手順は [CPU_RAM_OPTIMIZATION.md](../../CPU_RAM_OPTIMIZATION.md)。

## 条件

| 項目 | TB250 | 6f |
| --- | --- | --- |
| CPU | Celeron G3930、2C/2T、2.90 GHz、AVX なし | i5-3230M、2C/4T、2.60 GHz、AVX / F16C、AVX2 / FMA なし |
| RAM（OS が認識する容量） | 33,587,806,208 bytes | 16,117,338,112 bytes |
| OS | Debian、Linux 6.12.107+deb13-amd64 | Ubuntu、Linux 7.0.0-30-generic |
| コンパイラ | GCC 14.2.0 | GCC 15.2.0 |
| core のフラグ | `-O3 -DNDEBUG -fvisibility=hidden -fvisibility-inlines-hidden` | 同左 |

CPU の測定は単一スレッド、K=1024 / M=3072 / N=1,16、6 dtype、warmup 2 回、測定 7 回の中央値。逆量子化・scratch 確保を含む。基準・候補の benchmark ソースとビルド条件は同じ。CPU アフィニティやクロック固定は行っていない。6f は K=14336 / M=512 / N=16 の Q4_K も測定した。

GPU は既存の llama.cpp 共有 DL build と、それぞれの `libggml-cl11.so` を使う。adapter は b11384 向けだが、ホストのソースには `.git` がなく、実行 JSON の `build_commit` は `unknown`。バイナリ・モデル・候補ソースの SHA256 は [run.json](run.json) に記録した。候補ソースの hash はローカル / TB250 / 6f で一致する。

Qwen3-0.6B Q8_0 / Qwen3-0.6B-Base Q4_K_M、`-ngl 99 -ub 128 -b 128 -p 128 -n 64 -r 5 -t 2 -o json`。JSON の平均 t/s と標準偏差を保存した。`~/bench/gpu-bench.lock` を取得し、`gpu_exec.sh` の温度待機を使って 1 GPU / process ずつ実行した。GPU 温度は 85℃以上で待機し、78℃未満で再開する設定。個別実行の温度値は今回の JSON に記録していない。

## CPU 結果のファイル

- [6f/comparison.csv](6f/comparison.csv): 基準 `cpu-baseline.csv` と最終 `cpu-opt3.csv` の比較。N=16 の量子化重みは 4.75〜7.21×、F32 N=1 は 2.04×。
- [6f/comparison-large.csv](6f/comparison-large.csv): K=14336 の Q4_K は 5.46×。
- [tb250/comparison.csv](tb250/comparison.csv): 基準 `cpu-baseline.csv` と再コンパイル済み最終 `cpu-final.csv` の比較。N=16 の量子化重みは 6.86〜11.38×、F32 N=1 は 3.73×。
- 各ホストの `descriptor-*.csv`: 混合 divisor に対する `fastdiv_init()` 200万回 × 7 回の中央値。checksum は基準と最終候補で一致。
- `simd-evidence.txt`: 実機 core object の `mulps` / `addps`。AVX 指定なしで SSE 経路を生成した証拠。

`cpu-opt.csv` / `cpu-opt2.csv` と対応テストログは途中の候補の診断記録。最終比較には使用しない。TB250 で tar の保存時刻が既存 object より古くなった候補測定は、実機側で `cpu-opt3-stale.csv` に退避し、ソースを touch して再コンパイルした。`tb250/build-final.log` に `ops.cpp.o` と ref テストの再コンパイルを記録している。

## NVIDIA の GPU 結果

| GPU | モデル | 基準 pp128 | 候補 pp128 | 基準 tg64 | 候補 tg64 |
| --- | --- | ---: | ---: | ---: | ---: |
| GT 730 | Q8_0 | 242.040 | 241.790 | 34.807 | 34.786 |
| GT 730 | Base Q4_K_M | 231.372 | 231.174 | 39.128 | 39.131 |
| GT 710 | Q8_0 | 120.356 | 119.397 | 14.661 | 14.661 |
| GT 710 | Base Q4_K_M | 115.670 | 116.802 | 17.776 | 17.771 |
| GT 430 | Q8_0 | 89.371 | 89.372 | 22.296 | 22.292 |
| GT 430 | Base Q4_K_M | 89.927 | 89.926 | 23.668 | 23.668 |

単位 t/s、5 回の平均。NVIDIA 390.157。差は全ケース −0.80〜+0.98% で、従来の ±3% 許容内。ホスト側の記述子準備の高速化を、モデル全体の高速化と解釈しない。生 JSON / stderr は `tb250/gpu-final/`、比較表は [gpu-final/comparison.csv](tb250/gpu-final/comparison.csv)。

GT 730 で CPU thread 数 1、attention work-group 256、attention P=8、max work-group 64 を比較したが、生成速度の改善はなかった。既定設定を採用する。対応する JSON / log は `tb250/` 直下。`t2_poll0` はこの版の llama-bench が `-poll` に非対応で失敗した記録なので、速度の比較に使用しない。

## AMD の GPU 結果

| GPU / platform | モデル | 基準 pp128 | 候補 pp128 | 基準 tg64 | 候補 tg64 |
| --- | --- | ---: | ---: | ---: | ---: |
| WX 2100 Clover | Q8_0 | 331.406 | 337.382 | 20.564 | 21.580 |
| WX 2100 Clover | Base Q4_K_M | 327.866 | 333.961 | 18.555 | 18.930 |
| WX 2100 rusticl | Q8_0 | 450.134 | 448.165 | 14.920 | 14.984 |
| WX 2100 rusticl | Base Q4_K_M | 436.099 | 439.852 | 15.434 | 18.025 |
| RX 6400 rusticl | Q8_0 | 911.323 | 914.462 | 19.813 | 20.306 |
| RX 6400 rusticl | Base Q4_K_M | 928.870 | 924.638 | 22.003 | 20.724 |

単位 t/s、5 回の平均。Mesa 25.0.7-2+deb13u1。RX 6400 の Base Q4_K_M tg64 は −5.81% で、既存の 20% 許容内。WX 2100 rusticl の同ケースは +16.79% だが、候補の標準偏差 4.677 t/s が大きく、安定した改善とは判定しない。ほかの差は −0.46〜+4.94%。生 JSON / stderr と比較表は [gpu-amd/comparison.csv](tb250/gpu-amd/comparison.csv) と同じディレクトリに保存した。

WX 2100 rusticl / Base Q4_K_M のみ、候補→基準の順で各 10 回追加測定した。pp128 は 436.666 → 436.450 t/s（−0.05%）、tg64 は 15.264 ± 0.186 → 15.277 ± 0.162 t/s（+0.08%）。初回の大きな差は再現せず、GPU 全体の安定した速度向上は確認していない（[comparison-recheck.csv](tb250/gpu-amd/comparison-recheck.csv)）。初回 5 回のデータも保全した。

## 検証範囲

- 最終 core / ref / tensor CTest: ローカル arm64 8/8、6f 8/8、TB250 OpenCL あり 9/9 PASS。
- CPU 回帰: 6 dtype、M=1/3/4/5/9、N=1/3/4/7/16/17、odd K、量子化 block、非整列 offset、padded stride、次元 2 / 3 broadcast、入力不変と guard bytes、K の加算順。x86_64 cross build を Rosetta でも実行。
- `fastdiv_init()`: 旧 multiplier / shift との厳密比較（境界値・2 の冪・ランダム divisor 1万個）、既存 fastdiv 適用試験。
- GT 730 の関連 ggml 演算 `RMS_NORM,SOFT_MAX,ROPE,GET_ROWS,SET_ROWS,MUL_MAT`: 1105/1105 PASS。
- 独立 C API: NVIDIA 3 枚と AMD 3 構成の全対応演算が成功。unsupported の SKIP は NVIDIA 各 2 件、AMD 各 1 件で、ログに理由を保存した。
- `mvm_check` 18/18、`attn_check` 8/8: NVIDIA 3 枚でそれぞれ PASS。
- カーネルソース・backend / adapter は基準から変更なし。GT 730 でビルドオプション文字列と起動ログが厳密一致。pp8 は 617、tg は 452 launches / graph（[identity-summary.txt](tb250/gpu-final/identity-summary.txt)）。起動を増やしていない。

基準の全 ggml 演算試験は、大きな MoE ケースで長時間かかったため約 9 分で中断した。981 MiB の GT 730 に 1 GiB を確保する既存の失敗を 1 件確認した。全演算試験完了・全件 PASS とは扱わない。状態は [gpu-ggml-tests-status.txt](tb250/gpu-ggml-tests-status.txt)、全量の途中ログは TB250 の `/home/eightman/dev/tools/ScrapGpuLibrary-opt-20261011/gpu-ggml-tests.log` に保存している。

Devin SWE2 (`swe-2-high`) と Cursor の read-only 調査を設計候補の検討に使用した。採用の判断はソース、独立した確認、上記の実機測定に基づく。

## 実機で再利用する場所

両ホストとも `/home/eightman/dev/tools/ScrapGpuLibrary-opt-20261011/opt` が候補ソースと専用 build。`base` は比較用。TB250 の plugin は `opt/build/adapters/ggml/libggml-cl11.so`。6f の core は `opt/build/libscrapgpu_core.a`。測定した実装は [run.json](run.json) のソース SHA256 で識別できる。
