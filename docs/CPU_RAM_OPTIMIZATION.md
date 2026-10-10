# 旧式 CPU と RAM の行列演算最適化

2026-10-11、`eightman@6f` と TB250 で検証した CPU 参照実装の最適化。基準は `83d82d8386507fccda6566a826afab26c8c1160d`。

## 対象と実装

公開 C API の `scrapgpu_ref_execute()` が実行する `MUL_MAT` を高速化する。入力の重みは F32 / F16 / Q4_0 / Q8_0 / Q4_K / Q6_K、activation と出力は F32。llama.cpp の CPU backend や自動 CPU fallback の実装は今回の対象に含まれない。

- 4 列以上では activation を最大 16 列の小さな K-major panel に詰め、逆量子化した重み 1 行を複数の出力列で使い回す。量子化重み全体を F32 に展開しない。
- x86 では SSE2 の独立した lane を使用する。AVX / AVX2 / FMA は不要。ほかのアーキテクチャには同じ加算順の portable 経路がある。
- 各出力の K 方向の加算順を維持する。行・列の余り、非整列 offset、行間の padding、次元 2 / 3 の broadcast に対応する。
- 1〜3 列の F32 は 4 行をまとめて処理する。F16 と量子化重みの少数列処理は、panel の転置による余分な RAM 転送を避けるため従来の経路を使う。

一時領域は activation panel と重み 1 行で最大 `17 * K * sizeof(float)`。K=1024 なら 68 KiB、K=14336 なら 952 KiB。K×M の重み全体を展開する領域は不要。確保サイズの乗算前に上限を確認する。並列化や外部 BLAS への依存は追加していない。

## 6f の実測

i5-3230M（2 コア / 4 スレッド、RAM 約 15 GiB）、GCC 15.2、Release `-O3 -DNDEBUG`、ISA 追加フラグなし。単一スレッドの C API 呼び出しを warmup 2 回、測定 7 回実行した中央値。K=1024、M=3072。

| 重み | 出力列 N | 基準 ms | 最適化 ms | 速度比 |
| --- | ---: | ---: | ---: | ---: |
| F32 | 1 | 7.297 | 3.577 | 2.04× |
| F32 | 16 | 51.671 | 10.110 | 5.11× |
| F16 | 16 | 61.283 | 19.359 | 3.17× |
| Q4_0 | 16 | 48.524 | 6.846 | 7.09× |
| Q8_0 | 16 | 48.394 | 6.713 | 7.21× |
| Q4_K | 16 | 48.469 | 6.778 | 7.15× |
| Q6_K | 16 | 53.050 | 11.168 | 4.75× |

F16 / 量子化重みの N=1 は 0.998〜1.008×でほぼ同等。大きな内積長の Q4_K（K=14336、M=512、N=16）も 113.273 → 20.740 ms、5.46×。全測定ケースで出力 checksum が基準と一致した。これは行列演算単体の速度比で、モデル全体の推論速度比ではない。

生 CSV、比較結果、テストログは [実測記録](benchmarks/tb250-6f-20261011/) に保存した。テストは 6 dtype、端数、stride、offset、broadcast、桁落ちが起きる K 加算順を確認する。

## TB250 の CPU 実測

Celeron G3930（2 コア / 2 スレッド、RAM 約 31 GiB）、GCC 14.2。6f と同じ形状・反復数・ビルド条件。AVX を持たない CPU で SSE2 経路を実行した。

| 重み | 出力列 N | 基準 ms | 最適化 ms | 速度比 |
| --- | ---: | ---: | ---: | ---: |
| F32 | 1 | 9.856 | 2.642 | 3.73× |
| F32 | 16 | 73.900 | 10.617 | 6.96× |
| F16 | 16 | 78.143 | 14.931 | 5.23× |
| Q4_0 | 16 | 69.358 | 6.214 | 11.16× |
| Q8_0 | 16 | 69.249 | 6.083 | 11.38× |
| Q4_K | 16 | 69.272 | 6.158 | 11.25× |
| Q6_K | 16 | 73.990 | 10.780 | 6.86× |

F16 / 量子化重みの N=1 はほぼ同等で、全ケースの checksum が基準と一致した。最終測定は `ops.cpp` を再コンパイルした後の `cpu-final.csv` を使用する。

CL11 のホスト側 `fastdiv_init()` は TB250 で 17.836 → 4.982 ns（3.58×）、6f で 16.794 → 5.438 ns（3.09×）。NVIDIA 3 枚の Qwen3-0.6B の pp128 / tg64 は基準との差が −0.80〜+0.98% で、モデル全体の有意な高速化は確認していない。カーネル・GPU 調整値は維持した。GPU 構成別の値と検証範囲は [実測記録](benchmarks/tb250-6f-20261011/) を参照。

## ビルドと再測定

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DSCRAPGPU_BUILD_BENCHMARKS=ON
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
./build-cpu/scrapgpu_cpu_bench --k 1024 --m 3072 \
  --iterations 7 --warmup 2 > candidate.csv
python3 tools/compare_cpu_bench.py baseline.csv candidate.csv
./build-cpu/scrapgpu_descriptor_bench
```

CPU benchmark の既定は 6 dtype × N=1 / 16。`--dtype Q4_K --n 16 --k 14336 --m 512` のようにケースを限定できる。データ生成・量子化は計測外、C API 内の panel 確保・逆量子化・演算は計測内。比較ツールはケース・checksum が違う場合に失敗する。基準と候補は同じ benchmark ソース、コンパイラ、ビルド条件で測定する。

`scrapgpu_descriptor_bench` は CL11 が使う `fastdiv_init()` のホスト側準備を測定する。1・2 の冪を短絡し、GCC / Clang では leading-zero 命令で桁数を求める。GPU に渡す multiplier / shift は旧式の計算と厳密に一致し、カーネル ABI を維持する。GPU 全体への効果は別途 llama-bench で評価する。
