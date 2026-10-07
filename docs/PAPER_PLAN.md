# 論文化の検討（arXiv preprint）

状態: 検討中（2026-10-08）。

## 位置付け

- 「旧 GPU でも LLM が動く」は既出なので、新規性にしない。
- 主張の軸は性能工学に置く。「OpenCL 1.1 しかない Fermi に現行の LLM バックエンドを新しく実装し、GPU アーキテクチャに合わせた最適化で、実効メモリ帯域の近くまで性能を回収した」。
- 構図: CL11 がケーススタディ。ScrapGpuLibrary が、その方法論（GPU ごとに測り、合う物理配置・カーネル・転送を選んで回収する runtime）の一般化。
- 進め方: 短い preprint を先に出し、ScrapGpuLibrary の一般化と対象 GPU の拡大に合わせて育てる。

## 先行研究との差分

| 研究 | 内容 | 本研究との差 |
| --- | --- | --- |
| [M2Cache](https://arxiv.org/abs/2410.14740)（arXiv 2410.14740） | M40 などの旧 GPU で、混合精度と HBM / DRAM / SSD の多段キャッシュにより、VRAM に入らない LLM を推論する。速度と炭素排出の削減を報告 | 対象は CUDA が動く Maxwell 世代以降で、VRAM 容量の問題を扱う。本研究は CUDA も現行 API も使えない GPU で、カーネルの性能（帯域の回収）を扱う |
| [AIvailable](https://arxiv.org/abs/2511.11621)（arXiv 2511.11621） | 異種・旧世代の NVIDIA / AMD GPU 群を束ねる LLM-as-a-Service 基盤。VRAM に応じたモデル配置、CPU フォールバックなし | 対象はサービス基盤の層で、各 GPU では既存の推論エンジンが動くことが前提。本研究は、エンジンが動かない GPU のためのバックエンドとカーネル |
| [Relic](https://github.com/anamnesic/relic)（Anamnesic Labs、GitHub、MIT） | 1〜4GB 級の旧 GPU、iGPU 向けの独自 GGUF ランタイム。OpenCL 1.2 / 3.0。Q4_0 / Q8_0 / FP16 の詰め替え、自動調整されたカーネル登録表、ハードウェアのマイクロプロファイラ、dGPU + iGPU + ホストの異種構成、分散投機的デコード。対象例は GTX 1650、Intel UHD、Caicos XT（TeraScale） | **最も近い**。差は (1) OpenCL 1.1（Fermi）を明示的に対象とし、実機で検証していること（Relic のホスト API は 1.1 の範囲だが 1.2 / 3.0 を謳い、旧 GPU の実機は未検証）、(2) 初期性能 → ボトルネックの分析 → 各最適化の効果 → 実効帯域比、という定量的な性能工学（Relic は README に数値がなく、ADR に GTX 1650 での tok/s と CUDA カーネルの GB/s があるが、理論帯域に対する比と旧 GPU の数値はない）、(3) ggml / llama.cpp のプラグインとして既存の GGUF 生態系にそのまま載ること、(4) 正しさの評価方法、(5) ドライバの不具合の報告。ソースを読んだ詳細は [REUSE.md](REUSE.md) |
| [llama-fermi-clblast](https://github.com/eightman999/llama-fermi-clblast)（自作の旧フォーク） | 2023 年の llama.cpp を CLBlast で GT 430 に移植。OpenLLaMA-3B Q4_0 で 1.35〜2.40 t/s、CPU のみ（4.91 t/s）より遅い | 本研究のベースラインの一つ。「動くが遅い」から「CPU より速い」への変化を示せる |

注意:

- Relic の対象には Caicos XT（2011 年）が含まれ、GT 430（2010 年）とほぼ同世代。「最も古い世代を対象にした」という主張は避け、「OpenCL 1.1 の制約（サブグループなし、fp16 演算なし、1.2 以降の API なし）の下での設計」を差分として述べる。
- 「初めて」とは書かない。書くなら「我々の知る限り」とし、検索範囲を明記する。
- 投稿前に related work をもう一度洗う。2026-10-08 に下の「追加の先行研究」で一巡したが、Relic は 2026-05 に作られて活発に更新されており（最終 push 2026-10-06）、旧 GPU の実測が追加される可能性がある。投稿直前に再確認する。
- Relic のソースを読んだ限り（REUSE.md）、「auto-tuned kernel registry」は静的な表、「micro-profiler」はベンダー文字列から定数を埋める処理、「分散投機的デコード」は同一プロセス内の 2 デコーダである。論文では Relic の README ではなくソースと ADR を根拠に比較し、誇張した批判を避ける。
- 背景の「Fermi の 32bit 整数除算が約 100 命令」は、一次情報で確認できなかった（NVIDIA の資料は見出しのみ、他の資料は約 20〜30 命令と書く）。論文では、自前で SASS を実測した命令数を示す。

## 追加の先行研究（2026-10-08 調査）

上の 4 件に加えて、引用できる文献・技術報告・ソフトウェアを集めた。

- arXiv の ID は arXiv API でタイトルと著者を照合した。DOI は Crossref で、GitHub の PR / Discussion は `gh api` で確認した。
- 会場は arXiv ページの comment 欄または Crossref で確認できたものだけを書いた。書いていないものは arXiv 表記にとどめる。
- 内容の要約は arXiv の abstract に基づく。本文を読んでいないものは、投稿前に本文で確認する。ACM / IEEE のページは開けなかったので、該当する文献は DOI の登録情報（Crossref）の確認にとどまる。
- 「本研究との関係」は、何を引用するかと、差を 1 行で書いた。

### 旧 GPU・低予算 GPU での LLM 推論

| 文献 | 識別子 | 本研究との関係 |
| --- | --- | --- |
| Cao ほか, "DumpsterCluster: From Dumpster Diving to Serving LLaMA-70B on $60 GPUs", 2026 | [arXiv 2608.14614](https://arxiv.org/abs/2608.14614) | 退役 V100 の中古部品で 128 GPU のクラスタを作りパイプライン並列で 70B を配信。**旧 GPU は 1 トークンあたりのエネルギーが大きいと報告**。本研究の J/token の主張と正面から比べる必要がある。差は、データセンター級の V100 と CUDA、1 年間運用したクラスタであること |
| Shi ほか, "GreenLLM", 2024 | [arXiv 2412.20322](https://arxiv.org/abs/2412.20322) | 旧世代 GPU を再利用して炭素排出を最大 40.6% 削減。旧 GPU 再利用の環境面の根拠として引用。GPU の世代は特定されず、カーネルは扱わない |
| Xing, "Exploration of Cryptocurrency Mining-Specific GPUs in AI Applications: A Case Study of CMP 170HX", 2025 | [arXiv 2505.03782](https://arxiv.org/abs/2505.03782) | マイニング専用 GPU を CUDA 側の改変で AI 用に転用し、OpenCL ベンチと llama ベンチで評価。転用 GPU の先例。差は、新しめの NVIDIA で CUDA、バックエンドの新規実装ではない点 |
| Sheng ほか, "FlexGen", 2023 | [arXiv 2303.06865](https://arxiv.org/abs/2303.06865) | 単一 GPU で VRAM を超えるモデルを動かす古典（M2Cache と同系統）。本研究は、VRAM に収まる範囲で帯域を回収する点が違う |
| Song ほか, "PowerInfer", 2023（SOSP 2024 と arXiv ページに記載） | [arXiv 2312.12456](https://arxiv.org/abs/2312.12456) | コンシューマ GPU（RTX 4090）での CPU / GPU ハイブリッド推論。新世代 GPU を前提にしている点が違う |

### OpenCL での LLM 推論（モバイル GPU を含む）

| 文献 | 識別子 | 本研究との関係 |
| --- | --- | --- |
| llama.cpp PR #10693 "Introducing experimental OpenCL backend with support for Qualcomm Adreno GPUs"（lhez ほか、2024-12-13 マージ） | [GitHub](https://github.com/ggml-org/llama.cpp/pull/10693)、[docs/backend/OPENCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md) | **ggml 向け OpenCL の最も近い先行実装**。OpenCL 2.0 以上（`opencl_c_version.major < 2` で拒否）、`cl_khr_fp16` 必須、サブグループ必須。本研究の差は「OpenCL 1.1・サブグループなし・fp16 演算なし」 |
| Wang（Qualcomm）, "llama.cpp OpenCL backend update", IWOCL 2025 スライド | [PDF](https://www.iwocl.org/wp-content/uploads/iwocl-2025-hongqiang-wang-lamacpp-backend-update.pdf) | 上記の設計（Q4_0 の再配置、MatVec が鍵）。調査では、OpenCL 1.x への（サブグループなしの）移植は可能という趣旨の記述と、旧 CLBlast 版の deprecated の経緯が確認された。「1.1 版は未踏だった」という傍証になるが、引用する前にスライドの該当箇所を確認する |
| llama.cpp PR #1164 "CLBlast support" / #1459 "OpenCL dequant_mul_mat"（0cc4m、2023-04 / 05 マージ）、#7735 "ggml : remove OpenCL"（2024-06-04） | [#1164](https://github.com/ggml-org/llama.cpp/pull/1164)、[#1459](https://github.com/ggml-org/llama.cpp/pull/1459)、[#7735](https://github.com/ggml-org/llama.cpp/pull/7735) | 旧 OpenCL バックエンドの導入と削除。CLBlast による GEMM と dequantize 中心で、生成（mat-vec）の最適化と帯域比の報告はない。自作の旧フォーク（llama-fermi-clblast）の系譜 |
| Nugteren, "CLBlast: A tuned OpenCL BLAS library", IWOCL 2018 | [arXiv 1705.05249](https://arxiv.org/abs/1705.05249)、[doi:10.1145/3204919.3204924](https://doi.org/10.1145/3204919.3204924) | OpenCL BLAS の先例で、**GPU ごとのチューニングデータベース**の考え方の出典。README は OpenCL 1.1 ハードウェアで使えると明記。差は、量子化 mat-vec と 1.1 限定、旧世代を主対象にしている点 |
| Li ほか, "Transformer-Lite", 2024 | [arXiv 2403.20041](https://arxiv.org/abs/2403.20041) | モバイル GPU 向けの LLM エンジン（FP4、動的形状、KV のコピー排除）。OpenCL かどうかは abstract では確認できていない。差は、SoC の GPU でありデスクトップの旧 GPU ではない点 |
| Tang ほか, "Scaling On-Device GPU Inference for Large Generative Models（ML Drift）", 2025 | [arXiv 2505.00232](https://arxiv.org/abs/2505.00232) | 複数の GPU API にまたがるオンデバイス推論の枠組み。差は同上 |
| Lai ほか, "Relax"（MLC-LLM の基盤）, 2023 | [arXiv 2311.02103](https://arxiv.org/abs/2311.02103) | MLC-LLM の学術的な引用先（MLC-LLM 自体の論文は確認できなかった）。TVM の OpenCL は既定で 1.2。コンパイラ生成のカーネルである点が違う |
| Ruan ほか, "WebLLM", 2024 | [arXiv 2412.15803](https://arxiv.org/abs/2412.15803) | WebGPU でネイティブ性能の約 8 割を維持と報告。**「約 8 割」の分母はネイティブ性能であり、本研究の帯域比とは別の指標**。混同を避けて引用する |
| Xue ほか, "PowerInfer-2", 2024 | [arXiv 2406.06282](https://arxiv.org/abs/2406.06282) | スマートフォンでのメモリ階層と I/O の最適化。GPU / OpenCL ではない。関連研究の範囲 |

### 量子化 LLM の GPU カーネル・量子化形式

| 文献 | 識別子 | 本研究との関係 |
| --- | --- | --- |
| ikawrakow, llama.cpp PR #1684 "k-quants"（2023-06-05 マージ） | [GitHub](https://github.com/ggml-org/llama.cpp/pull/1684) | **必引用**。本研究が使う ggml の量子化形式（Q2_K〜Q6_K）の一次情報。論文ではなく PR |
| ggml-org, GGUF 仕様 | [docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | ファイル形式の仕様。**Q8_0 や k-quants の内部のブロック構造は定義されておらず、実装（`ggml-common.h`）が正**。ブロック配置はソースを引用する |
| Park ほか, "LUT-GEMM", ICLR 2024 | [arXiv 2206.09557](https://arxiv.org/abs/2206.09557) | LUT による逆量子化の省略。量子化 mat-vec の別解として対比。CUDA が対象 |
| Frantar ほか, "MARLIN", 2024 | [arXiv 2408.11743](https://arxiv.org/abs/2408.11743) | 4bit 重みの自己回帰推論向けカーネル。重みの事前再配置という点で R32 に近い。Ampere 以降・大バッチ向けで、本研究は旧 GPU・バッチ 1 |
| Guo ほか, "FLUTE", EMNLP 2024 Findings | [arXiv 2407.10960](https://arxiv.org/abs/2407.10960) | LUT 量子化の行列積と重みのオフライン再配置。同上 |
| Lin ほか, "QServe", 2024 | [arXiv 2405.04532](https://arxiv.org/abs/2405.04532) | W4A8KV4 と GPU システムの共同設計。逆量子化のオーバーヘッドの議論として補助的に引用 |
| Zhao ほか, "Atom", 2023 | [arXiv 2310.19102](https://arxiv.org/abs/2310.19102) | 低ビット量子化による LLM サービング。補助 |
| Lin ほか, "AWQ", MLSys 2024 | [arXiv 2306.00978](https://arxiv.org/abs/2306.00978) | 量子化手法の背景。本研究は量子化手法を新規に作らないので補助 |
| Frantar ほか, "GPTQ", ICLR 2023 | [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) | 同上 |
| Dettmers ほか, "LLM.int8()", NeurIPS 2022 | [arXiv 2208.07339](https://arxiv.org/abs/2208.07339) | Q8_0 との対比（Q8_0 はブロック単位のスケールで、外れ値の分離を行わない） |
| Xiao ほか, "SmoothQuant", ICML 2023 | [arXiv 2211.10438](https://arxiv.org/abs/2211.10438) | W8A8 の量子化。本研究の mat-vec は活性化が FP32 なので、直接の関係は薄い |
| Dao ほか, "FlashAttention", 2022 | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) | 対比のみ（CL11 は Flash Attention 非対応） |
| Kwon ほか, "PagedAttention / vLLM", SOSP 2023 | [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) | 対比のみ（単一ユーザの低帯域環境は対象外） |
| Abdelfattah ほか, "KBLAS", 2014 | [arXiv 1410.1726](https://arxiv.org/abs/1410.1726) | CUDA の GEMV / Level-2 BLAS の最適化。比較は cuBLAS 比の速度向上で、達成帯域比ではない |
| Abdelfattah ほか, "Systematic Approach in Optimizing Numerical Memory-Bound Kernels on GPU", Euro-Par 2012 Workshops | [doi:10.1007/978-3-642-36949-0_23](https://doi.org/10.1007/978-3-642-36949-0_23) | メモリ律速カーネル（GEMV を含む）の最適化。同上 |
| Mukunoki ほか, "Fast Implementation of General Matrix-Vector Multiplication (GEMV) on Kepler GPUs", PDP 2015 | [doi:10.1109/PDP.2015.66](https://doi.org/10.1109/PDP.2015.66) | **注意の項を参照**。旧 NVIDIA の GEMV をピーク近くまで出したとされる。abstract は未確認 |

### ルーフラインモデルと LLM 推論の律速

| 文献 | 識別子 | 本研究との関係 |
| --- | --- | --- |
| Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 52(4), 2009 | [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) | **必引用**。「帯域の約 8 割」の分母（実測した streaming 帯域）を置く枠組み。技術報告の前身は [UCB/EECS-2008-134](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)（題名が異なる） |
| Yuan ほか, "LLM Inference Unveiled: Survey and Roofline Model Insights", 2024 | [arXiv 2402.16363](https://arxiv.org/abs/2402.16363) | ルーフラインで LLM のデコードが帯域律速である理由を整理。本研究は個別の旧 GPU での達成率を実測する点が違う |
| Pope ほか, "Efficiently Scaling Transformer Inference", 2022 | [arXiv 2211.05102](https://arxiv.org/abs/2211.05102) | 大規模 Transformer 推論の解析モデル。TPU・500B 超が対象で、補助 |
| Chen, "Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode", 2026 | [arXiv 2605.30571](https://arxiv.org/abs/2605.30571) | バッチ 1 のデコードで、理論帯域に対する達成率を報告（L4 で約 81%、H100 で約 27%）。**新世代 GPU の達成率の比較対象**。差は、旧世代・OpenCL・量子化 mat-vec |
| Deakin ほか, "Evaluating attainable memory bandwidth of parallel programming models via BabelStream", IJCSE 2018 | [doi:10.1504/IJCSE.2018.095847](https://doi.org/10.1504/IJCSE.2018.095847) | OpenCL を含む STREAM 風の帯域測定。streaming 22.9 GB/s の測定方法の先例 |

### Fermi / Kepler のマイクロベンチマークと性能モデル

| 文献 | 識別子 | 本研究との関係 |
| --- | --- | --- |
| Mei, Chu, "Dissecting GPU Memory Hierarchy through Microbenchmarking", IEEE TPDS 2017 | [arXiv 1509.02308](https://arxiv.org/abs/1509.02308)、[doi:10.1109/TPDS.2016.2549523](https://doi.org/10.1109/TPDS.2016.2549523) | **最重要の引用候補**。Fermi / Kepler / Maxwell のキャッシュ構造と遅延の解析。GT 430 / GT 710 / GT 730 と同世代。差は、本研究がカーネルの最適化と達成帯域を扱う点 |
| Wong ほか, "Demystifying GPU microarchitecture through microbenchmarking", ISPASS 2010 | [doi:10.1109/ISPASS.2010.5452013](https://doi.org/10.1109/ISPASS.2010.5452013) | マイクロベンチの手法の先例。**対象は GT200 で Fermi ではない**（本文を確認した範囲） |
| Lai, Seznec, "Performance upper bound analysis and optimization of SGEMM on Fermi and Kepler GPUs", CGO 2013 | [doi:10.1109/CGO.2013.6494986](https://doi.org/10.1109/CGO.2013.6494986) | Fermi / Kepler で命令レベルの上限を導出して詰める手法の先例。演算律速の SGEMM で、mat-mm の 8×4 タイルと比べられる。本研究の mat-vec は帯域律速。本文は未確認 |
| Volkov, "Better Performance at Lower Occupancy", GTC 2010 | [PDF](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf) | 占有率を下げてスレッドあたりの ILP とレジスタを増やすほど速い例。8×4 のレジスタタイルの根拠。G80 時代の例で Fermi の数値ではない |
| Zhang, Owens, "A quantitative performance analysis model for GPU architectures", HPCA 2011 | [doi:10.1109/HPCA.2011.5749745](https://doi.org/10.1109/HPCA.2011.5749745) | マイクロベンチに基づく性能モデル（GT200）。補助 |
| Hong, Kim, "An analytical model for a GPU architecture with memory-level and thread-level parallelism awareness", ISCA 2009 | [doi:10.1145/1555754.1555775](https://doi.org/10.1145/1555754.1555775) | メモリ並列度と計算並列度のモデル。帯域飽和の議論の背景 |
| Lam, "Measuring GPU Memory Latency", Chips and Cheese, 2021（ブログ、参考扱い） | [URL](https://chipsandcheese.com/p/measuring-gpu-memory-latency) | OpenCL のポインタチェイスで AMD（TeraScale / GCN / RDNA2）の遅延を比較。AMD 側の参考。査読なし |

### 異種 GPU・低帯域でのパイプライン並列と分散推論

| 文献 | 識別子 | 本研究との関係 |
| --- | --- | --- |
| Borzunov ほか, "Petals", ACL 2023 System Demonstrations | [arXiv 2209.01188](https://arxiv.org/abs/2209.01188) | コンシューマ GPU を集めて層ごとに分けて推論する古典。差は、ネットワーク越しの分散で、同一ホストの PCIe x1 ではない点 |
| Borzunov ほか, "Distributed Inference and Fine-tuning of Large Language Models Over The Internet", NeurIPS 2023 | [arXiv 2312.08361](https://arxiv.org/abs/2312.08361) | Petals の続編（フォールトトレラント推論と負荷分散）。同上 |
| Ryabinin ほか, "SWARM Parallelism", ICML 2023 | [arXiv 2301.11913](https://arxiv.org/abs/2301.11913) | 低帯域（200 Mb/s 未満）でもパイプライン並列が成り立つ根拠。訓練であり推論ではない |
| Jiang ほか, "HexGen", ICML 2024 | [arXiv 2311.11514](https://arxiv.org/abs/2311.11514) | 異種 GPU・異種ネットワークでの推論の配置。データセンター級の GPU が対象 |
| Mei ほか, "Helix", ASPLOS 2025 | [arXiv 2406.01566](https://arxiv.org/abs/2406.01566) | GPU とネットワークの異種性を max-flow と MILP で最適化。差は同上 |
| Patel ほか, "Splitwise", ISCA 2024 | [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) | prefill は演算律速、decode はメモリ律速という前提の裏付け。PCIe x1 の議論はない |
| Li ほか, "prima.cpp", 2025 | [arXiv 2504.08791](https://arxiv.org/abs/2504.08791) | llama.cpp ベースで家庭の異種デバイスに 30〜70B を分散（pipelined-ring）。目的が近い。差は、Wi-Fi / Ethernet でつなぐ別マシンであること、OpenCL 1.1 ではないこと |
| Zhang ほか, "TD-Pipe", ICPP 2025 | [arXiv 2506.10470](https://arxiv.org/abs/2506.10470) | PCIe 接続のみの GPU ノード向けのパイプライン並列。PCIe x1 や旧 GPU の議論はない |
| Zhang ほか, "EdgeShard", 2024 | [arXiv 2405.14371](https://arxiv.org/abs/2405.14371) | エッジとクラウドの間でモデルを分割。関連研究の範囲 |
| b4rtaz, distributed-llama（MIT）、exo-explore, exo（Apache-2.0） | [GitHub](https://github.com/b4rtaz/distributed-llama)、[GitHub](https://github.com/exo-explore/exo) | 家庭のデバイスを束ねる実装。前者はテンソル並列でノード数が 2 のべき乗、後者は MLX が主体。P4 の比較対象として言及する程度 |

### LLM 推論のエネルギー測定

| 文献 | 識別子 | 本研究との関係 |
| --- | --- | --- |
| Samsi ほか, "From Words to Watts", 2023 | [arXiv 2310.03003](https://arxiv.org/abs/2310.03003) | LLaMA 推論のエネルギー測定の基本文献。V100 / A100 が対象 |
| Niu ほか, "TokenPowerBench", AAAI 2026 | [arXiv 2512.03024](https://arxiv.org/abs/2512.03024) | prefill と decode にエネルギーを帰属させて J/token を測る。**測定方法の参照**。差は、新世代 GPU と TensorRT-LLM / vLLM 系が対象である点 |
| Tschand ほか, "MLPerf Power", 2024 | [arXiv 2410.12032](https://arxiv.org/abs/2410.12032) | 電力測定の方法論の基準。個別の旧 GPU のデータではない |
| Chung ほか, "The ML.ENERGY Benchmark", NeurIPS 2025 Datasets & Benchmarks | [arXiv 2505.06371](https://arxiv.org/abs/2505.06371) | J/token のベンチマーク手法。旧 GPU は対象外 |
| Argerich ほか, "Watt Counts", 2026 | [arXiv 2604.09048](https://arxiv.org/abs/2604.09048) | 10 種の NVIDIA GPU・50 モデル・5000 以上の実験の J/token。**新旧 GPU のエネルギー比較として最も近い**。最古でも T4〜A30 級（調査での確認）で、Fermi〜Pascal は含まれない |
| Laskaridis ほか, "MELTing point: Mobile Evaluation of Language Transformers", MobiCom 2024 | [arXiv 2403.12844](https://arxiv.org/abs/2403.12844) | モバイルでの性能・エネルギー・熱の測定。LLM 推論がメモリ律速であるという結論の裏付け |
| Husom ほか, "Sustainable LLM Inference for Edge AI", 2025 | [arXiv 2504.03360](https://arxiv.org/abs/2504.03360) | Raspberry Pi 4 で量子化モデルのエネルギー・精度・遅延を測定。量子化とエネルギーの先例（CPU のみ） |

### 注意: 本研究と重なりが大きい先行研究

重なりが大きい順。「見つからなかった」ことは、存在しない証明ではない（検索範囲は下に記載）。

1. **llama.cpp の Vulkan 性能スレッド（Discussion #10879、2024-12-18）** — [GitHub](https://github.com/ggml-org/llama.cpp/discussions/10879)
   - 重なり: 旧 GPU で llama.cpp が動き、生成（tg128）の値が公開されている。Llama-2-7B Q4_0（Vulkan）で、Radeon RX 470 は 40.11 t/s、RX 570 / 580 は約 37〜39 t/s、Radeon Pro WX 4100（Polaris）は 16.05 t/s、Tesla K80（Kepler）は 9.39 t/s。
   - **帯域使用率は報告されていない。** モデルを約 3.8GB と仮定した概算（要検算）では、RX 470 は約 150 GB/s で理論帯域（約 211 GB/s、推測）の約 7 割、WX 4100 は約 61 GB/s で約 6 割。K80 は 1 ダイ 240 GB/s と仮定して約 15%。
   - 影響: 「旧 GPU で LLM を実用速度で動かす」は新規ではない（位置付けの節と同じ）。Polaris は Vulkan で既に 6〜7 割に達している可能性があるため、「帯域の 8 割」の新規性は、**Vulkan が使えない Fermi・OpenCL 1.1・PCIe x1 の環境で達成した点**に置く。WX 2100（2GB）は 7B が載らないので直接比較できず、小さなモデルで自前の Vulkan 計測が要る。
2. **Qualcomm の llama.cpp OpenCL バックエンド（PR #10693、IWOCL 2025）**
   - 重なり: ggml 向けの OpenCL で量子化 mat-vec（Q4_0）を最適化。重みの再配置、定数メモリ、ベクトル型、ループ展開など、手法が似ている。デコードは帯域律速という整理も同じ。
   - 差: Adreno が対象で、OpenCL 2.0 / 3.0 とサブグループが必須。スライドは「1.x への移植は可能」と述べるが実施していない。達成帯域率は報告せず tok/s のみ（調査での確認）。
3. **旧 CLBlast バックエンド（PR #1164 / #1459、2023）** — OpenCL で llama.cpp を動かした先例。MatMul と dequantize が中心で、mat-vec の最適化と帯域比はない。自作の旧フォークの系譜でもあるので、「動くが遅い」から変わった点として示せる。
4. **Mukunoki ほか（PDP 2015）の Kepler 向け GEMV（MUBLAS）**
   - 重なり: 旧 NVIDIA の GEMV をピーク近くまで出したという主張。RIKEN の資料の結論に Fermi / Kepler / Maxwell で「nearly peak」とある（二次資料。論文の abstract は未確認）。
   - 差: CUDA、量子化なしの GEMV、上位機（GTX 580 / K20c / GTX 980 など）。
   - 対応: **本文を入手して確認する**。旧 GPU の GEMV で帯域の大半を回収したという主張の新規性は、この論文の数値次第で弱まる可能性がある。本研究の新規性は「量子化ブロックのデコードを含む mat-vec」「OpenCL 1.1」「実装の公開」に置く。
5. **DumpsterCluster（arXiv 2608.14614）** — 結論が逆向き（旧 GPU は 1 トークンあたりのエネルギーが大きい）。本研究のエネルギー効率の主張は、これと直接比べる必要がある。
6. **新世代 GPU の帯域達成率（arXiv 2605.30571）** — L4 で約 81%。「8 割」だけを示すと新世代と変わらないように見えるので、旧世代・OpenCL 1.1・x1 の制約下で達成した点を書く。
7. **Relic** — 上の表と [REUSE.md](REUSE.md) を参照。

見つからなかったもの（検索した範囲）:

| 探した内容 | 結果 |
| --- | --- |
| Fermi 世代（GT 430、GTX 480 / 580 など）で LLM（Transformer）を動かした例 | 見つからなかった。Fermi に触れるのは AlexNet の歴史的な記事程度 |
| OpenCL 1.1 だけで LLM 推論バックエンドを実装した例 | 見つからなかった。CLBlast が 1.1 対応（README）という間接的な情報と、Qualcomm のスライドの「移植は可能」という記述のみ |
| 旧 GPU（Fermi / Kepler / Polaris）で、量子化 mat-vec の「達成帯域 / 理論帯域」を一次情報として報告した例 | 見つからなかった。新世代は arXiv 2605.30571。旧 NVIDIA の（量子化なしの）GEMV は Mukunoki ほかと KBLAS があるが、帯域比の数値は確認できていない。Polaris / Kepler の tg は #10879 にあるが帯域比はない |
| Mesa の rusticl / Clover や PoCL 上の llama.cpp | 見つからなかった（Clover の利用例は PR #1164 の RX 570 のみ） |
| PCIe x1 の GPU でのパイプライン並列推論、WX 2100 / RX 6400 の LLM ベンチ | 見つからなかった |
| 旧 GPU（Fermi〜Polaris）の tokens/J | 見つからなかった。旧世代の J/token は arXiv 2608.14614（V100）と arXiv 2604.09048（T4 以降）のみ |

検索の限界: Vulkan・Kompute・ncnn の旧 GPU 向け実装、TVM / MLC-LLM の旧 GPU での OpenCL の報告は、十分には調べていない。ACM / IEEE のページは直接開けなかった。

## 主張と根拠の状況

| 主張 | 根拠 | 状況 |
| --- | --- | --- |
| Fermi（OpenCL 1.1）で現行の LLM を全層オフロードできる | GT 430 で Qwen3-0.6B / TinyLlama / Qwen2 系 0.5B を全層オフロード | 済 |
| 性能を実効帯域の近くまで回収した | Q8_0 の mat-vec: 4.65 → 18.8 GB/s。単純なストリーミング読み出し 22.9 GB/s の約 82%。モデル内では 14.8〜17.6 GB/s | 済（論文ではマイクロベンチとモデル内の両方を出す） |
| 生成が CPU を上回る | Qwen3-0.6B Q8_0 の tg: 2.97 → 21.3 t/s（CPU 15.9）。pp は 88.6（CPU 26.0） | 済 |
| 各最適化の寄与 | 除算の除去、バイト → float の変換、R32 配置、定数キャッシュ、転置コピー（640 → 136µs）と融合、mat-mm の 8×4 | 一部済。最適化を 1 つずつ外したときの数値（ablation）を整理する必要がある |
| 方法論が他の GPU でも通用する | Kepler（GT 710 / GT 730）、AMD（WX 2100 / RX 6400）で実施中 | 実施中 |
| 正しさ | test-backend-ops、logits の KL ダイバージェンス、perplexity の差 | KL ダイバージェンスは未測定 |
| エネルギー効率 | AMD は amdgpu の sysfs で測れる。NVIDIA 390 は電力を取得できない | GeForce は外部の電力計（コンセント側）が必要 |
| ドライバの不具合の発見 | Kepler + 390 で定数メモリのサブバッファの開始位置が無視される。公称上限を超える確保が通る | 済（再現コードを整理する） |

## 追加で必要な実験

1. **ablation**: 最適化を 1 つずつ外したときの mat-vec の GB/s と tg（GT 430、可能なら全 GPU）。
2. **ルーフライン図**: GPU ごとのストリーミング帯域と理論演算性能に、各カーネル（mat-vec、mat-mm、attention）を重ねる。
3. **正しさ**: CPU や公式バックエンドとの KL ダイバージェンス、top-1 一致率、perplexity の差。
4. **ベースライン**: CPU（同じマシン）、旧 CLBlast 移植版、AMD では公式 Vulkan バックエンド。可能なら Relic も同じ GPU（GT 730 / WX 2100 は OpenCL 1.2 以上で動く可能性がある）で比べる。
5. **エネルギー**: GeForce は外部電力計で J/token を測る。AMD は sysfs の値で測る。アイドル電力も記録する。
6. **ばらつき**: 繰り返し回数、標準偏差、温度、クロックの状態を記録する。

## 題名の候補

- Recovering Latent Memory Bandwidth for LLM Inference on Legacy GPUs
- Modern LLM Inference on Pre-Modern GPU Compute: Architecture-Aware Kernels for OpenCL 1.1

## 投稿前の確認

- arXiv の初回投稿では、分野（cs.DC、cs.PF など）によって endorsement が必要になることがある。
- 再現性のため、投稿時にコードを公開するかどうかを決める（現在、リポジトリは非公開）。
- 検証機は 1 台で、各世代 1〜2 枚ずつ。一般化の主張はその範囲に留める。
