# ScrapGpuLibrary

**中古・旧世代の GPU（いわゆる「スクラップ GPU」）で、現行の ML ソフトウェアを実用速度で動かすための、汎用 tensor / runtime ライブラリです。**

ドライバも API も止まった GPU（NVIDIA Fermi / Kepler、古い Radeon など）は、現行の推論エンジンでは使えないか、使えても遅いことが多くあります。ScrapGpuLibrary は、そうした GPU のための runtime、tensor、op、量子化、バックエンドを、特定のエンジンに依存しない形で提供します。ggml（llama.cpp）、PyTorch、ONNX Runtime などは、その上の adapter として接続します。

```
ScrapGpuLibrary
├─ runtime    device / buffer / stream・event / allocator / multi-GPU transport
├─ tensor     dtype / layout / view・stride
├─ ops        matmul・matvec / softmax / norm / rope / attention / elementwise
├─ quant      q4・q8 / k-quant / device-specific packing
├─ backends   opencl11 / opencl12 / vulkan / cuda
└─ adapters   ggml / PyTorch extension / ONNX Runtime / standalone C API
```

各層の責務、依存の向き、設計の前提、移行計画は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) にまとめています。

## 現状

最初の成果は、OpenCL 1.1 だけで動く ggml backend「CL11」です。いまは 1 つの実装に全層が同居していて、`adapters/ggml/` に置いています。ここから runtime / tensor / ops / quant / backends/opencl11 を段階的に切り出します（ARCHITECTURE.md の P1〜P3）。

| 段階 | 内容 | 状態 |
| --- | --- | --- |
| P0 | CL11（ggml backend）の実装と GPU 別の最適化、ツリー外プラグイン化 | 済み |
| P1 | GPU 別の最適化の統合、性能の基準表と正しさの許容値の固定 | 済み（KL ダイバージェンスの許容値は、モデルと文章ごとに公式バックエンドで校正する） |
| P2 | runtime と backends/opencl11 の切り出し | 未着手 |
| P3 | tensor / ops / quant の切り出し、C API | 未着手 |
| P4 | multi-GPU transport | 未着手 |
| P5 | バックエンド（opencl12 / vulkan / cuda）と adapter（PyTorch / ONNX Runtime）の追加 | 未着手 |

### CL11

- OpenCL 1.1 のホスト API と OpenCL C 1.1 のカーネルだけで書いています。fp16 演算、サブグループ、OpenCL 1.2 以降の API は使いません。
- 検証した GPU は、NVIDIA 390 ドライバの GeForce GT 430（Fermi）・GT 710 / GT 730（Kepler）と、Mesa の Radeon Pro WX 2100（GCN、Clover と rusticl）・Radeon RX 6400（RDNA2、rusticl）です。調整値は GPU の世代ごとに `cl11_tuning_for()` に集めています（Fermi、Kepler、AMD GCN は測定した値、AMD RDNA2 は AMD 共通の設定だけ）。
- Llama / Qwen2 / Qwen3 系のモデルを全層オフロードできます（Flash Attention は非対応）。LFM2 / LFM2-MoE（短い畳み込み、`MUL_MAT_ID`）も全 op が載ります（Mac での確認まで。TB250 の GPU では未測定）。Q4_0 / Q8_0 / Q4_K / Q6_K の重みは GPU 向けの配置に並べ替えて持ちます。対応していない op は自動で CPU に回ります。
- 対応 op、性能、環境変数、制約、試して採用しなかった手法は [docs/CL11.md](docs/CL11.md) を参照してください。
- llama.cpp への組み込み方（ツリー外プラグイン、ツリー内パッチ）は [adapters/ggml/](adapters/ggml/) を参照してください。

llama-bench（`-ngl 99 -ub 128 -b 128 -p 128 -n 64`、CPU は同じマシンの Celeron G3930 の 2 スレッド。pp128 / tg64、単位 t/s）:

| モデル | GPU | CL11 pp128 | CL11 tg64 | CPU pp128 | CPU tg64 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q8_0 | GT 430 | 89.4 | 22.3 | 25.9 | 15.6 |
| | GT 730 | 241.9 | 34.9 | 25.9 | 15.9 |
| | GT 710 | 120.0 | 14.7 | 26.0 | 15.7 |
| Qwen3-0.6B-Base Q4_K_M | GT 430 | 89.9 | 23.7 | 6.1 | 4.6 |
| | GT 730 | 231.6 | 39.2 | 6.2 | 4.6 |
| | GT 710 | 115.6 | 17.8 | 6.2 | 4.6 |
| Qwen2-0.5B-ft Q4_0 | GT 430 | 118.5 | 36.5 | 47.5 | 25.8 |

Qwen3-1.7B Q4_K_M（1.2GB）は GT 710（2GB）で 29 層すべてをオフロードできます（pp128 39.7 / tg64 7.5、CPU は 2.0 / 1.6）。GT 710 は帯域が 11.7 GB/s しかなく、Q8_0 の生成は CPU より遅くなります（Q4_K_M なら速い）。Radeon は公式の Vulkan のほうが速いので、CL11 は使いません（[docs/HARDWARE.md](docs/HARDWARE.md)）。

## 検証機

BIOSTAR TB250-BTC PRO に、GeForce GT 430 / GT 710 / GT 730、Radeon Pro WX 2100、Radeon RX 6400 を PCIe x1 で載せた環境で開発しています。GPU ごとの帯域とドライバの既知の癖は [docs/HARDWARE.md](docs/HARDWARE.md) にあります。

## 出自

CL11 は [llama.cpp](https://github.com/ggml-org/llama.cpp) b11384（commit 0faee50）の上で開発しました。`adapters/ggml/` の初版は、その開発ブランチ（`fermi-cl11`、commit `9ab3f557c`）から取り込んだものです。

## ライセンス

MIT License（[LICENSE](LICENSE)）。CL11 は ggml / llama.cpp（MIT）のインターフェースと量子化形式の上に作られています。
