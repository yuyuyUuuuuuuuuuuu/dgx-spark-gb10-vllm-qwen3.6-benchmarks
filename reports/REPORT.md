# Qwen3.6-35B-A3B on DGX Spark (GB10) — 構築と最適化のノート(FP8 → NVFP4)

- **対象:** NVIDIA DGX Spark (GB10, SM121a / ドライバ 580.159.03 / 統合メモリ 128GB)
- **用途の前提:** 単一ユーザの常駐推論(同時リクエストは少ない)
- **構成ファイル:** FP8 = `../docker-compose.yml`、NVFP4 = `../docker-compose.nvfp4.yml`

このノートは 2 段階の構成を扱う。前半が NVFP4 構成(vLLM 0.23.0 系のサードパーティ SM121 ビルド)、
後半が最初に組んだ FP8 構成(公式 `cu130-nightly`、当時 v0.19.2 系)。その後の MTP・カーネル・エンジン比較は
`MTP_TUNING_REPORT.md` / `MTP_BENCHMARK_FULL.md` / `B12X_EXPERIMENT_REPORT.md` / `AEON0251_DFLASH_THINK_REPORT.md`。

---

## 1. NVFP4 構成(FP8 からの移行)

### 構成
| 項目 | 値 |
|---|---|
| イメージ | `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix` (40.7GB, vLLM 0.23.0 SM121 ビルド, サードパーティ) |
| モデル | `nvidia/Qwen3.6-35B-A3B-NVFP4` (公式 NVFP4 チェックポイント) |
| 投機 | MTP num_speculative_tokens=**1**(この節の計測で最速) |
| weight | **21.93 GiB** (FP8 の 35GiB から −13GiB) |
| 速度 | **~78 t/s** (日本語生成, thinking 無効, FP8 比 約2倍) |
| OCR/vision | 動作確認済み(画像内テキストを正しく読み取れた) |
| max-num-seqs | 8 / KV 並列 25.55x |

> この節の MTP スイープは `--async-scheduling` なしで、単一プロンプトの生成速度を見たもの。
> `--async-scheduling` を有効にし `vllm bench serve` で測り直すと num=3 が最速になった(`MTP_BENCHMARK_FULL.md`)。
> 条件が 2 点(async-scheduling の有無と計測方法)で違うため、どちらが効いたかは切り分けていない。
> 現在の `docker-compose.nvfp4.yml` は num=3 + async-scheduling。

### 実測比較
**量子化 (MTP num=3, 同一プロンプト):**
| | weight | 速度 | KV並列 | OCR |
|---|---|---|---|---|
| FP8 (公式 img cu130-nightly) | 35.0GiB | 37-40 t/s | 18.4x | OK |
| **NVFP4 (aeon img 0.23.0)** | **21.9GiB** | **~63 t/s** | 25-35x | OK |

**MTP num スイープ (NVFP4, 日本語要約, async-scheduling なし):**
| num | 速度 | 受理長 | 受理率 |
|---|---|---|---|
| 3 | ~63 t/s | 1.9 | 31% |
| 2 | ~73 t/s | 1.8 | 42% |
| **1** | **~78 t/s** | 1.62 | **62%** |

→ FP8 構成では num=3 が最適だったが、この条件の NVFP4 × 0.23 では **num=1 が最速**(後続トークンの受理率が低く、ドラフトが無駄になるため)。

### DFlash-ja 検証結果: 不採用
`zan/Qwen3.6-35B-A3B-DFlash-ja`(905MB)をドラフタとして試した。
- 日本語の自然文(要約)では受理率 **9-10%** と極端に低く、num=5 で ~20 t/s と MTP の 1/3 に低下。
- DFlash-ja は reasoning/コード寄り + bf16 で訓練 → NVFP4 で推論、という量子化のミスマッチが原因と考えられる。
- **日本語要約用途では MTP が明確に優位**。

### 注意
- イメージはサードパーティ製。使う前に中身(egress・privilege・公開ポート・env)を確認し、固定タグで参照する。`:latest` は使わない。
- モデルはイメージ同梱の uncensored 版ではなく、公式 `nvidia/...-NVFP4` を読ませる。
- NVFP4 のロードは公式 vLLM(当時の nightly)では `KeyError: w2_input_scale`(#44081)で失敗する。aeon イメージでは読める。
- FP8 構成に戻すときは `docker compose -f docker-compose.nvfp4.yml down` の後に `docker compose up -d`(`docker-compose.yml`)。
  `.env` の `MAX_NUM_SEQS` などは FP8 構成の値(§2.5)に合わせる。

---

## 2. FP8 構成(初期構成)

### 2.1 結論サマリ

| 項目 | 値 |
|---|---|
| モデル | `Qwen/Qwen3.6-35B-A3B-FP8` (FP8 block-128) |
| 推論エンジン | vLLM `cu130-nightly` (v0.19.2 系) |
| エンドポイント | `http://127.0.0.1:8000/v1/chat/completions` |
| 公開モデル名 (API `model` フィールド) | `Qwen/Qwen3.6-35B-A3B` |
| コンテキスト長 | 131,072 (131K) |
| 並列上限 | max-num-seqs=4 (KV 的には 18.29 並列まで可能) |
| KVキャッシュ | fp8, 31.68GB = 754,688 トークン |
| MTP (投機デコード) | num_speculative_tokens=3, moe_backend=triton |
| decode 速度 | コード 64 / 構造化 46 tok/s |
| GPU 確保メモリ | 69.4 GB (util=0.60) / システム空き 36GB |
| 精度の簡易確認 | 17×23=391 正解、素数判定(6k±1 最適化)のコードを正答 |

### 2.2 速度: GB10 で「150 tok/s」級が出ない理由

「150 tok/s」級の数字は **RTX 4090 / RTX 6000 クラスの数字**で、GB10 の FP8 構成では届かない。

#### 根本原因: メモリ帯域律速

decode は 1 トークンごとにアクティブ重み (A3B = FP8 で約 3-4GB) をメモリから読み出す **メモリ帯域律速** のワークロード。

| GPU | メモリ帯域 | 単一ユーザ decode |
|---|---|---|
| RTX 4090 | ~1000 GB/s | 120+ tok/s |
| RTX 6000 Blackwell | ~1800 GB/s | 200-240 tok/s |
| **DGX Spark GB10** | **~273 GB/s** (LPDDR5x 統合) | 理論 91 / 公式 28-30 / 本構成 46-64 (MTP 込み) |

#### 推論中の GPU プロファイル(nvidia-smi の読み値)

- GPU util: **94%**
- SM クロック: **2424 MHz** (最大ブースト)
- 電力: **33W**

コアは動いているのに電力が低い = 演算は余っていてメモリ読み出しを待っている、という帯域律速の症状と読んだ。

#### 評価

NVIDIA 公式ベンチ「GB10 FP8 単一ユーザ = 28-30 tok/s」に対し、**この構成は MTP 適用で 46-64 tok/s = 約 2 倍**。
MTP は 1 回の重み読み出しで複数トークンを投機的に確定させるので、帯域の制約を越えられる手段になる。
FP8 のままでは設定変更で大きく伸ばす余地は小さかった(その後、読み出す重みが小さい NVFP4 + MTP で
単発 decode 98〜101 tok/s まで伸びた。`MTP_BENCHMARK_FULL.md`)。

### 2.3 チューニング

#### 量子化: NVFP4 → FP8(当時は必須の切り替え)

`nvidia/Qwen3.6-35B-A3B-NVFP4` は当時の vLLM nightly の回帰バグでロードできなかった:

```
KeyError: 'layers.0.mlp.experts.w2_input_scale'
```

既知 issue [#44081](https://github.com/vllm-project/vllm/issues/44081) (v0.22.0 で破損) / [#38980](https://github.com/vllm-project/vllm/issues/38980) (GB10 で scale key 欠落)。
ModelOpt の NVFP4 スケールキーの命名と vLLM の Qwen3 MoE ローダーの期待が一致しない。**設定では直せないため FP8 版を採用した**(後に §1 の aeon イメージで解決)。
→ FP8 は `--quantization` フラグ不要(量子化はモデルに内蔵。付けると逆にエラーを誘発する)。

#### MTP num_speculative_tokens

| num | 平均 tok/s | 備考 |
|---|---|---|
| なし | ~44 | ベースライン |
| 2 | 48.5 | acceptance は高いが先読みが少ない |
| **3** | **51.1** | **最適** |
| 4 | 44.0 | 4 番目のドラフトが当たらず(acc 56%)オーバーヘッド負け |

per-position acceptance: 0.79 / 0.62 / 0.40 → 3 番目までが有効。**num=3 に決定。**
`moe_backend=triton` が必須(MTP ドラフタは unquantized MoE で、marlin/cutlass 非対応)。

#### メモリと 131K コンテキスト

| | util 0.85 (当初) | util 0.40 | **util 0.60 (確定)** |
|---|---|---|---|
| max_model_len | 65K | 65K | **131K** |
| vLLM GPU 確保 | 99.9 GB | 44.9 GB | **69.4 GB** |
| KV キャッシュ | 61 GB | 7.77 GB | **31.68 GB (75万トークン)** |
| 並列上限(KV) | 80x | 7.5x | **18.29x** |
| システム空き | 1 GB | 62 GB | **36 GB** |
| decode 速度 | 51 | 50.5 | **64 (不変)** |

当初の util=0.85 は KV キャッシュを 61GB(必要量の 20 倍超)確保し、システム空きが 1GB だった(過剰確保)。
util=0.60 + 131K に調整した結果、**速度を落とさずにコンテキストを倍増、KV 75 万トークン = 18 並列分の余裕、
システム側に 36GB を残せた**(同じ機械で他のプロセスを動かせる)。

#### KV 容量・コンテキスト長・並列数の関係

- KV 容量(トークン) ≒ max_model_len × 並列数。実測単価 **約 23,700 トークン/GB**。
- 必要 KV = max_model_len × 並列数 ÷ 23,700 [GB]。
- max_model_len を上げると同じ KV で並列上限が下がる。ただしこれは「全リクエストが同時にフル長を使う最悪ケース」の話で、
  実際には各リクエストが使った分だけ消費する。
- 131K で 18 並列の余裕があるので、`max-num-seqs` は 4 → 8〜16 に上げる余地がある(クライアント側の同時リクエスト数次第)。

### 2.4 vLLM serve フラグ(FP8 構成の全フラグ)

```
serve Qwen/Qwen3.6-35B-A3B-FP8
  --served-model-name=Qwen/Qwen3.6-35B-A3B
  --tensor-parallel-size=1
  --max-model-len=131072
  --max-num-seqs=4
  --gpu-memory-utilization=0.60
  --max-num-batched-tokens=8192
  --trust-remote-code
  --host=0.0.0.0 --port=8000
  --dtype=auto
  --kv-cache-dtype=fp8
  --attention-backend=flashinfer
  --enable-chunked-prefill
  --enable-prefix-caching
  --reasoning-parser=qwen3
  --speculative-config={"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}
```

環境変数: `CUTE_DSL_ARCH=sm_121a`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, `MAX_JOBS=1`, `FLASHINFER_JIT_MAX_PARALLEL_COMPILE=1`(JIT 並列 1 でホスト RAM の OOM を回避)

### 2.5 .env(FP8 構成で使った値)

```
HOST_PORT=8000
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=4
GPU_MEMORY_UTILIZATION=0.60
MAX_NUM_BATCHED_TOKENS=8192
```

### 2.6 起動と確認

```bash
# リポジトリ直下で(.env を用意してから)
docker compose up -d
curl -s http://127.0.0.1:8000/health
docker logs vllm-qwen3.6 --tail 20

# 動作テスト
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"17×23は?"}],"max_tokens":50}'

docker compose down
```

- 初回起動: モデルロード ~4 分 + コンパイル/CUDA グラフ ~2 分(合計 ~6-7 分)。
- **起動完了(health 200)まで推論リクエストを送らないこと。** コンパイル中に負荷をかけると EngineCore がハングした事例がある。
- ポートは `127.0.0.1` にだけバインドしている。
