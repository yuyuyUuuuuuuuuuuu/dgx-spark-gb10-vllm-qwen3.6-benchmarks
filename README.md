# Qwen3.6-35B-A3B on NVIDIA DGX Spark (GB10) with vLLM: benchmarks and notes

*English summary: measured on one DGX Spark (GB10, sm_121, TP=1): FP8 vs NVFP4, MTP and DFlash speculative decoding, Marlin vs FlashInfer NVFP4 MoE kernels, vLLM 0.23.0 vs 0.25.1, enabling thinking, and a DFlash startup assert on hybrid (GDN + full-attention) targets. The notes below are in Japanese.*

DGX Spark(GB10)の vLLM ベンチマーク。結論の例: NVFP4 は FP8 より速く 13GiB 軽い(MTP num=3 で FP8 37-40 t/s → NVFP4 ~63 t/s)。MTP は `--async-scheduling` ありで num=3 が最速(単発 98〜101 tok/s)。MoE カーネルを b12x にしても Marlin 比 +2.8% に留まる。

| finding (one DGX Spark / GB10, TP=1, vLLM) | number |
|---|---|
| NVFP4 vs FP8, same prompt, MTP num=3 | FP8 37-40 tok/s -> NVFP4 ~63 tok/s (about 1.6x); weights 35.0 GiB -> 21.9 GiB |
| MTP `num_speculative_tokens` | with `--async-scheduling`, num=3 is fastest: single stream 98-101 tok/s (+12-16% vs num=1); acceptance about 83/72/63% for num=1/2/3 |
| NVFP4 MoE kernel | Marlin 75.3 -> FlashInfer b12x 77.4 tok/s (+2.8%), matched A/B without MTP |
| scope | all numbers are from one GB10; not directly comparable with other hardware |

NVIDIA DGX Spark(GB10, sm_121)上で vLLM により **Qwen3.6-35B-A3B** を動かしたときの計測ノートと道具です。
単一ユーザ・単一ノード(TP=1)の推論を前提に、次の問いを実測で確かめています。

- FP8 と NVFP4 のどちらを使うか(メモリと速度)
- MTP 投機デコードの `num_speculative_tokens` はいくつが速いか(単発と並行で結論が変わるか)
- NVFP4 MoE のカーネル(Marlin と flashinfer b12x)で速度は変わるか
- vLLM 0.23.0 → 0.25.1 の更新や DFlash 投機で速くなるか
- thinking を確実に有効化する方法
- DFlash の外部 drafter を hybrid(GDN + full-attention)target に載せたときの起動時 assert の原因と対処

すべて GB10 1 台での計測です。他のハードウェアの数字とは直接比べられません。

## 前提

### ハードウェア

| 項目 | 値 |
|---|---|
| 機械 | NVIDIA DGX Spark(GB10, SM121a) |
| メモリ | 統合メモリ 128GB(LPDDR5x, 約 273 GB/s) |
| ドライバ | 580.159.03 |
| 構成 | 1 台, tensor-parallel-size=1 |

### ソフトウェア

Docker + NVIDIA Container Toolkit + Docker Compose v2 で、vLLM の OpenAI 互換サーバをコンテナとして起動します。

| イメージ | vLLM | 使いどころ |
|---|---|---|
| `vllm/vllm-openai:cu130-nightly` | 当時 v0.19.2 系(nightly なので動くタグ) | FP8 構成 |
| `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix` | 0.23.0(SM121 ビルド) | NVFP4 のベースライン、MTP 比較 |
| `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0` | 0.24.0 | DFlash(z-lab drafter)の起動調査 |
| `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-16-v0.25.1` | 0.25.1 | 0.25.1 比較、DFlash 比較 |
| `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | 0.25.2 + flashinfer 0.6.15 | b12x vs Marlin |

aeon / anemll のイメージはサードパーティ製です。使う前に中身を確認し、固定タグで参照してください。
タグ名の日付はイメージ側の命名です。

### モデル

| モデル | 用途 |
|---|---|
| `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 構成 |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | NVFP4 のベースライン(MoE experts だけ NVFP4、linear 層は FP8 の MIXED_PRECISION) |
| `unsloth/Qwen3.6-35B-A3B-NVFP4` | b12x の検証(MoE が FP8 のため b12x では起動しない) |
| `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` | DFlash 起動調査の body |
| `zan/Qwen3.6-35B-A3B-DFlash-ja` | DFlash drafter(日本語向け) |
| z-lab の Qwen3.6-35B-A3B 用 DFlash drafter | DFlash 起動調査の drafter |
| `unsloth/Qwen3.6-27B-NVFP4` | 27B の起動構成(計測なし) |

HF キャッシュ(`HF_CACHE_DIR`、既定 `~/.cache/huggingface`)にあるモデルは名前で、それ以外は
`MODELS_DIR`(既定 `~/models`)に置いたディレクトリを compose がマウントします。

## 主な結論

1. **NVFP4 は FP8 より速く、13GiB 軽い。** 同一プロンプト・MTP num=3 で FP8 37-40 t/s → NVFP4 ~63 t/s(約 1.6 倍。NVFP4 側を num=1 にすると ~78 t/s で約 2 倍)、
   weight 35.0GiB → 21.9GiB。公式 vLLM(当時の nightly)は NVFP4 checkpoint を読めず(`KeyError: w2_input_scale`, vLLM #44081)、
   SM121 ビルドのイメージが必要だった。([reports/REPORT.md](reports/REPORT.md))
2. **MTP は `--async-scheduling` ありなら num=3 が最速。** `vllm bench serve`(出力 512 固定)で num=1 比、
   単発 decode +12〜16%(98〜101 tok/s)、conc=4 でも +11〜16%。**num=2 は conc=4 で num=1 より 6〜11% 遅い**という谷がある。
   acceptance は num=1/2/3 で約 83/72/63%。`--async-scheduling` なしの初期計測では num=1 が最速だった。
   ([reports/MTP_BENCHMARK_FULL.md](reports/MTP_BENCHMARK_FULL.md), [reports/MTP_TUNING_REPORT.md](reports/MTP_TUNING_REPORT.md))
3. **MoE カーネルを b12x にしても速くならない。** 条件を揃えた A/B(MTP なし, Triton attention)で Marlin 75.3 → b12x 77.4 tok/s(+2.8%)。
   この checkpoint では NVFP4 なのが MoE experts だけで、active 3B の decode に占める MoE GEMM の割合が小さい。
   MTP と組み合わせると b12x 側は Triton attention を強いられ、conc=4 では全条件で Marlin より遅い(MTP num=3 同士で 4〜11%)。
   ([reports/B12X_EXPERIMENT_REPORT.md](reports/B12X_EXPERIMENT_REPORT.md), [reports/b12x-vs-marlin-report.html](reports/b12x-vs-marlin-report.html))
4. **vLLM 0.25.1(aeon イメージ)は 0.23.0 より速くならない。** 条件を揃えた再計測でも単発 decode の平均は -2.4%。
   ([reports/AEON0251_DFLASH_THINK_REPORT.md](reports/AEON0251_DFLASH_THINK_REPORT.md) の追記)
5. **DFlash は用途を選ぶ。** DFlash-ja drafter はコード単発だけ +14%(115.1 tok/s)、自然文・短文・並行では -5〜-25%。
   非因果 drafter のため KV を BF16 にする必要があり、並行時の KV 容量が大きく減る。日本語要約では受理率 9-10% の計測もある。
   ([reports/AEON0251_DFLASH_THINK_REPORT.md](reports/AEON0251_DFLASH_THINK_REPORT.md))
6. **thinking は chat 経路では有効にならない。** `enable_thinking:true` でも空の `<think></think>` で即答する(0.23.0 / 0.25.1 とも)。
   raw `/v1/completions` に `<think>\n` を種として付けると思考が生成される。
7. **FP8 構成の要点。** GB10 の decode はメモリ帯域律速(約 273 GB/s)。FP8 + MTP num=3 で 46-64 tok/s(公式の GB10 FP8 単一ユーザ 28-30 tok/s の約 2 倍)。
   `gpu-memory-utilization` 0.85 → 0.60 で速度を変えずに 131K コンテキスト・KV 75 万トークンを確保し、36GB をシステムに残せた。
8. **DFlash 外部 drafter の起動時 assert。** z-lab drafter を hybrid target に載せると `unify_kv_cache_spec_page_size` の assert で起動しない。
   原因(drafter の page サイズが target の page と非整数比で、pad パスに入るための `indexes_kv_by_block_stride` が立っていない)と、
   診断・パッチのスクリプトを [patches/](patches/) に置いた。v1 パッチでは起動しなかった。v2 パッチと heretic body での結果は記録されていない。

## ディレクトリ構成

```
.
├── README.md / LICENSE / env.example / .gitignore
├── docker-compose*.yml        # 構成ごとの compose(下の表)
├── reports/                   # 計測レポート(Markdown)と図表(HTML)
├── bench-datasets/            # ベンチのランナー・集計・データセット・生データ
│   ├── run_matrix.sh          # conc 1/4/8 × 3 ワークロード
│   ├── run_focus.sh           # conc 1/4 × 3 ワークロード
│   ├── aggregate.py           # mtp1/2/3 の集計 → results/_aggregate.json
│   ├── compare_engines.py     # mtp3 vs aeon0251fair
│   ├── compare3.py            # mtp3 vs aeon0251fair vs aeon0251noauto
│   ├── make_code_bcb.py       # コード用データセットの再生成
│   ├── ja_lecture.jsonl       # 自作の日本語プロンプト 30 件
│   └── results/               # vllm bench serve の生 JSON
└── patches/                   # DFlash page_size 問題の診断・パッチ(コンテナ内で起動前に実行)
```

## レポート

| ファイル | 内容 |
|---|---|
| [reports/REPORT.md](reports/REPORT.md) | FP8 構成の構築とチューニング、帯域律速の話、NVFP4 への移行、DFlash-ja の初期検証 |
| [reports/MTP_TUNING_REPORT.md](reports/MTP_TUNING_REPORT.md) | MTP num=1/2/3 の単発計測(後の並行計測で推奨が覆った経緯を含む) |
| [reports/MTP_BENCHMARK_FULL.md](reports/MTP_BENCHMARK_FULL.md) | MTP num=1/2/3 × 3 ワークロード × conc 1/4/8 の `vllm bench serve` 計測 |
| [reports/mtp-benchmark-report.html](reports/mtp-benchmark-report.html) | 上の図表(ブラウザで開く) |
| [reports/B12X_EXPERIMENT_REPORT.md](reports/B12X_EXPERIMENT_REPORT.md) | b12x vs Marlin の A/B(MTP なし) |
| [reports/b12x-vs-marlin-report.html](reports/b12x-vs-marlin-report.html) | b12x vs Marlin × MTP num=1/2/3(ブラウザで開く) |
| [reports/AEON0251_DFLASH_THINK_REPORT.md](reports/AEON0251_DFLASH_THINK_REPORT.md) | vLLM 0.25.1、DFlash、thinking 有効化、見送った手法 |

HTML は GitHub 上では描画されないので、ローカルでブラウザから開いてください(グラフは JavaScript で描画)。

## 構成(compose)と結果の対応

結果ラベルは `bench-datasets/results/<label>__<workload>__c<conc>.json` の `<label>` です。

| compose | イメージ | 内容 | 結果 |
|---|---|---|---|
| `docker-compose.yml` | cu130-nightly | FP8 + MTP num=3(初期構成、`.env` を使う) | REPORT.md §2 |
| `docker-compose.nvfp4.yml` | aeon 0.23.0 | ベースライン: NVFP4 + MTP num=3 + async-scheduling(`.env` を使う) | mtp3.yml と同等の構成で計測 |
| `docker-compose.nomtp.yml` | aeon 0.23.0 | NVFP4、MTP なし | ファイル内コメントに数値 |
| `docker-compose.mtp1.yml` / `mtp2.yml` / `mtp3.yml` | aeon 0.23.0 | MTP num=1/2/3 | `mtp1` `mtp2` `mtp3` |
| `docker-compose.marlin-test.yml` / `b12x-test.yml` | anemll | Marlin / b12x の A/B(MTP なし) | B12X_EXPERIMENT_REPORT.md |
| `docker-compose.unsloth-b12x.yml` | anemll | unsloth ckpt + b12x(起動しない記録) | B12X_EXPERIMENT_REPORT.md |
| `docker-compose.b12x-mtp1.yml` / `-mtp2` / `-mtp3` | anemll | b12x + MTP num=1/2/3 | `b12xmtp1` `b12xmtp2` `b12xmtp3` |
| `docker-compose.aeon0251-test.yml` | aeon 0.25.1 | 0.25.1 + MTP num=3(初回、一部設定が抜けている) | `aeon0251` |
| `docker-compose.aeon0251-fair.yml` | aeon 0.25.1 | 0.25.1 + MTP num=3(ベースラインと条件を揃えた版) | `aeon0251fair` |
| `docker-compose.aeon0251-noauto.yml` | aeon 0.25.1 | 上 + flashinfer autotune 無効 | `aeon0251noauto`(360 件中 194 件失敗) |
| `docker-compose.dflash-test.yml` | aeon 0.25.1 | DFlash(DFlash-ja drafter, num_spec=11, BF16 KV) | `dflash` |
| `docker-compose.dflash-triton-bf16.yml` / `-fp8.yml` | aeon 0.25.1 | DFlash + TRITON_ATTN(ポート 8001) | 記録なし |
| `docker-compose.dflash-repro-v024.yml` | aeon 0.24.0 | GB10 のコミュニティ投稿の再現(z-lab drafter, fp8 KV, TRITON_ATTN) | 起動せず(profile_cudagraph_memory の assert) |
| `docker-compose.dflash-official.yml` | aeon 0.24.0 | イメージ作者の公式レシピ(nvidia body) | 起動せず(page_size assert) |
| `docker-compose.dflash-fixed.yml` | aeon 0.24.0 | 上 + `patches/patch-dflash-pagesize.sh`(v1) | 起動せず |
| `docker-compose.dflash-heretic.yml` | aeon 0.24.0 | heretic body、パッチなし | 記録なし |
| `docker-compose.dflash-v2.yml` | aeon 0.24.0 | heretic body + 診断 + v2 パッチ | 記録なし |
| `docker-compose.dflash-v023.yml` | aeon 0.23.0 | heretic body、パッチなし | 記録なし |
| `docker-compose.27b.yml` | aeon 0.23.0 | Qwen3.6-27B NVFP4 の起動構成 | 記録なし |

ポートは `127.0.0.1:8000`(`.env` を使う構成は `HOST_PORT`)が基本で、`dflash-triton-*`, `dflash-repro-v024`, `dflash-official`,
`dflash-fixed`, `dflash-heretic`, `dflash-v2`, `dflash-v023` だけが `127.0.0.1:8001` です。同じポートの構成は同時に起動できないので、
1 つずつ `docker compose -f <file> up -d` / `down` してください。flashinfer / vLLM のキャッシュはリポジトリ直下の `.cache/` にできます。

## ベンチの動かし方

1. **変数を用意する。**
   ```bash
   cp env.example .env   # 値を確認・調整
   ```
2. **ベンチ用イメージを作る。** ランナーは `vllm bench serve` を持つイメージ(既定名 `vllm-bench:local`)を使います。
   計測では、サーバと同じイメージに pandas を足したものを使いました(custom データセットのローダが pandas を要する)。例:
   ```bash
   docker build -t vllm-bench:local - <<'DOCKERFILE'
   FROM ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix
   RUN pip install pandas
   DOCKERFILE
   ```
3. **コード用データセットを作る。** `bench-datasets/code_bcb.jsonl` は同梱していません(下の「データセット」)。
   ```bash
   python3 bench-datasets/make_code_bcb.py --split <BigCodeBench のリリース名>   # 要 datasets パッケージ
   ```
4. **サーバを起動し、health が 200 になるまで待つ。** 初回はモデルロードとコンパイルで数分〜十数分かかります。
   起動中に推論リクエストを送らないでください(コンパイル中の負荷で EngineCore がハングした例があります)。
   ```bash
   docker compose -f docker-compose.mtp3.yml up -d
   curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
   ```
5. **ベンチを流す。** ラベルは結果ファイル名になります。既存のラベルを使うと同梱の結果を上書きするので、新しいラベルを付けてください。
   ```bash
   bench-datasets/run_matrix.sh my-mtp3     # conc 1/4/8(9 run)
   bench-datasets/run_focus.sh  my-mtp3     # conc 1/4 だけ(6 run)
   ```
   変数 `BENCH_IMAGE` / `BASE_URL` / `MODEL` / `OPENAI_API_KEY` で既定値を変えられます(`env.example` の末尾)。
6. **集計する。**
   ```bash
   python3 bench-datasets/aggregate.py        # ラベル mtp1/mtp2/mtp3 を表にし results/_aggregate.json を書き直す
   python3 bench-datasets/compare_engines.py  # mtp3 vs aeon0251fair
   python3 bench-datasets/compare3.py         # mtp3 vs aeon0251fair vs aeon0251noauto
   ```
   集計スクリプトは比較するラベルがコード内に固定されています。自分のラベルと比べるときはスクリプト冒頭のラベルを書き換えてください。
7. `docker compose -f docker-compose.mtp3.yml down` で止めます。

指標: 「単発 decode tok/s」は `1000 / median TPOT`(prefill を除いた 1 リクエストあたりの decode 速度)、
「集約 output throughput」はサーバ全体の出力トークン/秒です。どのワークロードも出力長を 512 に固定(`--ignore-eos`)しています。

`MTP_TUNING_REPORT.md` と `B12X_EXPERIMENT_REPORT.md` の単発計測(completion_tokens / 経過時間)に使ったスクリプトは含まれていません。

## データセット

| ワークロード | ファイル | 件数 | 内容 |
|---|---|---|---|
| 自然文 `ja` | `bench-datasets/ja_lecture.jsonl` | 30 | 自作。「大学の講義を担当する教員として、機械学習のテーマを初学者向けに説明せよ」という日本語プロンプト(テーマだけが違う) |
| 短文 `short` | なし(vLLM の random データセット) | 50 | 入力 128 トークン |
| コード `code` | `bench-datasets/code_bcb.jsonl`(同梱なし) | 40 | BigCodeBench の BigCodeBench/0〜39 の `instruct_prompt` |

`code_bcb.jsonl` は BigCodeBench(Apache-2.0)のプロンプトをそのまま使ったもので、その中には公開テスト用 FTP サーバの
既定の資格情報がタスク文として含まれます。秘密情報の検出に引っかかるのを避けるため同梱せず、
`bench-datasets/make_code_bcb.py` で再生成する形にしました。計測に使ったファイルの SHA-256 はスクリプトに書いてあり、
生成後に一致するかが表示されます(どの BigCodeBench リリースから取ったかは記録されていません)。

`bench-datasets/results/*.json` は `vllm bench serve --save-result` の出力そのままです(実行時刻の `date` フィールドを含む)。

## 注意

- 数字はこの 1 台・このドライバ・これらのイメージでのものです。`cu130-nightly` は動くタグなので、同じタグでも中身が変わります。
- GPU メモリは多くの構成で `gpu-memory-utilization=0.44` に固定して比較しています(DFlash 版の一部は 0.50、FP8 構成は 0.60、27B は 0.85)。
- `patches/` のスクリプトはコンテナ内の vLLM のソース(`/usr/local/lib/python3.12/site-packages/vllm/...`)を書き換えます。
  compose からマウントして起動前に実行する前提で、イメージそのものは変更しません。対象の行が見つからなければ何もせず非 0 で終了します。
- ポートは `127.0.0.1` にだけバインドしています。外部に公開する場合は `VLLM_API_KEY` などで保護してください。

## ライセンス

このリポジトリのコードと文書は MIT License([LICENSE](LICENSE))です。
BigCodeBench 由来のプロンプト(再生成して使う `code_bcb.jsonl`)は BigCodeBench のライセンス(Apache-2.0)に従います。
モデルとコンテナイメージはそれぞれの配布元のライセンスに従います。
