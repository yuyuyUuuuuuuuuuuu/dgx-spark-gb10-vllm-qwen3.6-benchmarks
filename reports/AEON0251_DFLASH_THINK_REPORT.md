# aeon 0.25.1 / DFlash / thinking 有効化の検証レポート

**環境**: GB10 / `nvidia/Qwen3.6-35B-A3B-NVFP4` / ベースライン = aeon イメージ vLLM 0.23.0 + Marlin + MTP num=3(`../docker-compose.mtp3.yml`)
**問い**: ①`ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-16-v0.25.1` にすると 35B-A3B で DFlash などが使えて速くなるか ②thinking を確実に ON にする方法はあるか
**ハーネス**: `vllm bench serve`(出力長 512 固定 / ignore_eos、単発 + 並行 4 × 自然文/短文/コード、`../bench-datasets/run_focus.sh`)。全 run 失敗 0。

---

## 結論

1. **速度**: 高速化の候補(b12x / TRT-LLM / SGLang / 0.25.1 + MTP3 / DFlash)はどれも**ベースライン(0.23.0 Marlin + MTP3)を超えなかった**。DFlash はコード単発だけ +14% だが、自然文・短文・並行では全敗(-5〜-25%)。**自然文の長文要約が主用途なら 0.23.0 Marlin + MTP3 のままが正解。**
2. **thinking ON**: 0.25.1 でも chat 経路では直らない(reasoning_content が空)。Qwen3.6 は chat template 経由だと空の think で即答するモデル挙動。**raw `/v1/completions` に `<think>` を種として注入するのが唯一確実な発火経路**。

---

## 速度: 同条件比較 [単発 decode tok/s (=1000/median TPOT) と acceptance%]

| ワークロード | conc | **MTP3(ベースライン)** | 0.25.1+MTP3 | DFlash |
|---|--:|--:|--:|--:|
| 自然文 | 1 | **97.6 (62%)** | 93.1 (61%) | 92.8 (21%) |
| 自然文 | 4 | **61.1 (63%)** | 58.2 (62%) | 50.5 (22%) |
| 短文 | 1 | **98.6 (61%)** | 92.4 (59%) | 86.5 (18%) |
| 短文 | 4 | **59.0 (59%)** | 56.0 (60%) | 44.1 (18%) |
| コード | 1 | 100.8 (67%) | 98.2 (68%) | **115.1 (29%)** |
| コード | 4 | **62.0 (68%)** | 54.2 (68%) | 56.2 (28%) |

- 列と構成の対応: MTP3 = `../docker-compose.mtp3.yml`(結果ラベル `mtp3`)、0.25.1+MTP3 = `../docker-compose.aeon0251-test.yml`(`aeon0251`)、DFlash = `../docker-compose.dflash-test.yml`(`dflash`、drafter は `zan/Qwen3.6-35B-A3B-DFlash-ja`)。
- **ベースラインの MTP3 が総合で最速**。0.25.1+MTP3 は全条件でわずかに遅い(-2.6〜-12.5%)。ただしこの 0.25.1 構成はベースラインの一部の設定が抜けていた。条件を揃えた再計測は下の「追記」。
- **DFlash が勝つのはコード単発のみ(+14%)**。他は全敗。acceptance が 18-29%(MTP3 の 59-68% よりずっと低い)で、num_spec=11 で大量に draft しても 7 割以上外れる。
- **DFlash の並行は大きく落ちる**(-9〜-25%): 非因果 draft の要件で **KV = BF16 が必須** → KV 容量が激減(max concurrency 4.58x@131K。ベースラインは 8.73x@262K)。並行負荷で BF16 KV のコストが表に出る。

## 見送った高速化手法

| 手法 | 判定 | 根拠 |
|---|---|---|
| **b12x** (native FP4 MoE) | 見送り | Marlin からは抜けるが +2.8% 止まりで、ベースライン以下。GB10 では Marlin が実質ベスト(vLLM #43906) |
| **TensorRT-LLM** | 見送り | GitHub #16075: Qwen3.6-35B が GB10 でロード時に即死(量子化段の IndexError)。安定版なし |
| **SGLang** (r0b0tlab) | 見送り | 見つかった唯一の実測が単発 57-61 tok/s = ベースラインの -40%。acceptance は高い(93%)が速度に繋がらない(GB10 の decode 自体がカーネル律速)。別スタック + 手作業のパッチが必要で、ソースも 1 件のみで未再現 |
| **0.25.1 MTP3** | 見送り | ベースライン 0.23.0 より全条件で遅い(公平版でも平均 -2.4%、下の追記)。attention(flashinfer)は壊れていないが速度の利点が無い |
| **DFlash** | 条件付き | コード単発のみ +14%、他は全敗。自然文が主の用途には不利 |

## thinking ON 問題

- **原因**: Qwen3.6(nvidia / heretic checkpoint)は **chat template 経由(enable_thinking:true)だと `<think>` を開いた直後に空の `</think>` を出して即答**するモデル挙動。parser(qwen3)も max_tokens も原因ではない。0.23.0 / 0.25.1 の両方で同じ。streaming / 非 streaming の両方で reasoning_content が空。
- **確実に思考させる方法**: raw `/v1/completions` に `<think>\n` を種として注入 → 思考プロセスの生成を確認(600 トークン思考が続いた)。
- クライアント側で thinking の Auto/ON/OFF を切り替えたい場合も、モデルは 1 つのままで共存できる。ただし ON は chat 経路では効かないので、raw completions 経路(chat template を自前で組み `<think>\n` を付ける)を挟む必要がある。

## 提言
- **0.23.0 Marlin + MTP3 のまま**(0.25.1 に移っても速度・thinking の利点は無い)。
- コード専用の高速経路が欲しい場合に限り、DFlash を別インスタンスで検討する余地がある(コード単発 +14%)。自然文には不利。
- thinking ON が必要なら raw completions 経路を使う。

## 追記: 0.25.1 の公平版再計測(生データからの集計)

上の 0.25.1 列(`aeon0251`)は、ベースラインの `VLLM_MARLIN_USE_ATOMIC_ADD=1`・`--attention-backend=flashinfer`・tool-call 系フラグが抜けた構成だった。
これらを揃えた `../docker-compose.aeon0251-fair.yml`(結果ラベル `aeon0251fair`、`run_matrix.sh` で conc 1/4/8)の結果を、
`../bench-datasets/results/` の生データから集計すると次のとおり(`../bench-datasets/compare_engines.py` と同じ計算)。

| ワークロード | conc | ベースライン 0.23.0 | 0.25.1 公平版 | 差 | 公平版 acceptance |
|---|--:|--:|--:|--:|--:|
| 自然文 | 1 | 97.6 | 95.4 | -2.3% | 61% |
| 自然文 | 4 | 61.1 | 60.1 | -1.6% | 62% |
| 自然文 | 8 | 44.0 | 43.2 | -1.7% | 61% |
| 短文 | 1 | 98.6 | 94.4 | -4.3% | 60% |
| 短文 | 4 | 59.0 | 57.4 | -2.7% | 58% |
| 短文 | 8 | 40.7 | 41.5 | +1.9% | 59% |
| コード | 1 | 100.8 | 101.6 | +0.8% | 68% |
| コード | 4 | 62.0 | 58.9 | -4.9% | 68% |
| コード | 8 | 43.3 | 40.4 | -6.6% | 67% |

- 単発 decode の平均差は -2.4%、集約 output throughput の平均差は -4.3%(9 条件)。差は縮んだが、結論(0.25.1 に移る利点は無い)は変わらない。
- `--no-enable-flashinfer-autotune` を足した `../docker-compose.aeon0251-noauto.yml`(`aeon0251noauto`)は 360 リクエスト中 194 件が失敗した(conc=8 は全ワークロードで全失敗、コード conc=4 も全失敗、短文 conc=4 は 50 件中 34 件失敗)。比較には使えない。`compare3.py` はこの 3 者を並べる集計スクリプト。

## 再現
- compose: `../docker-compose.aeon0251-test.yml` / `../docker-compose.aeon0251-fair.yml` / `../docker-compose.aeon0251-noauto.yml` / `../docker-compose.dflash-test.yml`。ベースラインは `../docker-compose.mtp3.yml`。
- 生データ: `../bench-datasets/results/{mtp3,aeon0251,aeon0251fair,aeon0251noauto,dflash}__*.json`。
- 関連: `MTP_BENCHMARK_FULL.md`、`B12X_EXPERIMENT_REPORT.md`。
