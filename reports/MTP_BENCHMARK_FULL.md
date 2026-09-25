# Qwen3.6-35B-A3B GB10: MTP num=1/2/3 ベンチ(vllm bench serve)

**環境**: GB10 / vLLM 0.23.0(aeon イメージ)/ `nvidia/Qwen3.6-35B-A3B-NVFP4` / `--async-scheduling` あり
**ハーネス**: `vllm bench serve`(サーバと同じイメージに同梱)。出力長 512 固定(`--ignore-eos`)で揃えた。
**規模**: 3 構成(num=1/2/3)× 3 ワークロード(自然文=講義解説プロンプト 30 / 短文=random 50 / コード=BigCodeBench 40)× 3 並行度(1/4/8)= **27 run / 1080 リクエスト / 失敗 0**。
**指標**: median TPOT(prefill を除いた純 decode)から算出した単発 decode tok/s、集約 output throughput、spec-decode acceptance。
**図表**: `mtp-benchmark-report.html`(同じデータのグラフ)
**サーバ設定**: `MAX_MODEL_LEN=262144` / `MAX_NUM_SEQS=8` / `GPU_MEMORY_UTILIZATION=0.44` / `--max-num-batched-tokens=8192`(`../env.example`)

---

## 結論

- **単発(直列)なら num を上げるほど速い**: num=2 +11〜13%、num=3 +12〜16%。品質はロスレス(投機採択のみ)。
- **中並行(conc=4)で num=2 が全ワークロードで逆転(−6〜11%)。num=3 は逆に改善(+11〜16%)。** = **非単調**。単発の A/B(`MTP_TUNING_REPORT.md`)だけでは見えなかった谷。
- **高並行(conc=8)** では差が縮小、num=2/3 とも num=1 を僅かに上回る程度。
- **acceptance は num 増で単調低下**(num=1 ~83% → num=2 ~72% → num=3 ~63%)。**ワークロード順は コード > 自然文 > 短文**で一貫。
- **推奨: 用途で分岐**。長文要約のような**直列・単発主体**なら **num=3**(全並行度で num=1 以上。単発と conc=4 では最速、conc=8 では num=2 とほぼ同じ)。並行が読めない環境は num=1 のままが無難(**num=2 は中並行で最弱なので非推奨**)。

## 単発あたり純decode速度 [tok/s]  (=1000/median TPOT)

| workload | conc | num=1 | num=2 | num=3 | 2vs1 | 3vs1 |
|---|--:|--:|--:|--:|--:|--:|
| 自然文 | 1 | 87.1 | 96.7 | 97.6 | +11% | +12% |
| 自然文 | 4 | 55.2 | 49.3 | 61.1 | **−11%** | +11% |
| 自然文 | 8 | 41.4 | 44.7 | 44.0 | +8% | +6% |
| 短文 | 1 | 86.7 | 95.7 | 98.6 | +10% | +14% |
| 短文 | 4 | 52.1 | 48.9 | 59.0 | **−6%** | +13% |
| 短文 | 8 | 39.0 | 42.4 | 40.7 | +9% | +4% |
| コード | 1 | 88.2 | 99.8 | 100.8 | +13% | +14% |
| コード | 4 | 53.3 | 48.0 | 62.0 | **−10%** | +16% |
| コード | 8 | 39.7 | 43.7 | 43.3 | +10% | +9% |

## 集約 output throughput [tok/s]  (サーバ全体)

| workload | conc | num=1 | num=2 | num=3 |
|---|--:|--:|--:|--:|
| 自然文 | 4 | 208.7 | 189.4 | **225.2** |
| 自然文 | 8 | 304.7 | 306.5 | **325.8** |
| コード | 4 | 206.8 | 188.4 | **235.8** |
| コード | 8 | 307.5 | **335.1** | 320.9 |

→ 高並行スループットも概ね num=3 が最良。num=2 は中並行で最弱。

## acceptance rate [%]

| workload | num=1 | num=2 | num=3 |
|---|--:|--:|--:|
| 自然文(conc=1) | 82.2 | 72.3 | 61.7 |
| 短文(conc=1) | 79.4 | 68.7 | 61.2 |
| コード(conc=1) | 85.1 | 76.0 | 67.5 |

acceptance length: num=1≈1.82, num=2≈2.45, num=3≈2.87(理論上限 num+1 に届かない=Qwen単一MTP層反復のため末尾トークン精度低下)。

## なぜ num=2 だけ中並行で谷になるか(仮説)
num=2 は draft 2トークンを単一MTP層で生成→acceptance が num=1 より落ちる(72%)。単発ではMTP並列化の得が採択低下の損を上回るが、conc=4 では**バッチが投機ぶんで膨らみ**、採択されなかった draft の計算が無駄になってスループットを食う。num=3 は draft 3 でヒット時の一括前進が大きく、中並行でも得が勝つ。conc=8 は元々バッチ飽和で投機の相対利得が縮む。

## 手順・再現
- runner: `../bench-datasets/run_matrix.sh <label>`(label = mtp1 / mtp2 / mtp3)
- 集計: `../bench-datasets/aggregate.py` → `results/_aggregate.json`
- データセット: `../bench-datasets/ja_lecture.jsonl`、`../bench-datasets/code_bcb.jsonl`(`make_code_bcb.py` で再生成)、短文 = random
- bench image: `vllm-bench:local`(サーバと同じイメージに pandas を足したもの。custom データセットのローダが pandas を要する)
- 構成 compose: `../docker-compose.mtp1.yml` / `../docker-compose.mtp2.yml` / `../docker-compose.mtp3.yml`
- 生データ: `../bench-datasets/results/mtp{1,2,3}__*.json`

## 関連
- 先行の単発のみの計測: `MTP_TUNING_REPORT.md`(num=2 +18% と出たが単発限定。今回の conc=4 で覆った)
- b12x / TRT-LLM / SGLang の見送り: `B12X_EXPERIMENT_REPORT.md`、`AEON0251_DFLASH_THINK_REPORT.md`
