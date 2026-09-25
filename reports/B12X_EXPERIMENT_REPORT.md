# Qwen3.6-35B-A3B GB10: b12x vs Marlin 実測レポート

**環境**: GB10(単機, TP=1)/ b12x・Marlin の比較は anemll イメージ(vLLM 0.25.2 + flashinfer 0.6.15)、ベースラインは aeon イメージ(vLLM 0.23.0)/ `nvidia/Qwen3.6-35B-A3B-NVFP4`
**目的**: 「NVFP4 の MoE が Marlin にフォールバックしている(W4A16 相当で 2 倍遅い)ので、flashinfer b12x で脱出すれば速くなる」という仮説を GB10 実機で確かめる。

---

## 結論(3行)

1. **b12x MoEバックエンドはGB10で動く。Marlin脱出も出力品質(garbage無し)も確認できた。**(仮説のうち「動く」部分は成立)
2. **ただし高速化は ~+2.8%(75.3→77.4 tok/s)に過ぎない。** 「Marlin=2倍遅い」の想定は本checkpointでは**成立しなかった**。
3. **乗り換える価値は薄い。** リスク(image 差し替え・flashinfer decode ABI 破損の回避策・サポート外の構成)に見合う速度差ではない。**ベースライン構成(Marlin)のままでよい**。

---

## 実測(GB10 実機・単発 decode・completion_tokens / 経過時間)

計測方法: 単発リクエストの completion_tokens / 経過時間(この計測用スクリプトはリポジトリに含まれていない)。

| 構成 | MoEカーネル | MTP | attention | checkpoint | tok/s |
|---|---|---|---|---|---|
| **ベースライン(当時)** | Marlin | num=1 | flashinfer | nvidia | **~87.7** (86.2/89.7/87.1) |
| フェア比較A | **Marlin** | なし | triton | nvidia | **~75.3** (71.7/75.2/75.0/75.7/75.5) |
| フェア比較B | **b12x** | なし | triton | nvidia | **~77.4** (77.3/77.3/77.3/77.5) |

**b12x vs Marlin(単一変数=MoEカーネルのみ差替、他は完全同一): +2.8%**

> ベースライン 87.7 が実験の 77.4 より速いのは、ベースラインが **MTP 投機(num=1)** と **flashinfer attention** を使うから。実験は image の flashinfer decode ABI 破損を避けるため triton attention(遅い)+ MTP なし。したがって「ベースライン vs 実験」は公平な比較ではない。公平なのは A/B(75.3 vs 77.4)。

## なぜ「2倍」にならなかったか(実測から判明)

- nvidia checkpoint は **MIXED_PRECISION (modelopt)**: linear層の大半は**FP8(既にネイティブ高速)**。**NVFP4なのはMoE expertsだけ**で、そこだけがMarlinに落ちていた。
- **35B-A3Bはactive 3Bだけ**。MoE GEMMがdecode総時間に占める割合が小さい → MoEカーネルを2倍速くしても全体は数%しか動かない。
- ログ: `[nvfp4.py:239] Using 'FLASHINFER_B12X' NvFp4 MoE backend`(b12x 起動時) vs `Using 'MARLIN' NvFp4 MoE backend`(Marlin 時)。linear層は両方 `FlashInferFP8ScaledMMLinearKernel`(FP8)。

## 副次的に確定した事実

- **unsloth/Qwen3.6-35B-A3B-NVFP4 はb12xに使えない**: `compressed-tensors`量子化で **MoE expertsがFP8**。`ValueError: moe_backend='flashinfer_b12x' is not supported for FP8 MoE`。→ 「unsloth 版が b12x 向けの本命 checkpoint」という想定は**誤り**。nvidia MIXED_PRECISION版が唯一のNVFP4-MoE checkpoint(=唯一Marlinに落ち、唯一b12xが効く)。
- **b12xのgarbageバグ(#47365)は単機TP=1で発火しない**を実機確認(chat出力coherent)。
- **anemll image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM 0.25.2 + flashinfer 0.6.15)** はb12x実装を持つが、**flashinfer decode attentionのplan() ABIが壊れている**(19 vs 20 args mismatch)。回避=`--attention-backend=TRITON_ATTN`(CLI flag必須、env `VLLM_ATTENTION_BACKEND`は無視される)。
- **`--linear-backend=flashinfer_b12x` はnvidia ckptで即死**(GDN/linear層がFP8でb12xカーネル無し)。b12xは **`--moe-backend` のみ** に付ける。

## 提言

- **b12x への乗り換えは非推奨**。+2.8% のために ①vLLM 0.23→0.25 の image 差し替え ②triton attention の強制(flashinfer decode 破損の回避)③サポート外の b12x 経路 を背負う価値はない。ベースライン(Marlin + MTP + flashinfer, 87.7 tok/s)のままにする。
- b12xで意味のある差を狙うなら、①attention破損の直ったimage(flashinfer/vLLM整合版)で**flashinfer attention + b12x MoE + MTP**を揃える、②MoE比率の高い(activeの大きい)モデルで測る、のどちらか。今回のnvidia 35B-A3Bでは頭打ち。

## 再現方法(compose)
リポジトリ直下に:
- `docker-compose.b12x-test.yml` (b12x, nvidia ckpt)
- `docker-compose.marlin-test.yml` (marlin, 同 image・同 ckpt = フェア比較用)
- `docker-compose.unsloth-b12x.yml` (unsloth ckpt = FP8 MoE で起動しないことの記録)
- ベースラインは当時 `docker-compose.mtp1.yml` と同じ構成(MTP num=1)。

MTP と組み合わせた b12x の比較(`docker-compose.b12x-mtp{1,2,3}.yml`)は `b12x-vs-marlin-report.html`。
