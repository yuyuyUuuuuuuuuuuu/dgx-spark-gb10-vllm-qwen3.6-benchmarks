# Qwen3.6-35B-A3B GB10: MTP num=1/2/3 単発計測

**環境**: GB10 / vLLM 0.23.0(aeon イメージ)/ `nvidia/Qwen3.6-35B-A3B-NVFP4` / `--async-scheduling` あり
**動機**: TensorRT-LLM と SGLang の調査(どちらも見送り、下記)の結果、手元で伸ばせる余地は vLLM の MTP spec-decode しか残っていなかった。当時の構成は num=1 だったので num=2/3 を測った。
**構成**: `../docker-compose.mtp1.yml` / `../docker-compose.mtp2.yml` / `../docker-compose.mtp3.yml`(MTP num だけが違う)

> **注意**: 単発(直列)のみの計測。並行度を振った `MTP_BENCHMARK_FULL.md` で、下の「num=2 推奨」は覆った
> (num=2 は conc=4 で num=1 より遅い)。

---

## 結論(3行)

1. **MTP num=2 で単発 decode +18%(90→106 tok/s)、num=3 で +23%(90→111 tok/s)。品質劣化なし(自然文で確認)。**
2. **MTP は投機デコードなので数学的にロスレス**(draft は target と一致したときだけ採択)。「num=3 で自然文の品質が落ちる」という事前の懸念は、この計測では再現しなかった。
3. この時点の推奨は「num=2(+18%、1 フラグ)。num=3 は +5% 上乗せだが末尾 position の採択率が落ちる(pos2=54%)」。ただし単発・自然文中心の計測なので、長文要約やコードでの確認が必要とした(→ `MTP_BENCHMARK_FULL.md` で確認し、num=3 に落ち着いた)。

---

## 実測(GB10・単発 decode・同一 image/checkpoint/設定で MTP num のみ差し替え)

計測方法: 単発リクエストの completion_tokens / 経過時間(この計測用スクリプトはリポジトリに含まれていない)。

| MTP num | 単発 tok/s(steady) | vs num=1 | acceptance 率(全体) | per-pos 採択 | 自然文品質 |
|---|---|---|---|---|---|
| **1** | ~90 (89.5/90.0/90.4) | — | (num=1 は投機 1 段) | — | 良好 932ch |
| **2** | ~106 (105.3/106.5/107.2/108.0) | **+18%** | 73.7% | pos0 83% / pos1 64% | 同等 934ch |
| **3** | ~111 (107-116) | **+23%** | 69.1% | pos0 84% / pos1 69% / **pos2 54%** | 同等 929ch |

- **速度は投機の採択率しだいでプロンプトごとにばらつく**(num=2 で初回 65〜102、warm 後 106 で安定)。表の数字は warm 後の steady。
- **acceptance は num を上げるほど末尾 position が落ちる**(Qwen は単一の MTP 層を使い回すため。起動時に warning `num_speculative_tokens > 1 will run multiple times on same MTP layer` が出る)。num=3 の pos2=54% が伸びの頭打ち要因。
- **品質**: num=1/2/3 で出力長はほぼ同じ(932/934/929ch)、いずれも coherent な日本語。`enable_thinking:false` で測定。

## なぜ num=3 で頭打ちか
Qwen は独立した MTP ヘッドを N 個持たず、**単一の MTP 層を反復**するため、2 番目・3 番目の draft ほど精度が落ちる(pos2 採択 54%)。理論上限(num=3 で 4 トークン/step)には届かず、実効は num=2→3 で +5% 程度。

## 採用前に確認すべきとした点
- 本計測は単発・自然文(長文要約を想定)。①長文要約 ②コード生成 でも同じ伸びが出るか、品質が崩れないかを確認する。
- prefill 律速のワークロードでは投機が逆効果になり得る(GB10 で -7% の報告例あり)。長い prefill を伴う要約は要注意。

## 見送ったもの(同じ調査で判断)
- **TensorRT-LLM**: GitHub #16075(調査時点で未解決)で Qwen3.6-35B が GB10 でロード時に即死(量子化段の IndexError)。GB10 での TRT-LLM の tok/s の公開実測は見つからなかった。安定版なし(1.3.0rc のみ)。
- **SGLang**: 見送り(理由は `AEON0251_DFLASH_THINK_REPORT.md`)。なお vLLM #43906 によると、SM121 では Marlin が flashinfer より 16% 速く省メモリで、GB10 では Marlin は「フォールバック」ではなく現状のベスト。b12x の実測 +2.8% とも整合する。
- b12x は別レポート `B12X_EXPERIMENT_REPORT.md`(+2.8% で移行の価値なし)。
