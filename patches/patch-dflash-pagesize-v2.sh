#!/usr/bin/env bash
# ============================================================================
# patch-dflash-pagesize-v2.sh
# ----------------------------------------------------------------------------
# 目的:
#   hybrid target (Qwen3_5MoeForConditionalGeneration = GDN/linear "mamba"
#   + full-attn) に 外部 z-lab DFlash drafter を載せると engine 起動時に
#     vllm/v1/core/kv_cache_utils.py:unify_kv_cache_spec_page_size
#     assert new_spec.page_size_bytes == max_page_size   AssertionError
#   で落ちる問題を根治する。
#
# 真因(v1 が効かなかった理由):
#   z-lab drafter の config は layer_types = [SWA×5, full_attention×1]。
#   AEONイメージの DFlash loader (qwen3_dflash.py:DFlashAttention.get_kv_cache_spec)
#   は "SlidingWindowSpec のときだけ" FullAttentionSpec に作り替える。
#   v1 パッチはこの SWA 分岐にだけ indexes_kv_by_block_stride=True を足したが:
#     (A) drafter の "本物の full_attention 層"(layer 5)は SWA 分岐を通らず
#         `return spec`(素の FullAttentionSpec, flag=False)で出て行くため、
#         unify の pad パスに入れず NotImplementedError/assert 経路に落ちる。
#     (B) SWA→Full 変換で `page_size_padded=spec.page_size_padded` を素通し
#         コピーしている。入力 spec に page_size_padded が既に載っていると、
#         unify が「割り切れる」分岐(block_size だけ差し替え)を取ったとき
#         古い padded 値が page_size_bytes に残り、max_page_size と一致せず
#         assert に落ちる。
#   hybrid target 側は mamba 整合のため attention block_size が 1136 に
#   引き上げられ(interface.py:"Setting attention block size to 1136 tokens")、
#   drafter の attention page が target の page(2,326,528B)と非整数比になる
#   ため、pad パスへ必ず入れる必要がある。pad は AttentionSpec かつ
#   indexes_kv_by_block_stride=True の層のみ許可される。
#
# 対策(2箇所):
#   [1] qwen3_dflash.py の DFlashAttention.get_kv_cache_spec を書き換え、
#       - SWA→Full 変換にも
#       - 素の full_attention(AttentionSpec)にも
#       indexes_kv_by_block_stride=True を付け、かつ page_size_padded は None に
#       落として(unify に padded を再計算させる)返す。
#       flash_attn / flashinfer / triton_attn いずれも layered stride order が
#       num-blocks-first なので block-stride index は正しい。
#   [2] kv_cache_utils.py の unify_kv_cache_spec_page_size の
#       「割り切れる」分岐を安全化:block_size を差し替える際に
#       stale な page_size_padded を None にリセットしてから replace する
#       (B の再発防止・防御的)。padded パス側は元から max に上書きするので不要。
#
# 冪等:既に patched なら何もしない。対象が見つからなければ非0で終了。
# mount 実行専用:イメージは無改変。GPU 不要の静的パッチ。
#
# 使い方(entrypoint 前に container 内で実行):
#   bash /patch/patch-dflash-pagesize-v2.sh && exec vllm serve ...
# ============================================================================
set -euo pipefail

F=/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
U=/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py

[ -f "$F" ] || { echo "[patch-dflash-v2] ERROR: $F not found (image layout changed?)" >&2; exit 3; }
[ -f "$U" ] || { echo "[patch-dflash-v2] ERROR: $U not found (image layout changed?)" >&2; exit 3; }

# ------------------------------------------------------------------ [1] loader
python3 - "$F" <<'PY'
import sys, re
fn = sys.argv[1]
src = open(fn, encoding="utf-8").read()

MARK = "# AEON-pagesize-fix-v2"
if MARK in src:
    print("[patch-dflash-v2] loader already patched, skip.")
    sys.exit(0)

# 置換対象: DFlashAttention.get_kv_cache_spec 全体。
# アンカーは、v1 適用済み・未適用どちらの本体でも一致するよう、メソッド
# シグネチャから "return spec\n" までを丸ごと差し替える。
sig = "    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:\n"
if src.count(sig) != 1:
    sys.stderr.write("[patch-dflash-v2] ERROR: expected exactly 1 get_kv_cache_spec sig, found %d\n" % src.count(sig))
    sys.exit(4)

start = src.index(sig)
# メソッド本体の終端 = 最初に現れる "        return spec\n"(このメソッド内でのみ使用)
end_marker = "        return spec\n"
end_rel = src.find(end_marker, start)
if end_rel == -1:
    sys.stderr.write("[patch-dflash-v2] ERROR: could not find 'return spec' terminator of get_kv_cache_spec\n")
    sys.exit(5)
end = end_rel + len(end_marker)

new_method = (
    "    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:\n"
    "        " + MARK + "\n"
    "        # DFlash draft layers pre-write全context KVで evict しないため full-\n"
    "        # attention KV を確保する。hybrid target と page が非整数比になるため、\n"
    "        # SWA / full どちらの層でも block-stride index を有効化し、pad path を\n"
    "        # 通せるようにする。stale な page_size_padded は None にして unify に\n"
    "        # 再計算させる(古い padded 値が残ると割り切れ分岐で assert に落ちる)。\n"
    "        spec = super().get_kv_cache_spec(vllm_config)\n"
    "        if isinstance(spec, SlidingWindowSpec):\n"
    "            return FullAttentionSpec(\n"
    "                block_size=spec.block_size,\n"
    "                num_kv_heads=spec.num_kv_heads,\n"
    "                head_size=spec.head_size,\n"
    "                head_size_v=getattr(spec, \"head_size_v\", spec.head_size),\n"
    "                dtype=spec.dtype,\n"
    "                kv_quant_mode=spec.kv_quant_mode,\n"
    "                page_size_padded=None,\n"
    "                indexes_kv_by_block_stride=True,\n"
    "            )\n"
    "        from dataclasses import replace as _replace\n"
    "        from vllm.v1.kv_cache_interface import AttentionSpec as _AttnSpec\n"
    "        if isinstance(spec, _AttnSpec):\n"
    "            return _replace(\n"
    "                spec,\n"
    "                page_size_padded=None,\n"
    "                indexes_kv_by_block_stride=True,\n"
    "            )\n"
    "        return spec\n"
)

src = src[:start] + new_method + src[end:]
open(fn, "w", encoding="utf-8").write(src)
print("[patch-dflash-v2] loader patched OK (SWA + full both get indexes_kv_by_block_stride=True, padded reset).")
PY

python3 -c "import ast; ast.parse(open('$F', encoding='utf-8').read())" \
  && echo "[patch-dflash-v2] loader syntax OK." \
  || { echo "[patch-dflash-v2] ERROR: loader syntax broke" >&2; exit 6; }

# ------------------------------------------------ [2] unify divisible-branch safety
python3 - "$U" <<'PY'
import sys
fn = sys.argv[1]
src = open(fn, encoding="utf-8").read()

MARK = "# AEON-pagesize-fix-v2-unify"
if MARK in src:
    print("[patch-dflash-v2] unify already patched, skip.")
    sys.exit(0)

# 割り切れる分岐で block_size 差し替え時に stale padded をクリアする。
old = (
    "            if max_page_size % layer_page_size == 0:\n"
    "                ratio = max_page_size // layer_page_size\n"
    "                new_block_size = layer_spec.block_size * ratio\n"
    "                new_spec = replace(layer_spec, block_size=new_block_size)\n"
)
if src.count(old) != 1:
    sys.stderr.write("[patch-dflash-v2] ERROR: unify divisible-branch anchor not found (found %d). Image changed?\n" % src.count(old))
    sys.exit(7)

new = (
    "            if max_page_size % layer_page_size == 0:\n"
    "                " + MARK + "\n"
    "                ratio = max_page_size // layer_page_size\n"
    "                new_block_size = layer_spec.block_size * ratio\n"
    "                # stale page_size_padded を持ち越すと page_size_bytes が\n"
    "                # scaled 値でなく古い padded を返し、assert に落ちる。\n"
    "                if getattr(layer_spec, \"page_size_padded\", None) is not None:\n"
    "                    new_spec = replace(\n"
    "                        layer_spec, block_size=new_block_size, page_size_padded=None\n"
    "                    )\n"
    "                else:\n"
    "                    new_spec = replace(layer_spec, block_size=new_block_size)\n"
)
src = src.replace(old, new, 1)
open(fn, "w", encoding="utf-8").write(src)
print("[patch-dflash-v2] unify divisible-branch hardened OK.")
PY

python3 -c "import ast; ast.parse(open('$U', encoding='utf-8').read())" \
  && echo "[patch-dflash-v2] unify syntax OK." \
  || { echo "[patch-dflash-v2] ERROR: unify syntax broke" >&2; exit 8; }

echo "[patch-dflash-v2] DONE."
