#!/usr/bin/env bash
# ============================================================================
# patch-dflash-pagesize.sh
# ----------------------------------------------------------------------------
# 目的:
#   hybrid target (Qwen3_5MoeForConditionalGeneration = GDN/mamba + full-attn)
#   に 外部 z-lab DFlash drafter を載せると engine 起動時に
#     vllm/v1/core/kv_cache_utils.py:unify_kv_cache_spec_page_size
#     assert new_spec.page_size_bytes == max_page_size   AssertionError
#   で落ちる。
#
# 真因:
#   drafter は SWA(sliding_window=4096)。AEONイメージの DFlash loader
#   (vllm/model_executor/models/qwen3_dflash.py) は SWA spec を
#   FullAttentionSpec に作り替えるが、その際 indexes_kv_by_block_stride を
#   コピーし忘れている(default False になる)。
#   drafter の page(block16 => 65,536B)は target の mamba整合page
#   (2,326,528B)を割り切れない(比 35.5、非整数)。flag が True なら
#   unify は page_size_padded パスで pad して通せるが、False だと
#   NotImplementedError/AssertionError で落ちる。
#
# 対策:
#   drafter の SWA->Full 変換で indexes_kv_by_block_stride=True を明示。
#   flash_attn / flashinfer / triton_attn いずれも layered stride order が
#   num-blocks-first(stride_order[0]!=0)なので True が正しい。
#   これで unify は drafter page を max_page_size(2,326,528)へ物理pad して
#   起動が通る(論理block16は不変、物理pageのみ拡大)。
#
# 使い方(entrypoint 前に container 内で実行):
#   bash /patch/patch-dflash-pagesize.sh && exec vllm serve ...
#   (compose の command でこの順に呼ぶ。リポジトリ直下の docker-compose.dflash-fixed.yml 参照)
#
# 冪等: 既に patched なら何もしない。対象行が見つからなければ非0で終了。
# ============================================================================
set -euo pipefail

F=/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py

if [ ! -f "$F" ]; then
  echo "[patch-dflash] ERROR: $F not found (image layout changed?)" >&2
  exit 3
fi

if grep -q "indexes_kv_by_block_stride=True,  # AEON-pagesize-fix" "$F"; then
  echo "[patch-dflash] already patched, skip."
  exit 0
fi

# SWA->Full 変換の FullAttentionSpec(...) 呼び出しに欠けている
# indexes_kv_by_block_stride=True を、page_size_padded 行の直後に挿入する。
# 対象行(v0.24.0/v0.25.1 共通):
#     page_size_padded=spec.page_size_padded,
python3 - "$F" <<'PY'
import sys, re, io
fn = sys.argv[1]
src = open(fn, encoding="utf-8").read()

needle = "page_size_padded=spec.page_size_padded,\n"
if needle not in src:
    sys.stderr.write("[patch-dflash] ERROR: anchor line not found; refusing to guess.\n")
    sys.exit(4)

# 挿入は「最初に現れる SWA->Full 変換ブロック内の」page_size_padded 行の直後のみ。
# qwen3_dflash.py では該当箇所は1つだけ。念のため count を検証。
if src.count(needle) != 1:
    sys.stderr.write("[patch-dflash] ERROR: expected exactly 1 anchor, found %d.\n" % src.count(needle))
    sys.exit(5)

# 直前行のインデントを継承(通常16スペース)
idx = src.index(needle)
line_start = src.rfind("\n", 0, idx) + 1
indent = re.match(r"[ \t]*", src[line_start:]).group(0)

insertion = needle + indent + "indexes_kv_by_block_stride=True,  # AEON-pagesize-fix\n"
src = src.replace(needle, insertion, 1)
open(fn, "w", encoding="utf-8").write(src)
print("[patch-dflash] patched OK: indexes_kv_by_block_stride=True inserted.")
PY

# 検証: 構文が壊れていないこと
python3 -c "import ast; ast.parse(open('$F', encoding='utf-8').read())" \
  && echo "[patch-dflash] syntax OK." \
  || { echo "[patch-dflash] ERROR: syntax broke after patch" >&2; exit 6; }
