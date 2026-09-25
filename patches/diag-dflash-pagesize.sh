#!/usr/bin/env bash
# ============================================================================
# diag-dflash-pagesize.sh   --- DIAGNOSTIC ONLY, no behavior fix
# ----------------------------------------------------------------------------
# Purpose:
#   Make unify_kv_cache_spec_page_size DUMP every layer's spec (class,
#   block_size, page_size_bytes, page_size_padded, indexes_kv_by_block_stride)
#   right before it would assert, so we can see EXACTLY which drafter layer(s)
#   fail and why. It then still raises (so the engine stops as before) but with
#   a readable dump instead of a bare AssertionError.
#
#   Run this once, instead of or before the fix patch, to confirm the root cause
#   on the real image (docker-compose.dflash-v2.yml runs it before the v2 patch).
#   It changes NO allocation behavior; it only adds logging + a clearer error.
#   Idempotent.
#
# Usage (inside container, before entrypoint):
#   bash /patch/diag-dflash-pagesize.sh && exec vllm serve ...
#
# NOTE: mount-only. Does NOT modify the image. GPU is used only because the
#       engine runs to the kv-cache step; the dump happens at profiling time.
# ============================================================================
set -euo pipefail

U=/usr/local/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
[ -f "$U" ] || { echo "[diag] ERROR: $U not found" >&2; exit 3; }

if grep -q "# AEON-pagesize-DIAG" "$U"; then
  echo "[diag] already instrumented, skip."
  exit 0
fi

python3 - "$U" <<'PY'
import sys, re
fn = sys.argv[1]
src = open(fn, encoding="utf-8").read()

anchor = "    max_page_size = max(page_sizes)\n"
if src.count(anchor) != 1:
    sys.stderr.write("[diag] ERROR: expected exactly 1 anchor 'max_page_size = max(...)', found %d\n" % src.count(anchor))
    sys.exit(4)

dump = (
    anchor
    + "    # AEON-pagesize-DIAG: dump specs before unify\n"
    + "    import sys as _sys\n"
    + "    _sys.stderr.write('[AEON-DIAG] unify_kv_cache_spec_page_size: max_page_size=%d\\n' % max_page_size)\n"
    + "    for _ln, _ls in kv_cache_spec.items():\n"
    + "        _sys.stderr.write('[AEON-DIAG]   layer=%s cls=%s block_size=%s page=%s padded=%s idx_stride=%s divisible=%s\\n' % (\n"
    + "            _ln, type(_ls).__name__, getattr(_ls,'block_size',None), _ls.page_size_bytes,\n"
    + "            getattr(_ls,'page_size_padded',None), getattr(_ls,'indexes_kv_by_block_stride',None),\n"
    + "            (max_page_size % _ls.page_size_bytes == 0)))\n"
)
src = src.replace(anchor, dump, 1)

# Replace the bare assert with a descriptive error including the mismatch.
bad = "            assert new_spec.page_size_bytes == max_page_size\n"
if src.count(bad) != 1:
    sys.stderr.write("[diag] ERROR: expected exactly 1 assert line, found %d\n" % src.count(bad))
    sys.exit(5)
good = (
    "            if new_spec.page_size_bytes != max_page_size:  # AEON-pagesize-DIAG\n"
    "                import sys as _sys\n"
    "                _sys.stderr.write('[AEON-DIAG] UNIFY-FAIL layer=%s cls=%s block_size=%s page=%s padded=%s idx_stride=%s max=%s\\n' % (\n"
    "                    layer_name, type(new_spec).__name__, getattr(new_spec,'block_size',None),\n"
    "                    new_spec.page_size_bytes, getattr(new_spec,'page_size_padded',None),\n"
    "                    getattr(new_spec,'indexes_kv_by_block_stride',None), max_page_size))\n"
    "                raise AssertionError('AEON-DIAG page mismatch: %s got %d want %d' % (layer_name, new_spec.page_size_bytes, max_page_size))\n"
)
src = src.replace(bad, good, 1)

open(fn, "w", encoding="utf-8").write(src)
print("[diag] instrumented OK.")
PY

python3 -c "import ast; ast.parse(open('$U', encoding='utf-8').read())" \
  && echo "[diag] syntax OK." \
  || { echo "[diag] ERROR: syntax broke after instrument" >&2; exit 6; }
