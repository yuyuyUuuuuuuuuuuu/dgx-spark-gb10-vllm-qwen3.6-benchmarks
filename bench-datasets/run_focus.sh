#!/bin/bash
# Focused bench: concurrency 1 and 4 only, 3 workloads (ja / short / code), output length fixed at 512.
# Used for the b12x x MTP num=1/2/3 comparison and the aeon 0.25.1 / DFlash runs.
# Usage: run_focus.sh <label>
# Same environment overrides as run_matrix.sh (BENCH_IMAGE, BASE_URL, MODEL, OPENAI_API_KEY).
set -u
LABEL="${1:?usage: run_focus.sh <label>}"
IMG="${BENCH_IMAGE:-vllm-bench:local}"
DS="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DS/results"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"; BASE="${BASE_URL:-http://localhost:8000}"; OUTLEN=512
run(){ local wl="$1" dsargs="$2" np="$3" cc="$4"
  echo ">>> $LABEL $wl conc=$cc"
  docker run --rm --network host ${OPENAI_API_KEY:+-e OPENAI_API_KEY} -v "$DS":/ds --entrypoint vllm "$IMG" bench serve \
    --backend openai-chat --endpoint /v1/chat/completions --base-url "$BASE" \
    --model "$MODEL" $dsargs --ignore-eos --max-concurrency "$cc" --num-prompts "$np" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles "50,90,99" \
    --save-result --result-dir /ds/results --result-filename "${LABEL}__${wl}__c${cc}.json" \
    2>&1 | grep -E "Output token throughput:|Median TPOT|Acceptance rate|Failed requests"
  echo; }
for CC in 1 4; do
  run "ja"    "--dataset-name custom --dataset-path /ds/ja_lecture.jsonl --custom-output-len $OUTLEN" 30 "$CC"
  run "short" "--dataset-name random --random-input-len 128 --random-output-len $OUTLEN"             50 "$CC"
  run "code"  "--dataset-name custom --dataset-path /ds/code_bcb.jsonl --custom-output-len $OUTLEN"   40 "$CC"
done
echo "=== $LABEL focus done ==="
