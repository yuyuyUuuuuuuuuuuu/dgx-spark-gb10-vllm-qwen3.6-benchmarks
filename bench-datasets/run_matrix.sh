#!/bin/bash
# MTP benchmark matrix runner — one server config at a time.
# 3 workloads (ja / short / code) x concurrency 1, 4, 8, output length fixed at 512 (--ignore-eos).
# Usage: run_matrix.sh <label>      e.g. run_matrix.sh mtp1
# Requires the target vLLM server to be healthy at $BASE_URL (default http://localhost:8000).
# Results: results/<label>__<workload>__c<concurrency>.json
#
# Environment overrides (all optional):
#   BENCH_IMAGE     image that provides `vllm bench serve` (default vllm-bench:local, see README)
#   BASE_URL        server base URL (default http://localhost:8000)
#   MODEL           served model name (default Qwen/Qwen3.6-35B-A3B)
#   OPENAI_API_KEY  passed into the bench container only if set (for a server started with VLLM_API_KEY)
set -u
LABEL="${1:?usage: run_matrix.sh <label e.g. mtp1>}"
IMG="${BENCH_IMAGE:-vllm-bench:local}"
DS="$(cd "$(dirname "$0")" && pwd)"
OUT="$DS/results"
mkdir -p "$OUT"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
BASE="${BASE_URL:-http://localhost:8000}"
OUTLEN=512

run() {  # run <workload> <dataset-args> <num-prompts> <concurrency>
  local wl="$1" dsargs="$2" np="$3" cc="$4"
  local fn="${LABEL}__${wl}__c${cc}.json"
  echo ">>> $LABEL $wl conc=$cc np=$np"
  docker run --rm --network host ${OPENAI_API_KEY:+-e OPENAI_API_KEY} -v "$DS":/ds --entrypoint vllm "$IMG" bench serve \
    --backend openai-chat --endpoint /v1/chat/completions --base-url "$BASE" \
    --model "$MODEL" $dsargs --ignore-eos \
    --max-concurrency "$cc" --num-prompts "$np" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles "50,90,99" \
    --save-result --result-dir /ds/results --result-filename "$fn" \
    2>&1 | grep -E "Successful|Failed|Output token throughput:|Median TPOT|P99 TPOT|Mean TPOT|Acceptance rate|Acceptance length|Request throughput"
  echo
}

for CC in 1 4 8; do
  run "ja"    "--dataset-name custom --dataset-path /ds/ja_lecture.jsonl --custom-output-len $OUTLEN" 30 "$CC"
  run "short" "--dataset-name random --random-input-len 128 --random-output-len $OUTLEN"             50 "$CC"
  run "code"  "--dataset-name custom --dataset-path /ds/code_bcb.jsonl --custom-output-len $OUTLEN"   40 "$CC"
done
echo "=== $LABEL matrix done. results in $OUT ==="
