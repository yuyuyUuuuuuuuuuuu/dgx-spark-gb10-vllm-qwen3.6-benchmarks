"""Rebuild code_bcb.jsonl for the "code" workload.

The code workload is the first 40 BigCodeBench tasks (BigCodeBench/0 .. BigCodeBench/39),
one JSON object per line: {"prompt": <instruct_prompt>}, written with ensure_ascii=False.
BigCodeBench (https://github.com/bigcode-project/bigcodebench) is Apache-2.0 licensed.

The file is not shipped in this repository (see README). The file used for the published
results had 40 lines and SHA-256
    05aad5606ffc5e2ae75e996e616240c4834d9465f3fab6f5620f10926bee3e14
Which BigCodeBench release it came from was not recorded. Pick a --split (release name on the
Hugging Face Hub, e.g. v0.1.4) and compare the printed hash with the one above.

Requires the `datasets` package and access to the Hugging Face Hub.
"""
import argparse
import hashlib
import json
import os

EXPECTED_SHA256 = "05aad5606ffc5e2ae75e996e616240c4834d9465f3fab6f5620f10926bee3e14"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="bigcode/bigcodebench split (release), e.g. v0.1.4")
    ap.add_argument("-n", type=int, default=40, help="number of tasks from the start (default 40)")
    ap.add_argument("-o", "--output", default=os.path.join(here, "code_bcb.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so --help works without the package

    ds = load_dataset("bigcode/bigcodebench", split=args.split)
    rows = sorted(ds, key=lambda r: int(r["task_id"].split("/")[1]))[: args.n]
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r["instruct_prompt"]}, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    status = "matches" if digest == EXPECTED_SHA256 else "DIFFERS from"
    print(f"wrote {args.output} ({len(rows)} prompts), sha256 {digest} {status} the published file")


if __name__ == "__main__":
    main()
