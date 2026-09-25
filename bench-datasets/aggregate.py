import json, glob, os
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows=[]
for f in sorted(glob.glob(f"{RES}/mtp*__*.json")):
    b=os.path.basename(f).replace(".json","")
    mtp,wl,cc=b.split("__")  # mtp1, ja, c1
    d=json.load(open(f))
    tpot=d["median_tpot_ms"]
    rows.append(dict(
        mtp=int(mtp[-1]), wl=wl, cc=int(cc[1:]),
        tpot=tpot, tpot_p90=d["p90_tpot_ms"], tpot_p99=d["p99_tpot_ms"],
        decode_tps=1000.0/tpot if tpot else 0,           # per-request pure-decode tok/s
        out_tps=d["output_throughput"],                   # aggregate output tok/s
        acc=d.get("spec_decode_acceptance_rate"),
        acclen=d.get("spec_decode_acceptance_length"),
        completed=d["completed"], failed=d["failed"],
    ))

def get(mtp,wl,cc,k):
    for r in rows:
        if r["mtp"]==mtp and r["wl"]==wl and r["cc"]==cc: return r[k]
    return None

wls=["ja","short","code"]; ccs=[1,4,8]; mtps=[1,2,3]
wlname={"ja":"自然文(講義)","short":"短文","code":"コード"}

print("="*78)
print("MTP num=1/2/3 ベンチ結果  (out_len=512固定, ignore_eos, GB10)")
print("="*78)
print("\n### A. 単発あたり純decode速度  = 1000/median_TPOT [tok/s]  (高いほど速い)\n")
hdr=f"{'workload':<14}{'conc':>5} | {'num=1':>8}{'num=2':>8}{'num=3':>8} | {'2 vs 1':>8}{'3 vs 1':>8}"
print(hdr); print("-"*len(hdr))
for wl in wls:
    for cc in ccs:
        v1=get(1,wl,cc,"decode_tps"); v2=get(2,wl,cc,"decode_tps"); v3=get(3,wl,cc,"decode_tps")
        d2=f"{(v2/v1-1)*100:+.0f}%"; d3=f"{(v3/v1-1)*100:+.0f}%"
        print(f"{wlname[wl]:<12}{cc:>5} | {v1:>8.1f}{v2:>8.1f}{v3:>8.1f} | {d2:>8}{d3:>8}")

print("\n### B. 集約 output throughput [tok/s]  (サーバ全体・並行の総処理量)\n")
print(hdr); print("-"*len(hdr))
for wl in wls:
    for cc in ccs:
        v1=get(1,wl,cc,"out_tps"); v2=get(2,wl,cc,"out_tps"); v3=get(3,wl,cc,"out_tps")
        d2=f"{(v2/v1-1)*100:+.0f}%"; d3=f"{(v3/v1-1)*100:+.0f}%"
        print(f"{wlname[wl]:<12}{cc:>5} | {v1:>8.1f}{v2:>8.1f}{v3:>8.1f} | {d2:>8}{d3:>8}")

print("\n### C. spec-decode acceptance rate [%]\n")
print(hdr); print("-"*len(hdr))
for wl in wls:
    for cc in ccs:
        v1=get(1,wl,cc,"acc"); v2=get(2,wl,cc,"acc"); v3=get(3,wl,cc,"acc")
        print(f"{wlname[wl]:<12}{cc:>5} | {v1:>8.1f}{v2:>8.1f}{v3:>8.1f} |")

# sanity: any failures?
tf=sum(r["failed"] for r in rows); tc=sum(r["completed"] for r in rows)
print(f"\n[health] total completed={tc}, failed={tf}, runs={len(rows)}")
# dump machine-readable
json.dump(rows, open(f"{RES}/_aggregate.json","w"), ensure_ascii=False, indent=1)
