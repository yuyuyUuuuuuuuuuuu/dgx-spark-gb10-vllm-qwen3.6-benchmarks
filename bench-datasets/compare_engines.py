import json, glob, os
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# 比較: mtp3 (ベースライン 0.23.0 + MTP3 + atomic_add) vs aeon0251fair (0.25.1 + MTP3 + atomic_add, 全env移植)
A="mtp3"        # baseline = 0.23.0
B="aeon0251fair" # fair 0.25.1

def load(label):
    d={}
    for f in sorted(glob.glob(f"{RES}/{label}__*.json")):
        b=os.path.basename(f).replace(".json","")
        _,wl,cc=b.split("__")
        j=json.load(open(f))
        tpot=j["median_tpot_ms"]
        d[(wl,int(cc[1:]))]=dict(
            tpot=tpot,
            decode_tps=1000.0/tpot if tpot else 0,
            out_tps=j["output_throughput"],
            acc=j.get("spec_decode_acceptance_rate"),
            completed=j["completed"], failed=j["failed"],
        )
    return d

da=load(A); db=load(B)
wls=["ja","short","code"]; ccs=[1,4,8]
wlname={"ja":"自然文(講義)","short":"短文","code":"コード"}

print("="*82)
print(f"エンジン比較: {A}=基準0.23.0+MTP3  vs  {B}=公平0.25.1+MTP3(全env移植)")
print("out_len=512固定, ignore_eos, GB10, 同一checkpoint(MoE=Marlin), atomic_add両方ON")
print("="*82)

def section(key, title, unit):
    print(f"\n### {title} [{unit}]  (高いほど速い)\n")
    hdr=f"{'workload':<14}{'conc':>5} | {'0.23.0(基準)':>13}{'0.25.1(公平)':>13} | {'0.25.1 vs 基準':>16}"
    print(hdr); print("-"*len(hdr))
    for wl in wls:
        for cc in ccs:
            a=da.get((wl,cc)); b=db.get((wl,cc))
            if not a or not b:
                print(f"{wlname[wl]:<12}{cc:>5} | {'(欠測)':>13}"); continue
            va=a[key]; vb=b[key]
            delta=f"{(vb/va-1)*100:+.1f}%" if va else "n/a"
            print(f"{wlname[wl]:<12}{cc:>5} | {va:>13.1f}{vb:>13.1f} | {delta:>16}")

section("decode_tps","A. 単発あたり純decode速度 = 1000/median_TPOT","tok/s")
section("out_tps","B. 集約 output throughput","tok/s")

print("\n### C. spec-decode acceptance rate [%]\n")
hdr=f"{'workload':<14}{'conc':>5} | {'0.23.0':>9}{'0.25.1':>9}"
print(hdr); print("-"*len(hdr))
for wl in wls:
    for cc in ccs:
        a=da.get((wl,cc)); b=db.get((wl,cc))
        if not a or not b: continue
        aa=a["acc"] if a["acc"] is not None else 0
        bb=b["acc"] if b["acc"] is not None else 0
        print(f"{wlname[wl]:<12}{cc:>5} | {aa:>9.1f}{bb:>9.1f}")

# health
fa=sum(v["failed"] for v in da.values()); fb=sum(v["failed"] for v in db.values())
ca=sum(v["completed"] for v in da.values()); cb=sum(v["completed"] for v in db.values())
print(f"\n[health] 0.23.0: completed={ca} failed={fa} | 0.25.1: completed={cb} failed={fb}")

# 総括
deltas=[db[k]["decode_tps"]/da[k]["decode_tps"]-1 for k in da if k in db and da[k]["decode_tps"]]
if deltas:
    avg=sum(deltas)/len(deltas)*100
    print(f"[総括] 単発decode速度: 0.25.1は基準0.23.0比で平均 {avg:+.1f}% (全{len(deltas)}条件)")
