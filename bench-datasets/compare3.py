import json, glob, os
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LABELS=[("mtp3","0.23.0基準"),("aeon0251fair","0.25.1 autoON"),("aeon0251noauto","0.25.1 autoOFF")]

def load(label):
    d={}
    for f in sorted(glob.glob(f"{RES}/{label}__*.json")):
        _,wl,cc=os.path.basename(f).replace(".json","").split("__")
        j=json.load(open(f)); tpot=j["median_tpot_ms"]
        d[(wl,int(cc[1:]))]=dict(decode_tps=1000.0/tpot if tpot else 0,
            out_tps=j["output_throughput"], acc=j.get("spec_decode_acceptance_rate"),
            completed=j["completed"], failed=j["failed"])
    return d

data={lab:load(lab) for lab,_ in LABELS}
wls=["ja","short","code"]; ccs=[1,4,8]
wlname={"ja":"自然文","short":"短文","code":"コード"}

def section(key,title,unit):
    print(f"\n### {title} [{unit}]  (高いほど速い)\n")
    hdr=f"{'workload':<10}{'conc':>5} | {'0.23.0':>9}{'25.1 ON':>9}{'25.1 OFF':>10} | {'OFF vs 基準':>12}{'OFF vs ON':>11}"
    print(hdr); print("-"*len(hdr))
    for wl in wls:
        for cc in ccs:
            v={lab:data[lab].get((wl,cc)) for lab,_ in LABELS}
            if any(x is None for x in v.values()):
                a=v["mtp3"]; b=v["aeon0251fair"]; c=v.get("aeon0251noauto")
                if not c: print(f"{wlname[wl]:<8}{cc:>5} | (OFF欠測)"); continue
            a=v["mtp3"][key]; b=v["aeon0251fair"][key]; c=v["aeon0251noauto"][key]
            d1=f"{(c/a-1)*100:+.1f}%"; d2=f"{(c/b-1)*100:+.1f}%"
            print(f"{wlname[wl]:<8}{cc:>5} | {a:>9.1f}{b:>9.1f}{c:>10.1f} | {d1:>12}{d2:>11}")

print("="*80)
print("3-way: 基準0.23.0 / 0.25.1(autotune ON=公平版) / 0.25.1(autotune OFF)")
print("out512固定 ignore_eos GB10 同checkpoint MoE=Marlin atomic_add全ON")
print("="*80)
section("decode_tps","A. 単発あたり純decode速度 = 1000/median_TPOT","tok/s")
section("out_tps","B. 集約 output throughput","tok/s")

# 総括
for lab,name in LABELS[1:]:
    ds=[data[lab][k]["decode_tps"]/data["mtp3"][k]["decode_tps"]-1 for k in data["mtp3"] if k in data[lab] and data["mtp3"][k]["decode_tps"]]
    os_=[data[lab][k]["out_tps"]/data["mtp3"][k]["out_tps"]-1 for k in data["mtp3"] if k in data[lab] and data["mtp3"][k]["out_tps"]]
    if ds:
        print(f"\n[{name} vs 基準0.23.0] 単発decode平均{sum(ds)/len(ds)*100:+.1f}% / 集約平均{sum(os_)/len(os_)*100:+.1f}% (全{len(ds)}条件)")

fa={lab:sum(v['failed'] for v in data[lab].values()) for lab,_ in LABELS}
print(f"\n[health] failed: {fa}")
