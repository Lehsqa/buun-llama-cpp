import json, statistics, sys, re
for name in sys.argv[1:]:
    recs = [json.loads(l) for l in open(f"/home/leshqa/claude-opt/{name}.bench") if l.startswith("{")]
    out = []
    for case in ("pp6k", "pp32k", "tg512f"):
        rs = [r for r in recs if r["case"] == case]
        if not rs: continue
        key = "tg_tok_s" if case.startswith("tg") else "pp_tok_s"
        vals = [r[key] for r in rs]
        use = vals[1:] if case.startswith("pp") and len(vals) > 2 else vals
        acc = sum(r["draft_accepted"] or 0 for r in rs) / max(1, sum(r["draft_n"] or 0 for r in rs))
        out.append(f"{case} med {statistics.median(use):.1f} (n={len(use)}, all={[round(v,1) for v in vals]})" + (f" acc={acc:.2f}" if case.startswith("tg") else ""))
    log = open(f"/home/leshqa/claude-opt/{name}.log", errors="ignore").read()
    hits = re.findall(r"\[moe-cache\][^\n]*hits[^\n]*", log)
    pools = re.findall(r"pool\[\d\]: type=(\S+) .*?total=(\d+) MiB", log)
    print(f"== {name}\n  " + "\n  ".join(out))
    print("  pools:", pools, "sum", sum(int(p[1]) for p in pools))
    for h in hits[-2:]: print("  ", h[:300])
