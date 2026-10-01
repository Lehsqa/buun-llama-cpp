#!/usr/bin/env bash
# start server, fire a ~24k prompt, sample nvidia pcie + rocm activity during prefill
D=~/claude-opt; export LLAMA_LOG=$D/dmon.log LV=3
KV_K=q4_0 KV_V=q4_0 BUUN_ROOT=$HOME/Projects/buun-opt-rtx-mi50 UBATCH_SIZE=1280 setsid bash $D/launcher.sh > $D/dmon.launcher.out 2>&1 < /dev/null &
LPID=$!
until grep -q "listening on http" $D/dmon.log 2>/dev/null; do sleep 1; kill -0 $LPID || exit 1; done
python3 - <<PY &
import json, urllib.request
words = ("alpha beta gamma delta epsilon zeta eta theta " * 3000).split()[:24000]
b = json.dumps({"prompt": " ".join(words), "max_tokens": 4}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:10000/v1/completions", b, {"Content-Type": "application/json"}), timeout=900))
print("pp", r["timings"]["prompt_per_second"])
PY
RQ=$!
sleep 15
timeout 30 nvidia-smi dmon -s ut -d 1 > $D/dmon.nv.txt
top -b -n 3 -d 2 | grep -E "llama-server|Cpu" | tail -4 > $D/dmon.top.txt
wait $RQ
kill -TERM $LPID; sleep 20; pkill -KILL -x llama-server; echo done
