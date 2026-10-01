#!/usr/bin/env bash
# run_arm.sh NAME [REQUEST_TOKENS]  -- env passes through to the launcher (BUUN_ROOT, KV_K, UBATCH_SIZE, ...)
# Starts the server, waits for "listening" (or failure), snapshots VRAM, optionally sends one
# prompt of ~REQUEST_TOKENS tokens, then stops the server and keeps the log as ~/claude-opt/NAME.log.
set -u
NAME=$1; REQ=${2:-0}
D=~/claude-opt; LOG=$D/$NAME.log
export LLAMA_LOG=$LOG LV=${LV:-4}
setsid bash $D/launcher.sh > $D/$NAME.launcher.out 2>&1 < /dev/null &
LPID=$!
state=timeout
for i in $(seq 1 360); do
  sleep 1
  if grep -q "listening on http" "$LOG" 2>/dev/null; then state=ready; break; fi
  if ! kill -0 $LPID 2>/dev/null; then state=died; break; fi
done
echo "== $NAME: $state after ${i}s"
{ echo "== vram after load ($state)"; rocm-smi --showmeminfo vram | grep "GPU\[0\]"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; } | tee $D/$NAME.vram
SPID=$(pgrep -x llama-server | head -1)
MAJ0=$(awk '/^pgmajfault/{print $2}' /proc/vmstat)
RB0=$( [[ -n "$SPID" ]] && awk '/^read_bytes/{print $2}' /proc/$SPID/io || echo 0)
IOPID=
if [[ $state == ready && "${IOSTAT:-1}" == 1 ]]; then
  iostat -x -y 5 > $D/$NAME.iostat 2>&1 &
  IOPID=$!
fi
if [[ $state == ready && $REQ -gt 0 ]]; then
  python3 - "$REQ" <<'PY' | tee -a $D/$NAME.vram
import json, sys, time, urllib.request
n = int(sys.argv[1])
words = ("the quick brown fox jumps over the lazy dog while counting prime numbers " * (n // 12 + 1)).split()[:n]
body = json.dumps({"prompt": " ".join(words) + "\nSummarize in one sentence:", "max_tokens": 64, "temperature": 0}).encode()
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:10000/v1/completions", body, {"Content-Type": "application/json"}), timeout=1800))
    tm = r.get("timings", {})
    print("request ok %.1fs prompt_n=%s pp=%.1f t/s tg=%.2f t/s" % (time.time() - t, tm.get("prompt_n"), tm.get("prompt_per_second", 0), tm.get("predicted_per_second", 0)))
except Exception as e:
    print("request FAILED after %.1fs: %r" % (time.time() - t, e))
PY
  { echo "== vram after request"; rocm-smi --showmeminfo vram | grep "GPU\[0\]"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; } | tee -a $D/$NAME.vram
fi
if [[ $state == ready && -n "${BENCH_ARGS:-}" ]]; then
  python3 ~/Projects/qwen38-perf/bench.py --url http://127.0.0.1:10000 $BENCH_ARGS --tag $NAME > $D/$NAME.bench 2>&1
  tail -20 $D/$NAME.bench
  grep -E "moe-cache.*(hits|evict|session)" "$LOG" | tail -3
fi
if [[ $state == ready && "${QUALITY:-0}" == 1 ]]; then
  K=~/Projects/qwen38-perf/buun-rtx-mi-kit
  python3 $K/quality.py --url http://127.0.0.1:10000 --tag $NAME --output $D/$NAME.quality.json > $D/$NAME.quality.out 2>&1
  tail -5 $D/$NAME.quality.out
  python3 $K/longctx.py --port 10000 --reps 7900 --needles PASS-A,PASS-B,PASS-C --positions 0.1,0.5,0.9 --max-tokens 512 --output $D/$NAME.longctx.json > $D/$NAME.longctx.out 2>&1
  tail -8 $D/$NAME.longctx.out
fi
if [[ $state == ready && "${GREEDY:-0}" == 1 ]]; then
  python3 $D/greedy.py --url http://127.0.0.1:10000 --output $D/$NAME.greedy.json > $D/$NAME.greedy.out 2>&1
  tail -6 $D/$NAME.greedy.out
fi
if [[ -n "$SPID" ]]; then
  MAJ1=$(awk '/^pgmajfault/{print $2}' /proc/vmstat)
  RB1=$(awk '/^read_bytes/{print $2}' /proc/$SPID/io 2>/dev/null || echo $RB0)
  echo "majflt=$((MAJ1-MAJ0)) read_bytes=$((RB1-RB0))" | tee $D/$NAME.io
fi
[[ -n "$IOPID" ]] && kill $IOPID 2>/dev/null
MDIR=${MODEL_DIR:-$HOME/Models/gguf/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64}
python3 $D/fincore.py $MDIR/*.gguf > $D/$NAME.fincore 2>&1; tail -1 $D/$NAME.fincore
kill -TERM $LPID 2>/dev/null
for i in $(seq 1 40); do kill -0 $LPID 2>/dev/null || break; sleep 1; done
pkill -KILL -x llama-server 2>/dev/null; sleep 2
pgrep -a -x llama-server || echo "== stopped"
