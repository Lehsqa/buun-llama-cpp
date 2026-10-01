#!/usr/bin/env bash
# Qwen3.8-Flash-Next on buun-llama-cpp, branch opt/strata-port (Lehsqa/buun-llama-cpp) --
# RTX 3080 Ti 12 GB (CUDA0) + Instinct MI50 32 GB (ROCm0), Ryzen 7600X, 30 GiB RAM.
#
# The opt/rtx-mi50 launcher (RTX+MI_Qwen3.8-Flash-Next.sh) plus the Strata-port features (P0, P1):
#   - PLE direct reader (P1, default on here): the 36.6 GiB PLE n-gram table is read with O_DIRECT
#     positioned reads through a 16-thread pool and a 1M-row cache, with read-ahead of later ubatches,
#     instead of 4 KiB page faults through the lazy mmap. It needs lazy PLE (-lzm on, kept below) and
#     falls back to the mmap path with a warning if it cannot open.
#   - GGML_SCHED_TIMING (P0, diagnostic, default off): SCHED_TIMING=N logs one
#     "sched-timing:" line every N graph computes (input-copy / ids-sync / compute ms, expert MiB).
#     It synchronises after every split, so it slows the server down: use it for measurement only.
#
# Runtime: ~/Projects/buun-opt-rtx-mi50/runtime, built from ~/Projects/buun-opt-src on a branch that
#   contains strata/p1-ple-direct (bash scripts/rtx-mi50/build.sh, see scripts/rtx-mi50/README.md).
#
# Measured 2026-10-01 (REPORT §9; 150k ctx, q4_0/q4_0 KV, ub 1280, MTP n=3, 21 CPU blocks,
# cache auto; bench medians of 5, boot sample excluded):
#   --ple-io mmap   : pp6k 335.2  pp32k 300.2  tg 36.4
#   --ple-io direct : pp6k 330.6  pp32k 298.8  tg 37.8   (reader hit 99.2%, p50 126 us, blocked 1.1 s total)
#   direct: quality.py 10/10; 142k needles 3/3. Page cache is no longer filled with PLE pages.
#   Threads (tg, same P1 session): -t 5 36.0, -t 6 36.4, -t 12 29.3 -> 5 and 6 are level, 12 loses; 6 stays.
#   MTP draft KV q8_0: tg 36.8 vs 37.1 (no gain, +139 MiB expert cache); f16 stays the default.
#   --draft-p-min 0.5/0.7 lose ~3 t/s; not set.
# Note: greedy output is not run-to-run reproducible on this host even with the cache off (2/5
# identical in P0), so the exactness of direct vs mmap was gated on quality + needles.
#
# KV default below is turbo4/turbo4 (as in the main launcher, chosen by the user; unmeasured at
# ub 1280). The measured reference is KV_K=q4_0 KV_V=q4_0.
#
# Env overrides (as RTX+MI_Qwen3.8-Flash-Next.sh, plus):
#   PLE_IO=direct|mmap        PLE row fetch (default direct)
#   PLE_IO_THREADS=16         reader threads (1..256)
#   PLE_ROW_CACHE=1048576     cached PLE rows (0 = none; ~120 B per row -> ~120 MiB RAM at 1M)
#   SCHED_TIMING=0|N          scheduler timing summary every N graphs (0 = off)
#   CTX=150000 UBATCH_SIZE=1280 KV_K=turbo4 KV_V=turbo4 DRAFT_KV_K=f16 DRAFT_KV_V=f16
#   TS=3,32 CPU_BLOCKS='0|1|2|3|3[1-9]|4[0-7]' OUT_DEVICE=CUDA0 N_MAX=3 TOPK_OVERRIDE=
#   MTP=0|1 MTP_QUANT=shared-Q8_0|Q4_K_M API_PORT=10000 LV=3
#   MOE_CACHE=auto|off|soft|on|<MiB> MOE_CPU_OVERLAP=auto|0..8 MOE_CACHE_PROFILE=0|1
#   BUUN_ROOT=<build root> EXTRA_ARGS="..." DRY_RUN=1 STOP_GRACE=15
#
# Runtime knobs read by the server (environment, optional):
#   LLAMA_QSA_BLOCK_TOPK=0     per-cell QSA selection (old graph, larger compute buffers)
#   LLAMA_MTP_DRAFT_UBATCH=N   MTP draft ubatch cap (default 256)
#
# Shutdown: same as the main launcher (setsid; TERM, then SIGKILL after STOP_GRACE seconds;
# leftovers recorded in the pid file are stopped before starting).
#
# Rollback: bash ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh      (opt/rtx-mi50 defaults, PLE via mmap)
#           PLE_IO=mmap bash ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next_opt.sh
set -euo pipefail

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BUUN_ROOT="${BUUN_ROOT:-$HOME/Projects/buun-opt-rtx-mi50}"
LLAMA_BIN="$BUUN_ROOT/runtime/llama-server"

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
[[ -d "$CUDA_ROOT" ]] || CUDA_ROOT=/usr/lib/cuda
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"

MODEL_DIR="$HOME/Models/gguf/Qwen3.8-Flash-Next"
MODEL="$MODEL_DIR/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"

MTP_QUANT="${MTP_QUANT:-shared-Q8_0}"
MTP_MODEL="$MODEL_DIR/mtp-Qwen3.8-Flash-Next-$MTP_QUANT.gguf"
MTP="${MTP:-1}"                        # 0 = run target-only
N_MAX="${N_MAX:-3}"                    # MTP draft length

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-10000}"
ALIAS="${ALIAS:-Qwen3.8-Flash-Next}"

LOG_DIR="${LOG_DIR:-$HOME/.local/state/llama-launcher}"
LLAMA_LOG="${LLAMA_LOG:-$LOG_DIR/llama-server-qwen38next-buun-strata.log}"

CTX="${CTX:-150000}"
KV_K="${KV_K:-turbo4}"
KV_V="${KV_V:-turbo4}"
DRAFT_KV_K="${DRAFT_KV_K:-f16}"
DRAFT_KV_V="${DRAFT_KV_V:-f16}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-1280}"     # measured best; 768/1024 also fit, 1536 leaves ~0.5 GiB on ROCm0
THREADS="${THREADS:-6}"
THREADS_BATCH="${THREADS_BATCH:-6}"

TS="${TS:-3,32}"                       # CUDA0 : ROCm0 layer split
OUT_DEVICE="${OUT_DEVICE:-CUDA0}"      # shared MTP head borrows output.weight -> must be CUDA0
# Tensor overrides are matched with std::regex_search (unanchored), so a bare
# 'output\.weight' also pins every blk.N.attn_output.weight to CUDA0 — measured:
# 13 tensors land there instead of 1, costing 22 graph splits per ubatch and
# 175 MiB of CUDA0 model buffer (pp6k 270.8 -> 286.5, tg 32.87 -> 34.31).
# The anchored default keeps only the global output head on CUDA0; reproduce the
# older frozen arm with OUT_PATTERN='output\.weight'.
OUT_PATTERN="${OUT_PATTERN:-^output\\.weight$}"
# 21 host blocks: 0,1,2 and 3 (CUDA0's four layers) + 31..47 (ROCm0's tail).
CPU_BLOCKS="${CPU_BLOCKS:-0|1|2|3|3[1-9]|4[0-7]}"
CPU_EXP_PATTERN='ffn_(up|down|gate|gate_up|up_s|gate_s|down_s|up_exps|gate_exps|down_exps|gate_up_exps)\.weight'

# Expert cache: the fork's headline feature. auto keeps weight repacking and
# uses (free VRAM - reserve); forced numeric budgets disable repacking.
MOE_CACHE="${MOE_CACHE:-auto}"
MOE_CPU_OVERLAP="${MOE_CPU_OVERLAP:-auto}"
MOE_CACHE_PROFILE="${MOE_CACHE_PROFILE:-0}"   # 0 = no saved heatmaps
# Expert parallelism inside the cache. 0 = serial admission (the measured default):
# a fanout can only span devices the *cache owner* owns, and here CUDA owns one device,
# so auto resolves to min(3, devices) = 1 and only switches admission to the bundle policy
# (inserts_per_plan 8 -> 16, readmit_after -> 40), not to real cross-device splitting.
MOE_CACHE_EP="${MOE_CACHE_EP:-0}"

LV="${LV:-3}"                          # 4 = trace (tensor map, cache pools/stats)
STOP_GRACE="${STOP_GRACE:-15}"         # seconds to wait after TERM before SIGKILL
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
TOPK_OVERRIDE="${TOPK_OVERRIDE:-}"     # empty = GGUF default indexer top_k (2048)

# PLE direct reader (Strata P1). direct needs the lazily read table (-lzm on below).
PLE_IO="${PLE_IO:-direct}"
PLE_IO_THREADS="${PLE_IO_THREADS:-16}"
PLE_ROW_CACHE="${PLE_ROW_CACHE:-1048576}"
# Scheduler timing (Strata P0): diagnostic only, it synchronises after every split.
SCHED_TIMING="${SCHED_TIMING:-0}"

# Hide the AMD iGPU (gfx1036) so ROCm0 is the MI50.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
if [[ "$SCHED_TIMING" != "0" ]]; then
  export GGML_SCHED_TIMING="$SCHED_TIMING"
else
  unset GGML_SCHED_TIMING
fi
export LD_LIBRARY_PATH="$BUUN_ROOT/runtime:$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ------------------------------------------------------------------
# Stop helper: TERM, then KILL. The fork's graceful shutdown can wedge and
# then ignores every further SIGINT/SIGTERM, so SIGKILL is the only exit.
# ------------------------------------------------------------------
stop_server() {   # $1 = pid, $2 = label
  local pid="$1" label="$2"
  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 "$STOP_GRACE"); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  echo "  $label (pid $pid) did not exit on SIGTERM; sending SIGKILL" >&2
  kill -KILL "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "  ERROR: $label (pid $pid) survived SIGKILL" >&2
    return 1
  fi
  echo "  $label (pid $pid) stopped with SIGKILL" >&2
  return 0
}

# ------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------
mkdir -p "$LOG_DIR"

for f in "$LLAMA_BIN" "$MODEL" "$BUUN_ROOT/runtime/libggml-hip.so" "$BUUN_ROOT/runtime/libggml-cuda.so"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: missing file: $f" >&2
    echo "       build it with: cd ~/Projects/buun-opt-src && git pull && BUILD_JOBS=6 bash scripts/rtx-mi50/build.sh" >&2
    exit 1
  fi
done
case "$PLE_IO" in
  mmap|direct) ;;
  *) echo "ERROR: PLE_IO must be mmap or direct (got '$PLE_IO')" >&2; exit 1 ;;
esac
if [[ "$PLE_IO" == "direct" ]]; then
  HELP_TEXT="$("$LLAMA_BIN" --help 2>/dev/null || true)"
fi
if [[ "$PLE_IO" == "direct" && "$HELP_TEXT" != *--ple-io* ]]; then
  echo "ERROR: '$LLAMA_BIN --help' does not list --ple-io: the runtime predates strata/p1-ple-direct (or did not start)." >&2
  echo "       Rebuild it from a branch with the PLE reader, or run with PLE_IO=mmap." >&2
  exit 1
fi
if [[ "$MTP" == "1" && ! -e "$MTP_MODEL" ]]; then
  echo "ERROR: missing draft model: $MTP_MODEL" >&2
  exit 1
fi

# The measured-good cache placement: blk 3 (CUDA0's last layer) on the host
# frees the ~1.5 GB the cache needs for its 1024 MiB slab floor.
CACHE_PLACEMENT='0|1|2|3|3[1-9]|4[0-7]'
if [[ "$MOE_CACHE" != "off" && "$MOE_CACHE" != "0" && "$CPU_BLOCKS" != "$CACHE_PLACEMENT" ]]; then
  echo "NOTE: MOE_CACHE=$MOE_CACHE with CPU_BLOCKS='$CPU_BLOCKS' is not the measured placement." >&2
  echo "      With the production 18-block set the session reports 'dormant' (granted 812 MiB" >&2
  echo "      < 1024 MiB slab floor); use CPU_BLOCKS='$CACHE_PLACEMENT' for a ~1.95 GB cache." >&2
fi

# ------------------------------------------------------------------
# Build the argument list
# ------------------------------------------------------------------
OVERRIDES="${OUT_PATTERN}=${OUT_DEVICE}"
if [[ -n "$CPU_BLOCKS" ]]; then
  OVERRIDES="${OVERRIDES},blk\\.(${CPU_BLOCKS})\\.${CPU_EXP_PATTERN}=CPU"
fi

SPEC_ARGS=()
if [[ "$MTP" == "1" ]]; then
  SPEC_ARGS=(
    -md "$MTP_MODEL"
    --spec-type draft-mtp
    --spec-draft-n-max "$N_MAX"
    --spec-draft-device CUDA0
    --spec-draft-ngl all
    --threads-draft 6
    --threads-batch-draft 6
  )
else
  SPEC_ARGS=(--spec-type none)
fi

CACHE_ARGS=(--moe-cache "$MOE_CACHE" --moe-cache-expert-parallel "$MOE_CACHE_EP")
[[ "$MOE_CPU_OVERLAP" != "auto" ]] && CACHE_ARGS+=(--moe-cache-cpu-overlap "$MOE_CPU_OVERLAP")
[[ "$MOE_CACHE_PROFILE" == "0" ]] && CACHE_ARGS+=(--no-moe-cache-profile)

ARGS=(
  -m "$MODEL"
  -lm mmap
  -lzm on
  --mmap-prefetch off
  --ple-io "$PLE_IO"
  --fit off
  --override-tensor "$OVERRIDES"
  "${SPEC_ARGS[@]}"
  "${CACHE_ARGS[@]}"
  -ngl all
  --device CUDA0,ROCm0
  --split-mode layer
  --tensor-split "$TS"
  -c "$CTX"
  --n-predict -1
  -ctk "$KV_K" -ctv "$KV_V"
  -ctkd "$DRAFT_KV_K" -ctvd "$DRAFT_KV_V"
  ${TOPK_OVERRIDE:+--override-kv qwen4exp.attention.indexer.top_k=int:$TOPK_OVERRIDE}
  --flash-attn on
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --threads "$THREADS"
  --threads-batch "$THREADS_BATCH"
  --threads-http 4
  --parallel 1
  --cache-prompt
  --cache-ram 4096
  --cache-idle-slots
  --reasoning-preserve
  --alias "$ALIAS"
  -lv "$LV"
  --host "$API_HOST"
  --port "$API_PORT"
)

if [[ "$PLE_IO" == "direct" ]]; then
  ARGS+=(--ple-io-threads "$PLE_IO_THREADS" --ple-row-cache "$PLE_ROW_CACHE")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "$LLAMA_BIN"
  printf ' %q' "${ARGS[@]}"; echo
  [[ -n "$EXTRA_ARGS" ]] && echo "EXTRA_ARGS: $EXTRA_ARGS"
  exit 0
fi

# ------------------------------------------------------------------
# Stop leftovers of *this* launcher instance (recorded in a pid file) and make
# sure the port is ours to take. A foreign llama-server on the port is reported,
# never killed: it may be the production build sharing the machine.
# ------------------------------------------------------------------
PIDFILE="$LOG_DIR/llama-server-buun-strata-$API_PORT.pid"
if [ -f "$PIDFILE" ]; then
  OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    if [ "$(readlink -f "/proc/$OLD_PID/exe" 2>/dev/null)" = "$(readlink -f "$LLAMA_BIN")" ]; then
      echo "Stopping this launcher's previous server (pid $OLD_PID from $PIDFILE)..."
      stop_server "$OLD_PID" "previous llama-server" || true
    else
      echo "NOTE: stale pid file: pid $OLD_PID is not this runtime; ignoring it." >&2
    fi
  fi
  rm -f "$PIDFILE"
fi

PORT_PID="$(ss -ltnpH "sport = :$API_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
if [ -n "$PORT_PID" ]; then
  OWNER="$(ps -o comm= -p "$PORT_PID" 2>/dev/null || true)"
  echo "ERROR: port $API_PORT is already held by '$OWNER' (pid $PORT_PID)." >&2
  echo "       Stop that process deliberately, or run this launcher with API_PORT=<other>." >&2
  echo "       (deliberately not killing it: it may be another build's deployment)" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Start
# ------------------------------------------------------------------
echo "Starting buun llama-server (opt/strata-port) on $API_HOST:$API_PORT ..."
echo "  binary: $LLAMA_BIN"
echo "  model:  $MODEL"
if [[ "$PLE_IO" == "direct" ]]; then
  echo "  PLE:    lazy rows, direct reader (O_DIRECT, $PLE_IO_THREADS threads, $PLE_ROW_CACHE cached rows)"
else
  echo "  PLE:    mmap + lazy rows (SSD-backed), prefetch off"
fi
echo "  MTP:    $([[ "$MTP" == 1 ]] && echo "on ($MTP_MODEL, n_max $N_MAX, CUDA0)" || echo off)"
echo "  ctx:    $CTX | ubatch: $UBATCH_SIZE | tensor-split: $TS | CPU expert blocks: ${CPU_BLOCKS:-none}"
echo "  KV:     $KV_K / $KV_V (target) | $DRAFT_KV_K / $DRAFT_KV_V (draft) | indexer top_k: ${TOPK_OVERRIDE:-gguf default}"
echo "  cache:  $MOE_CACHE (expert-parallel 0, cpu-overlap $MOE_CPU_OVERLAP, profile $MOE_CACHE_PROFILE)"
[[ "$SCHED_TIMING" != "0" ]] && echo "  diag:   GGML_SCHED_TIMING=$SCHED_TIMING (slows the server; measurement only)"

# setsid puts the server in its own session: a terminal Ctrl+C then hits only
# this script, and the trap below is the single authority for stopping the
# model. tee stays out of the job, so $! is the server itself.
if command -v setsid >/dev/null 2>&1; then
  setsid "$LLAMA_BIN" "${ARGS[@]}" $EXTRA_ARGS > >(tee "$LLAMA_LOG") 2>&1 &
else
  "$LLAMA_BIN" "${ARGS[@]}" $EXTRA_ARGS > >(tee "$LLAMA_LOG") 2>&1 &
fi
LLAMA_PID=$!

# setsid can fork when it is already a group leader; make sure we hold the
# server's pid and not a short-lived wrapper.
sleep 0.3
if [ "$(readlink -f "/proc/$LLAMA_PID/exe" 2>/dev/null)" != "$(readlink -f "$LLAMA_BIN")" ]; then
  RESOLVE_PATTERN="^$(printf '%s' "$LLAMA_BIN" | sed 's/[.[\*^$(){}?+|/]/\\&/g')( |\$)"
  RESOLVED="$(pgrep -u "$(id -u)" -f "$RESOLVE_PATTERN" | head -1 || true)"
  if [ -n "$RESOLVED" ]; then
    LLAMA_PID="$RESOLVED"
  else
    echo "ERROR: started server but could not determine its pid" >&2
    exit 1
  fi
fi
echo "$LLAMA_PID" > "$PIDFILE"

cleanup() {
  [[ -n "${CLEANED:-}" ]] && return          # TERM and EXIT both fire; run once
  CLEANED=1
  # Ignore further INT/TERM while stopping: a second impatient Ctrl+C must not
  # abort the escalation and orphan the model process.
  trap '' INT TERM HUP
  trap - EXIT
  echo
  echo "Shutting down (pid $LLAMA_PID)..."
  stop_server "$LLAMA_PID" "llama-server" || true
  wait "$LLAMA_PID" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "Done."
}
trap cleanup EXIT INT TERM HUP

echo
echo "----------------------------------------------------------------"
echo "  llama-server PID $LLAMA_PID  log: $LLAMA_LOG"
echo "  API:  http://$API_HOST:$API_PORT   (loads in ~1-2 min; /health returns 503 until ready)"
echo "  Ctrl+C to stop."
echo "----------------------------------------------------------------"
echo

wait "$LLAMA_PID"
