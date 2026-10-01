#!/usr/bin/env bash
# Qwen3.8-Flash-Next on buun-llama-cpp (spiritbuun fork @ ed774445cd69) —
# RTX 3080 Ti 12 GB (CUDA0) + Instinct MI50 32 GB (ROCm0), Ryzen 7600X, 30 GiB RAM.
#
# Same model, draft, ports and aliases as the production launcher
# (RTX+MI_Qwen3.8-Flash-Next.sh), so it is a drop-in stand-in; the only
# differences are the engine (separate CUDA/HIP backend modules with the
# single-provider MoE expert cache) and the measured placement below.
#
# Measured 2026-09-19 (same-session, 150k ctx / ub 768 / q4_0+q4_0 KV /
# f16 draft KV / MTP n=3 shared-Q8, layer split 3,32, output head on CUDA0):
#   production fork 2857e5114, 18 CPU blocks, cache n/a : pp6k 256.5  pp32k 237.7  tg 27.07
#   buun, 21 CPU blocks, MOE_CACHE=off                  : pp6k 270.4               tg 27.86
#   buun, 21 CPU blocks, MOE_CACHE=auto  <-- defaults   : pp6k 266.9               tg 33.66 / 32.89
#   expert cache session: 1950 MiB granted, pools iq4_nl 825 / iq3_s 714 /
#   iq2_s 407 MiB, 46.8% row hits (1132353/2417350), 0 dispatch/collect failures.
#   142k-token request OK (pp 162.3, tg 22.2) and a needle at 50% depth retrieved.
#
# Why these flags:
#   - The fork's target context requests a ~1493 MiB CUDA0 pp compute buffer on
#     the first ub-768 request (production needed ~381 MiB). CUDA0 only has
#     ~360 MiB free with the production CPU-block set, so ub 768 there fails
#     with "failed to allocate compute pp buffers". Two extra CUDA0 expert
#     blocks on the host (blk 3, plus 1 and 2 via CPU_BLOCKS) free ~3.3 GB and
#     both make ub 768 work and give the expert cache a pool budget.
#   - --moe-cache auto needs free >= reserve (1024 MiB) + the 1024 MiB slab
#     floor. With the production 18-block set it reports "session dormant"
#     (granted 812 MiB); with the 21-block set it admits ~1.95 GB.
#   - Cache pools are given back to the allocator on pressure, so prefill still
#     gets its buffers; prefill nodes are bypassed by design (>64 routed rows).
#   - PLE stays demand-paged: -lm mmap -lzm on, --mmap-prefetch off, no mlock,
#     no whole-model prefault (88 GiB mapped vs 28 GiB RAM).
#   - Expert parallelism stays 0: CUDA and HIP compile the same backend GUID,
#     and the cache has exactly one provider per process (CUDA here). The
#     owner guard admits only the owning registry's devices.
#   - Output head on CUDA0 is required: the shared MTP head borrows the
#     target's output.weight and the draft scheduler only owns CUDA0 + CPU.
#
# Known fork defects (both off this model's path, not worked around here):
#   - MUL_MAT_ID with f16/f32 weights faults the ROCm0 GPU (also in a HIP-only
#     build). All expert types this model uses (iq2_s/iq3_s/iq4_nl) pass.
#   - The stable/tie top-k kernel assumes 32-lane warps and is wrong on wave64;
#     the model calls ggml_top_k(..., stable=false).
#
# Env overrides (same names as the production launcher where they overlap):
#   CTX=150000 UBATCH_SIZE=768 KV_K=q4_0 KV_V=q4_0 DRAFT_KV_K=f16 DRAFT_KV_V=f16
#   TS=3,32 CPU_BLOCKS='0|1|2|3|3[1-9]|4[0-7]' OUT_DEVICE=CUDA0 N_MAX=3
#   MTP=0|1 MTP_QUANT=shared-Q8_0|Q4_K_M API_PORT=10000 LV=3
#   MOE_CACHE=auto|off|soft|on|<MiB> MOE_CPU_OVERLAP=auto|0..8 MOE_CACHE_PROFILE=0|1
#   BUUN_ROOT=<build root> EXTRA_ARGS="..." DRY_RUN=1 STOP_GRACE=15
#
# Shutdown: the fork's signal_handler runs a graceful shutdown that can wedge
# (it calls non-async-signal-safe code; when it deadlocks, SIGINT/SIGTERM are
# swallowed forever and the process keeps its VRAM). This launcher therefore
#   - starts the server with setsid, in its own session, so a terminal Ctrl+C
#     reaches this script only and never the model process directly,
#   - on INT/TERM/EXIT/HUP sends TERM, waits STOP_GRACE seconds, then SIGKILL,
#   - and before starting, stops any leftover llama-server of this runtime so a
#     previously wedged run cannot hold VRAM or the port.
# Only `kill -9` on the *script* can still leave the server behind.
#
# Rollback (production llama.cpp build, unchanged):
#   bash ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next_old.sh
set -euo pipefail

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BUUN_ROOT="${BUUN_ROOT:-$HOME/Projects/buun-rtx-mi-ed774445cd69-cuda}"
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
LLAMA_LOG="${LLAMA_LOG:-$LOG_DIR/llama-server-qwen38next-buun.log}"

CTX="${CTX:-150000}"
KV_K="${KV_K:-turbo4}"
KV_V="${KV_V:-turbo4}"
DRAFT_KV_K="${DRAFT_KV_K:-f16}"
DRAFT_KV_V="${DRAFT_KV_V:-f16}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-768}"      # 512 is the safe fallback (needs fewer CUDA0 MB)
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

# Hide the AMD iGPU (gfx1036) so ROCm0 is the MI50.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
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
    echo "       build it with: cd ~/Projects/qwen38-perf/buun-rtx-mi-kit && CACHE_PROVIDER=cuda bash build-linux.sh" >&2
    exit 1
  fi
done
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
  OVERRIDES="${OVERRIDES},blk\\.(${CPU_BLOCKS})\\.${CPU_EXP_PATTERN}=${EXP_BUFT:-CPU}"
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
PIDFILE="$LOG_DIR/llama-server-buun-$API_PORT.pid"
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
echo "Starting buun llama-server on $API_HOST:$API_PORT ..."
echo "  binary: $LLAMA_BIN"
echo "  model:  $MODEL"
echo "  PLE:    mmap + lazy rows (SSD-backed), prefetch off"
echo "  MTP:    $([[ "$MTP" == 1 ]] && echo "on ($MTP_MODEL, n_max $N_MAX, CUDA0)" || echo off)"
echo "  ctx:    $CTX | ubatch: $UBATCH_SIZE | tensor-split: $TS | CPU expert blocks: ${CPU_BLOCKS:-none}"
echo "  KV:     $KV_K / $KV_V (target) | $DRAFT_KV_K / $DRAFT_KV_V (draft)"
echo "  cache:  $MOE_CACHE (expert-parallel 0, cpu-overlap $MOE_CPU_OVERLAP, profile $MOE_CACHE_PROFILE)"

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
