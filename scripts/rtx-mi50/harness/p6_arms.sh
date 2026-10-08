#!/usr/bin/env bash
# Controlled P6 arms; inspect artifacts before accepting results (see README).
set -euo pipefail
cd "$HOME/claude-opt"

assert_idle() {
    if pgrep -a -x llama-server; then
        printf '%s\n' 'A server is already running; stop it through its launcher before running P6 arms.' >&2
        exit 1
    fi
}
assert_idle

ubatches=${P6_UBATCHES:-1280}
for ub in $ubatches; do
    if [[ "$ub" != 1280 && "$ub" != 768 ]]; then
        printf 'Unsupported P6 ubatch: %s (use 1280, optionally 768 separately)\n' "$ub" >&2
        exit 1
    fi
done

export BUUN_ROOT="${BUUN_ROOT:-$HOME/Projects/buun-p6-rtx-mi50}"
export CTX=150000 BATCH_SIZE=2048 UBATCH_SIZE=1280 THREADS=6 THREADS_BATCH=6
export KV_K=q4_0 KV_V=q4_0 DRAFT_KV_K=f16 DRAFT_KV_V=f16
export N_MAX=3 MTP=1 MOE_CACHE_EP=0 LLAMA_MTP_DRAFT_UBATCH=256
export TOPK_OVERRIDE= GGML_SCHED_TIMING=0
export GREEDY=0 QUALITY=0 IOSTAT=1
export BENCH_ARGS='--only pp6k,pp32k,tg512f --reps 5'

run_arm() {
    # run_arm.sh can kill all llama-server processes during cleanup.
    assert_idle
    bash ./run_arm.sh "$@"
}

# Greedy comparisons are informational; saved-input CPU tests gate exactness.
export EXTRA_ARGS='--ple-io mmap'
for arm in off-a off-b on; do
    value=0; [[ "$arm" == on ]] && value=1
    GGML_CPU_IQ_MT="$value" MOE_CACHE=off GREEDY=1 \
        BENCH_ARGS= IOSTAT=0 run_arm "p6-u1280-greedy-$arm" 0
done
python3 ./greedy.py --compare p6-u1280-greedy-off-a.greedy.json p6-u1280-greedy-off-b.greedy.json || true
python3 ./greedy.py --compare p6-u1280-greedy-off-a.greedy.json p6-u1280-greedy-on.greedy.json || true

for ple in mmap direct; do
    export EXTRA_ARGS="--ple-io $ple"
    if [[ "$ple" == direct ]]; then
        export EXTRA_ARGS='--ple-io direct --ple-io-threads 16 --ple-row-cache 1048576'
    fi
    for ub in $ubatches; do
        for round in 1 2; do
            order='0 1'; [[ "$round" == 2 ]] && order='1 0'
            for value in $order; do
                GGML_CPU_IQ_MT="$value" UBATCH_SIZE="$ub" MOE_CACHE=auto \
                    run_arm "p6-u${ub}-${ple}-r${round}-mt${value}" 0
            done
        done
    done
    for value in 0 1; do
        GGML_CPU_IQ_MT="$value" UBATCH_SIZE=1280 MOE_CACHE=off \
            run_arm "p6-u1280-${ple}-cpu-mt${value}" 0
        GGML_CPU_IQ_MT="$value" UBATCH_SIZE=1280 MOE_CACHE=auto QUALITY=1 \
            BENCH_ARGS= run_arm "p6-u1280-${ple}-quality-mt${value}" 0
    done
done

# Keep the deployed codec's results separate from the q4_0 reference.
if [[ "${P6_TURBO4_CONTROL:-0}" == 1 ]]; then
    export EXTRA_ARGS='--ple-io mmap'
    for value in 0 1; do
        GGML_CPU_IQ_MT="$value" UBATCH_SIZE=1280 MOE_CACHE=auto QUALITY=1 \
            KV_K=turbo4 KV_V=turbo4 run_arm "p6-u1280-turbo4-mt${value}" 0
    done
fi
