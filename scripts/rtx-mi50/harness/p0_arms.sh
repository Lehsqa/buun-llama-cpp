#!/usr/bin/env bash
# p0_arms.sh -- Strata port P0 measurement arms (see docs/superpowers/plans/2026-10-01-strata-p0-p1.md, Task 3)
set -u
cd ~/claude-opt
export BUUN_ROOT=$HOME/Projects/buun-opt-rtx-mi50 KV_K=q4_0 KV_V=q4_0 UBATCH_SIZE=1280
B="--only pp6k,pp32k,tg512f --reps 5"
BENCH_ARGS="$B" ./run_arm.sh p0-ref 0
GGML_SCHED_TIMING=1 ./run_arm.sh p0-timing 32000
MOE_CACHE=off GREEDY=1 IOSTAT=0 ./run_arm.sh p0-det-a 0
MOE_CACHE=off GREEDY=1 IOSTAT=0 ./run_arm.sh p0-det-b 0
DRAFT_KV_K=q8_0 DRAFT_KV_V=q8_0 BENCH_ARGS="$B" ./run_arm.sh p0-draftkv-q8 0
for P in 0.3 0.5 0.7; do
  EXTRA_ARGS="--draft-p-min $P" BENCH_ARGS="--only tg512f --reps 5" ./run_arm.sh p0-pmin-$P 0
done
EXTRA_ARGS="-t 6 -tb 6" BENCH_ARGS="--only tg512f --reps 5" ./run_arm.sh p0-t6 0
touch p0.done
