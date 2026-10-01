#!/usr/bin/env bash
# p1_arms.sh -- PLE direct reader A/B (plan 2026-10-01-strata-p0-p1, Task 8)
set -u
cd ~/claude-opt
export BUUN_ROOT=$HOME/Projects/buun-opt-rtx-mi50 KV_K=q4_0 KV_V=q4_0 UBATCH_SIZE=1280
B="--only pp6k,pp32k,tg512f --reps 5"
MOE_CACHE=off GREEDY=1 IOSTAT=0 ./run_arm.sh p1-det-mmap 0
MOE_CACHE=off GREEDY=1 IOSTAT=0 EXTRA_ARGS="--ple-io direct" ./run_arm.sh p1-det-direct 0
BENCH_ARGS="$B" ./run_arm.sh p1-mmap 0
BENCH_ARGS="$B" EXTRA_ARGS="--ple-io direct" ./run_arm.sh p1-direct 0
QUALITY=1 EXTRA_ARGS="--ple-io direct" ./run_arm.sh p1-direct-quality 0
EXTRA_ARGS="-t 5 -tb 5" BENCH_ARGS="--only tg512f --reps 5" ./run_arm.sh p1-t5 0
EXTRA_ARGS="-t 12 -tb 12" BENCH_ARGS="--only tg512f --reps 5" ./run_arm.sh p1-t12 0
touch p1.done
