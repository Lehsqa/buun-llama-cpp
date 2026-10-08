# rtx-mi50 harness (mirror of target `~/claude-opt/`)

Target: RTX 3080 Ti (CUDA0) + MI50 (ROCm0), `ssh -i ~/.ssh/id_llm -p 2222 leshqa@176.96.135.203`.
These files are the source of truth; the target copy is a deployment.

| Script        | What it does |
|---------------|--------------|
| `launcher.sh` | Starts `llama-server` for Qwen3.8-Flash-Next with the measured placement (env: `BUUN_ROOT KV_K KV_V UBATCH_SIZE CTX MOE_CACHE EXTRA_ARGS ...`, see its header). |
| `run_arm.sh NAME [REQ_TOKENS]` | One A/B arm: launch, wait for "listening", snapshot VRAM, optional ~REQ_TOKENS request, bench / quality / greedy, I/O + page-cache capture, stop. |
| `fincore.py FILE...` | Page-cache residency per file (resident MiB / total MiB + TOTAL) via `mmap` + `mincore` (util-linux `fincore` is not installed on target). |
| `greedy.py`   | `--url URL --output F`: 5 fixed chat prompts at temperature 0 / top_k 1, no prompt cache. `--compare A B`: prints `diff <id> at char <k>` per differing prompt and `identical N/M`; exit 0 iff all identical. |
| `summ.py NAME...` | Median pp6k / pp32k / tg512f from `NAME.bench`, MoE-cache pools and hit lines from `NAME.log`. |
| `dmon_arm.sh` | One-off: ~24k-token prefill while sampling `nvidia-smi dmon` and `top`. |
| `p6_arms.sh` | P6 off/on controls in separate mmap/direct modes, plus cache-off, quality/needles and informational greedy arms. |

## `run_arm.sh` env knobs

- `BENCH_ARGS` - if set, runs `~/Projects/qwen38-perf/bench.py ... $BENCH_ARGS` -> `NAME.bench`
- `QUALITY=1` - runs the kit's `quality.py` + `longctx.py` -> `NAME.quality.json`, `NAME.longctx.json`
- `GREEDY=1` - runs `greedy.py` -> `NAME.greedy.json` (default 0)
- `IOSTAT=1` - `iostat -x -y 5` for the whole arm -> `NAME.iostat` (default 1; `IOSTAT=0` disables)
- `MODEL_DIR` - shard directory scanned by `fincore.py` (default the Q4_K_M-M64 shard dir)
- anything else (`BUUN_ROOT`, `KV_K`, `LV`, ...) passes through to `launcher.sh`

## Outputs (`~/claude-opt/NAME.*`)

`log` (server), `launcher.out`, `vram`, `bench`, `quality.*`, `longctx.*`, `greedy.json`/`greedy.out`,
`io` (`majflt=<delta> read_bytes=<delta>`: system-wide major faults and the server's `read_bytes`, from
after load to end of the arm), `iostat`, `fincore` (model shard residency at end of the arm).

## P6 controls (prepared locally; target acceptance pending)

Build the current P6 source into an isolated runtime first; do not rebuild from the pristine pin or
discard the validated QSA/mask/MTP batching rewrite. From the target source checkout, after recording
its actual revision and kit patches:

```bash
OUT=$HOME/Projects/buun-p6-rtx-mi50 BUILD_JOBS=6 bash scripts/rtx-mi50/build.sh
# Later CPU-only iterations: the dynamically loaded MODULE needs an explicit rebuild.
cmake --build ~/Projects/buun-p6-rtx-mi50/cuda -j6 \
  --target ggml-cpu llama-server test-backend-ops
# Copy only while no llama-server from this runtime is running.
cp -a ~/Projects/buun-p6-rtx-mi50/cuda/bin/. ~/Projects/buun-p6-rtx-mi50/runtime/
```

The first build compiles both CUDA and HIP backends. Preserve CUDA cache ownership, gfx906 flags and
the kit compatibility patches; never copy into the deployed runtime. `test-cpu-iq-mt` uses private
symbols and needs a separate CPU-only `GGML_BACKEND_DL=OFF` AVX2/FMA build. Windows shared builds skip
that private-symbol test; a Windows static build can register it. The production MODULE build uses
`test-backend-ops test -b CPU -o MUL_MAT_ID` in fresh flag-off/on processes instead.

After deliberately stopping the deployed server through its launcher:

```bash
bash ~/claude-opt/p6_arms.sh
# Optional historical ubatch, reported separately:
P6_UBATCHES='1280 768' bash ~/claude-opt/p6_arms.sh
# Optional deployed codec control (two additional arms):
P6_TURBO4_CONTROL=1 bash ~/claude-opt/p6_arms.sh
```

The driver runs in `~/claude-opt`, defaults `BUUN_ROOT` to `~/Projects/buun-p6-rtx-mi50` (override
allowed), and refuses to start if `pgrep -a -x llama-server` finds a server, including before each arm.
This matters because the existing `run_arm.sh` cleanup can kill every process named `llama-server`.
It pins context 150000, batch 2048, ubatch 1280, target/draft threads 6, target q4_0/q4_0, draft f16/f16,
MTP n=3, draft ubatch 256, cache expert parallelism 0, default top_k (empty override) and scheduler
timing off. Throughput uses `--only pp6k,pp32k,tg512f --reps 5`. Inherited greedy, quality, I/O and
benchmark selections are reset so they cannot leak into another arm. Existing artifact names are
reused on another invocation; archive a completed run before rerunning to preserve session evidence.

| Arm names | Configuration / purpose |
|---|---|
| `p6-u1280-greedy-{off-a,off-b,on}` | mmap, cache off, greedy only, no bench or iostat; off/off then off/on comparisons are informational (`|| true`) |
| `p6-u{1280,768}-{mmap,direct}-r{1,2}-mt{0,1}` | auto cache, five-repetition throughput; round 1 off/on, round 2 on/off; ub768 only when requested |
| `p6-u1280-{mmap,direct}-cpu-mt{0,1}` | cache-off throughput controls |
| `p6-u1280-{mmap,direct}-quality-mt{0,1}` | auto cache, existing quality + ~142k-token needles, no throughput benchmark |
| `p6-u1280-turbo4-mt{0,1}` | optional mmap, turbo4/turbo4, auto cache, throughput plus quality/needles; separate codec results |

Direct arms use 16 I/O threads and row cache 1048576. There are 19 default arms, 27 with ub768 added,
and two more with the turbo4 control. Each call starts a fresh server. The driver leaves production
launchers and their defaults untouched.

**Artifacts determine success.** `run_arm.sh` can exit successfully after a startup/request failure.
Inspect `NAME.launcher.out`, `NAME.log` for listening/startup/request/OOM/cache dispatch or collect
errors; inspect benchmark errors and usable repetitions, `NAME.quality.{out,json}` for **10/10**,
`NAME.longctx.{out,json}` for **3/3** needles at 10/50/90%, and `NAME.vram` for **about 500 MiB or more
ROCm0 free after load**. No OOM or new cache errors are allowed. Greedy identity is never an adoption
gate: exactness comes from kernel and saved-input graph parity.

Compare matched same-session medians and spreads separately for mmap/direct and every codec/ubatch.
Require a reproducible ub1280 auto-cache tg gain beyond roughly ±2 t/s per-repetition noise, with no
prefill/latency regression beyond noise; pp32k can drift about 2% across sessions. Exclude boot prefill
and retain five usable repetitions (collect more if needed). Record cache hits/pools, draft acceptance,
I/O, peak RAM/VRAM and separate matched CPU-op profiling. Kernel microbenchmarks alone do not justify
adoption. Roll back with `GGML_CPU_IQ_MT=0` or unset it, then start a fresh server.

## Deploy

```bash
scp -P 2222 -i ~/.ssh/id_llm scripts/rtx-mi50/harness/* leshqa@176.96.135.203:claude-opt/
```
