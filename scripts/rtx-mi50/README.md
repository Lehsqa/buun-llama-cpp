# opt/rtx-mi50 — Qwen3.8-Flash-Next on RTX 3080 Ti + MI50

Branch rooted at the production pin `ed774445` (see `HANDOFF.md`), so the only differences from the
deployed runtime are the commits on this branch plus the kit patches the build script applies.

## What changed

1. **QSA block-level top-k (prefill / scan path)** — `src/models/qwen4exp.cpp`,
   `src/llama-memory-hybrid-idx.*`. The indexer used to expand every block score to its cells and
   rank all `n_kv` cells per query. That costs several F32 `[n_kv, n_ubatch]` surfaces — each
   150016 × 736 × 4 B ≈ **421 MiB**, the exact size of the unexplained ub704→736 CUDA0 step — plus a
   top-k over 150k columns. The scan path now ranks the `n_kv/ratio` blocks directly and expands only
   the chosen blocks to cells. It selects the forced tail block plus the best `top_k/ratio` whole
   blocks, which is the reference budget. The old path also took `ratio-1-tail` arbitrary cells
   from the next tied block. The decode/gather path is unchanged.
2. **Per-head indexer scoring in prefill** — the 4-head score surface is no longer materialised at once.
3. **Scan mask without the full add** — selected cells copy their causal-mask value into the `-inf`
   mask. The result is bit-identical and needs one `[n_kv, n_ubatch]` f16 surface less.
4. **CUDA top-k radix select for CCCL < 3.2** — CUDA 12.0 ships CUB 2.0.1, so `ggml_top_k` with more
   than 1024 columns used to run a full segmented argsort plus about 3 row-sized pool buffers. It now
   uses the same radix select HIP already runs, which also means less VMM pool growth on CUDA0.
5. **Wave64 fix for stable top-k** — the tie-collection kernel strode by `warpSize` (64 on gfx906)
   while it was launched with 32 threads, so it skipped half of every chunk.

Dry-run compute buffer at the shipped geometry (CPU-only no_alloc probe, 150k ctx, q4_0 KV,
ratio 4), so these are graph-allocator numbers and not per-device CUDA0/ROCm0 values:

| ubatch | before | after |
|---|---|---|
| 512 | 1137 MiB | 550 MiB |
| 704 | 1549 MiB | 757 MiB |
| 736 | 1617 MiB | 791 MiB |
| 768 | 1686 MiB | 825 MiB |
| 1536 | 3153 MiB | 1650 MiB |

`LLAMA_QSA_BLOCK_TOPK=0` restores the old per-cell selection with the same binary. Items 3–5 are
always on. Use that variable for a clean A/B.

## Build (target host)

```bash
git clone -b opt/rtx-mi50 git@github.com:Lehsqa/buun-llama-cpp.git ~/Projects/buun-opt-src
cd ~/Projects/buun-opt-src
BUILD_JOBS=6 bash scripts/rtx-mi50/build.sh      # ~25 min; runtime -> ~/Projects/buun-opt-rtx-mi50/runtime
```

The script applies kit patches 0001/0003/0004/0005 from `~/Projects/qwen38-perf/buun-rtx-mi-kit`
and uses the HANDOFF §3 flags. It merges CUDA `bin/*` and HIP `libggml-hip.so*` into a new runtime,
then runs `ldd` and `--list-devices`. The production runtime is not touched.

Kernel checks:

```bash
R=~/Projects/buun-opt-rtx-mi50/runtime
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
$R/test-backend-ops test -b CUDA0 -o TOP_K
$R/test-backend-ops test -b ROCm0 -o TOP_K
$R/test-backend-ops test -b CUDA0 -o GET_ROWS
$R/test-backend-ops test -b CUDA0 -o SET_ROWS
$R/test-backend-ops test -b CUDA0 -o MUL_MAT_ID
```

## Launch the new runtime with the deployed launcher

```bash
L=~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh
sed 's#buun-rtx-mi-ed774445cd69-cuda/runtime#buun-opt-rtx-mi50/runtime#g' "$L" > /tmp/qwen-opt.sh
grep -n 'buun-opt-rtx-mi50/runtime' /tmp/qwen-opt.sh     # must show the replaced path
DRY_RUN=1 bash /tmp/qwen-opt.sh                          # argv check
```

Only one server can run at a time, so stop the current `:10000` server first. It is running the
unvalidated turbo4 + `top_k=4096` arm. Use the reference config: q4_0/q4_0 KV and the default `top_k`.

## A/B plan (medians of ≥5, drop the boot sample)

```bash
B=~/Projects/qwen38-perf/bench.py
# arm 1: old QSA path, same binary, ub768 (should reproduce HANDOFF §5: 286 / 256 / 34.2)
LLAMA_QSA_BLOCK_TOPK=0 bash /tmp/qwen-opt.sh
python3 $B --url http://127.0.0.1:10000 --only pp6k,pp32k,tg512f --reps 5 --tag opt-cell-ub768
# arm 2: new path, ub768 (cache pools should grow: compare the [moe-cache] pool lines)
bash /tmp/qwen-opt.sh
python3 $B --url http://127.0.0.1:10000 --only pp6k,pp32k,tg512f --reps 5 --tag opt-blk-ub768
# arm 3/4: spend the freed memory on prefill
UBATCH_SIZE=1024 bash /tmp/qwen-opt.sh   # then the same bench, --tag opt-blk-ub1024
UBATCH_SIZE=1536 bash /tmp/qwen-opt.sh   # then the same bench, --tag opt-blk-ub1536
```

In each server log, grep:

```bash
LOG=~/.local/state/llama-launcher/llama-server-qwen38next-buun.log
grep -E "compute buffer size|moe-cache\].*(pool|enabled|dormant)|hits" "$LOG"
```

Look for CUDA0/ROCm0 compute buffers about 0.8–0.9 GiB smaller at ub768, the ub704→736 step gone,
larger `[moe-cache]` pools, and a higher hit rate. Run quality and long context on the winning arm:

```bash
python3 ~/Projects/qwen38-perf/buun-rtx-mi-kit/quality.py --url http://127.0.0.1:10000 --tag opt-blk --output /tmp/q.json
python3 ~/Projects/qwen38-perf/buun-rtx-mi-kit/longctx.py --port 10000 --reps 7900 \
        --needles PASS-A,PASS-B,PASS-C --positions 0.1,0.5,0.9 --max-tokens 512 --output /tmp/n.json
```

A follow-up the freed ROCm0 memory may allow: one fewer CPU expert block (each is about 0.9 GiB).
For example, set `CPU_BLOCKS` to drop layer 31, if ROCm0 has that headroom after arm 2.

## Known limitation

Block selection needs the direct cache layout: one cell per position, which covers all text. An
M-RoPE image prompt with repeated positions falls back to the per-cell graph, which is ~0.8 GiB
larger at ub768. That graph is reallocated at runtime, so on a nearly full GPU keep
`LLAMA_QSA_BLOCK_TOPK=0` if images are served.
