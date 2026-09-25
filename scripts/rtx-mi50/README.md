# opt/rtx-mi50 — Qwen3.8-Flash-Next on RTX 3080 Ti + MI50

Branch rooted at the production pin `ed774445` (see `HANDOFF.md`), so the only differences from the
deployed runtime are the commits on this branch plus the kit patches the build script applies.

## Result (measured on the target, 2026-09-25)

All arms use q4_0/q4_0 KV, the default indexer `top_k`, 150k ctx, MTP n_max 3 and 21 CPU expert
blocks. Each figure is a `bench.py` median of 5 reps, with the boot rep dropped for pp.

| arm | pp6k | pp32k | tg512f | cache hits | cache pools |
|---|---|---|---|---|---|
| production runtime, ub768 | 285.9 | 259.5 | 34.4 | 49.9% | 2020 MiB |
| this branch, ub768 | 279.5 | 255.5 | 36.9 | 64.9% | 3904 MiB |
| this branch, ub1024 | 332.1 | 302.3 | 35.7 | 63.8% | 3734 MiB |
| **this branch, ub1280** | **340.5** | **305.2** | **36.5** | 64.0% | 3506 MiB |
| this branch, ub1536 | 335.3 | 304.7 | 37.6 | 61.2% | 3323 MiB |

**Recommended: `UBATCH_SIZE=1280`.** Against production that is +19% pp6k, +18% pp32k and +6% tg.
Prefill plateaus above 1280, and the tg spread between the 1024–1536 arms is within draft-acceptance
noise.

Quality at ub1280: `quality.py` passed 10/10. `longctx.py` at 142 266 tokens, with needles at
10/50/90%, found 3/3. That long request ran at pp 187.4 t/s and tg 23.7 t/s, against the HANDOFF
baseline of pp 164.9 and tg 20.8.

Compute buffers (target, from the server log, ub768): CUDA0 1754 → 944 MiB, ROCm0 1649 → 1014 MiB.
MTP draft CUDA0: 1493 → 498 MiB. ROCm0 free after load: production ub768 591 MiB. On this branch
it is 1225 / 985 / 745 / 505 MiB at ub 768 / 1024 / 1280 / 1536.

## What changed

1. **QSA block-level top-k (prefill / scan path)** — `src/models/qwen4exp.cpp`,
   `src/llama-memory-hybrid-idx.*`. The indexer used to expand every block score to its cells and
   rank all `n_kv` cells per query. Each of those F32 `[n_kv, n_ubatch]` surfaces is
   150016 × 736 × 4 B ≈ 421 MiB. The scan path now ranks the `n_kv/ratio` blocks directly and
   expands only the chosen blocks. It selects the forced tail block plus the best `top_k/ratio`
   whole blocks, which is the reference budget. The old path also took `ratio-1-tail` arbitrary
   tied cells. Decode (gather path) is unchanged. `LLAMA_QSA_BLOCK_TOPK=0` restores the old
   selection.
2. **Per-head indexer scoring in prefill**, and a scan mask built by copying the selected cells'
   causal-mask values. The result is bit-identical, with one mask surface less.
3. **One shared KQ-mask view per graph.** The mask is a host input, and the scheduler copies every
   distinct view of it to each device, so a per-layer view cost 219 MiB per layer on ROCm0. Find
   this kind of thing with `GGML_SCHED_DEBUG=2` plus `llama-fit-params --fit-print on -lv 5`.
4. **MTP draft context ubatch capped at 256** (`LLAMA_MTP_DRAFT_UBATCH`). The draft only replays
   target hidden rows, but it inherited the target ubatch and reserved like a target prefill graph.
5. **CUDA radix top-k for CCCL < 3.2.** CUDA 12.0 / CUB 2.0.1 used a full segmented argsort.
6. **Wave64 fix in the stable top-k tie kernel.** It strode by `warpSize` (64) with 32 launched
   threads.

## Build (target host)

```bash
git clone -b opt/rtx-mi50 git@github.com:Lehsqa/buun-llama-cpp.git ~/Projects/buun-opt-src
cd ~/Projects/buun-opt-src
BUILD_JOBS=6 bash scripts/rtx-mi50/build.sh      # ~25 min; runtime -> ~/Projects/buun-opt-rtx-mi50/runtime
```

This applies kit patches 0001/0003/0004/0005 from `~/Projects/qwen38-perf/buun-rtx-mi-kit`, uses
the HANDOFF §3 flags, and merges CUDA `bin/*` + HIP `libggml-hip.so*` into a new runtime. The
production runtime is untouched. For incremental rebuilds after `git pull`, most changes touch only
`libllama` and the server:

```bash
O=~/Projects/buun-opt-rtx-mi50
cmake --build $O/cuda -j12 --target llama-server llama-fit-params && cp -a $O/cuda/bin/. $O/runtime/
```

Only copy libraries while no server from that runtime is running.

## Run

The deployed launcher already has a `BUUN_ROOT` knob. It hardcodes
`--override-kv qwen4exp.attention.indexer.top_k=int:4096` and defaults the KV to turbo4, which
are the two unvalidated operator experiments. Run the measured configuration from a copy without
those:

```bash
sed 's#^  --override-kv qwen4exp.attention.indexer.top_k=int:4096$#  ${TOPK_OVERRIDE:+--override-kv qwen4exp.attention.indexer.top_k=int:$TOPK_OVERRIDE}#' \
    ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh > ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next_opt.sh
BUUN_ROOT=~/Projects/buun-opt-rtx-mi50 KV_K=q4_0 KV_V=q4_0 UBATCH_SIZE=1280 \
    bash ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next_opt.sh
```

Estimate per-device memory for any ubatch without loading weights:

```bash
R=~/Projects/buun-opt-rtx-mi50/runtime
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 $R/llama-fit-params --fit-print on -m <model> \
  -lm mmap -lzm on --mmap-prefetch off -ot '<same -ot as the launcher>' -ngl all \
  --device CUDA0,ROCm0 -sm layer -ts 3,32 -c 150000 -ctk q4_0 -ctv q4_0 -fa on -b 2048 -ub 1280
```

It prints `device model context compute` in MiB. The MTP draft context is not included.

## Where prefill time goes now

With ub1280, RTX PCIe RX peaks at 15–20 GB/s but averages ~5 GB/s. Every CPU-resident expert
matmul is op-offloaded to CUDA0 (backend 0), about 21 GB per ubatch. The CPU is ~92% idle, and the
RTX mostly waits on the MI50 part of the layer pipeline. Pinning the expert weights would gain
little. The remaining levers are overlapping the next offloaded layer's weight upload with MI50
compute, and rebalancing layers between the GPUs.

## Known limitations

- Block selection needs the direct cache layout: one cell per position, which covers all text.
  An M-RoPE image prompt falls back to the per-cell graph, which is larger and reallocated at
  runtime. Keep `LLAMA_QSA_BLOCK_TOPK=0` if images are served on nearly full GPUs.
- ub1536 fits today with ~0.5 GiB left on ROCm0. Leave that headroom for the hipBLAS handle,
  which is created lazily at the first prefill.
