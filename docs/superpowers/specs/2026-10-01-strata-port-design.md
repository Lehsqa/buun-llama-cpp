# Strata feature port — design (2026-10-01)

Branch: `opt/strata-port` (from `opt/rtx-mi50` @ `dd86ebea2`). Source studied: `Niko1221/Strata` @ `rc/0.1.30`
(`86fd27f`). Host: RTX 3080 Ti 12 GB (CUDA0) + MI50 32 GB (ROCm0, gfx906 wave64, no VMM), Ryzen 7600X (AVX2 only),
30 GiB RAM, NVMe; model `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64` (`qwen4exp`). Host facts: `HANDOFF.md`,
`scripts/rtx-mi50/REPORT.md`.

## 1. Intent

The user asked for the useful optimisation features of Strata to be ported into buun-llama-cpp, implemented and tested
on the target host. After the comparison (§2) the user selected eight features (P1–P8 below); a measurement package
(P0) comes first.

Success: each package, behind its own flag, measured on the target host against the REPORT §2 reference arm
(ub1280, q4_0/q4_0 KV, default indexer top_k, MTP n=3, 21 CPU blocks, cache auto) with
`quality.py` 10/10 and the 142k needle test 3/3 still passing. Flags default off. The deployed launcher
(`~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh`) and the rollback launchers are not edited; promotion of a flag
is the user's decision after seeing the numbers.

## 2. Comparison summary (Strata vs buun)

Strata is a standalone engine (~51 k lines, CUDA-first, ggml kernels borrowed) for the same model; nothing merges
directly — every feature is re-implemented inside llama.cpp/ggml. Strata numbers are from its RTX 5070 / 64 GB box;
gains for this host are estimates.

| Strata feature | buun today | Selected |
|---|---|---|
| Prefill: experts streamed through a slot ring on a copy stream, pinned staging, next layer during current layer (+14 % alone) | per-ubatch ~21 GB upload from pageable mmap on the compute stream, ids readback sync per node, no lookahead (`ggml-backend.cpp` compute_splits) | P4 |
| PLE: O_DIRECT pread pool (16), 8-way row cache 1 M rows, next-chunk prefetch | `get_rows` on a lazily mmapped CPU tensor, serial 4 KiB faults (34× read amplification), first split, no overlap | P1 |
| Decode `--pcie-frac`: part of the cache misses DMA'd and computed on GPU | misses always computed on CPU; copy engine idle in decode | P5 |
| Indexer keys stored pooled per 4-cell block | raw f16 per cell, re-pooled over all n_kv each ubatch (`build_qsa_top_k`) | P7 |
| Cost-aware draft window (`DraftPolicy`), lookup drafts up to 5 deep | priority list of drafters, CopySpec capped at `n_max`, 3→2 heuristic | P3 |
| Greedy verify argmax in the GPU graph | device argmax exists for DFlash only; MTP copies T×248 320 logits to host | P2 |
| AVX2 multi-token iq2_s/iq3_s expert kernels | per-token `vec_dot` | P6 |
| `--kv-resident`: KV in pinned RAM, page cache in VRAM | none | P8 (low payoff here: CUDA0 holds 1 of 12 attention layers) |
| Conversation checkpoints, penalties over verified rows, coupled sampling | equal or better (`--ctx-checkpoints`, exact-q rejection sampling) | — |
| Draft-vocab subset, heat-profile prewarm, calibrate.py, multi-GPU per-card caches | partial / VRAM-blocked | not in scope |

Config-only items found along the way (no code; to be measured in P0): MTP draft KV f16 → q8_0 (~137 MiB CUDA0),
`--draft-p-min` sweep, `--ctx-checkpoints` vs RAM, `-t` sweep.

## 3. Structure

- One integration branch `opt/strata-port`; one branch per package `strata/pN-<name>` cut from it and merged back only
  after its gate passes. Failed packages stay as archived branches. Commit titles `<module> : <title>`.
- Each package gets its own implementation plan (writing-plans) and is executed in order P0 → P8. P5 reuses P4's copy
  stream / staging; P8 is last because it is the largest risk.
- Every flag defaults to off; with the flag off, code paths are unchanged (greedy output identical to the reference).
- `scripts/rtx-mi50/REPORT.md` is updated after every package (results, commands, open issues) so a later session can
  resume.

### Shared test protocol

1. Dev box (CPU build `build-cpu/`, `-DLLAMA_BUILD_TESTS=ON`): package unit/parity test + `test-llama-archs -a qwen4exp`.
2. Target host: rebuild per REPORT §1; `test-backend-ops` on CUDA0 and ROCm0 for any new/changed GPU op;
   `run_arm.sh` arms: smoke, bench (`pp6k,pp32k,tg512f --reps 5`, boot sample excluded), `QUALITY=1`.
3. Exactness: packages declared exact (P1, P2, P3, P4, P6, P7-f32, P8) must give token-identical greedy output against
   the flag-off arm on the same build; P5 and P7-f16 are inexact and are gated on quality, needles and NMSE.
4. Memory: `llama-fit-params --fit-print` per device; ≥ ~500 MiB free on ROCm0 after load (REPORT §6.3).

## 4. Packages

### P0 — measurement

- `GGML_SCHED_TIMING=1` (env): in `ggml_backend_sched_compute_splits`, per split: wall time of input copies, of the
  MUL_MAT_ID ids readback sync, of compute (synchronising after each split — diagnostic only). One summary line per
  graph compute at INFO: backend, n_splits, copy ms, ids-sync ms, compute ms, bytes copied.
- Harness (`~/claude-opt` on target, mirrored into `scripts/rtx-mi50/harness/`): `run_arm.sh` records
  `/proc/vmstat pgmajfault` delta, `/proc/PID/io read_bytes`, an `iostat -x` sample during bench, `fincore` of the PLE
  shard and expert shards after the run.
- Run the reference arm with timing and the config-only items (§2) once; record in REPORT.
- Gate: timing off = no behaviour change; numbers recorded.

### P1 — PLE direct reader

- Flags: `--ple-io mmap|direct` (default `mmap`), `--ple-io-threads N` (16), `--ple-row-cache N` rows (1 048 576).
  New `llama_model_params` fields; `common` plumbing in `common/arg.cpp`, `common/common.{h,cpp}`.
- `src/llama-ple-reader.{h,cpp}` (added to `src/CMakeLists.txt`), owned by `llama_model`, opened after
  `init_mappings` on the real load only (not on the fit / no_alloc pass):
  - `O_DIRECT` fd on the shard that holds `per_layer_token_embd`, absolute offset from the loader's weights map;
    row size `ggml_row_size(type, ne0)`. Fallback chain: `O_DIRECT` refused → buffered `pread`; non-Linux → mmap path.
  - pread thread pool, 4 KiB-aligned slots, rows on the same page deduped, jobs sorted by offset;
    8-way set-associative row cache with round-robin replacement; thread-safe (target ctx, MTP ctx, server).
  - API: `issue(rows, n, out) → ticket`, `collect(ticket)`, `prefetch(rows, n)` (cache-warm only), `stats()`.
  - Stats line on context free and every N requests: hit %, reads, bytes, p50/p99 µs, blocked ms.
- `src/models/qwen4exp.cpp`: with direct I/O, `build_inp_ple` replaces `ggml_get_rows(per_layer_tok_embd, rows)` by an
  F32 input `[ne0, n_rows]` (scale/bias ops unchanged); `llm_graph_input_ple::set_input` computes the hashes as today,
  `issue` + `collect`, dequantises with `ggml_get_type_traits(type)->to_float` and `ggml_backend_tensor_set`;
  `can_reuse` checks the new tensor's shape. Image-token rows keep their path.
- Prefetch: after ubatch k is submitted, the context calls a model hook that computes ubatch k+1's hashes
  speculatively and calls `prefetch`. `set_input` remains authoritative; a wrong guess only misses the cache.
- Gate: dev-box test — rows from mmap and direct paths byte-identical over random row sets incl. page-straddling rows,
  and identical logits on a tiny qwen4exp model; target: token-identical greedy output, pp/tg, pgmajfault, iostat.

### P2 — greedy MTP verify on GPU

- Flag `--spec-gpu-argmax` (default off).
- Add the MTP draft type to the device-argmax verify path in `tools/server/server-context.cpp` (today DFlash only),
  after confirming that MTP's draft step reads only the nextn embeddings, not host logits.
- Used only when the request is greedy (temperature ≤ 0 or top_k 1, no penalties, no grammar, no logit bias, no
  n_probs); otherwise the CPU sampler path is used unchanged.
- Gate: token-identical greedy output on/off; tg512f at temperature 0.

### P3 — cost-aware draft length, deeper copy drafts

- Flags `--spec-policy fixed|cost` (default `fixed`), `--spec-mtp-depth N` (default = `--draft-max` when unset,
  i.e. today's behaviour).
- `common/speculative-draft-policy.h` (+ `.cpp`): port of Strata `DraftPolicy` — EMA of verify-round cost per window
  size T (ms), acceptance per source (MTP per depth; copy drafts per match-length bucket, with priors), choose T
  maximising expected accepted tokens per ms, probe untried sizes when confidence ≥ 0.85.
- `common_speculative_draft`: MTP drafts up to `--spec-mtp-depth`; if the copy drafter's first token equals MTP's
  first draft, its continuation may extend the window up to `--draft-max`; the policy then picks T.
  `common_speculative_accept` gains the measured round time (ms) passed by the server.
- With `cost`, replaces the 3→2 adaptive heuristic. Verification unchanged → exact.
- Gate: dev-box unit tests for the policy (synthetic cost/acceptance traces); target: token-identical greedy output,
  tg on a code-edit corpus and on prose, acceptance + round-time logs.

### P4 — prefill expert streaming (CUDA0 op-offload)

- Flags `--prefill-weight-stream off|stage|ahead` (default off), `--prefill-weight-stream-staging N×MiB` (8×32).
- P4a `stage` (no extra VRAM): pinned host staging ring (`cudaHostAlloc`), dedicated non-blocking copy stream on
  CUDA0, 2–3 helper threads `memcpy` from the mmap into the ring with `madvise(WILLNEED)` ahead; DMA ring → the
  scheduler's input-copy tensor; compute stream waits on an event. For MUL_MAT_ID weight inputs with
  `n_tokens·n_used ≥ 2·n_expert`, skip the ids readback and copy the whole tensor.
- P4b `ahead`: offloaded weight inputs get dedicated device slots outside gallocr's reuse, so layer L+1's upload runs
  during layer L and the next ubatch's first CPU layers load during the previous ubatch's tail. Slots are borrowed
  from moe-cache slabs for the duration of a prompt; afterwards evicted experts are refilled in heat order on the
  cache's fill stream.
- Interface `ggml_backend_weight_stream_iface` (new header `ggml/src/ggml-backend-weight-stream.h`, next to `ggml-backend-moe-cache.h`), resolved by proc
  address like the moe-cache; CUDA implementation `ggml/src/ggml-cuda/weight-stream.cu`; scheduler hooks in
  `ggml/src/ggml-backend.cpp` (`split_graph` marks streamable inputs; `compute_splits` waits/releases).
- Optional P4c: experts resident in the moe-cache copied device-to-device instead of over PCIe.
- Gate: byte-identical weights → token-identical greedy output; pp6k/pp32k up; tg512f not lower after refill;
  P0 timing shows upload wait reduced.

### P5 — decode: share of cache misses computed on GPU

- Flags `--moe-cache-pcie-frac F` (default 0), `--pin-experts-gib N` (default 0).
- Step 0: target probe — `cudaHostRegister(ReadOnly|Portable)` on a file-backed `MAP_SHARED` range with CUDA 12.0.
  If it fails, fetched experts are staged through P4a's ring by a helper thread.
- Pinning: per-layer expert slices registered up to N GiB (hottest-miss CPU layers first, from the heat profile);
  refused if `MemAvailable` would fall below 6 GiB.
- In the moe-cache plan: the last ⌈F·misses⌉ misses that are pinned or stageable are marked fetch; `dispatch` issues
  their H2D on the copy stream into ~64 staging pseudo-slots (~128 MiB CUDA0); after the hits' kernel, the compute
  stream waits on the copy event and runs the same kernel over the staged pointers; the CPU drops those rows.
  Files: `ggml/src/ggml-cuda/moe-cache.cu`, `ggml/src/ggml-backend-moe-cache.h`, `ggml/src/ggml-cpu/ggml-cpu.c`,
  `src/llama-model-loader.cpp`, `common/arg.cpp`.
- Inexact (GPU vs CPU activation quantisation). Gate: `test-backend-ops` NMSE GPU-vs-CPU for the fetched path,
  quality 10/10, needles 3/3, tg512f sweep F ∈ {0, .2, .35, .55, .75}.

### P6 — AVX2 multi-token iq2_s / iq3_s kernels

- Env `GGML_CPU_IQ_MT=1` (default off).
- `ggml/src/ggml-cpu/arch/x86/quants.c`: `vec_dot_iq2_s_q8_K_mt` / `vec_dot_iq3_s_q8_K_mt` — decode each weight block
  once, accumulate up to 8 activation columns. Per column, the accumulation order is identical to the single-token
  kernel → bit-identical results.
- `mul_mat_id` CPU path (incl. moe-cache miss rows) uses it when an expert serves ≥ 2 rows. iq4_nl unchanged.
- Gate: dev-box test — bit-exact vs per-column `vec_dot` over random blocks and 1–8 columns;
  `test-backend-ops -b CPU -o MUL_MAT_ID`; target tg512f.

### P7 — pooled indexer keys

- Flag `--qsa-pooled-keys off|f32|f16` (default off).
- Indexer memory (`src/llama-memory-hybrid-idx.{h,cpp}`): per attention layer, a pooled store `[idx_dim, max_blocks]`
  (f32 or f16) holding, per finished block, the pooled → RMS-normed → roped key exactly as `build_qsa_top_k` computes
  it today; the raw per-cell cache shrinks to a window of the most recent `n_ubatch + 2·r + draft_max` cells.
- Graph (`src/models/qwen4exp.cpp`): per ubatch, compute pooled keys only for blocks the ubatch touches, from
  f16-rounded raw keys with today's op order, `set_rows` them into the store; scoring reads the store. The O(n_kv)
  gather/pool/norm/rope disappears.
- Truncation: inside the raw window → exact recompute of the tail block. Beyond it, inside a block whose raw keys
  are gone → `seq_rm` returns false (server re-processes, as for recurrent state). With P7 on, server checkpoint
  positions are rounded down to a multiple of r.
- Scope: single sequence (`--parallel 1`); otherwise refused at startup. State save/load includes store + window.
- Memory at 150k (12 layers): raw 440 MiB → f32 ~220 MiB / f16 ~110 MiB.
- Gate: dev-box tiny random qwen4exp — f32 pooled vs raw bit-identical logits through prefill, decode, draft rollback
  and checkpoint restore; target: fit-print, quality, needles, pp/tg.

### P8 — long-context KV streaming

- Flag `--kv-resident N` cells (default 0 = off; minimum 20 480).
- Host copy: per attention layer full K/V in pinned mapped memory allocated by the owning backend (CUDA
  `cudaHostAlloc(Mapped|Portable)`, HIP `hipHostMalloc(Mapped)`); device page pool of N cells (page = 4-cell block)
  plus page table, in the KV buffer.
- New op `GGML_OP_KV_PAGE_RESOLVE` (CPU reference, CUDA, HIP): selected block ids → hits marked, misses claimed once by
  CAS, CLOCK victims excluding pages used by this call, misses read over PCIe as zero-copy loads, output slot indices
  for the existing gather. No warp-size assumptions (block atomics + shared memory) — valid on wave64. Host pointer
  passed via op params + a proc-address vtable, never as a graph tensor.
- Writes: write-through to host and to the resident page. Prefill: identity mapping while n_kv ≤ N; beyond it, a
  per-device staging buffer (one layer's K/V) filled by async copy one attention layer ahead of flash-attention.
- Scope: single sequence; KV types q4_0/q8_0/f16; turbo types refused. `seq_rm`, save/load, checkpoints operate on the
  host copy + page table. Refused if pinning drops `MemAvailable` below 6 GiB.
- Expected: ROCm0 −0.55…0.72 GiB net, CUDA0 ~−70 MiB, ~1 GB pinned RAM.
- Gate: `test-backend-ops` for the new op on CPU/CUDA0/ROCm0; token-identical greedy output; 142k needles; pp/tg;
  RAM and VRAM numbers.

## 5. Invariants (unchanged from REPORT §8)

`--fit off`; anchored `^output\.weight$=CUDA0`; 21 CPU blocks with cache auto; PLE lazy mmap remains the default;
output head on CUDA0; one server at a time on the host; `HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0`; old launchers
runnable; never copy into `runtime/` while a server from it runs; ≥ ~500 MiB free on ROCm0 after load.

## 6. Risks

- P4b/P5 touch the scheduler and moe-cache hot paths shared by all models — flag-off must be a strict no-op.
- P5 pinning and P8 pinned KV compete with the page cache on a 30 GiB host; both enforce a MemAvailable floor.
- P7/P8 change memory-module semantics (`seq_rm` may refuse); single-sequence restriction contains it.
- P8 adds a ggml op across three backends; HIP correctness on gfx906 must be shown by `test-backend-ops` on ROCm0.
- Gains are estimates; any package that does not beat the reference stays an archived branch.
