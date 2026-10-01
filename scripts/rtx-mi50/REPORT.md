# opt/rtx-mi50 — work report and handoff (2026-09-24 … 2026-09-25)

Audience: the next agent continuing optimisation of buun-llama-cpp for the RTX 3080 Ti + MI50 host.
Read `HANDOFF.md` (repo root, untracked, original brief) for host/model/toolchain facts first; this
file records what was done on top of it, how it was measured, and what to do next.

---

## 1. Access and environments

| what | where / how |
|---|---|
| Dev box (this repo) | `~/Projects/buun-llama-cpp`, **no GPU, no CUDA/ROCm**. CPU-only build in `build-cpu/` (`-DLLAMA_BUILD_TESTS=ON`) for compile checks + `test-llama-archs`. |
| Target host (LLM server) | `ssh -i ~/.ssh/id_llm -p 2222 leshqa@176.96.135.203` (hostname `leshqa-MS-7D76`). User granted access for testing. Can be offline (No route to host on 2026-10-01). |
| GitHub fork | `git@github.com:Lehsqa/buun-llama-cpp.git`, git remote `fork`. Push needs `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_github -o IdentitiesOnly=yes"`. No `gh` CLI. |
| Branch | `opt/rtx-mi50`, rooted at production pin `ed774445cd69` (NOT master — master has 21 newer commits). |
| Target source clone | `~/Projects/buun-opt-src` (tracks `fork/opt/rtx-mi50`; has kit patches 0001/0003/0004/0005 applied as uncommitted changes — expected, do not commit them). |
| Target build root | `~/Projects/buun-opt-rtx-mi50/{cuda,hip,runtime}`. `runtime/` = merged CUDA bin + HIP `libggml-hip.so`. |
| Production runtime (untouched) | `~/Projects/buun-rtx-mi-ed774445cd69-cuda/runtime` |
| **Deployed launcher (user promoted it, verified 2026-10-01)** | `~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh` = the generated opt launcher (BUUN_ROOT `buun-opt-rtx-mi50`, ub1280, `TOPK_OVERRIDE` empty, log `llama-server-qwen38next-buun-opt.log`, pid file `llama-server-buun-opt-10000.pid`). The **only user edit**: `KV_K/KV_V` defaults changed back to **turbo4**. turbo4 at ub1280 is NOT benchmarked or quality-checked (all §2 numbers are q4_0). `_opt.sh` no longer exists. |
| Rollback launchers | `RTX+MI_Qwen3.8-Flash-Next_old.sh` = the former buun production launcher (production runtime, ub768, turbo4, hardcoded `top_k=4096`). `_buun_old.sh` = an older buun launcher (2026-09-20). No launcher runs the pre-buun llama.cpp `2857e5114` binary any more; only the comments mention it. |
| systemd | `~/.config/systemd/user/qwen-flash-llama.service`: `ExecStart=%h/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh`, so it now **starts the opt runtime**. Currently inactive. |
| Remote harness | `~/claude-opt/` on target (see §5). |

### Rebuild on target after pushing new commits
```bash
ssh -i ~/.ssh/id_llm -p 2222 leshqa@176.96.135.203
cd ~/Projects/buun-opt-src && git pull --ff-only
O=~/Projects/buun-opt-rtx-mi50
# src/, common/, tools/ changes only (~20 s):
cmake --build $O/cuda -j12 --target llama-server llama-fit-params && cp -a $O/cuda/bin/. $O/runtime/
# ggml-cuda changes: add --target ggml-cuda (~10 min). HIP-side changes:
cmake --build $O/hip -j12 --target ggml-hip && cp -a $O/hip/bin/libggml-hip.so* $O/runtime/   # ~15 min
# full clean build: bash scripts/rtx-mi50/build.sh
```
Never copy into `runtime/` while a server from it is running (`pgrep -a -x llama-server`).

---

## 2. Final results (target, 150k ctx, q4_0/q4_0 KV, default top_k, MTP n=3, 21 CPU blocks, cache auto)

`bench.py` medians of 5 (pp: boot rep dropped).

| arm | pp6k | pp32k | tg512f | cache hits | pools | ROCm0 free after load |
|---|---|---|---|---|---|---|
| production runtime, ub768 | 285.9 | 259.5 | 34.4 | 49.9% | 2020 MiB | 591 MiB |
| branch, ub768 | 279.5 | 255.5 | 36.9 | 64.9% | 3904 MiB | 1225 MiB |
| branch, ub1024 | 332.1 | 302.3 | 35.7 | 63.8% | 3734 MiB | 985 MiB |
| **branch, ub1280 (default)** | **340.5** | **305.2** | **36.5** | 64.0% | 3506 MiB | 745 MiB |
| branch, ub1536 | 335.3 | 304.7 | 37.6 | 61.2% | 3323 MiB | 505 MiB |

Gates at ub1280: `quality.py` 10/10; `longctx.py` 142 266 tokens, needles 10/50/90% → 3/3 found,
pp 187.4 / tg 23.7 (HANDOFF baseline pp 164.9 / tg 20.8).

Compute buffers ub768 (server log): target CUDA0 1754 → 944 MiB, ROCm0 1649 → 1014 MiB;
MTP draft CUDA0 1493 → 498 MiB.

Note: HANDOFF §5 tg 34.2 matches today's production 34.4; production pp6k today was 285.9 (HANDOFF 286).

---

## 3. Commits on `opt/rtx-mi50` (oldest first)

1. `c4fe38c4d ggml-cuda : radix top-k without CCCL DeviceTopK, fix wave64 stable ties`
   - `ggml/src/ggml-cuda/top-k.cu`: radix select (`top_k_radix_cuda`) now compiled for every CUDA+CUB
     build; used when `ncols > 1024` on CCCL < 3.2 (CUDA 12.0 / CUB 2.0.1). Narrow rows keep bitonic.
   - `stable_top_k_collect_ties`: stride `blockDim.x` instead of `warpSize` (wave64 skipped half a chunk).
     The 32-bit ballot mask issue on wave64 is NOT fully audited — the stable path is still unused by this model.
2. `e4662b5d4 qwen4exp : rank QSA blocks directly in the scan path`
   - `src/models/qwen4exp.cpp`: `qwen4_qsa_block_select()` (~L799) decides block mode: needs `blk_bias`,
     ratio > 1, scan path (not gather), env `LLAMA_QSA_BLOCK_TOPK != 0`, and `qsa_layout_direct()`.
     In `build_qsa_top_k` (~L904): top-k over `n_blocks` (n_sel = ceil(width/r)) then
     `get_rows(blk_cells as [r, n_blocks, ns])` → per-cell indices of width `r*n_sel`.
     Prefill (n_tokens > 64) scores indexer heads one at a time.
     `llm_graph_input_qsa` gained `block_select` (+ `can_reuse` check) and `cells_of_blk`.
   - `src/llama-memory-hybrid-idx.{h,cpp}`: `qsa_layout_direct()` (~L478) mirrors set_input_qsa's direct
     pass; `set_input_qsa` (~L516): tolerates unallocated `cell_blk` (scratch vector), empty gather slots of
     a block point at that block's own representative cell, blocks past the causal tail get `-inf`
     (only the tail block itself gets the forced 1e9).
   - scan mask: `-inf` fill then `set_rows` of the gathered causal-mask values (bit-identical, no full add).
3. `b18fea51e scripts : rtx-mi50 build and A/B guide` — `scripts/rtx-mi50/build.sh` (+README).
4. `5542623ce qwen4exp : share one KQ-mask view across QSA layers`
   - `build_qsa_mask_cells()` (~L1116) caches one `[1, n_kv, n_q]` view of the KQ mask per graph
     (`qsa_mask_cells` map in `models.h`); scan and gather both use it; the `-inf` mask is
     `ggml_fill(mask_cells)` so no other view of the host mask exists.
   - **This fixed the target crash** (`hipblasCreate` CUBLAS_STATUS_ALLOC_FAILED): per-layer views made the
     scheduler copy the 219 MiB mask to ROCm0 8 times.
5. `3ab2be1c6 server : bound the MTP draft context's ubatch`
   - `tools/server/server-context.cpp`: `bound_mtp_draft_ubatch()` (~L5231) caps MTP ctx n_ubatch to 256
     (env `LLAMA_MTP_DRAFT_UBATCH`), applied in `create_mtp_context()` and the standalone-sidecar MTP path
     (~L7161). n_batch unchanged (replay decodes split by ubatch).
6. `99b64afc9 scripts : rtx-mi50 measured results and run instructions` — README results.

Runtime env knobs added: `LLAMA_QSA_BLOCK_TOPK=0` (old per-cell selection), `LLAMA_MTP_DRAFT_UBATCH=N`.

---

## 4. Verification methods (reuse these)

### Local (dev box, CPU only)
```bash
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu --target test-llama-archs llama llama-server -j16
./build-cpu/bin/test-llama-archs -a qwen4exp     # QSA layout, cache admission, MTP sidecar, backend NMSE
```
Ad-hoc tools (scratchpad, NOT in repo — recreate if needed; source is in this session's scratchpad dir
`/tmp/claude-1000/.../scratchpad/`, lost on reboot):
- `qsa-numerics.cpp`: tiny random qwen4exp (4 layers, ratio arg), prefills 900 tokens + 8 decodes, dumps
  logits; compare `LLAMA_QSA_BLOCK_TOPK=1` vs `=0` vs dense (`top_k=100000`) with `cmp.py` (pure python,
  no numpy on dev box). Reference: block-vs-cell NMSE 2.6e-11, cell-vs-dense 1.9e-9, argmax 100%.
- `qsa-membench.cpp`: no_alloc model at shipped geometry via `llama_model_saver` +
  `llama_model_init_from_user(no_alloc=true)`, prints CPU compute buffer per ubatch.
  **Caveat: misses multi-GPU split copies** (it showed ~825 MiB while ROCm0 really wanted 2337 MiB before fix #4).
  Build: `g++ -std=c++17 -O1 x.cpp -Iinclude -Iggml/include -Isrc -Lbuild-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-cpu/bin`

### Target: memory without loading weights (seconds, authoritative per device)
```bash
R=~/Projects/buun-opt-rtx-mi50/runtime
M=~/Models/gguf/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf
OT='^output\.weight$=CUDA0,blk\.(0|1|2|3|3[1-9]|4[0-7])\.ffn_(up|down|gate|gate_up|up_s|gate_s|down_s|up_exps|gate_exps|down_exps|gate_up_exps)\.weight=CPU'
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$R:/opt/rocm/lib
$R/llama-fit-params --fit-print on -m "$M" -lm mmap -lzm on --mmap-prefetch off -ot "$OT" -ngl all \
  --device CUDA0,ROCm0 -sm layer -ts 3,32 -c 150000 -ctk q4_0 -ctv q4_0 -fa on -b 2048 -ub 1280
# prints "device model context compute" MiB; excludes the MTP draft context (~0.5 GiB CUDA0 now)
# per-split inputs/copies: prefix GGML_SCHED_DEBUG=2 and add -lv 5 (≈9 MB output; grep "SPLIT #")
```
Production runtime has no `llama-fit-params` binary.

### Target: real runs
```bash
cd ~/claude-opt
KV_K=q4_0 KV_V=q4_0 BUUN_ROOT=$HOME/Projects/buun-opt-rtx-mi50 UBATCH_SIZE=1280 ./run_arm.sh NAME 6000   # smoke: one ~6k request
... BENCH_ARGS="--only pp6k,pp32k,tg512f --reps 5" ./run_arm.sh NAME 0                                     # bench (~15 min)
... QUALITY=1 ./run_arm.sh NAME 0                                                                          # quality.py + 142k needles (~15 min)
python3 summ.py NAME [NAME...]    # medians, cache pools, hit rate
```
Run long arms detached (`setsid nohup ... & disown`, write a `.done` marker) and poll from the dev box —
SSH commands time out at 120 s otherwise.

---

## 5. Remote harness `~/claude-opt/` (target)

- `launcher.sh` — copy of the deployed buun launcher with: `TOPK_OVERRIDE` (empty = GGUF default) instead of
  the hardcoded `top_k=4096`, and `EXP_BUFT` (buffer type for CPU expert blocks; default CPU).
- `run_arm.sh NAME [REQ_TOKENS]` — starts launcher with `LLAMA_LOG=~/claude-opt/NAME.log LV=4`, waits for
  "listening", snapshots VRAM (`NAME.vram`), optional single request / `BENCH_ARGS` (→ `NAME.bench`) /
  `QUALITY=1` (→ `NAME.quality.*`, `NAME.longctx.*`), then TERM → SIGKILL.
- `summ.py` — summarises `.bench` + log (pools, hits).
- `gen_opt_launcher.py` — generated the opt launcher from the old buun launcher (header + defaults:
  BUUN_ROOT opt runtime, q4_0 KV, ub1280, TOPK_OVERRIDE, own log/pid). Its `src`/`dst` paths are stale now
  (source became `_old.sh`; the output was promoted to the main name). Edit the paths before reusing it,
  and don't overwrite the user's turbo4 edit.
- `dmon_arm.sh` — 24k prefill while sampling `nvidia-smi dmon` + `top`.
- Result files kept: `prod-ub768-bench`, `opt2-ub{768,1024,1280}-bench` (before MTP cap),
  `opt3-ub{768,1024,1280,1536}-bench`, `opt3-ub1280-quality`, `sched0.txt/sched1.txt` (split dumps).
- Log lines worth grepping: `compute buffer size`, `\[moe-cache\].*(capacity|pool\[|hits)`, `failed`.

---

## 6. Lessons / gotchas learned this session (in addition to HANDOFF §4)

1. **Host-input views are copied per device per view.** Any new graph code reading a host input (KQ mask,
   QSA bias, blk_cells…) from many layers must build ONE view per graph and reuse it. Check with
   `GGML_SCHED_DEBUG=2` → `## SPLIT #N: ROCm0 # K inputs` listing repeated `attn_inp_kq_mask (view) (219M)`.
2. CPU no_alloc probes understate per-device buffers; always confirm with `llama-fit-params --fit-print` on target.
3. ROCm0 needs real slack after load: hipBLAS handle is created lazily at first prefill GEMM. ≲30 MiB free →
   `CUBLAS_STATUS_ALLOC_FAILED`. Keep ≥ ~500 MiB.
4. The MTP draft ctx re-reserves on first request (log shows 245 → 498 MiB CUDA0 now; was 735 → 1493).
5. `-ot ...=CUDA_Host` is not accepted (only device bufts registered in `parse_tensor_buffer_overrides`);
   adding host bufts parses but with `-lm mmap` tensors stay `CPU_Mapped` — no pinning. Reverted, not committed.
6. Prefill profile at ub1280 (24k prompt): RTX PCIe RX bursts 15–20 GB/s, avg ~5 GB/s; SM ~30–50% duty;
   CPU ~92% idle. All CPU-block expert MUL_MAT_IDs in prefill are op-offloaded to CUDA0 (backend 0),
   ~21 GB weights/ubatch; RTX mostly waits for the MI50 pipeline stage. PCIe is Gen4 x16 on both GPUs under load.
7. Expert cache budget = free CUDA0 − 1024 MiB reserve; every MiB freed on CUDA0 becomes cache (3506 MiB at ub1280).
8. Long SSH commands: avoid single quotes inside `ssh '...'` from zsh; send scripts via `ssh host bash -s <<'EOF'`
   or `scp` a file.
9. Target `/tmp` is wiped on reboot; keep artifacts in `~/claude-opt`.

---

## 7. Open leads (ranked)

1. **Overlap offloaded expert weight upload with MI50 compute in prefill.** CPU blocks 0–3 and 31–47 are
   uploaded to CUDA0 every ubatch (~21 GB); the upload for layer L+1 could proceed while ROCm0 runs. Look at
   `ggml_backend_sched` op-offload (`GGML_OP_OFFLOAD_MIN_BATCH`, `ggml_backend_sched_compute_splits`) and
   whether CPU-block layers for ROCm0's tail could offload to ROCm0 instead (MI50 has PCIe Gen4 x16 too,
   and 745 MiB free; HBM2 1 TB/s). Measure with `dmon_arm.sh`.
2. **Layer rebalancing / CPU block count.** With freed memory, try `TS` variants or dropping one CPU block
   (e.g. remove 31 from `CPU_BLOCKS`, ~0.9 GiB to ROCm0) — ROCm0 now has 745 MiB at ub1280 (not enough for a
   full block; ub1024 gives 985 MiB). Check with fit-print first.
3. **QSA input build is O(n_kv) on CPU per ubatch** (`set_input_qsa` + `qsa_layout_direct` each loop all cells;
   ~4 ms at 150k). Could move bias/blk_cells build to GPU or cache across ubatches of one prompt.
   `qsa_layout_direct` duplicates work — could be folded into set_input_qsa's first pass result.
4. **Decode (tg) path**: gather path unchanged; cache hits ~64%. Ideas: heatmap persistence
   (`MOE_CACHE_PROFILE=1`, never measured), `--moe-cache-cpu-overlap` sweep at the new budget.
5. **Wave64 portability**: stable top-k still uses a 32-bit ballot mask assumption; f16/f32 `MUL_MAT_ID`
   page fault on ROCm0 (HANDOFF §4.10) — both off-path for this model.
6. **turbo4 KV at ub1280 is now the deployed default (user choice) but unmeasured.** First thing to run:
   bench + QUALITY arms with `KV_K=turbo4 KV_V=turbo4` against the q4_0 ub1280 numbers in §2. turbo4
   uses fused FA decode and changes ROCm0/CUDA0 KV size (4.125 vs 4.5 bpv), so recheck free VRAM after load.
   `TOPK_OVERRIDE=4096` is also still unmeasured.
7. Image (M-RoPE) prompts fall back to per-cell graph (bigger, reallocated at runtime). Making block select
   work for the ranked layout would remove `LLAMA_QSA_BLOCK_TOPK=0` advice.
8. Port these commits to current master (21 commits ahead of the pin) if production moves off the pin;
   touched files were identical between pin and master on 2026-09-24.

---

## 8. Invariants to keep (from HANDOFF + this session)

`--fit off`; anchored `^output\.weight$=CUDA0`; 21 CPU blocks `0|1|2|3|3[1-9]|4[0-7]` with cache auto;
PLE lazy (`-lm mmap -lzm on --mmap-prefetch off`, no mlock); output head on CUDA0; one server at a time;
`HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0`; old launchers stay runnable as rollback;
q4_0/q4_0 KV + default top_k as the reference config; ≥ ~500 MiB free on ROCm0 after load.

---

## 9. Strata port (opt/strata-port)

### 9.1 P0 measurement

Run 2026-10-01 13:11 → 13:53 (all nine arms in ~41 min) by `harness/p0_arms.sh`. Binary: `strata/p0-measure` at
2539d2487 (adds the `GGML_SCHED_TIMING` summary, off unless the env is set) plus the uncommitted kit patches.
Reference config as in §2: q4_0/q4_0 KV, ub1280, default top_k, MTP n=3, 21 CPU blocks, cache auto. `summ.py` medians of 5
(pp: boot rep dropped). Raw files are `~/claude-opt/p0-*.{bench,log,io,fincore,greedy.json}`.

**Reference arm (`p0-ref`)**, timing off:

| arm | pp6k | pp32k | tg512f | accept | cache hits | pools |
|---|---|---|---|---|---|---|
| §2 branch, ub1280 | 340.5 | 305.2 | 36.5 | — | 64.0% | 3506 MiB |
| **p0-ref** | **333.8** | **299.8** | **37.1** | 0.77 | 62.6% | 3506 MiB (iq4_nl 1485 / iq3_s 1291 / iq2_s 730) |

The arm is within run-to-run noise of §2 (pp −2%, tg +2%; per-rep spread pp6k 313.7–338.9, tg 35.6–39.9). Pools are
identical, so with timing off the timing build changes nothing.

**I/O and residency (`p0-ref`, whole bench window, after load):** `majflt=60593` (system-wide `pgmajfault` delta) and
`read_bytes=3.26 GiB` (server process). Per request, the first one reads 2191 MiB (cold touch). After that a pp6k reads
3–77 MiB, a pp32k 7–108 MiB and a tg512f 19–319 MiB. Page-cache residency (`fincore.py`, taken before stop, caches not
dropped between arms):

| shard(s) | content | resident / size |
|---|---|---|
| 00002 | `per_layer_token_embd.weight` (PLE table, lazy mmap) | **67.2 / 36 621 MiB (0.18%)** |
| 00003–00006, 00022–00033 | CPU expert blocks 0–3, 31–47 (+ token_embd, blk 30 edges) | 22 238 / 26 776 MiB (83%) |
| 00007–00021 | GPU-resident blocks | 0 / 26 088 MiB |
| total | | 22 306 / 90 146 MiB |

The PLE table is touched sparsely: lazy rows keep it at ~67 MiB of page cache. The CPU expert shards are almost fully
cached and account for nearly all residency.

**Prefill timing (`p0-timing`, `GGML_SCHED_TIMING=1`, one 32 008-token request).** It ran at 306.4 t/s, the same as p0-ref
pp32k, so the per-split synchronisation costs no visible throughput. The log has 271 `sched-timing:` lines: 31 are prefill
ubatches (84 splits, `wmib>0`), 215 are 2-split MTP-draft graphs (8 per 2048 batch) and 25 are 46-split decode graphs.
With b2048/ub1280, each batch is a 1280 + 768 ubatch pair. Medians (first ubatch excluded from the per-size rows):

| ubatch | n | in_ms | ids_ms | wmib | cmp_ms CUDA0 | cmp_ms ROCm0 | cmp_ms CPU | sum ms | in_ms share |
|---|---|---|---|---|---|---|---|---|---|
| 1280 | 15 | 796.4 | 13.0 | 11 700 | 206.6 | 2737.7 | 2.7 | 3745 | 21.4% |
| 768 | 15 | 681.4 | 8.2 | 10 866 | 145.7 | 1874.1 | 1.6 | 2712 | 25.5% |
| all prefill | 31 | 777.5 | 12.8 | 11 090 | 190.2 | 2216.6 | 2.6 | 3222 | **23.3%** |

- `in_ms` is all input copies of a ubatch, including the expert-id readback (`ids_ms`, which is part of `in_ms`). It is
  **~23% of a prefill ubatch** (21% at 1280, 25% at 768). This sizes P4.
- `wmib` is the offloaded CPU-expert bytes uploaded to CUDA0: ~11 GiB per ubatch at ~15.7 GB/s effective. It is nearly the
  same at 768 and 1280 tokens because almost every expert is hit either way, so the upload is a fixed per-ubatch cost.
- ROCm0 compute dominates (69–73%) and grows with depth (2166 → 3404 ms at 1280 over the 32k prompt). CUDA0 compute is ~5.5%.
- The timing path synchronises after every split, so these are serial sums. Any real copy/compute overlap is hidden
  inside them.

**Config items** (single arms, tg512f medians of 5; the noise band is about ±1 t/s):

| item | tg512f | accept | other |
|---|---|---|---|
| reference (draft KV f16, p-min 0 = off, `-t 6`) | 37.1 | 0.77 | |
| draft KV q8_0 (`-ctkd/-ctvd q8_0`) | 36.8 | 0.77 | draft KV 293.0 → 155.7 MiB; CUDA0 cache 3506 → **3645 MiB** (+139), hits 64.1%; pp6k 330.5, pp32k 300.8 |
| `--draft-p-min 0.3` | 37.0 | 0.78 | |
| `--draft-p-min 0.5` | 34.0 | 0.85 | |
| `--draft-p-min 0.7` | 33.9 | 0.92 | |
| `-t 6 -tb 6` | 37.5 | 0.77 | the launcher already defaults to `THREADS=6`, so this arm is a tg repeat of p0-ref (noise check), not a thread test |

Draft KV q8_0 adds ~139 MiB of expert cache at equal tg and pp, a free memory win. Higher p-min raises acceptance but cuts
the draft length more than it saves, so tg falls. Keep p-min off.

**Determinism verdict: FAIL, `identical 2/5` with `MOE_CACHE=off`** (`p0-det-a` vs `p0-det-b`; the log confirms
`MoE cache requested=off resolved=off`). `list` and `long` match. `code` diverges at char 257, `prose` at 467 and
`math` at 429. Each is a plausible near-tie word choice after a long identical prefix. The expert cache is therefore not
the only source of run-to-run variation on this host. The remaining source is not identified; MTP n=3 was on, so
verify-batch shapes vary between runs. **Exactness gates fall back to quality + needles for this host.** Byte-identical
greedy output cannot be required of a flag-off/flag-on pair.
