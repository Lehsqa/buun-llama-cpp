# HANDOFF — buun-llama-cpp on RTX 3080 Ti + MI50

Audience: the next agent, whose task is to **rewrite/optimise buun-llama-cpp for this machine**.
Everything here was measured on this host. Read §4 (gotchas) and §7 (leads) before touching code.

---

## 1. System specification

| Component | Detail |
|---|---|
| Host | `leshqa-MS-7D76`, Ubuntu 24.04, single NVMe (~170 GB free on `/`) |
| CPU | AMD Ryzen 5 7600X, 6C/12T (Zen4). Backend built `GGML_NATIVE=OFF` → **AVX2+FMA only, no AVX-512** |
| RAM | **30 GiB total** — the binding constraint for CPU-resident experts and PLE paging |
| GPU 0 | **RTX 3080 Ti 12 GB** → `CUDA0` (11909 MiB), sm_86, VMM yes. PCIe Gen1 at idle, upshifts under load |
| GPU 1 | **Instinct MI50 32 GB** → `ROCm0` (34342961152 B), `gfx906:sramecc+:xnack-`, **wave64**, VMM **no** |
| iGPU | Raphael `gfx1036` — must be hidden with `HIP_VISIBLE_DEVICES=0` / `ROCR_VISIBLE_DEVICES=0`, otherwise `ROCm0` is the iGPU |
| Profiling | `perf_event_paranoid=4` + no passwordless sudo → **no perf**; `ptrace_scope=1` → **no gdb attach** (gdb-as-parent via FIFO works; use `pgrep -x llama-server`, never `-f bin/llama-server`, which also matches gdb) |

Toolchains (unchanged from the working production setup — do not replace them):

- **CUDA 12.0** — `nvcc` V12.0.140, `/usr/bin/nvcc`, toolkit at `/usr/lib/cuda` (**not** `/usr/local/cuda`), CUB 2.0.1 (so the CUCCL ≥ 3.2 top-k path is unavailable).
- **ROCm 6.3.3** — `/opt/rocm` → `/opt/rocm-6.3.3`, hipcc = `/opt/rocm/lib/llvm/bin/clang` (clang 18, `roc-6.3.3`).
- gcc/g++ 13.3.0, CMake 3.28.3.

## 2. Model and runtime configuration

- **Target**: `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64` — 33 shards, 88.02 GiB, 4.27 BPW, 176.94 B params (A3B), arch **`qwen4exp`**.
  - 48 layers, n_embd 2560, n_head 24, n_head_kv 2, head k/v 256; full attention every 4th layer (`compress_ratios`), **indexer** (4 heads, key 128, `top_k 2048` in metadata), **SSM** (conv 4, state 128, groups 16, inner 6144), **hyper-connections** (4, low_rank 320).
  - MoE: 512 experts, **10 used/token**, expert FFN 640 + shared expert 640.
  - **PLE**: `per_layer_token_embd.weight` = 36 621 MiB, ngram 3, 8 heads/ngram, 16 heads (head_offsets/vocab_sizes ≈ 20 M each), vocab 248 320.
  - Tensor types: f32 388, f16 1, q5_1 1, q8_0 666, **iq4_nl 48, iq3_s 24, iq2_s 72**, bf16 24.
    **Expert tensors are iq2_s / iq3_s / iq4_nl only — no q4_K, no f16/f32 experts.**
- **Draft**: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.79 GB; q8_0 19, f32 11, bf16 2). Shared-tensor MTP that **borrows** the target's `token_embd`/`output`. A Q4_K_M full-module sidecar also exists as a fallback.
- **Deployed config** (2026-09-24 state; **superseded by §9** — the deployed launcher now runs the `opt` runtime at ubatch 1280): ctx 150 000, ubatch 768, batch 2048, target KV q4_0/q4_0, draft KV f16/f16, MTP n_max 3, split `3,32`, output+draft on CUDA0, 21 CPU expert blocks `0|1|2|3|31..47`, cache auto, PLE lazy.

## 3. Building both backends (mixed-vendor recipe)

One pinned source tree, **two separate builds**, one merged runtime — CUDA and HIP compile the same
`ggml-cuda/*.cu` sources with different toolchains, so they must not share a build directory.

```bash
cd ~/Projects/qwen38-perf/buun-rtx-mi-kit
CACHE_PROVIDER=cuda BUILD_JOBS=6 bash build-linux.sh    # deployed variant (CUDA owns the cache)
CACHE_PROVIDER=hip  BUILD_JOBS=6 bash build-linux.sh    # applies patch 0002, separate build root
```

- Pins `spiritbuun/buun-llama-cpp @ ed774445cd696cd45dcf590b29baf2b36a01874f` into
  `~/Projects/buun-rtx-mi-ed774445cd69-src/` (a pristine read-only clone also sits at `~/Projects/qwen38-forks/buun/`). The script is **idempotent**: it reuses the tree when it is already at pin+patches, and emits `local-changes.patch` + `build-info.txt`.
- CUDA: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF`
- HIP: `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang -DGGML_HIP_NO_VMM=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF -DCMAKE_HIP_FLAGS=-DHIP_ENABLE_WARP_SYNC_BUILTINS`
- **Merged runtime** = CUDA `bin/*` + HIP `bin/libggml-hip.so*` → `~/Projects/buun-rtx-mi-ed774445cd69-cuda/runtime/` (~594 MB `libggml-cuda.so`, ~108 MB `libggml-hip.so`).
- Build time ≈ 10 min CUDA + ≈ 15 min HIP at `-j6`.
- Post-build checks: `runtime/llama-server --list-devices` must show `CUDA0` (RTX) and `ROCm0` (MI50); `ldd` both modules for `not found`; then `test-backend-ops test -b CUDA0|ROCm0 -o MUL_MAT_ID` and `-o TOP_K`.

### Applied patches (hashes in `build-info.txt`, diff in `local-changes.patch`)

| patch | sha256 (short) | why it exists |
|---|---|---|
| `0001-cache-owner-guard.patch` | `efea2c6d…` | cache sessions admit only the **registered owner's** devices — CUDA and HIP compile the *same backend GUID*, so `ggml_backend_is_cuda()` alone would admit HIP's device |
| `0002-hip-cache-owner.patch` | `0541fb1c…` | **available, NOT applied**: loads `hip` before `cuda` so HIP owns the cache (dormant while MI50 has ~600–800 MiB free) |
| `0003-dl-test-link-guard.patch` | `3ea65d6b…` | three test link sites need `NOT GGML_BACKEND_DL` — DL builds make `ggml-cpu` a MODULE library and CMake configure fails |
| `0004-fla-cubin-symbol-scope.patch` | `8bd1894f…` | hoists the FLA cubin `extern "C"` block out of an anonymous namespace: **nvcc 12.0/cudafe rejects incomplete-array externs there** ("declared with a never-completed type") |
| `0005-hip-syncwarp-shim.patch` | `db2fc8cc…` | `#define __syncwarp(...) __builtin_amdgcn_wave_barrier()` — **ROCm 6.3.3 has no `__syncwarp`** (EXL3 kernels use it) |
| build flag | — | `-DHIP_ENABLE_WARP_SYNC_BUILTINS` — ROCm 6.3.3 gates `__ballot_sync`/`__shfl_sync` behind it (top-k needs them) |

`local-changes.patch` sha256: `117bd69289bf5e743d199d2b254ea02c1699db46131686d3f564c16a4046c234`.

## 4. Runtime gotchas (each one cost hours to find — do not re-derive them)

1. **Expert cache is a single-provider singleton.** The first module to register owns it; `ggml_backend_load_all_from_path()` loads `cuda` before `hip`, so **CUDA owns it**. `--moe-cache-expert-parallel` fans out only across devices *the owner* has — with one CUDA device, `auto` resolves to `1` and merely switches admission to the bundle policy (`inserts 8→16`, `readmit_after →40`). Measured neutral (34.57 vs 34.29 tg) → default is `0` (`MOE_CACHE_EP` knob available).
2. **Cache admission**: `budget = free VRAM − reserve` (reserve = 6 % of VRAM clamped to 1024–3072 MiB → **1024 MiB** on the RTX). Auto mode also requires a **1024 MiB slab floor**; below it the session reports *dormant*. A numeric budget bypasses the floor (`min-slab=0`, `granted=cap`) but **forces weight repacking off**.
3. **Cache eligibility**: ≤10 tokens/batch, ≤64 routed rows/node, ≥64 slots/pool, expert ≥512 KiB (cc ≥ 8.0). Prefill nodes always bypass it — gains are decode-only.
4. **CUDA0 pp compute at 150k**: 1027 (ub512) / 1171 (640) / 1244 (704) / **1701 (736)** / 1754 MiB (768). There is a **discrete +421 MiB step between ub 704 and 736**, CUDA0-only, with graph splits (84) and fused-op resolution unchanged. **Cause unidentified** — prime optimisation target (§7).
5. **`--fit off` is mandatory**: the auto-fit pass loads the shared draft standalone (no target to borrow `token_embd` from) and aborts.
6. **`output.weight` must stay on CUDA0**: the shared MTP head borrows it and the draft scheduler owns only CUDA0+CPU.
7. **Tensor overrides use `std::regex_search` (unanchored).** A bare `output\.weight=CUDA0` also pins all 12 `blk.N.attn_output.weight`: +175 MiB on CUDA0, **graph splits 106 → 84**, −5 % pp / −4 % tg. Always anchor: `^output\.weight$`.
8. **PLE must stay lazy**: `-lm mmap -lzm on --mmap-prefetch off`; never `mlock`; never pin `per_layer_token_embd` to CPU (88 GiB mapped vs 28 GiB RAM).
9. **Server shutdown can wedge**: `signal_handler` calls a non-async-signal-safe `shutdown_handler`; after a first signal the process can end up with INT/TERM **blocked forever** (`SigBlk 0x4002`) and only `SIGKILL` ends it (it keeps its VRAM). The launcher starts it with `setsid` and escalates TERM→SIGKILL (`STOP_GRACE=15`, immune to a second Ctrl+C).
10. **Off-path HIP defects** (do not report "HIP passes"): `MUL_MAT_ID` with **f16/f32** weights page-faults ROCm0 (reproduces in a HIP-only build); the **stable/ties TOP_K** kernel assumes 32-lane warps (32-bit ballot mask + `<<<WARP_SIZE>>>` with `base += warpSize`) and is wrong on wave64. This model dispatches neither (no f16/f32 experts; `ggml_top_k(..., stable=false)`).
11. **One server at a time** at 150k — both GPUs are nearly full (CUDA0 ~400–800 MiB, ROCm0 ~600–800 MiB free).
12. Measurement noise: first request after boot is ~30 % slower; tg varies ±5 %; use medians of ≥5 reps for tg and exclude the boot sample for pp. P0 refined this: per-rep tg spread is about ±2 t/s, and pp32k can shift ~2 % between sessions on different days, so only same-session A/B deltas beyond that are real.
13. **Greedy output is not reproducible run to run** (P0: only 2/5 prompts identical between two flag-off runs with `MOE_CACHE=off`; the source is unidentified, MTP verify-batch shapes are the suspect). Byte-identical output cannot gate a flag. Gate on `quality.py` 10/10 + `longctx.py` needles 3/3 + throughput within noise, and read greedy diffs as information only.

## 5. Measured baselines

Judge a change against the **newest** table (§5.2, ub1280, same-session A/B). §5.1 is the original pin baseline at ubatch 768, kept for history.

### 5.1 Pin baseline (2026-09-24, ub768)

| arm | pp6k | pp32k | tg512f | notes |
|---|---|---|---|---|
| production `danielhanchen@2857e5114`, 18 blocks | 256–260 | 237–239 | 26.8–27.1 | served by `…-Flash-Next_old.sh` |
| buun, 21 blocks, cache off | ~270 | ~244 | 27.9 | |
| **buun, 21 blocks, cache auto (current best)** | **286** | **256** | **34.2–34.3** | +10 % pp / +28 % tg vs production |
| quality corpus, 10 mixed prompts | — | — | — | buun 10/10 in 68.0 s vs production 10/10 in 91.7 s |

Cache session detail: pools `iq4_nl 907 + iq3_s 785 + iq2_s 447 = 2139 MiB`, 3097 slots, hits **47–52 %**, 0 dispatch/collect failures, ~143 k evictions/session. Long context: 142 321-token request OK (pp 164.9, tg 20.8), needles at 10/50/90 % all retrieved; TTFT recovers to 0.72–1.34 s after a 129 s prefill.

Tuning already done (detail in the report): ub 512/640/768 trade cache capacity for prefill (ub640 = +8.3 % tg / −8.7 % pp32k; ub768 wins 32k-prompt total latency); CPU overlap `auto` ≈ `0`; 20 blocks + numeric 512/768 MiB cap = +5 %/+6 % over its own no-repack control but far below the 21-block auto arm; turbo4 KV (4.125 bpv vs q4_0's 4.5) measured **neutral** at 8k/32k — its 140k arms were stopped before running.

### 5.2 `opt/rtx-mi50` and Strata P0/P1 (2026-09-25 … 2026-10-01)

150k ctx, q4_0/q4_0 KV, default `top_k`, MTP n=3, 21 CPU blocks, cache auto. `bench.py` medians of 5 (pp: boot rep dropped). Sources: `scripts/rtx-mi50/REPORT.md` §2 and §9.

| arm | ub | pp6k | pp32k | tg512f | cache hits | pools | gates |
|---|---|---|---|---|---|---|---|
| production runtime (pin) | 768 | 285.9 | 259.5 | 34.4 | 49.9 % | 2020 MiB | |
| `opt/rtx-mi50` | 768 | 279.5 | 255.5 | 36.9 | 64.9 % | 3904 MiB | |
| `opt/rtx-mi50` | 1024 | 332.1 | 302.3 | 35.7 | 63.8 % | 3734 MiB | |
| **`opt/rtx-mi50` (deployed default)** | **1280** | **340.5** | **305.2** | **36.5** | 64.0 % | 3506 MiB | quality 10/10; 142 266-token needles 3/3 (pp 187.4 / tg 23.7) |
| `opt/rtx-mi50` | 1536 | 335.3 | 304.7 | 37.6 | 61.2 % | 3323 MiB | |
| P0 reference re-run (`p0-ref`) | 1280 | 333.8 | 299.8 | 37.1 | 62.6 % | 3506 MiB | |
| P1 `--ple-io mmap` (`p1-mmap`) | 1280 | 335.2 | 300.2 | 36.4 | 63.7 % | 3506 MiB | |
| **P1 `--ple-io direct`** (`p1-direct`) | 1280 | 330.6 | 298.8 | 37.8 | 61.9 % | 3506 MiB | quality 10/10; needles 3/3 (pp 185.0 / tg 23.2) |

- `opt/rtx-mi50` at ub1280 vs same-day production: **+19 % pp6k, +18 % pp32k, +6 % tg**. Long context: pp 187.4 vs 164.9, tg 23.7 vs 20.8 (HANDOFF §5.1 baseline).
- The gain comes from memory, not kernels: the compute buffers at ub768 fell CUDA0 1754 → 944 MiB and ROCm0 1649 → 1014 MiB (block-ranked QSA selection `e4662b5d4` plus one shared KQ-mask view across QSA layers `5542623ce`; per-layer mask views had made the scheduler copy the 219 MiB mask to ROCm0 8×), and the MTP draft context on CUDA0 fell 1493 → 498 MiB (draft ubatch capped at 256, `3ab2be1c6`). That freed VRAM became expert cache (pools 2020 → 3506 MiB at ub1280, hits ~50 % → 62–65 %) and allowed a larger ubatch.
- Rows are not all one session: the ub768–1536 rows come from 2026-09-25, the P0/P1 rows from 2026-10-01. The ub1280 rows differ by up to ~2 % on pp32k; REPORT §9.1 attributes that to a shift between sessions (reps within a session sit within ~1 %), so compare within one session only (§4.12).

### 5.3 Strata P0 — where the time goes (`GGML_SCHED_TIMING`, one 32 008-token prefill)

- **Prefill ubatch input copies are 23–24 % of the ubatch** (777.5 ms of a 3222 ms median; 21 % at ub1280, 25 % at ub768). About **11 GiB of offloaded CPU-expert weights** is uploaded to CUDA0 per ubatch at ~15 GB/s effective, nearly independent of ubatch size because almost every expert is hit either way. This sizes P4.
- ROCm0 compute is **69–73 %** of prefill compute and grows with depth (2166 → 3404 ms at ub1280 over the 32k prompt); CUDA0 compute is ~5.5 %, CPU ~0. The timing path syncs after every split, so these are serial sums and any real copy/compute overlap is hidden. The timing itself costs no visible throughput (306.4 vs 299.8 t/s).
- Page cache after the bench: the lazy PLE table keeps only ~67 of 36 621 MiB resident (0.18 %); the CPU expert shards are 83 % resident (22.2 of 26.8 GiB) on the 30 GiB host. The first request reads 2191 MiB cold; later requests read single- to low-hundreds of MiB.
- Config items (tg512f medians of 5, ±1 t/s between arms): draft KV q8_0 is neutral on tg/pp and frees **+139 MiB of expert cache** (3506 → 3645 MiB), a free memory win that the launchers do not take yet (`DRAFT_KV_K/V` still default `f16`). `--draft-p-min 0.3` is neutral; 0.5 and 0.7 raise acceptance (0.85 / 0.92) but cut tg to 34.0 / 33.9, so keep p-min off. `-t 5` ≈ `-t 6` (36.0 vs 36.4); **`-t 12` loses ~7 t/s (29.3)** and was not profiled further. Keep `THREADS=6`.
- Determinism: see §4.13.

### 5.4 Strata P1 — `--ple-io direct` (verdict PASS)

- vs `--ple-io mmap`: pp6k −1.4 %, pp32k −0.5 %, tg +3.8 % — all inside the per-rep spread (gate: pp −2 %, tg −5 %).
- PLE page-cache residency 56 → **0 MiB**; major faults 74 079 → 56 226 (**−24 %**); `read_bytes` unchanged (5.64 vs 5.67 GiB, dominated by CPU-expert shard re-reads).
- Reader: row-cache hit 99.2 %, 23 440 disk reads, p50 126 µs, p99 8.4 ms; the compute thread waited 1.13 s in total over the ~11-minute bench. Short greedy prompts start cold (hit 59 % at the end of the det run).
- Quality 10/10, needles 3/3, greedy 3/5 identical vs mmap (information only, §4.13).
- In code `--ple-io` still defaults to `mmap`; the `_opt.sh` launcher defaults `PLE_IO=direct`. Whether the main launcher takes it is the user's decision (not done).

## 6. Where everything lives

| path | contents |
|---|---|
| `~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh` | **deployed buun launcher** (anchored override, cache auto, 21 blocks). Knobs: `CTX UBATCH_SIZE KV_K KV_V DRAFT_KV_K DRAFT_KV_V TS CPU_BLOCKS OUT_DEVICE OUT_PATTERN MOE_CACHE MOE_CACHE_EP MOE_CPU_OVERLAP MOE_CACHE_PROFILE N_MAX MTP MTP_QUANT API_PORT LV STOP_GRACE EXTRA_ARGS DRY_RUN` |
| `~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next_old.sh` | production launcher → **rollback path** |
| `~/.config/systemd/user/qwen-flash-llama.service` | `ExecStart` points at the buun launcher; currently disabled/inactive. `KillSignal=SIGINT`, `KillMode=mixed`, `TimeoutStopSec=30` (compatible with the launcher's escalation) |
| `~/Projects/qwen38-perf/buun-rtx-mi-kit/` | `build-linux.sh`, `launch-linux.sh` (experiment driver, port 18081), the 5 patches, and tools: `smoke.py quality.py sequence.py longctx.py kv-compare.sh tuning.sh validate.sh tuning-analyze.py kv-analyze.py` |
| `~/Projects/buun-rtx-mi-ed774445cd69-cuda/` | build tree (`cuda/`, `hip/`), **`runtime/` (live binaries)**, `build-info.txt`, `local-changes.patch`, `devices.txt`, `deps-*.txt`, every arm log and JSON |
| `~/Projects/qwen38-perf/buun-rtx-mi-REPORT.md` | full report (build → verification → override audit → tuning → long context → hashes → rollback) |
| `~/Projects/qwen38-perf/results.jsonl` | every timed request, tagged (`audit-*`, `tuning-*`, `validate-*`, `kvcmp-*`, `control-*`) |

## 7. Highest-value leads for the rewrite

1. **The +421 MiB CUDA0 step between ub 704 and 736.** Largest single lever: removing it drops ub768's 1754 MiB to ~1315 MiB, i.e. **+440 MiB of cache budget** (~+20 % cached expert rows) or headroom for a bigger ubatch. Needs per-node gallocr/allocator instrumentation at 704 vs 736, target and draft separately (the fork exposes no runtime scheduler-debug knob).
   *Status 2026-10-08:* **mostly captured without being diagnosed.** `opt/rtx-mi50` cut the ub768 CUDA0 compute buffer 1754 → 944 MiB and the draft context 1493 → 498 MiB (§5.2), which moved the cache pools to 3506 MiB at ub1280. The discrete 704 → 736 step itself was never re-bisected, so its cause is still unknown.
2. **Prefill is not GPU-bound.** ROCm0 compute scales 2.144 MiB/ubatch at 150k (indexer/QSA term), and the fork's own TODO in `llama-memory-hybrid-idx.cpp` says the QSA input build is O(n_kv) per ubatch per stream (~865 µs at 33k → ~3.9 ms at 150k).
   *Status 2026-10-08:* **open, and P0 changed the picture.** Prefill is bounded by the ROCm0 stage (69–73 % of compute) and by input copies (23–24 % of a ubatch, ~11 GiB of expert weights uploaded per ubatch). The QSA input build is still O(n_kv) on the CPU (~4 ms per ubatch at 150k). P4 (overlap/stream the uploads) targets the copies and P7 (pooled indexer keys) the indexer/QSA term; neither is started.
3. **Cache hit rate is only ~50 %** with 2139 MiB / 3075 slots. Levers: the step above; allocating the indexer-score buffer on the device with slack; shrinking per-token pp buffers.
   *Status 2026-10-08:* **now 62–65 %** with 3506 MiB pools at ub1280 (§5.2). Remaining levers: draft KV q8_0 (+139 MiB, §5.3), heatmap persistence, and P5.
4. **Wave64 portability**: fix stable top-k and audit other 32-lane assumptions; fix the f16/f32 `MUL_MAT_ID` ROCm fault (both with reproductions + tests).
   *Status 2026-10-08:* **partly done, off-path either way.** `c4fe38c4d` fixed the stable-ties collect stride on wave64 and added a radix top-k for CCCL < 3.2. The 32-bit ballot-mask assumption is still **not audited**, and the f16/f32 `MUL_MAT_ID` ROCm0 fault is untouched.
5. **Deferred**: HIP-owner cache (patch 0002), VBR (`GGML_HIP_NO_VMM=ON` by design), heatmap persistence (`MOE_CACHE_PROFILE=1`), turbo4 at long context.
   *Status 2026-10-08:* all still unmeasured. turbo4 is now the **deployed KV default** (user edit) but has not been benchmarked or quality-checked at ub1280, at any context length; the §5 numbers are all q4_0.
6. **Invariants to keep**: the pinned source + five patches, `--fit off`, the anchored override, 21 CPU blocks with cache auto, PLE lazy + prefetch off, the shared output head on CUDA0, and `_old.sh` runnable as rollback.

## 8. Quick start

```bash
# start the deployed server (log: ~/.local/state/llama-launcher/llama-server-qwen38next-buun.log)
bash ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh
DRY_RUN=1 bash ~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh     # print argv only
# stop: Ctrl+C, or kill -TERM <script pid>  (trap escalates; a wedged server needs SIGKILL)

# measure (one server at a time)
python3 ~/Projects/qwen38-perf/bench.py --url http://127.0.0.1:10000 --only pp6k,pp32k,tg512f --reps 5 --tag my-arm
python3 ~/Projects/qwen38-perf/buun-rtx-mi-kit/quality.py --url http://127.0.0.1:10000 --tag my-arm --output /tmp/q.json
python3 ~/Projects/qwen38-perf/buun-rtx-mi-kit/longctx.py --port 10000 --reps 7900 \
        --needles PASS-A,PASS-B,PASS-C --positions 0.1,0.5,0.9 --max-tokens 512 --output /tmp/n.json

# rebuild after source edits
cd ~/Projects/qwen38-perf/buun-rtx-mi-kit && CACHE_PROVIDER=cuda BUILD_JOBS=6 bash build-linux.sh
```

## 9. State at handoff (2026-09-24, updated 2026-10-08)

The 2026-09-24 snapshot (a server on `:10000` with turbo4 and `top_k 4096`) is **obsolete**: the work in §5.2 replaced that setup.
Everything below is from `scripts/rtx-mi50/REPORT.md` as of 2026-10-01 and the repo as of 2026-10-08.
**The target host was not re-checked on 2026-10-08** — it may be offline, and a server may be running. Run `pgrep -a -x llama-server`
before starting anything; only one server fits at 150k.

**Deployed on the target**
- Main launcher `~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next.sh` is the generated `opt` launcher: runtime
  `~/Projects/buun-opt-rtx-mi50/runtime`, ub1280, `TOPK_OVERRIDE` empty (GGUF default 2048), log `llama-server-qwen38next-buun-opt.log`.
  The only user edit: `KV_K/KV_V` default **turbo4** — **unmeasured at ub1280** (every §5.2 number is q4_0/q4_0). The systemd unit
  `qwen-flash-llama.service` starts it; it was inactive on 2026-10-01.
- `~/Scripts/llama.cpp/RTX+MI_Qwen3.8-Flash-Next_opt.sh` (repo copy: `scripts/rtx-mi50/`) is the same launcher plus the P1 knobs:
  `PLE_IO=direct` default, `PLE_IO_THREADS=16`, and `SCHED_TIMING=N` (exports `GGML_SCHED_TIMING`, default 0 = off). Its KV default is also turbo4.
  Rollback: `PLE_IO=mmap`, or `_old.sh` (pin runtime, ub768, turbo4, hardcoded `top_k=4096`).
- The target runtime holds the P1 build (`d87edd0aa` + the kit patches). The later `6594ea436` (Windows test guards, O_DIRECT short-read guard) is not rebuilt there.

**Branches**
- `opt/rtx-mi50` — validated (§5.2), rooted at the pin `ed774445`, not at master. On remote `fork`.
- `opt/strata-port` — P0 + P1 merged (`7be8c7400`); results in `REPORT.md` §9. On remote `fork`, with `strata/p0-measure` and `strata/p1-ple-direct`.
- `strata/p6-avx2-compute` — **current implementation branch**: opt-in IQ2_S/IQ3_S AVX2 multi-token kernels, CPU dispatch, exact parity tests and target harness implemented locally on the integration base. Release AVX2/non-AVX2 tests pass; CPU MODULE builds. Local evidence and pending target gates are in `REPORT.md` §10 and the plan execution ledger. Target acceptance, rollout and merge remain pending; no target P6 measurements.

**Open questions**
- turbo4 KV: never measured at ub1280 or at ≥64k populated context, yet it is the deployed default. First check: bench + quality arms with `KV_K=turbo4 KV_V=turbo4` against the q4_0 ub1280 row in §5.2, and re-check free VRAM after load.
- A larger indexer `top_k` (`TOPK_OVERRIDE=4096`): quality and throughput effect never measured.
- Cause of the ub704 → 736 CUDA0 compute step: undiagnosed (§7.1).
- Source of the run-to-run greedy non-determinism (§4.13).
- The off-path HIP kernel defects (§7.4): wave64 ballot mask unaudited; f16/f32 `MUL_MAT_ID` fault.
- Heatmap persistence (`MOE_CACHE_PROFILE=1`) and a `--moe-cache-cpu-overlap` sweep at the new cache budget: never measured.
- Why `-t 12` loses ~7 t/s versus `-t 6`: not profiled.
- Code follow-up: in `src/llama-ple-reader.cpp:143` the O_DIRECT short-read break is `got < job.len`; the proposed guard is `got % PLE_ALIGN != 0`. Not changed.

## 10. Next steps — status of the Strata packages (2026-10-08)

Packages come from `docs/superpowers/specs/2026-10-01-strata-port-design.md`: one plan per package, executed in order P0 → P8.
"Rebuild" is the rebuild time on the host.

| # | Package | Flag | Where | Rebuild | Status |
|---|---|---|---|---|---|
| P0 | Measurement: per-split scheduler timing, harness with I/O capture and greedy check | `GGML_SCHED_TIMING=N` | ggml-backend scheduler, `scripts/rtx-mi50/harness/` | n/a | **Done**, merged. Results §5.3 |
| P1 | PLE table read by an O_DIRECT positioned-read reader instead of mmap | `--ple-io direct` | `src/llama-ple-reader.cpp`, qwen4exp | n/a | **Done, PASS**, merged. Results §5.4 |
| P2 | Greedy MTP verify on the GPU | `--spec-gpu-argmax` | server, common | 20 s | Not started, no plan |
| P3 | Draft length by measured cost, plus copy-from-context drafts deeper than 3 | `--spec-policy cost`, `--spec-mtp-depth` | common/speculative | 20 s | Not started, no plan |
| P4 | Prefill expert streaming: pinned staging ring, copy stream, prefetched weight slots | `--prefill-weight-stream R` | ggml-backend scheduler + CUDA | ~10 min | Not started, no plan. P0 sizes it (below) |
| P5 | Decode: share of cache misses computed on the GPU. Reuses P4's copy stream and staging, plus a capped registration of the expert pages as pinned memory | `--moe-cache-pcie-frac F`, `--pin-experts-gib N` | moe-cache.cu, ggml-cpu, loader | ~10 min | Not started, no plan. Depends on P4 |
| P6 | AVX2 kernels that compute iq2_s/iq3_s experts for all tokens of a verify window in one pass | `GGML_CPU_IQ_MT=1` | ggml-cpu x86 | ~2 min | **Implemented and validated locally 2026-10-08**, branch `strata/p6-avx2-compute`; exact CPU tests + harness ready. Target quality/needles/memory/performance gates pending; default off, not adopted/merged |
| P7 | Pooled indexer keys: finished blocks stored once (f32 by default for exactness, f16 optional), open tail recomputed from a small raw buffer | `--qsa-pooled-keys` | indexer memory, qwen4exp | CPU+GPU | Not started, no plan |
| P8 | Long-context KV streaming: full KV in pinned RAM, a page cache in VRAM, a resolve kernel (CUDA and HIP wave64) | `--kv-resident N` | KV cache, new ggml op, CUDA+HIP | ~25 min | Not started, no plan |

None of the P2–P5, P7 and P8 flags exist in the code yet. Write each package's plan first.

**What the P0/P1 results change**
- **P4 sizing:** input copies are ~24 % of a prefill ubatch (~11 GiB of expert weights per ubatch at ~15 GB/s), and ROCm0 compute is 69–73 % of prefill compute (§5.3). The sched timing is a serial sum, so it does not give the gain from overlap. Measure P4 end to end.
- **Exactness gates:** the spec declares P2, P3, P4, P6, P7-f32 and P8 exact (token-identical greedy output against the flag-off arm). On this host that gate cannot be passed even flag-off against flag-off (2/5 identical, §4.13). End-to-end gate instead: `quality.py` 10/10, `longctx.py` needles 3/3, throughput within noise, same-session A/B. Kernel-level bit-parity tests (for example P6's AVX2 dot products against buun's existing ones) are still meaningful and stay exact.
- **P6 implementation:** opt-in `GGML_CPU_IQ_MT=1`, x86 AVX2 only, IQ2_S/IQ3_S only, batches of 2–8 tokens (the MTP verify window) with at least two routed rows for an expert; larger prefill batches and the existing IQ panel path (≥ 8 rows per expert, which keeps precedence) keep the current code. It is a CPU-side decode feature, so it can only move tg, and it has no expected number yet. Controls: P6 off/on on the same build, ub768 and ub1280 reported separately, and `--ple-io mmap` and `direct` controls with PLE mode fixed while toggling.
- **PLE direct as a default:** `--ple-io direct` is throughput-neutral and removes the 36.6 GiB table from the page cache (§5.4). The `_opt.sh` launcher already uses it; promoting it to the main launcher is the user's call.

**Small wins outside the package list**
- Draft KV q8_0 (`DRAFT_KV_K=q8_0 DRAFT_KV_V=q8_0`): +139 MiB expert cache at equal tg/pp (§5.3). Not adopted by the launchers.
- Measure turbo4 against q4_0 at ub1280, then decide whether it stays the deployed default (§9).
