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
- **Deployed config**: ctx 150 000, ubatch 768, batch 2048, target KV q4_0/q4_0, draft KV f16/f16, MTP n_max 3, split `3,32`, output+draft on CUDA0, 21 CPU expert blocks `0|1|2|3|31..47`, cache auto, PLE lazy.

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
12. Measurement noise: first request after boot is ~30 % slower; tg varies ±5 %; use medians of ≥5 reps for tg and exclude the boot sample for pp.

## 5. Measured baseline (judge any change against this)

| arm | pp6k | pp32k | tg512f | notes |
|---|---|---|---|---|
| production `danielhanchen@2857e5114`, 18 blocks | 256–260 | 237–239 | 26.8–27.1 | served by `…-Flash-Next_old.sh` |
| buun, 21 blocks, cache off | ~270 | ~244 | 27.9 | |
| **buun, 21 blocks, cache auto (current best)** | **286** | **256** | **34.2–34.3** | +10 % pp / +28 % tg vs production |
| quality corpus, 10 mixed prompts | — | — | — | buun 10/10 in 68.0 s vs production 10/10 in 91.7 s |

Cache session detail: pools `iq4_nl 907 + iq3_s 785 + iq2_s 447 = 2139 MiB`, 3097 slots, hits **47–52 %**, 0 dispatch/collect failures, ~143 k evictions/session. Long context: 142 321-token request OK (pp 164.9, tg 20.8), needles at 10/50/90 % all retrieved; TTFT recovers to 0.72–1.34 s after a 129 s prefill.

Tuning already done (detail in the report): ub 512/640/768 trade cache capacity for prefill (ub640 = +8.3 % tg / −8.7 % pp32k; ub768 wins 32k-prompt total latency); CPU overlap `auto` ≈ `0`; 20 blocks + numeric 512/768 MiB cap = +5 %/+6 % over its own no-repack control but far below the 21-block auto arm; turbo4 KV (4.125 bpv vs q4_0's 4.5) measured **neutral** at 8k/32k — its 140k arms were stopped before running.

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
2. **Prefill is not GPU-bound.** ROCm0 compute scales 2.144 MiB/ubatch at 150k (indexer/QSA term), and the fork's own TODO in `llama-memory-hybrid-idx.cpp` says the QSA input build is O(n_kv) per ubatch per stream (~865 µs at 33k → ~3.9 ms at 150k).
3. **Cache hit rate is only ~50 %** with 2139 MiB / 3075 slots. Levers: the step above; allocating the indexer-score buffer on the device with slack; shrinking per-token pp buffers.
4. **Wave64 portability**: fix stable top-k and audit other 32-lane assumptions; fix the f16/f32 `MUL_MAT_ID` ROCm fault (both with reproductions + tests).
5. **Deferred**: HIP-owner cache (patch 0002), VBR (`GGML_HIP_NO_VMM=ON` by design), heatmap persistence (`MOE_CACHE_PROFILE=1`), turbo4 at long context.
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

## 9. State at handoff (2026-09-24)

- A buun server **is running** on `:10000`, started 10:27 from the deployed launcher, with two
  operator modifications on top of the validated baseline:
  `-ctk turbo4 -ctv turbo4` and `--override-kv qwen4exp.attention.indexer.top_k=int:4096`.
  Both are **unvalidated experiments** on this host: turbo4 was neutral at 8k/32k and never measured
  at 140k; the indexer `top_k 4096` (GGUF default 2048) has not been benchmarked or quality-checked.
  Do not treat this running configuration as the reference — the reference is §5 (q4_0/q4_0 KV,
  `top_k` at its metadata default).
- Nothing else of ours is running: no experiment drivers, no monitors, port free apart from that server.
- Open questions left by the previous session: cause of the ub704→736 CUDA0 step; whether turbo4 helps
  at ≥64k populated context; whether a larger indexer `top_k` changes quality or throughput; the two
  off-path HIP kernel defects; heatmap persistence (stage D) never measured.

## 10. Next steps what to do next
┌─────┬────────────────────────────────────────────┬──────────────────────────────┬──────────────────────────┬──────────┐
│  #  │                  Package                   │             Flag             │          Where           │ Rebuild  │
│     │                                            │                              │                          │ on host  │
├─────┼────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────┼──────────┤
│ P2  │ Greedy MTP verify on the GPU               │ --spec-gpu-argmax            │ server, common           │ 20 s     │
├─────┼────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────┼──────────┤
│ P3  │ Draft length by measured cost, plus        │ --spec-policy cost,          │ common/speculative       │ 20 s     │
│     │ copy-from-context drafts deeper than 3     │ --spec-mtp-depth             │                          │          │
├─────┼────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────┼──────────┤
│ P4  │ Prefill expert streaming: pinned staging   │ --prefill-weight-stream R    │ ggml-backend scheduler + │ ~10 min  │
│     │ ring, copy stream, prefetched weight slots │                              │  CUDA                    │          │
├─────┼────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────┼──────────┤
│     │ Decode: share of cache misses computed on  │                              │                          │          │
│ P5  │ the GPU. Reuses P4's copy stream and       │ --moe-cache-pcie-frac F,     │ moe-cache.cu, ggml-cpu,  │ ~10 min  │
│     │ staging, plus a capped registration of the │ --pin-experts-gib N          │ loader                   │          │
│     │  expert pages as pinned memory             │                              │                          │          │
├─────┼────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────┼──────────┤
│     │ AVX2 kernels that compute iq2_s/iq3_s      │                              │                          │          │
│ P6  │ experts for all tokens of a verify window  │ GGML_CPU_IQ_MT=1             │ ggml-cpu x86             │ ~2 min   │
│     │ in one pass                                │                              │                          │          │
├─────┼────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────┼──────────┤
│     │ Pooled indexer keys: finished blocks       │                              │                          │          │
│ P7  │ stored once (f32 by default for exactness, │ --qsa-pooled-keys            │ indexer memory, qwen4exp │ CPU+GPU  │
│     │  f16 optional), open tail recomputed from  │                              │                          │          │
│     │ a small raw buffer                         │                              │                          │          │
├─────┼────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────┼──────────┤
│     │ Long-context KV streaming: full KV in      │                              │ KV cache, new ggml op,   │          │
│ P8  │ pinned RAM, a page cache in VRAM, a        │ --kv-resident N              │ CUDA+HIP                 │ ~25 min  │
│     │ resolve kernel (CUDA and HIP wave64)       │                              │                          │          │
└─────┴────────────────────────────────────────────┴──────────────────────────────┴──────────────────────────┴──────────┘
