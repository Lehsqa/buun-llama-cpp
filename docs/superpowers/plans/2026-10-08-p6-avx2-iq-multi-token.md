# P6 AVX2 IQ Multi-Token Implementation Plan

> **For agentic workers:** The user requested subagent-driven implementation on the existing `strata/p6-avx2-compute` branch. Use subagents with spec and quality reviews. Steps use checkbox syntax for tracking.

**Goal:** Make CPU IQ2_S/IQ3_S expert projections decode weights once for up to eight routed activation rows, with bit-identical results to buun's existing AVX2 dot products and an opt-in `GGML_CPU_IQ_MT=1` switch.

**Architecture:** Reuse the existing `MUL_MAT_ID` expert grouping and thread chunks. Add two internal x86 dot-product functions that preserve buun's per-column integer lanes, FMA order, and horizontal reduction. Invoke them inside the current chunk evaluator for CPU rows remaining after CUDA cache planning; leave the existing IQ panel path, activation quantization, GPU kernels, and graph structure intact.

**Tech Stack:** C11/C++17, AVX2/FMA, ggml CPU backend, CMake/CTest; CUDA 12.0 and ROCm 6.3.3 remain the deployed GPU toolchains.

---

## Execution constraints (2026-10-08)

The user authorized implementation, local build/test validation, and pushing this branch to `fork`; no pull request is requested. The LLM target is unavailable, so do not connect to it, deploy a runtime, run target benchmarks, or claim throughput/quality/long-context results. Implement the target harness for later use. Remote acceptance and adoption checks in Tasks 5–6 remain pending; keep `GGML_CPU_IQ_MT` disabled by default and do not merge into the integration branch.


## Execution ledger (2026-10-08)

The unchecked task instructions below remain the implementation/target recipe. This ledger records
actual evidence, rather than treating local work as target acceptance. All local validation used the
Ryzen 7 PRO 7840U development host; the required 7600X/RTX 3080 Ti/MI50 measurements remain pending.

- [x] Existing `strata/p6-avx2-compute` branch retained; preexisting user `HANDOFF.md` edits preserved.
- [x] Tasks 1–3 implementation: internal exact IQ2_S/IQ3_S kernels, opt-in CPU dispatch, test registration
  and direct/saved-input parity fixtures implemented. Existing IQ-panel precedence retained.
- [x] Release native-OFF AVX2/FMA final build: server, CPU/backend and qwen4exp architecture test tools
  built; fresh flag-off/on CTest passed 2/2. Graph fixture covers 288 configurations with threads 1/6/12,
  token counts 1/2/3/4/8/9, slots 1/10, both formats, input layouts and expert windows.
- [x] Release invalid-input guards: both formats reject columns 0/9 and width 257 with GGML_ASSERT.
- [x] Final non-AVX2 build passed, fresh CTest 2/2; dynamic `ggml-cpu` MODULE rebuilt successfully.
- [x] Temporary CPU-library coverage probe: 29184 enabled calls; unset/0/true/empty each 0, exact parity
  throughout. CPU backend IQ2_S/IQ3_S/IQ4_NL MUL_MAT_ID 11/11 passed off and on.
- [x] qwen4exp CPU PLE resize/rollback and MTP handoff contracts passed; CUDA VBR skipped without GPU.
- [x] Task 5 microbenchmark tool implemented and smoke-run: all 10 rows plus finite checksum completed.
  Different CPU/concurrent build load; no target speedup claim.
- [x] Task 5 harness implementation and local validation: executable `p6_arms.sh`, bash syntax check,
  temporary launcher/run_arm/greedy stubs verified 19 default and 29 optional arms, counterbalancing,
  environment isolation and existing-server guard. Task 6 docs updated with measured local evidence.
- [ ] Windows shared/static compilation (shared private-symbol test is excluded; static is supported).
- [ ] Task 5 remote work: actual source/runtime audit, isolated CUDA+HIP build and explicit CPU MODULE
  deployment; separate non-DL 7600X CPU tests/microbenchmarks; merged-runtime backend/cache checks;
  every throughput/quality/needle/memory/profiling arm and artifact inspection.
- [ ] Task 6 remote evidence/adoption: five usable same-session medians/spreads by PLE mode and codec,
  quality 10/10, needles 3/3, no OOM/cache errors, ROCm0 ≥~500 MiB free after load, reproducible tg gain
  beyond ~±2 t/s repetition noise, no pp/latency regression beyond noise (~2% cross-session pp32k drift).
- [ ] Adoption/merge. Keep the feature default off; rollout needs target gates and separate turbo4 checks.

Detailed local results and exact target evidence placeholders: `scripts/rtx-mi50/REPORT.md` §10.
Rollback is `GGML_CPU_IQ_MT=0` or unset followed by a fresh server; no production default changed.

## Source of truth and inspected revisions

Re-read all 241 lines of the updated `HANDOFF.md`, including hardware, model, build recipe, gotchas, newest P0/P1 measurements, revised deployment state, and package gates. The user confirms **ub1280**; all primary controls below use it. Also inspected the existing package specification, deployment scripts, P0/P1 report, buun CPU dispatch, and Strata kernels/parity tests. Updated HANDOFF §4.13 and §10 supersede the older spec's server-level token-identity gate; exact kernel and saved-input graph tests remain mandatory.

- Buun: `/home/aslusarchuk/Projects/buun-llama-cpp`, HEAD `0eb737653b49dfd8822d92ca3095e944a1f40150`; current branch `strata/p6-avx2-compute`.
- Strata reference: `/home/aslusarchuk/Projects/Strata`, HEAD `6674a0065fb96bacde33e3eb10f91a1df86f95f2`.
- Package contract: `docs/superpowers/specs/2026-10-01-strata-port-design.md`, P6 section at line 151.
- Current measurements and build procedure: `scripts/rtx-mi50/REPORT.md`, sections 1, 2, 8, and 9.

No engine code is changed by writing this plan. No remote server state has been checked. Updated HANDOFF §9 explicitly marks the September 24 snapshot obsolete and says the target was not re-checked on October 8; inspect its actual process/runtime state before implementation.

## Configuration this work must serve

| Item | Required configuration |
|---|---|
| Host | Ubuntu 24.04; Ryzen 5 7600X, 6 cores/12 threads; 30 GiB RAM |
| CPU instructions | `GGML_NATIVE=OFF`, AVX2/FMA, no AVX-512 or AVX-VNNI dependency; **THREADS=6** in target arms |
| GPUs | RTX 3080 Ti 12 GB = CUDA0; MI50 32 GB = ROCm0, gfx906 wave64, no VMM |
| Device visibility | `HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0` hides the Raphael iGPU |
| Model | qwen4exp, 48 layers, embedding width 2560, 512 experts, 10 selected/token, expert FFN 640 |
| Actual expert formats | IQ2_S/IQ3_S gate/up projections; IQ4_NL remains on its existing path; do not assume Q4_K or f16/f32 experts |
| Placement | 21 CPU expert blocks: `0|1|2|3|31..47`; split `3,32`; shared output and draft on CUDA0 |
| Context and MTP | context 150000, batch 2048, **ubatch 1280 (user's current configuration)**, MTP maximum draft length 3; nominal verify window includes the current token plus up to 3 drafts |
| KV | measured P6 controls: target q4_0/q4_0, draft f16/f16, indexer top_k 2048; deployed turbo4 assessed in a separate control |
| Loading | `--fit off`; anchored `^output\.weight$=CUDA0`; lazy PLE, prefetch off, no whole-model mlock/pinning |
| Cache | CUDA-owned singleton, auto budget, expert-parallel setting held fixed; retain dispatch/collect fallback |

**Primary control: ub1280.** Preserve the existing RTX 3080 Ti + MI50 batching/memory rewrite and measure P6 off/on on the same new build. Specifically retain block-ranked QSA (`e4662b5d4`), the shared KQ-mask view (`5542623ce`), and the MTP draft ubatch cap at 256 (`3ab2be1c6`). These reduced compute buffers and enabled the larger expert cache; rebuilding from the pristine pin would discard the validated base.

Updated HANDOFF §5.2 supplies the relevant historical reference points, all at ub1280, q4_0/q4_0, default top_k, MTP n=3, 21 CPU blocks, and cache auto:

| Reference | pp6k | pp32k | tg512f | Cache pools |
|---|---|---|---|---|
| `opt/rtx-mi50` | 340.5 | 305.2 | 36.5 | 3506 MiB |
| P0 rerun | 333.8 | 299.8 | 37.1 | 3506 MiB |
| P1 mmap | 335.2 | 300.2 | 36.4 | 3506 MiB |
| P1 direct | 330.6 | 298.8 | 37.8 | 3506 MiB |

The opt rows and P0/P1 rows come from different sessions. Establish a fresh off baseline in the same session as on; tg per-repetition spread is about ±2 t/s, and pp32k can drift ~2% between sessions. The old ub768/34.2–34.3 tg figures are historical diagnostics only. If a ub768 control is run, report it separately.

PLE mode is a controlled variable: run paired P6 off/on at ub1280 with mmap, then another pair with direct (16 reader threads and the existing row cache). The engine defaults to mmap; `_opt.sh` defaults to direct. Do not promote direct to the main launcher as part of P6. P1 is already merged and passed quality/needles; no PLE-reader rewrite is needed here.

The main deployed launcher uses `~/Projects/buun-opt-rtx-mi50/runtime`, ub1280, empty `TOPK_OVERRIDE` (2048), and **turbo4** KV, which remains unmeasured at this ubatch. Keep the primary measured P6 controls on q4_0; add a separate optional turbo4 off/on benchmark + quality/long-context comparison before enabling P6 in the deployed turbo4 environment. Do not mix codec changes into a P6 delta.

The target currently documented runtime is P1 `d87edd0aa` plus kit patches; later `6594ea436` was not rebuilt there. Record actual target and new-build revisions, and compare P6 off/on on the same new build so unrelated baseline differences are not attributed to P6. The proposed O_DIRECT `got % PLE_ALIGN` follow-up is outside this package.

## Design decisions and boundaries

1. **Adapt the algorithm, not Strata's arithmetic.** `Strata/src/kernels/cpu/iq_avx2.cpp` and `iq_avx2_rows.inl` decode one weight chunk outside the token loop. Their generic `row_dot` folds scaling into each block and changes accumulation order. Its parity test permits relative error against ggml. P6 requires bit identity against buun, so retain buun's exact expressions. IQ2_S must apply `0.125f` after `hsum_float_8`; IQ3_S has no such final factor.
2. **Use existing grouping.** `ggml_compute_forward_mul_mat_id_impl` already builds `matrix_row_counts`/`matrix_rows`, excluding successful GPU cache hits. No new expert grouping, activation packing, scratch buffer, thread pool, scheduler flag, or public llama API is needed.
3. **Bound the fast path.** Enable only for exact environment value `1`, x86 AVX2 builds, IQ2_S/IQ3_S, and `2 <= ids->ne[1] <= 8`. An expert/chunk needs at least two remaining rows. Larger prefill batches, including the user's ub1280 prefill, retain the rewritten batching path. The kernel width is the expert's routed rows inside a small verify graph; it is not the server's `--ubatch-size`. Evaluate a trailing singleton with the existing vec_dot.
4. **Retain IQ panel precedence.** `iqp.cpp` already selects a panel kernel for at least eight rows/expert when eligible. Keep that branch before the chunk path. Its presence means an eight-token graph can use panels even though a direct eight-column MT kernel test passes. Do not disable or reorder panels in production to manufacture a gain.
5. **Preserve pointer semantics.** Activation pointer uses `id % ne11` and either packed `row_size` or `nb11`/`nb12`. Output pointer uses routed slot `id`, not expert index. Never assume consecutive activation/output columns or `ne11 == 1`.
6. **Keep operations separate.** Port dot products only; do not copy Strata's fused gate/up SiLU calculation or expert pool. Existing ggml graph controls activation and down projection.
7. **No P2–P5 dependency.** P6 works with current grouping. If those packages are merged later, rerun cache fallback and row-mask checks. No P4 streaming/P5 pinning is introduced here.

Alternatives rejected: direct Strata port fails the exactness contract; routing small verify groups through IQ panels changes arithmetic and scratch policy. Extending buun's existing AVX2 loops is the smallest compatible change.

## Files and responsibilities

Paths below are relative to `/home/aslusarchuk/Projects/buun-llama-cpp` unless a full Strata path is shown.

| Action | File | Responsibility |
|---|---|---|
| Modify | `ggml/src/ggml-cpu/quants.h` | Two internal multi-column declarations; no public API |
| Modify | `ggml/src/ggml-cpu/arch/x86/quants.c` | Exact AVX2 MT kernels and x86 non-AVX2 fallback |
| Modify | `ggml/src/ggml-cpu/ggml-cpu.c` | Environment gate and pointer-aware chunk dispatch |
| Create | `tests/test-cpu-iq-mt.cpp` | Bitwise kernel and full MUL_MAT_ID tests, optional microbenchmark |
| Modify | `tests/CMakeLists.txt` | CPU test registration with MODULE-library guard |
| Create | `scripts/rtx-mi50/harness/p6_arms.sh` | Sequential controlled off/on target measurements |
| Modify | `scripts/rtx-mi50/harness/README.md` | P6 arm invocation and artifact meanings |
| Modify | `scripts/rtx-mi50/REPORT.md` | Actual results, applicability, rollback, unresolved gates |

Read-only Strata references:

- `/home/aslusarchuk/Projects/Strata/src/kernels/cpu/iq_avx2.cpp`: `Fmt32<22>`, `Fmt32<21>`, gather variants, reduction.
- `/home/aslusarchuk/Projects/Strata/src/kernels/cpu/iq_avx2_rows.inl`: `row_dot`, `dot_rows`, token-count dispatch.
- `/home/aslusarchuk/Projects/Strata/src/kernels/cpu/native_expert.cpp`: `native_gu_mt_min`; comments documenting rounding changes with group width.
- `/home/aslusarchuk/Projects/Strata/src/kernels/cpu/iq_avx2_parity.cpp`: deterministic block fixtures and DRAM-sized benchmark method.
- `/home/aslusarchuk/Projects/Strata/src/kernels/native_expert_parity.cpp`: native expert shapes and per-format checks.

### Task 1: Pin the exact numerical contract in CPU tests

**Files:** create `tests/test-cpu-iq-mt.cpp`; modify `tests/CMakeLists.txt`; declare internal functions in `ggml/src/ggml-cpu/quants.h`.

- [ ] Confirm the implementation branch and clean worktree. The current branch already matches P6; do not create another branch or switch away from user changes.

```bash
git status --short
git branch --show-current
```

- [ ] Add these declarations inside the existing C linkage block in `quants.h`:

```c
// One weight row against 1..8 Q8_K activation rows; s[t] is the t-th dot.
void ggml_vec_dot_iq2_s_q8_K_mt(int n, float * s, const void * vx,
                              const void * const * vy, int ncols);
void ggml_vec_dot_iq3_s_q8_K_mt(int n, float * s, const void * vx,
                              const void * const * vy, int ncols);
```

- [ ] Start the test with this complete kernel-parity program. Use explicit failures, not C `assert`, because Release can define `NDEBUG`.

```cpp
#include "ggml.h"
#include "ggml-cpu.h"
#include "quants.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static unsigned rng_state = 0x6a09e667u;
static unsigned next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static std::vector<unsigned char> weights(ggml_type type, int k, int rows) {
    const size_t row = ggml_row_size(type, k);
    std::vector<unsigned char> bytes(row * rows);
    for (auto & b : bytes) b = static_cast<unsigned char>(next_u32());
    const size_t block = ggml_type_size(type);
    for (size_t off = 0; off < bytes.size(); off += block) {
        const float d = float(int(next_u32() % 65) - 32) / 64.0f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(bytes.data() + off, &h, sizeof(h));
    }
    return bytes;
}
static void kernel_parity() {
    for (ggml_type type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}) {
        const auto * traits = ggml_get_type_traits_cpu(type);
        const auto quant = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
        for (int k : {256, 512, 2560}) {
            for (int trial = 0; trial < 64; ++trial) {
                auto w = weights(type, k, 1);
                std::vector<std::vector<unsigned char>> q(8);
                const void * p[8];
                float expected[8], got[8];
                std::vector<float> x(k);
                for (int t = 0; t < 8; ++t) {
                    for (float & v : x) v = float(int(next_u32() % 2049) - 1024) / 128.0f;
                    if (trial == 0) for (float & v : x) v = 0.0f;
                    if (trial == 1) for (int j = 0; j < k; ++j) x[j] = (j & 1) ? 8.0f : -8.0f;
                    q[t].resize(ggml_row_size(GGML_TYPE_Q8_K, k));
                    quant(x.data(), q[t].data(), k);
                    p[t] = q[t].data();
                    traits->vec_dot(k, expected + t, 0, w.data(), 0, p[t], 0, 1);
                }
                for (int nt = 1; nt <= 8; ++nt) {
                    if (type == GGML_TYPE_IQ2_S) ggml_vec_dot_iq2_s_q8_K_mt(k, got, w.data(), p, nt);
                    else ggml_vec_dot_iq3_s_q8_K_mt(k, got, w.data(), p, nt);
                    for (int t = 0; t < nt; ++t) {
                        if (std::memcmp(got + t, expected + t, sizeof(float)) != 0) {
                            std::fprintf(stderr, "type=%d k=%d trial=%d nt=%d token=%d\n", type, k, trial, nt, t);
                            require(false, "MT differs from existing vec_dot in at least one bit");
                        }
                    }
                }
            }
        }
    }
}
int main() {
    kernel_parity();
    std::puts("cpu-iq-mt: kernel parity PASS");
    return 0;
}
```

- [ ] Register a test only where the internal x86 symbols exist and linking the CPU library is valid:

```cmake
if (TARGET ggml-cpu AND NOT GGML_BACKEND_DL AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    llama_build_and_test(test-cpu-iq-mt.cpp)
    target_include_directories(test-cpu-iq-mt PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
    target_link_libraries(test-cpu-iq-mt PRIVATE ggml-cpu)
endif()
```

Do not link a `ggml-cpu` MODULE into an executable. The target production build has `GGML_BACKEND_DL=ON`, so this private-symbol unit test belongs to the local non-DL build; production integration uses `test-backend-ops` instead.

- [ ] Configure with the deployed CPU instruction set and run the build before adding MT definitions:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX512=OFF -DGGML_AVX_VNNI=OFF \
  -DGGML_BACKEND_DL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu --target test-cpu-iq-mt -j6
```

Expected initial failure: undefined references to the two new `_mt` functions. Commit the test, declarations, and registration as `ggml-cpu : specify exact IQ multi-token dots`.

### Task 2: Add exact AVX2 kernels

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c`, beside the existing IQ2_S and IQ3_S dot products.

- [ ] Insert the complete functions in Appendix A. The appendix was derived from this revision's existing AVX2 branches. Keep the original single-token functions unchanged.
- [ ] Verify these invariants while reviewing the added code:
  - Weight grids, high-bit unpacking, sign masks, and integer scales are computed outside the token loop.
  - Every token has independent `sumi1`, `sumi2`, and `accumf` lanes.
  - Each token performs the original `_mm256_maddubs_epi16`, `_mm256_madd_epi16`, integer adds, and one FMA per weight block.
  - The scale product remains `GGML_CPU_FP16_TO_FP32(x[i].d) * y[t][i].d`.
  - IQ2_S retains final `0.125f * hsum_float_8`; IQ3_S retains `hsum_float_8`.
  - The non-AVX2 branch loops over the existing architecture vec_dot; no new ISA is required to link the x86 test.

- [ ] Run the direct bitwise test:

```bash
cmake --build build-cpu --target test-cpu-iq-mt -j6
./build-cpu/bin/test-cpu-iq-mt
```

Expected: `cpu-iq-mt: kernel parity PASS`, exit 0. Any bit mismatch blocks integration; do not change this gate to an NMSE/relative tolerance. Commit as `ggml-cpu : add exact AVX2 IQ multi-token dots`.

### Task 3: Dispatch grouped CPU rows without changing cache behavior

**File:** `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_mul_mat_id_one_chunk` near line 2279.

- [ ] Add this environment helper before the chunk evaluator. Its atomic cache is thread-safe and reads the flag once per process; tests must use fresh processes for off/on.

```c
#if defined(__AVX2__)
static bool ggml_cpu_iq_mt_enabled(void) {
    static atomic_int setting = -1;
    int value = atomic_load_explicit(&setting, memory_order_relaxed);
    if (value < 0) {
        const char * env = getenv("GGML_CPU_IQ_MT");
        value = env != NULL && strcmp(env, "1") == 0;
        atomic_store_explicit(&setting, value, memory_order_relaxed);
    }
    return value == 1;
}
#endif
```

- [ ] Insert this block immediately after the existing `vec_dot_type` declaration inside the chunk evaluator. The original scalar chunk loops follow unchanged.

```c
#if defined(__AVX2__)
    if ((type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S) &&
        ids->ne[1] >= 2 && ids->ne[1] <= 8 &&
        ir1_end - ir1_start >= 2 && ggml_cpu_iq_mt_enabled()) {
        for (int64_t first = ir1_start; first < ir1_end; first += 8) {
            const int count = (int) MIN(INT64_C(8), ir1_end - first);
            const void * acts[8];
            float * outputs[8];
            float dots[8];
            for (int t = 0; t < count; ++t) {
                const struct mmid_row_mapping mapping = MMID_MATRIX_ROW(cur_a, first + t);
                const int64_t col = mapping.i1 % ne11;
                acts[t] = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (col + mapping.i2 * ne11) * row_size
                    : col * nb11 + mapping.i2 * nb12);
                outputs[t] = (float *) ((char *) dst->data + mapping.i1 * nb1 + mapping.i2 * nb2);
            }
            for (int64_t row = ir0_start; row < ir0_end; ++row) {
                const void * weight = src0_cur + row * nb01;
                if (count == 1) {
                    vec_dot(ne00, dots, 0, weight, 0, acts[0], 0, 1);
                } else if (type == GGML_TYPE_IQ2_S) {
                    ggml_vec_dot_iq2_s_q8_K_mt(ne00, dots, weight, acts, count);
                } else {
                    ggml_vec_dot_iq3_s_q8_K_mt(ne00, dots, weight, acts, count);
                }
                for (int t = 0; t < count; ++t) outputs[t][row] = dots[t];
            }
        }
        return;
    }
#endif
```

The caller already validates tensor layouts and block sizes and already quantizes F32 activations into shared Q8_K scratch. This block consumes that scratch after the existing barrier and does not change its allocation. It respects the current two-dimensional work chunks, including split expert-row ranges; no new scheduling scheme is introduced.

- [ ] Keep `ggml_compute_forward_mul_mat_id_impl` cache planning, dispatch, barriers, panel precedence, and collection untouched. Successful hits never appear in CPU row groups. Failed dispatch restores hit rows before chunk work. Failed collection invokes this same evaluator for singleton fallback rows, which retain vec_dot.
- [ ] Keep the masked CPU wrapper behavior: it groups only requested logical rows. Expert-window exclusions still write zeros before chunk work.
- [ ] Build and run existing backend checks in fresh processes:

```bash
cmake --build build-cpu --target test-backend-ops test-llama-archs test-cpu-iq-mt -j6
env -u GGML_CPU_IQ_MT ./build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID
GGML_CPU_IQ_MT=1 ./build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID
GGML_CPU_IQ_MT=0 ./build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID
GGML_CPU_IQ_MT=1 ./build-cpu/bin/test-llama-archs -a qwen4exp
```

Expected: no new failed backend cases; qwen4exp architecture test passes. Existing backend checks use tolerance, so Task 4 supplies bitwise integration coverage. Commit as `ggml-cpu : batch small IQ expert row groups`.

### Task 4: Prove layout and graph integration bit for bit

**Files:** expand `tests/test-cpu-iq-mt.cpp`; modify test registration.

- [ ] Add this full integration fixture above `main`, with the additional includes shown. It compares the CPU graph result to the exact existing per-row dot using independently quantized copies of the original F32 inputs. It covers broadcast gate/up inputs, per-slot down-style inputs, strided prequantized Q8_K, strided IDs, expert windows, singleton groups, shared groups, more than eight tokens, and output-row/thread tails.

```cpp
#include "ggml-alloc.h"
#include "ggml-backend.h"

static void graph_parity(ggml_type type, int nt, int slots, bool q8_view,
                         bool expert_window, bool singleton, int threads) {
    constexpr int k = 2560, rows = 19, experts = 16, used = 10;
    auto * ctx = ggml_init({ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true});
    require(ctx != nullptr, "context allocation failed");
    auto * w = ggml_new_tensor_3d(ctx, type, k, rows, experts);
    auto * ids_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used + 3, nt);
    auto * ids = ggml_view_2d(ctx, ids_storage, used, nt, ids_storage->nb[1], sizeof(int32_t));
    const ggml_type xt = q8_view ? GGML_TYPE_Q8_K : GGML_TYPE_F32;
    auto * x_storage = ggml_new_tensor_3d(ctx, xt, k, slots + 1, nt);
    auto * x = ggml_view_3d(ctx, x_storage, k, slots, nt,
                          x_storage->nb[1], x_storage->nb[2], x_storage->nb[1]);
    auto * out = ggml_mul_mat_id(ctx, w, x, ids);
    const int lo = expert_window ? 7 : 0;
    if (expert_window) ggml_mul_mat_id_set_expert_window(out, lo, experts);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr, "CPU backend allocation failed");
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "tensor allocation failed");
    auto wb = weights(type, k, rows * experts);
    ggml_backend_tensor_set(w, wb.data(), 0, wb.size());
    const size_t qrow = ggml_row_size(GGML_TYPE_Q8_K, k);
    const auto quant = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    const auto dot = ggml_get_type_traits_cpu(type)->vec_dot;
    std::vector<std::vector<unsigned char>> q(nt * slots);
    std::vector<unsigned char> xb(ggml_nbytes(x_storage), 0);
    std::vector<float> input(k);
    for (int t = 0; t < nt; ++t) for (int s = 0; s < slots; ++s) {
        for (float & v : input) v = float(int(next_u32() % 1025) - 512) / 64.0f;
        q[t * slots + s].resize(qrow);
        quant(input.data(), q[t * slots + s].data(), k);
        const size_t off = x_storage->nb[1] + s * x->nb[1] + t * x->nb[2];
        if (q8_view) std::memcpy(xb.data() + off, q[t * slots + s].data(), qrow);
        else std::memcpy(xb.data() + off, input.data(), k * sizeof(float));
    }
    ggml_backend_tensor_set(x_storage, xb.data(), 0, xb.size());
    std::vector<int32_t> route_storage((used + 3) * nt, 0);
    for (int t = 0; t < nt; ++t) for (int s = 0; s < used; ++s) {
        // Repeated IDs deliberately create larger groups; singleton case uses just one token.
        const int local = singleton ? s : (s + t) % 4;
        int id = lo + local;
        if (expert_window && s == used - 1) id = lo - 1;
        route_storage[t * (used + 3) + s + 1] = id;
    }
    ggml_backend_tensor_set(ids_storage, route_storage.data(), 0,
                            route_storage.size() * sizeof(int32_t));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    const size_t wrow = ggml_row_size(type, k);
    for (int t = 0; t < nt; ++t) for (int s = 0; s < used; ++s) for (int r = 0; r < rows; ++r) {
        const int local = route_storage[t * (used + 3) + s + 1] - lo;
        float expected = 0.0f;
        if (local >= 0 && local < experts) {
            const void * wp = wb.data() + (local * rows + r) * wrow;
            dot(k, &expected, 0, wp, 0, q[t * slots + s % slots].data(), 0, 1);
        }
        require(std::memcmp(&expected, &got[(t * used + s) * rows + r], sizeof(float)) == 0,
                "MUL_MAT_ID differs from existing per-row vec_dot");
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}
```

- [ ] Replace `main` with this runner. Panels are disabled only for the graph fixture so the test's exact vec_dot oracle also applies to groups of eight or more rows; production still keeps panels. Run panel-enabled coverage separately below.

```cpp
int main() {
#if defined(_WIN32)
    _putenv_s("GGML_NO_IQ_PANEL", "1");
#else
    setenv("GGML_NO_IQ_PANEL", "1", 1);
#endif
    kernel_parity();
    for (ggml_type type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S})
        for (int nt : {1, 2, 3, 4, 8, 9})
            for (int slots : {1, 10})
                for (bool q8 : {false, true})
                    for (bool window : {false, true})
                        for (int threads : {1, 6, 12})
                            graph_parity(type, nt, slots, q8, window, nt == 1, threads);
    std::puts("cpu-iq-mt: kernel and graph parity PASS");
    return 0;
}
```

- [ ] Register off/on tests with fixed environments and execute fresh processes; additionally check `0`, `true`, and empty strings choose the disabled path. Use unset/`1` for the mandatory CTest registrations:

```cmake
llama_test(test-cpu-iq-mt NAME test-cpu-iq-mt-enabled)
set_tests_properties(test-cpu-iq-mt PROPERTIES ENVIRONMENT "GGML_CPU_IQ_MT=0")
set_tests_properties(test-cpu-iq-mt-enabled PROPERTIES ENVIRONMENT "GGML_CPU_IQ_MT=1")
```

```bash
cmake --build build-cpu --target test-cpu-iq-mt test-backend-ops -j6
ctest --test-dir build-cpu -R '^test-cpu-iq-mt' --output-on-failure
GGML_CPU_IQ_MT=true ./build-cpu/bin/test-cpu-iq-mt
GGML_CPU_IQ_MT= ./build-cpu/bin/test-cpu-iq-mt
env -u GGML_NO_IQ_PANEL GGML_CPU_IQ_MT=1 ./build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID
```

Expected: all parity checks pass; normal panel-enabled backend tests show no new failures. Review cache-dispatch failure and collect failure branches against Task 3 after these tests. No synthetic CUDA cache mock is needed because P6 does not change their control flow; target arms supply real cache integration.

- [ ] Configure a separate x86 non-AVX2 build and run the same private-symbol test to ensure the fallback links and no AVX2 instructions leak into the disabled architecture:

```bash
cmake -S . -B build-cpu-p6-noavx2 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_AVX2=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF \
  -DGGML_BACKEND_DL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu-p6-noavx2 --target test-cpu-iq-mt -j6
GGML_CPU_IQ_MT=1 ./build-cpu-p6-noavx2/bin/test-cpu-iq-mt
```

Expected: bitwise parity passes against that build's existing AVX/generic dot functions. Non-x86 builds compile no calls to the new x86 symbols. Commit as `tests : cover IQ multi-token routing and fallbacks`.

### Task 5: Measure kernels and the deployed mixed-vendor engine

**Files:** extend `tests/test-cpu-iq-mt.cpp` for optional timing; create `scripts/rtx-mi50/harness/p6_arms.sh`.

- [ ] Add this optional microbenchmark helper before `main`, with its includes. It uses a 256 MiB weight corpus, exceeding the 7600X's L3, and reports time per 640-row projection. Timing has no pass/fail threshold; end-to-end speed decides adoption. Do not claim Strata's measured speedup applies to the 7600X.

```cpp
#include <algorithm>
#include <chrono>

static volatile double bench_sink = 0.0;
static void kernel_benchmark() {
    constexpr int k = 2560, expert_rows = 640;
    for (ggml_type type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}) {
        const size_t row_bytes = ggml_row_size(type, k);
        const int count = int((256ull * 1024 * 1024) / row_bytes / expert_rows) * expert_rows;
        auto w = weights(type, k, count);
        const auto quant = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
        const auto dot = ggml_get_type_traits_cpu(type)->vec_dot;
        std::vector<std::vector<unsigned char>> q(8);
        std::vector<float> input(k);
        const void * acts[8];
        for (int t = 0; t < 8; ++t) {
            for (float & v : input) v = float(int(next_u32() % 1025) - 512) / 64.0f;
            q[t].resize(ggml_row_size(GGML_TYPE_Q8_K, k));
            quant(input.data(), q[t].data(), k);
            acts[t] = q[t].data();
        }
        for (int nt : {1, 2, 3, 4, 8}) {
            std::vector<double> old_times, mt_times;
            for (int pass = 0; pass <= 5; ++pass) {
                // Alternate order to reduce bias from warming and CPU clock changes.
                for (int j = 0; j < 2; ++j) {
                    const bool mt = ((j + pass) & 1) != 0;
                    double checksum = 0.0;
                    float result[8];
                    const auto start = std::chrono::steady_clock::now();
                    for (int r = 0; r < count; ++r) {
                        const void * wr = w.data() + size_t(r) * row_bytes;
                        if (!mt) {
                            for (int t = 0; t < nt; ++t) dot(k, result + t, 0, wr, 0, acts[t], 0, 1);
                        } else if (type == GGML_TYPE_IQ2_S) {
                            ggml_vec_dot_iq2_s_q8_K_mt(k, result, wr, acts, nt);
                        } else {
                            ggml_vec_dot_iq3_s_q8_K_mt(k, result, wr, acts, nt);
                        }
                        for (int t = 0; t < nt; ++t) checksum += result[t];
                    }
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count() * expert_rows / count;
                    bench_sink = checksum;
                    if (pass > 0) (mt ? mt_times : old_times).push_back(ms);
                }
            }
            std::sort(old_times.begin(), old_times.end());
            std::sort(mt_times.begin(), mt_times.end());
            std::printf("type=%d nt=%d rows=640 old_ms=%.6f mt_ms=%.6f speedup=%.3f\n",
                        type, nt, old_times[2], mt_times[2], old_times[2] / mt_times[2]);
        }
    }
}
```

Change `main`'s signature and insert the benchmark branch before the existing parity runner:

```diff
-int main() {
+int main(int argc, char ** argv) {
+    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
+        kernel_benchmark();
+        return 0;
+    }
```

```bash
cmake --build build-cpu --target test-cpu-iq-mt -j6
./build-cpu/bin/test-cpu-iq-mt --bench
```

Expected: one timing row for each format/window-width pair. This single-thread test measures projections, not all 10 selected experts or a full verify pass. Run it on the target 7600X as well as the dev box, using a separate non-DL CPU build for the private-symbol test. Do not substitute dev-box timings for target measurements.
- [ ] Build the target CPU module explicitly. Rebuilding only `llama-server` does not rebuild a dynamically loaded CPU module. Inspect actual target paths first; REPORT uses `~/Projects/buun-opt-src` and a separate build root. After getting the implementation onto that source checkout, use:

```bash
cd ~/Projects/buun-opt-src
OUT=$HOME/Projects/buun-p6-rtx-mi50 BUILD_JOBS=6 bash scripts/rtx-mi50/build.sh
```

The first isolated build compiles both backends. For later CPU-only iterations, rebuild the CPU module and test tools in the isolated CUDA build, then copy them only while no server from that runtime runs:

```bash
cmake --build ~/Projects/buun-p6-rtx-mi50/cuda -j6 \
  --target ggml-cpu llama-server test-backend-ops
cp -a ~/Projects/buun-p6-rtx-mi50/cuda/bin/. ~/Projects/buun-p6-rtx-mi50/runtime/
```

Never copy over the deployed runtime. Keep the original CUDA/HIP build directories separate, CUDA cache ownership, gfx906 flags, nvcc 12.0 extern workaround, backend MODULE link guard, and HIP syncwarp shim. The available HIP-owner patch stays unapplied. P6 changes no GPU kernel. Updated HANDOFF §7.4 records that `c4fe38c4d` fixed the stable-ties collect stride and added radix top-k for CCCL < 3.2; preserve that fix. The remaining wave64 ballot-mask audit and f16/f32 MUL_MAT_ID fault stay outside P6.

- [ ] On the target, ensure one server at a time, validate devices/library dependencies with the existing build script, then run CPU backend checks in the merged runtime:

```bash
env -u GGML_CPU_IQ_MT ~/Projects/buun-p6-rtx-mi50/runtime/test-backend-ops test -b CPU -o MUL_MAT_ID
GGML_CPU_IQ_MT=1 ~/Projects/buun-p6-rtx-mi50/runtime/test-backend-ops test -b CPU -o MUL_MAT_ID
```

- [ ] Add this complete target driver, executed from `~/claude-opt`. It reuses the existing launcher/run_arm harness without replacing launchers or changing production defaults:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/claude-opt"
if pgrep -a -x llama-server; then
    printf '%s\n' 'A server is already running; stop it through its launcher before running P6 arms.' >&2
    exit 1
fi
export BUUN_ROOT="$HOME/Projects/buun-p6-rtx-mi50"
export CTX=150000 BATCH_SIZE=2048 UBATCH_SIZE=1280 THREADS=6 THREADS_BATCH=6
export KV_K=q4_0 KV_V=q4_0 DRAFT_KV_K=f16 DRAFT_KV_V=f16
export N_MAX=3 MTP=1 MOE_CACHE_EP=0 LLAMA_MTP_DRAFT_UBATCH=256
export TOPK_OVERRIDE= GGML_SCHED_TIMING=0
export BENCH_ARGS="--only pp6k,pp32k,tg512f --reps 5"

# Greedy differences are diagnostic only; strict parity is tested on saved inputs.
export EXTRA_ARGS="--ple-io mmap"
for arm in off-a off-b on; do
    value=0; [[ "$arm" == on ]] && value=1
    GGML_CPU_IQ_MT="$value" MOE_CACHE=off GREEDY=1 \
        BENCH_ARGS= IOSTAT=0 bash ./run_arm.sh "p6-u1280-greedy-$arm" 0
done
python3 ./greedy.py --compare p6-u1280-greedy-off-a.greedy.json p6-u1280-greedy-off-b.greedy.json || true
python3 ./greedy.py --compare p6-u1280-greedy-off-a.greedy.json p6-u1280-greedy-on.greedy.json || true

# Each PLE mode has its own counterbalanced P6 off/on comparison.
# Optional historical control: P6_UBATCHES="1280 768"; keep its results separate.
for ple in mmap direct; do
    export EXTRA_ARGS="--ple-io $ple"
    if [[ "$ple" == direct ]]; then
        export EXTRA_ARGS="--ple-io direct --ple-io-threads 16 --ple-row-cache 1048576"
    fi
    for ub in ${P6_UBATCHES:-1280}; do
        for round in 1 2; do
            order="0 1"; [[ "$round" == 2 ]] && order="1 0"
            for value in $order; do
                GGML_CPU_IQ_MT="$value" UBATCH_SIZE="$ub" MOE_CACHE=auto \
                    bash ./run_arm.sh "p6-u${ub}-${ple}-r${round}-mt${value}" 0
            done
        done
    done
    # CPU/cache control and quality/long-context checks always use ub1280.
    for value in 0 1; do
        GGML_CPU_IQ_MT="$value" UBATCH_SIZE=1280 MOE_CACHE=off \
            bash ./run_arm.sh "p6-u1280-${ple}-cpu-mt${value}" 0
        GGML_CPU_IQ_MT="$value" UBATCH_SIZE=1280 MOE_CACHE=auto QUALITY=1 \
            BENCH_ARGS= bash ./run_arm.sh "p6-u1280-${ple}-quality-mt${value}" 0
    done
done

# Optional deployed-codec check. Same ubatch, PLE mode, and source build in both arms.
# The preceding q4_0 arms are the measured reference; turbo4 results stay separate.
if [[ "${P6_TURBO4_CONTROL:-0}" == 1 ]]; then
    export EXTRA_ARGS="--ple-io mmap"
    for value in 0 1; do
        GGML_CPU_IQ_MT="$value" UBATCH_SIZE=1280 MOE_CACHE=auto QUALITY=1 \
            KV_K=turbo4 KV_V=turbo4 \
            bash ./run_arm.sh "p6-u1280-turbo4-mt${value}" 0
    done
fi
```

`run_arm.sh` currently reports failures in logs and can finish after a failed request. Inspect every arm's listening state, request errors, benchmark output, and quality JSON; its exit code alone is not a pass. The driver deliberately keeps both greedy comparisons even on a mismatch; comparison failures remain informational artifacts. It checks for an existing server before starting any arm. For a deployed-codec comparison, run with `P6_TURBO4_CONTROL=1` and evaluate turbo4 off versus q4_0/mmap off first, then turbo4 off versus on, with quality/needles and post-load VRAM inspected separately.

- [ ] Analyze the existing JSONL benchmark results using the current `summ.py`/report workflow. Exclude boot prefill samples; keep at least five usable repetitions per comparison. Capture cache hits/pools/failures, draft acceptance/window lengths, major faults/read bytes, and peak RSS/VRAM. Keep `GGML_SCHED_TIMING=0` in throughput arms; run separate matched off/on profiling arms with the existing timing enabled to examine CPU op sums. P0 timing synchronizes each split, so it cannot establish copy/compute overlap. Do not add a new global profiling subsystem.
- [ ] Confirm the shared-group optimization is exercised. In a local debug build, count coverage of the MT call lines in Task 3 while running the enabled graph fixture. On the target, use kernel microbenchmarks and existing CPU-op timing; do not assume every expert in a four-token window is reused. Actual cache misses can reduce a shared group to one row.
- [ ] Record quality 10/10; needles at 10/50/90% all found at approximately 142k tokens; no new cache dispatch/collect errors; no OOM; ROCm0 retains about 500 MiB or more after load. P6 should add only bounded stack arrays, with no model-sized RAM/VRAM growth.

### Task 6: Decide adoption, document evidence, preserve rollback

**Files:** `scripts/rtx-mi50/REPORT.md`, `scripts/rtx-mi50/harness/README.md`.

- [ ] Append a P6 report section containing the preserved batching rewrite's source/build revision, explicit ub1280, instruction flags, all arm names/configurations, five-repetition medians, spread, CPU-op timing, cache statistics, quality/needles, and numerical test results. Separate primary ub1280 off/on results by PLE mode, the cache-off control, any optional ub768 diagnostic, and any turbo4 control. Historical §5.2 numbers give context; fresh same-session off arms are the comparison baseline.
- [ ] Apply these acceptance gates:
  1. Direct MT vs current vec_dot: zero differing bits for both formats, widths 1–8, random blocks and edge inputs.
  2. Saved graph inputs: zero differing bits with the new flag enabled; disabled/singleton/large-batch/non-AVX2 paths retain existing behavior.
  3. Normal IQ panels and CUDA cache coexist without new backend failures, missing rows, or double writes.
  4. Greedy comparisons are **information only**, per updated HANDOFF §4.13/§10: off/off already differed on this host, even with cache off. Retain off/off/on artifacts to diagnose changes, but do not make text/token identity an adoption gate or claim server exactness. Strict correctness comes from kernel and saved-input graph parity plus the quality/needle gates.
  5. Quality and long-context gates pass. Memory remains within the deployed limits.
  6. Same-session ub1280 cache-auto tg512f improves reproducibly beyond observed variation; pp6k/pp32k and total latency must not regress beyond noise. Updated P0 reports about ±2 t/s per tg repetition and ~2% pp32k drift across sessions. Report medians/spread, counterbalance off/on order, and rerun borderline results. Kernel speed alone does not pass adoption; P6 targets decode CPU work and promises no prefill speedup.
- [ ] Keep `GGML_CPU_IQ_MT` unset by default. If measurements justify rollout, use `GGML_CPU_IQ_MT=1` in an explicitly selected launch environment. Roll back with `GGML_CPU_IQ_MT=0` or by unsetting it and starting a fresh server. Keep all old launchers/runtimes runnable.
- [ ] Commit report and harness changes as `scripts : record P6 IQ multi-token validation`. Merge only after the gates pass; preserve an unsuccessful experiment branch with its measurements.

## Scope deliberately deferred

IQ4_NL multi-token down projection, gather/VNNI tuning, AVX-512 kernels, CPU affinity policy, larger prefill windows, fused SiLU/gate-up, IQ panel changes, GPU streaming/pinning, CUDA allocator step investigation, pooled QSA keys, KV streaming, and HIP kernel repairs are outside P6. No speedup estimate substitutes for measurements on the 7600X.

## Appendix A: Complete kernel definitions

Insert both definitions in `ggml/src/ggml-cpu/arch/x86/quants.c`. The test executable links their x86 fallback even when AVX2 is disabled; normal CPU dispatch references them only inside its AVX2 guard.

```c
void ggml_vec_dot_iq2_s_q8_K_mt(int n, float * s, const void * vx,
                                const void * const * vy, int ncols) {
    GGML_ASSERT(n % QK_K == 0 && ncols >= 1 && ncols <= 8);
#if defined(__AVX2__)
    const block_iq2_s * x = vx;
    const block_q8_K * y[8];
    for (int t = 0; t < ncols; ++t) y[t] = vy[t];
    const int nb = n / QK_K;


   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf[8];
    for (int t = 0; t < ncols; ++t) accumf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1[8], sumi2[8];
        for (int t = 0; t < ncols; ++t) {
            sumi1[t] = _mm256_setzero_si256();
            sumi2[t] = _mm256_setzero_si256();
        }
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);

            const __m256i sc1 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0));
            const __m256i sc2 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1));
            signs += 4;
            for (int t = 0; t < ncols; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) (q8 + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = _mm256_madd_epi16(dot1, sc1);
            const __m256i p2 = _mm256_madd_epi16(dot2, sc2);
            sumi1[t] = _mm256_add_epi32(sumi1[t], p1);
            sumi2[t] = _mm256_add_epi32(sumi2[t], p2);
            }
        }

        for (int t = 0; t < ncols; ++t) {
            const float d = dx * y[t][i].d;
            accumf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[t], sumi2[t])), accumf[t]);
        }

    }

    for (int t = 0; t < ncols; ++t) s[t] = 0.125f * hsum_float_8(accumf[t]);
#else
    for (int t = 0; t < ncols; ++t) {
        ggml_vec_dot_iq2_s_q8_K(n, s + t, 0, vx, 0, vy[t], 0, 1);
    }
#endif
}
```

```c
void ggml_vec_dot_iq3_s_q8_K_mt(int n, float * s, const void * vx,
                                const void * const * vy, int ncols) {
    GGML_ASSERT(n % QK_K == 0 && ncols >= 1 && ncols <= 8);
#if defined(__AVX2__)
    const block_iq3_s * x = vx;
    const block_q8_K * y[8];
    for (int t = 0; t < ncols; ++t) y[t] = vy[t];
    const int nb = n / QK_K;


   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf[8];
    for (int t = 0; t < ncols; ++t) accumf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        __m256i sumi1[8], sumi2[8];
        for (int t = 0; t < ncols; ++t) {
            sumi1[t] = _mm256_setzero_si256();
            sumi2[t] = _mm256_setzero_si256();
        }
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);

            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >> 4;
            const __m256i sc1 = _mm256_set1_epi16(2*ls1+1);
            const __m256i sc2 = _mm256_set1_epi16(2*ls2+1);
            signs += 4;
            for (int t = 0; t < ncols; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) (q8 + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i p1 = _mm256_madd_epi16(dot1, sc1);
            const __m256i p2 = _mm256_madd_epi16(dot2, sc2);
            sumi1[t] = _mm256_add_epi32(sumi1[t], p1);
            sumi2[t] = _mm256_add_epi32(sumi2[t], p2);
            }
        }

        for (int t = 0; t < ncols; ++t) {
            const float d = dx * y[t][i].d;
            accumf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[t], sumi2[t])), accumf[t]);
        }

    }

    for (int t = 0; t < ncols; ++t) s[t] = hsum_float_8(accumf[t]);
#else
    for (int t = 0; t < ncols; ++t) {
        ggml_vec_dot_iq3_s_q8_K(n, s + t, 0, vx, 0, vy[t], 0, 1);
    }
#endif
}
```

## Plan validation performed

- Re-read all 241 lines of updated HANDOFF; applied the user's corrected ub1280 setting, newest §5.2 P0/P1 references, and revised exactness/noise/deployment guidance.
- Extracted both Appendix A functions into scratch files under `/tmp`, compiled them with AVX2/FMA/F16C, and ran the proposed direct kernel parity test against the existing CPU type-trait dot functions: PASS, zero differing bits.
- Compiled the proposed graph fixture and ran it against the existing backend with panels disabled: PASS. This validates the fixture/oracle; it does **not** validate the proposed dispatch, which has not been added to engine source.
- Compiled the Appendix A x86 non-AVX2 fallback successfully. A full non-AVX2 backend build remains an implementation check.
- Syntax-checked the complete proposed test including its microbenchmark and checked the benchmark driver with `bash -n`; both passed.
- Checked Markdown fences, declarations, task references, and placeholder scan. Target performance, actual ub1280 runtime settings, cache integration, quality, and long-context gates remain unmeasured.
