#!/usr/bin/env bash
# Build this tree for the RTX 3080 Ti (CUDA 12.0, sm_86) + Instinct MI50 (ROCm 6.3.3, gfx906) host
# into a separate merged runtime, leaving the deployed runtime untouched.
#
#   bash scripts/rtx-mi50/build.sh            # CUDA + HIP builds, merged runtime in $OUT/runtime
#
# Knobs: KIT (patch kit dir), OUT (build root), BUILD_JOBS, SKIP_HIP=1, SKIP_CUDA=1,
#        KIT_PATCHES (space-separated patch number prefixes to apply, default "0001 0003 0004 0005").
set -euo pipefail

SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
KIT=${KIT:-$HOME/Projects/qwen38-perf/buun-rtx-mi-kit}
OUT=${OUT:-$HOME/Projects/buun-opt-rtx-mi50}
JOBS=${BUILD_JOBS:-6}
KIT_PATCHES=${KIT_PATCHES:-"0001 0003 0004 0005"}

# the same toolchains as the production build (HANDOFF.md section 1)
NVCC=${NVCC:-/usr/bin/nvcc}
CUDA_ROOT=${CUDA_ROOT:-/usr/lib/cuda}
HIP_CLANG=${HIP_CLANG:-/opt/rocm/lib/llvm/bin/clang}

TARGETS=(llama-server llama-cli llama-bench llama-perplexity llama-fit-params test-backend-ops)

log() { printf '\n== %s\n' "$*"; }

# ---- kit patches: apply once, skip when already present --------------------------------------
apply_kit_patches() {
    local p f
    for p in $KIT_PATCHES; do
        f=$(find "$KIT" -maxdepth 2 -name "${p}-*.patch" | head -n1 || true)
        if [[ -z "$f" ]]; then
            echo "warning: kit patch $p not found under $KIT" >&2
            continue
        fi
        if git -C "$SRC" apply --reverse --check "$f" >/dev/null 2>&1; then
            echo "kit patch already applied: $(basename "$f")"
        elif git -C "$SRC" apply --check "$f" >/dev/null 2>&1; then
            git -C "$SRC" apply "$f"
            echo "applied kit patch: $(basename "$f")"
        else
            echo "error: kit patch does not apply cleanly: $f" >&2
            exit 1
        fi
    done
}

common_flags=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF
    -DLLAMA_BUILD_TESTS=ON
)

build_cuda() {
    log "CUDA build -> $OUT/cuda"
    cmake -S "$SRC" -B "$OUT/cuda" "${common_flags[@]}" \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CUDA_COMPILER="$NVCC" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
        -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF
    # backend modules are dlopen()ed, so they are not dependencies of the tools: name them
    cmake --build "$OUT/cuda" -j"$JOBS" --target ggml-cpu ggml-cuda "${TARGETS[@]}"
}

build_hip() {
    log "HIP build -> $OUT/hip"
    cmake -S "$SRC" -B "$OUT/hip" "${common_flags[@]}" \
        -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
        -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
        -DGGML_HIP_NO_VMM=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_HIP_MMQ_MFMA=ON \
        -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
        -DCMAKE_HIP_FLAGS=-DHIP_ENABLE_WARP_SYNC_BUILTINS
    cmake --build "$OUT/hip" -j"$JOBS" --target ggml-hip
}

merge_runtime() {
    log "merged runtime -> $OUT/runtime"
    rm -rf "$OUT/runtime"
    mkdir -p "$OUT/runtime"
    cp -a "$OUT/cuda/bin/." "$OUT/runtime/"
    cp -a "$OUT"/hip/bin/libggml-hip.so* "$OUT/runtime/"
    {
        echo "source: $SRC"
        echo "commit: $(git -C "$SRC" rev-parse HEAD) ($(git -C "$SRC" rev-parse --abbrev-ref HEAD))"
        echo "local diff sha256: $(git -C "$SRC" diff | sha256sum | cut -d' ' -f1)"
        echo "built: $(date -Is)"
    } > "$OUT/runtime/build-info.txt"
    cat "$OUT/runtime/build-info.txt"
}

check_runtime() {
    log "post-build checks"
    for m in "$OUT"/runtime/libggml-cuda.so "$OUT"/runtime/libggml-hip.so; do
        if ldd "$m" | grep -q "not found"; then
            ldd "$m" | grep "not found" >&2
            echo "error: unresolved libraries in $m" >&2
            exit 1
        fi
    done
    HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 "$OUT/runtime/llama-server" --list-devices
}

mkdir -p "$OUT"
apply_kit_patches
[[ "${SKIP_CUDA:-0}" == 1 ]] || build_cuda
[[ "${SKIP_HIP:-0}"  == 1 ]] || build_hip
merge_runtime
check_runtime
