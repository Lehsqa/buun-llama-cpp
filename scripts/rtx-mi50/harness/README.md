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

## Deploy

```bash
scp -P 2222 -i ~/.ssh/id_llm scripts/rtx-mi50/harness/* leshqa@176.96.135.203:claude-opt/
```
