#!/bin/bash
# Interleaved kernel timing of the FlyDSL grouped MoE kernels (Yan's DSR1 conc6144 prefill benchmark,
# run_a4w4_conc6144.sh). Default since 2026-11-02: gemm1 only, 2048 tokens (256 rows per expert = one full M tile),
# weights rotated over 4 copies so every launch reads the experts' data from HBM. The earlier benchmark (1536 tokens,
# all three kernels, no rotation): MOE_EXP_TOKENS=1536 MOE_EXP_KERNELS=all MOE_EXP_ROTATE=1.
#
# usage: op_tests/flydsl_tests/moe_exp/bench.sh [ROUNDS] [VARIANT ...]
#   ROUNDS   timed rounds (default 3); round 0 is an extra warm-up/compile round, not timed
#   VARIANT  name[:ENV=VAL,ENV=VAL,...]   (default: base). Each variant adds its env on top of
#            run_a4w4_conc6144.sh's env.
#            Variants run interleaved (r0: a b c, r1: a b c, ...), one process per (variant, round).
# env:
#   PY           python with torch (default: python3)
#   SITE         directory with AITER (this branch) and FlyDSL installed with pip --target; prepended to PYTHONPATH
#   LOCK         command prefix for every GPU process, e.g. LOCK=gpu-lock on a shared node (default: none)
#   IDLE_CHECK=1 inside LOCK, before each process: wait up to 20 s for rocm-smi to show 0% GPU use and less than
#                IDLE_VRAM_MB (default 1024) of VRAM in use, else stop the series (a loaded but idle model server
#                shows 0% use, so the VRAM check matters on a shared node)
#   OUT          output directory (default: moe_exp_runs/<UTC stamp> in the current directory)
#   MOE_EXP_TOKENS  tokens (default 2048: x top-8 / 64 experts = 256 rows per expert; run_a4w4_conc6144.sh has 1536)
#   MOE_ARGS     test arguments (default: run_a4w4_conc6144.sh's shape with MOE_EXP_TOKENS, below)
#   MOE_EXP_KERNELS  timed kernels: comma list of quant_a1,gemm1,gemm2 or all (default gemm1; bench_driver.py)
#   MOE_EXP_ROTATE / MOE_EXP_ROTATE_A  copies of the timed GEMM's weights (default 4) / also rotate A (default 0)
#   AMD_SMI      amd-smi binary for the firmware/clock lines in meta.txt (default: amd-smi; skipped if missing)
#   PMC_ROCPROF  counter mode: path of rocprofv3; each process runs under `rocprofv3 --pmc $PMC_COUNTERS` and the
#                driver replaces all timing with MOE_EXP_PMC_LAUNCHES (default 8) spaced launches per timed kernel plus
#                a 1 GiB copy control; pmc_hbm.py then reports HBM read bytes per launch (use ROUNDS=1)
#   MOE_EXP_GRAPH_ITERS / MOE_EXP_GRAPH_REPS / MOE_EXP_GRAPH_GAP_S   graph timer (bench_driver.py; default 128
#                launches x 10 replays, 1 s idle before each kernel's graph timing; ITERS=0 turns it off)
# Writes OUT/meta.txt (node, versions, firmware), OUT/<variant>_r<round>.log, OUT/out/*.pt (outputs),
# OUT/dump-<variant>/ (FlyDSL IR dumps, named after the kernels), and OUT/summary.txt (summarize.py).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
ROUNDS=${1:-3}; shift || true
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(base)
PY=${PY:-python3}; LOCK=${LOCK:-}
OUT=${OUT:-moe_exp_runs/$(date -u +%Y%m%d_%H%M%S)}
mkdir -p "$OUT/out" && OUT=$(cd "$OUT" && pwd)
export MOE_EXP_KERNELS=${MOE_EXP_KERNELS:-gemm1} MOE_EXP_ROTATE=${MOE_EXP_ROTATE:-4} MOE_EXP_ROTATE_A=${MOE_EXP_ROTATE_A:-0}
MOE_EXP_TOKENS=${MOE_EXP_TOKENS:-2048}
PMC_ROCPROF=${PMC_ROCPROF:-}
PMC_COUNTERS=${PMC_COUNTERS:-GL2C_EA_RDREQ GL2C_EA_RDREQ_128B GL2C_READ_SECTORS GL2C_REQ}
PMC_REGEX=${PMC_REGEX:-a8w4_tdm|copyBuffer}
if [ -n "$PMC_ROCPROF" ]; then export MOE_EXP_PMC_LAUNCHES=${MOE_EXP_PMC_LAUNCHES:-8}; else export MOE_EXP_PMC_LAUNCHES=0; fi
MOE_ARGS=${MOE_ARGS:-"--scenario kernel --data-format a4w4 --experts 64 --tokens $MOE_EXP_TOKENS --topk 8 --model-dim 7168
  --inter-dim 2048 --act silu --no-bias --iters 128 --no-check-aot-cache --data-init uniform --scale-init uniform"}
# run_a4w4_conc6144.sh's environment (the A layout comes from the config CSV; run_dsr1.sh forced
# AITER_GROUPED_A4W4_A_PRESHUFFLE=1 and had no MMA-group / DS-first settings)
BASE_ENV=(AITER_FLYDSL_MMA_FIRST_GROUP=4 AITER_FLYDSL_MMA_GROUP=4 AITER_FLYDSL_DS_FIRST_N=8
          AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0
          FLYDSL_DUMP_IR=1 AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
          AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}   # FlyDSL 0.3.3.dev882 finds ld.lld only through ROCM_PATH
PYPATH=${SITE:+$SITE:}${PYTHONPATH:-}
LOG=$OUT/run.log
log() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG"; }

# Idle check, run inside LOCK so nothing of ours can start between the check and the run. Exit 99 = busy.
IDLE_VRAM_MB=${IDLE_VRAM_MB:-1024}
IDLE_SH='for i in $(seq 1 10); do
  use=$(rocm-smi --showuse 2>/dev/null | grep -o "GPU use (%): [0-9]*" | grep -o "[0-9]*$" | head -1)
  vb=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -o "Used Memory (B): [0-9]*" | grep -o "[0-9]*$" | head -1)
  vram=$(( ${vb:-0} / 1048576 ))
  [ "${use:-x}" = 0 ] && [ -n "$vb" ] && [ "$vram" -lt '"$IDLE_VRAM_MB"' ] && {
    echo "[moe-exp] idle: GPU use 0%, VRAM ${vram} MB"; exec "$@"; }
  sleep 2
done
echo "[moe-exp] busy: GPU use ${use:-?}%, VRAM ${vram:-?} MB in use; not running"; exit 99'

{
  echo "start      $(date -u '+%F %T UTC')"
  echo "host       $(hostname)   boot $(uptime -s) (system clock, $(date +%Z))"
  echo "repo       $(git -C "$REPO" log -1 --format='%h %cd %s' --date=iso | cut -c1-120)"
  echo "dirty      $(git -C "$REPO" status --short -- aiter op_tests | grep -v '^??' | wc -l) modified tracked files"
  # package metadata only: importing aiter would JIT-build its host modules (and touch the GPU) outside LOCK;
  # run from $HERE so the repository's own aiter/ source tree is not on sys.path
  echo "python     $(cd "$HERE" && PYTHONPATH=$PYPATH $PY -c 'import sys,importlib.metadata as m,importlib.util as u
print(sys.version.split()[0], "| torch", m.version("torch"), "| flydsl", m.version("flydsl"),
      "| aiter", m.version("amd-aiter"), "@", u.find_spec("aiter").origin)' 2>&1 | tail -1)"
  echo "site       ${SITE:-<none>}"
  AMD_SMI=${AMD_SMI:-amd-smi}
  if command -v "$AMD_SMI" >/dev/null; then
    echo "firmware   $($AMD_SMI firmware 2>/dev/null | grep -A1 -E 'FW_ID: (CP_MEC1|PM)$' | grep -oE '(CP_MEC1|PM|FW_VERSION: .*)' | paste -sd' ')"
    echo "clock      $($AMD_SMI metric -g 0 --clock 2>/dev/null | grep -m1 MAX_CLK | tr -s ' ')"
    echo "power cap  $($AMD_SMI static -g 0 --limit 2>/dev/null | grep -E 'SOCKET_POWER_LIMIT' | tr -s ' ' | paste -sd' ')"
  fi
  echo "variants   ${VARIANTS[*]}"
  echo "rounds     $ROUNDS timed (+ warm-up round 0)"
  echo "args       $(echo $MOE_ARGS)"
  echo "kernels    $MOE_EXP_KERNELS"
  if [ "$MOE_EXP_ROTATE" -gt 1 ]; then
    echo "rotation   $MOE_EXP_ROTATE copies of W + W scales$([ "$MOE_EXP_ROTATE_A" = 1 ] && echo ' + A + A scales')"
  else echo "rotation   none"; fi
  [ -n "$PMC_ROCPROF" ] && echo "pmc        $MOE_EXP_PMC_LAUNCHES launches, $PMC_ROCPROF --pmc $PMC_COUNTERS"
  echo "base env   ${BASE_ENV[*]}"
} > "$OUT/meta.txt"
cat "$OUT/meta.txt" | tee -a "$LOG"

for r in $(seq 0 "$ROUNDS"); do
  for spec in "${VARIANTS[@]}"; do
    name=${spec%%:*}; extra=()
    [ "$spec" != "$name" ] && IFS=',' read -r -a extra <<< "${spec#*:}"
    pre=(); [ "${IDLE_CHECK:-0}" = 1 ] && pre=(bash -c "$IDLE_SH" _)
    prof=(); [ -n "$PMC_ROCPROF" ] && prof=("$PMC_ROCPROF" --pmc $PMC_COUNTERS --kernel-include-regex "$PMC_REGEX"
                                            --output-format csv -d "$OUT/pmc-${name}_r$r" -o pmc --)
    log "${name}_r$r queued ${extra[*]}"
    f=$OUT/${name}_r$r.log
    (cd "$REPO" && $LOCK "${pre[@]}" env "${BASE_ENV[@]}" "${extra[@]}" PYTHONPATH="$PYPATH" \
        FLYDSL_RUNTIME_CACHE_DIR="$OUT/cache-$name" FLYDSL_DUMP_DIR="$OUT/dump-$name" \
        MOE_EXP_SAVE_OUT="$OUT/out/${name}_r$r.pt" \
        "${prof[@]}" $PY -u "$HERE/bench_driver.py" $MOE_ARGS > "$f" 2>&1)
    rc=$?
    log "${name}_r$r rc=$rc $(grep -hoE '\[moe-exp\] (idle|busy):.*' "$f" | head -1 | cut -c10-) |" \
        "$(grep -hoE '(quant_a1|gemm[12]): (profiler|graph)_us = [0-9.]+' "$f" | tr '\n' ' ')"
    [ $rc = 99 ] && { log "GPU busy inside the lock, stopping"; exit 1; }
  done
done
log DONE
if [ -n "$PMC_ROCPROF" ]; then
  PYTHONPATH=$PYPATH $PY "$HERE/pmc_hbm.py" "$OUT" | tee "$OUT/pmc_summary.txt"
else
  PYTHONPATH=$PYPATH $PY "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
fi
