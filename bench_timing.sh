#!/usr/bin/env bash
# Compares sequential vs parallel wall time for 6 benchmark runs.
#
# Runs (10 epochs each, uniform noise 40%):
#   baseline_resnet32_s1234   plain CE,  ResNet-32, seed 1234
#   l2rw_resnet32_s1234       L2RW,      ResNet-32, seed 1234
#   l2rw_resnet32_s0          L2RW,      ResNet-32, seed 0
#   baseline_resnet20_s1234   plain CE,  ResNet-20, seed 1234
#   l2rw_resnet20_s1234       L2RW,      ResNet-20, seed 1234
#   l2rw_resnet20_s0          L2RW,      ResNet-20, seed 0
#
# Usage:
#   bash bench_timing.sh [wandb_project]
#
# Note: on a single-GPU machine (e.g. Colab) parallel runs share the same
# device and will contend for memory — expect OOM or slowdown vs sequential.
#
# Sequential logs → logs/<name>_seq.log
# Parallel  logs → logs/<name>_par.log
# Checkpoints    → checkpoints/seq/ and checkpoints/par/

set -euo pipefail

if   command -v python3 &>/dev/null; then PY=$(command -v python3)
elif command -v python  &>/dev/null; then PY=$(command -v python)
else echo "ERROR: no python3 found on PATH"; exit 1; fi

if [[ ! -f bench_train.py ]]; then
  echo "ERROR: bench_train.py not found — run from the project root"; exit 1
fi

echo "Python: $PY  ($(${PY} --version 2>&1))"
echo ""

PROJECT=${1:-}
mkdir -p logs checkpoints/seq checkpoints/par

# ── Shared args ────────────────────────────────────────────────────────────────
COMMON=(
  --noise_rate 0.4 --noise_type uniform
  --epochs 3
  --batch_size 100 --val_size 1000 --val_batch_size 100
  --warmup_epochs 0
  --lr 0.1 --weight_decay 2e-4 --lr_milestones 82 123
  --log_sample_rate 0.1 --output_dir ./pairs_data/
)
[[ -n "$PROJECT" ]] && COMMON+=(--wandb_project "$PROJECT")

# ── Run definitions: parallel arrays ──────────────────────────────────────────
NAMES=(
  baseline_resnet32_s1234
  l2rw_resnet32_s1234
  l2rw_resnet32_s0
  baseline_resnet20_s1234
  l2rw_resnet20_s1234
  l2rw_resnet20_s0
)
MODELS=(resnet32 resnet32 resnet32 resnet20 resnet20 resnet20)
SEEDS=(1234 1234 0 1234 1234 0)
IS_BASELINE=(1 0 0 1 0 0)

# ── Sequential ─────────────────────────────────────────────────────────────────
echo "===== SEQUENTIAL ====="
T_SEQ=$SECONDS

for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  extra=(--model "${MODELS[$i]}" --seed "${SEEDS[$i]}" --wandb_run_name "$name")
  [[ ${IS_BASELINE[$i]} -eq 1 ]] && extra+=(--baseline)

  echo "  → $name"
  "$PY" bench_train.py "${COMMON[@]}" "${extra[@]}" \
    --checkpoint_dir ./checkpoints/seq/ \
    2>&1 | tee "logs/${name}_seq.log" | grep --line-buffered "^\[" || true
done

SEQ_SECS=$(( SECONDS - T_SEQ ))
echo "Sequential total: ${SEQ_SECS}s"
echo ""

# ── Parallel ───────────────────────────────────────────────────────────────────
echo "===== PARALLEL ====="
T_PAR=$SECONDS
PIDS=()

for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  extra=(--model "${MODELS[$i]}" --seed "${SEEDS[$i]}" --wandb_run_name "$name")
  [[ ${IS_BASELINE[$i]} -eq 1 ]] && extra+=(--baseline)

  "$PY" bench_train.py "${COMMON[@]}" "${extra[@]}" \
    --checkpoint_dir ./checkpoints/par/ \
    >"logs/${name}_par.log" 2>&1 &
  PIDS+=($!)
  echo "  → $name (pid $!)"
done

echo "Waiting for all runs to finish..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "  WARNING: pid $pid exited non-zero"
done

PAR_SECS=$(( SECONDS - T_PAR ))
echo "Parallel total: ${PAR_SECS}s"
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "===== RESULTS ====="
printf "  Sequential : %ds\n" "$SEQ_SECS"
printf "  Parallel   : %ds\n" "$PAR_SECS"
if (( SEQ_SECS > 0 && PAR_SECS > 0 )); then
  RATIO=$("$PY" -c "print(f'{$SEQ_SECS / $PAR_SECS:.2f}')")
  printf "  Speedup    : %sx\n" "$RATIO"
fi
echo ""

echo "Best val_acc per run (sequential):"
for name in "${NAMES[@]}"; do
  acc=$(grep -o "best_val_acc=[0-9.]*" "logs/${name}_seq.log" 2>/dev/null | tail -1)
  printf "  %-32s %s\n" "$name" "${acc:-n/a}"
done
