#!/usr/bin/env bash
# Runs 5 experiments in parallel:
#   1 baseline (plain CE, no L2RW)
#   3 L2RW uniform noise seeds (0, 1, 1234)
#   1 L2RW asymmetric noise
#
# Usage:
#   bash run_experiments.sh [wandb_project]
#
# All output is tee'd to logs/<run_name>.log so you can tail any run live.
# Final wall times are printed at the end for the L2RW vs baseline comparison.

# ── Python auto-detection (works in Colab, venv, and plain installs) ─────────
if   command -v python3  &>/dev/null; then PYTHON=$(command -v python3)
elif command -v python   &>/dev/null; then PYTHON=$(command -v python)
else echo "ERROR: no python3 or python found on PATH"; exit 1
fi

# Sanity-check: make sure train.py is actually reachable before firing 5 jobs
if [[ ! -f train.py ]]; then
  echo "ERROR: train.py not found in $(pwd)"
  echo "cd to the directory that contains train.py before running this script."
  exit 1
fi

echo "Using Python: $PYTHON  ($(${PYTHON} --version 2>&1))"
echo ""
# ─────────────────────────────────────────────────────────────────────────────

PROJECT=${1:-l2rw-cifar10}
COMMON="--noise_rate 0.4 --epochs 164 --batch_size 100 --val_size 1000 --val_batch_size 100
        --warmup_epochs 0 --lr 0.1 --weight_decay 2e-4 --lr_milestones 82 123
        --log_sample_rate 0.1 --output_dir ./pairs_data/ --wandb_project ${PROJECT}"

mkdir -p logs

echo "Launching 5 experiments..."
echo "Tail any run: tail -f logs/<name>.log"
echo ""

# 1. Baseline — plain cross-entropy, no L2RW
{ time $PYTHON train.py $COMMON \
    --noise_type uniform --seed 1234 \
    --baseline \
    --wandb_run_name baseline_uniform_s1234 ; } \
  2>&1 | tee logs/baseline_uniform_s1234.log &
PID_BASE=$!

# 2. L2RW — uniform, seed 1234 (paper settings)
{ time $PYTHON train.py $COMMON \
    --noise_type uniform --seed 1234 \
    --wandb_run_name l2rw_uniform_s1234 ; } \
  2>&1 | tee logs/l2rw_uniform_s1234.log &
PID_L2RW_1234=$!

# 3. L2RW — uniform, seed 0
{ time $PYTHON train.py $COMMON \
    --noise_type uniform --seed 0 \
    --wandb_run_name l2rw_uniform_s0 ; } \
  2>&1 | tee logs/l2rw_uniform_s0.log &
PID_L2RW_0=$!

# 4. L2RW — uniform, seed 1
{ time $PYTHON train.py $COMMON \
    --noise_type uniform --seed 1 \
    --wandb_run_name l2rw_uniform_s1 ; } \
  2>&1 | tee logs/l2rw_uniform_s1.log &
PID_L2RW_1=$!

# 5. L2RW — asymmetric noise, seed 1234
{ time $PYTHON train.py $COMMON \
    --noise_type asymmetric --seed 1234 \
    --wandb_run_name l2rw_asymmetric_s1234 ; } \
  2>&1 | tee logs/l2rw_asymmetric_s1234.log &
PID_ASYM=$!

echo "PIDs: baseline=$PID_BASE  l2rw_s1234=$PID_L2RW_1234  l2rw_s0=$PID_L2RW_0  l2rw_s1=$PID_L2RW_1  asym=$PID_ASYM"
echo ""

# Wait for all and collect exit codes
wait $PID_BASE;        EC_BASE=$?
wait $PID_L2RW_1234;   EC_L2RW_1234=$?
wait $PID_L2RW_0;      EC_L2RW_0=$?
wait $PID_L2RW_1;      EC_L2RW_1=$?
wait $PID_ASYM;        EC_ASYM=$?

echo ""
echo "===== All runs finished ====="
echo ""
echo "Wall times (real) — look for 'real' in each log:"
echo ""
grep -h "^real" \
  logs/baseline_uniform_s1234.log \
  logs/l2rw_uniform_s1234.log \
  logs/l2rw_uniform_s0.log \
  logs/l2rw_uniform_s1.log \
  logs/l2rw_asymmetric_s1234.log \
  2>/dev/null | paste - - - - - | \
  awk 'BEGIN{OFS="\t"} {
    names[NR]=$0
    print NR, $0
  }' || true

echo ""
echo "baseline_uniform_s1234 wall time:"
grep "^real" logs/baseline_uniform_s1234.log 2>/dev/null || echo "  (check log)"
echo "l2rw_uniform_s1234 wall time:"
grep "^real" logs/l2rw_uniform_s1234.log 2>/dev/null || echo "  (check log)"
echo ""
echo "Exit codes: baseline=$EC_BASE  l2rw_s1234=$EC_L2RW_1234  l2rw_s0=$EC_L2RW_0  l2rw_s1=$EC_L2RW_1  asym=$EC_ASYM"

# Surface any non-zero exit codes clearly
FAILED=0
for name_ec in "baseline:$EC_BASE" "l2rw_s1234:$EC_L2RW_1234" "l2rw_s0:$EC_L2RW_0" "l2rw_s1:$EC_L2RW_1" "asym:$EC_ASYM"; do
  name=${name_ec%%:*}; ec=${name_ec##*:}
  if [[ $ec -ne 0 ]]; then
    echo "FAILED: $name exited with code $ec — check logs/${name}*.log"
    FAILED=1
  fi
done
exit $FAILED