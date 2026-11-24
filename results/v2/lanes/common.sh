# Shared setup for the v2 rerun lanes (results/v2/RERUN_PLAN.md). Source from a lane script.
set -uo pipefail
cd /nvme0n1-disk/trimmed-mmd
export PYTHONPATH=src
mkdir -p results/v2/checkpoints results/v2/logs
cp -n results/checkpoints/{e1_calibration,e1_power,e2_oracle,e2_paths,e6_calibrations,e6_oraclenull}__* results/v2/checkpoints/ 2>/dev/null || true
export TRIMMEDMMD_CHECKPOINTS=$PWD/results/v2/checkpoints
LOG=results/v2/logs/lanes.log
step() {  # step NAME cmd...  -> logs to results/v2/logs/NAME.log; stops the lane on failure
  local name=$1; shift
  local t0=$(date +%s)
  echo "$(date -u +%FT%TZ) START $LANE $name" >> $LOG
  if env "$@" > results/v2/logs/$name.log 2>&1; then
    echo "$(date -u +%FT%TZ) DONE  $LANE $name $(( $(date +%s) - t0 ))s" >> $LOG
  else
    echo "$(date -u +%FT%TZ) FAIL  $LANE $name $(( $(date +%s) - t0 ))s (see results/v2/logs/$name.log)" >> $LOG
    echo "$(date -u +%FT%TZ) STOP  $LANE" >> $LOG
    exit 1
  fi
}
