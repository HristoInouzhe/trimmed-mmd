#!/bin/bash
# Reordered chain (DESIGN.md section 9): finish the running tier-2 cell, then the post hoc
# runs, then the remaining tier-2 cells in pre-registered order under the same launch cutoff.
cd /nvme0n1-disk/trimmed-mmd
while kill -0 956699 2>/dev/null; do sleep 15; done
echo "D1_d2_a20 n=1600 finished $(date -u)"
echo "posthoc start $(date -u)"
PYTHONPATH=src python3 -I explore/harder_sims/run.py prep --workers 1 2>&1 | grep -v Warn | tail -1
PYTHONPATH=src python3 -I explore/harder_sims/run.py run --designs N0_D5b_mixed N0_D2_d100 N0_D6_t3 N0_D7_narrow --ns 100 200 400 800 --reps 200 --workers 15 2>&1 | grep -v Warn | tail -1
echo "N0 done $(date -u)"
PYTHONPATH=src python3 -I explore/harder_sims/posthoc_soft.py --workers 15 2>&1 | grep -v Warn | tail -1
echo "soft done $(date -u)"
CUTOFF=$(date -u -d "2026-11-08 02:45:00" +%s)
for D in D4_small D1_d10_a20 D8_N4 D6_t3 D3b_corr D7_wide D2_d20 D1_d2_a10; do
  if [ "$(date -u +%s)" -ge "$CUTOFF" ]; then echo "cutoff reached before $D $(date -u)"; break; fi
  echo "start $D n=1600 $(date -u)"
  PYTHONPATH=src python3 -I explore/harder_sims/run.py run --designs $D --ns 1600 --reps 200 --workers 15 2>&1 | grep -v Warn | tail -1
done
echo "chain3 done $(date -u)"
