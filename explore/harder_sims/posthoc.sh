#!/bin/bash
# POST HOC runs (DESIGN.md section 9), after tier 2 has finished.
cd /nvme0n1-disk/trimmed-mmd
while pgrep -f "harder_sims/tier2.sh|\./tier2.sh" >/dev/null; do sleep 20; done
echo "posthoc start $(date -u)"
PYTHONPATH=src python3 -I explore/harder_sims/run.py prep --workers 1 2>&1 | grep -v Warn | tail -1
PYTHONPATH=src python3 -I explore/harder_sims/run.py run --designs N0_D5b_mixed N0_D2_d100 N0_D6_t3 N0_D7_narrow --ns 100 200 400 800 --reps 200 --workers 16 2>&1 | grep -v Warn | tail -2
echo "N0 done $(date -u)"
PYTHONPATH=src python3 -I explore/harder_sims/posthoc_soft.py --workers 16 2>&1 | grep -v Warn | tail -2
echo "posthoc done $(date -u)"
