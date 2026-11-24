#!/bin/bash
# Tier 2 (pre-registered order). Waits for tier 1, then runs one cell at a time;
# no new cell is launched after the cutoff (budget end 03:51 UTC minus 60 min, rounded down).
cd /nvme0n1-disk/trimmed-mmd
T1=$(cat explore/harder_sims/out/tier1.pid)
while kill -0 "$T1" 2>/dev/null; do sleep 20; done
CUTOFF=$(date -u -d "2026-11-08 02:45:00" +%s)
for D in D1_d2_a20 D4_small D1_d10_a20 D8_N4 D6_t3 D3b_corr D7_wide D2_d20 D1_d2_a10; do
  if [ "$(date -u +%s)" -ge "$CUTOFF" ]; then echo "cutoff reached before $D $(date -u)"; break; fi
  echo "start $D n=1600 $(date -u)"
  PYTHONPATH=src python3 -I explore/harder_sims/run.py run --designs $D --ns 1600 --reps 200 --workers 15 2>&1 | grep -v Warn | tail -2
done
if [ "$(date -u +%s)" -lt "$CUTOFF" ]; then
  echo "start D5a_imb $(date -u)"
  PYTHONPATH=src python3 -I explore/harder_sims/run.py run --designs D5a_imb --ns 100 200 400 --reps 200 --workers 15 2>&1 | grep -v Warn | tail -2
fi
echo "tier2 done $(date -u)"
