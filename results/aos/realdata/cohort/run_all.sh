#!/bin/bash
# Sequence of cohort solves (resumable: finished labels are skipped, except the forced rerun of the primary).
cd /home/hristouam/pCloudDrive/research-lines/trimmed-mmd
export PYTHONPATH=src
python3 -I scripts/aos_realdata_cohort.py --run batch0 --seed 1000 --threads 4 --force
python3 -I scripts/aos_realdata_cohort.py --run pooled old --threads 4
python3 -I scripts/aos_realdata_cohort.py --run batch0 --seed 1001 1002 1003 1004 1005 1006 1007 --threads 4
echo ALL_DONE
