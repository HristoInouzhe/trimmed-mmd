#!/bin/bash
# Lane C: fairness (F1-F4), sequential certificate with per-level witnesses.  8 threads.
LANE=C; source "$(dirname "$0")/common.sh"
step F1adult python3 scripts/fair_case_study.py adult --stage rethreshold --threshold plugin --outdir results/v2
step F1acs python3 scripts/fair_case_study.py acs --stage rethreshold --threshold plugin --outdir results/v2
step F2a OMP_NUM_THREADS=4 python3 scripts/fair_case_study.py adult --stage audit --threshold plugin --outdir results/v2 --audit-alphas 0.05 0.10 0.175 0.25 ahat 0.35 0.45 0.50 0.55 0.60 0.70
step F2b OMP_NUM_THREADS=4 python3 scripts/fair_case_study.py acs --stage audit --threshold plugin --outdir results/v2 --audit-alphas 0.15 ahat 0.35 0.40 0.50
step F3a OMP_NUM_THREADS=1 python3 scripts/fair_split_certificate.py adult --certificate sequential --outdir results/v2 --splits 6 --workers 8 --alphas ahat 0.35 0.45 0.50 0.60
step F3b OMP_NUM_THREADS=1 python3 scripts/fair_split_certificate.py acs --certificate sequential --outdir results/v2 --splits 6 --workers 8 --alphas ahat 0.35 0.40 0.50
step F4 FAIR_THREADS=1 python3 scripts/fair_bandwidth_tune.py adult --workers 8 --threshold plugin --outdir results/v2
echo "$(date -u +%FT%TZ) END   C" >> $LOG
