#!/bin/bash
# Lane B: synthetic study (S1-S4), then the regime certificate replay (G4, G5) and the small-size run.  8 threads.
LANE=B; source "$(dirname "$0")/common.sh"
step S1 OMP_NUM_THREADS=1 TRIMMEDMMD_E7_REPS=200 python3 -m trimmedmmd.cli run --profile modest --only E1,E1b,E2,E6,E7,E8 --threshold plugin --certificate all --outdir results/v2 --workers 8
step S2 OMP_NUM_THREADS=1 python3 -I scripts/exp_classifier_baseline.py --workers 8 --certificate all --outdir results/v2
step S3 python3 scripts/exp_onesided_null.py --threshold plugin --outdir results/v2 --from-csv results/tables/onesided_null.csv
step S4 OMP_NUM_THREADS=1 python3 -m trimmedmmd.experiments.kernel_selection --reps 24 --workers 8 --threshold plugin --outdir results/v2
step G4 OMP_NUM_THREADS=1 python3 scripts/export_fig2_data.py --from-rows results/figures/fig2_data.json --threshold plugin --certificate all --outdir results/v2 --workers 8
step G5 OMP_NUM_THREADS=1 python3 scripts/export_fig2_data.py --from-rows results/figures/fig2_null.json --threshold plugin --certificate all --outdir results/v2 --workers 8
step G4small OMP_NUM_THREADS=1 python3 scripts/export_fig2_data.py --sizes 100 200 300 400 --threshold plugin --certificate all --outdir results/v2 --out fig2_small.json --workers 8
echo "$(date -u +%FT%TZ) END   B" >> $LOG
