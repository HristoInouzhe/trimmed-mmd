#!/bin/bash
# Lane A: real data (R1-R4), then the Figure 2 RHC exports (G1-G3).  8 threads.
LANE=A; source "$(dirname "$0")/common.sh"
step R1 OMP_NUM_THREADS=8 python3 -m trimmedmmd.experiments.realdata_run --reps 8 --workers 8 --threshold plugin --certificate all --outdir results/v2 --reuse-paths results
step R2 python3 scripts/realdata_extras.py --threshold plugin --certificate all --outdir results/v2 --only r6,r7,r8
step R3 OMP_NUM_THREADS=4 python3 -I scripts/exp_rhc_balancing.py --threads 4 --alpha certified --threshold plugin --outdir results/v2
step R4 python3 -I scripts/exp_valid_bandwidth.py --workers 2 --threads 4 --threshold plugin --outdir results/v2
RHC_LEVEL=$(python3 -c "import pandas as pd; d=pd.read_csv('results/v2/tables/r2_lower_bounds.csv').set_index('dataset'); print(d.loc['rhc','median_lower'])")
echo "$(date -u +%FT%TZ) INFO  A RHC certified median = $RHC_LEVEL" >> $LOG
step G1 OMP_NUM_THREADS=4 python3 scripts/export_fig1_data.py --threshold plugin --certificate all --alpha-read certified --outdir results/v2
step G2 OMP_NUM_THREADS=4 python3 scripts/export_fig1_regimes.py --threshold plugin --certificate all --alpha-read $RHC_LEVEL --outdir results/v2
step G3 OMP_NUM_THREADS=2 python3 scripts/export_fig1_tradeoff.py --threshold plugin --outdir results/v2
echo "$(date -u +%FT%TZ) END   A" >> $LOG
