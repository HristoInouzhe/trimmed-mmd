#!/usr/bin/env bash
# AoS restructure: Figure 2(a) of the main text (Laplace ramp, analytic with tau = min{tau^P, tau^A}, classifiers on the
# batches of the ramp) and the S6 versions (Gaussian ramp; Figure 2(b) with the new analytic threshold), drawn by
# scripts/v3_fig_recovery.py from the data written by scripts/aos_main_tables.py. Print size as aos/tools/figs_aos.sh
# (imsart text width 5.62 in, smallest type 8 pt). Writes results/aos/main/figs/. To place them in the paper, copy
# results/aos/main/figs/fig_recovery_main.pdf to aos/figs_aos/ (done at the end).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
OUT=results/aos/main/figs
mkdir -p "$OUT"
COMMON="TMMD_FIG_NOTATION=aos2 TMMD_FIG_ONLY=1 TMMD_FIG_WIDTH=5.62 TMMD_FIG_MINPT=8 TMMD_FIG_LEFT=0.355 TMMD_FIG_RATIO=1.15 TMMD_FIG_ROWH=0.215"
# main text: panel (a) alone, Laplace ramp
env $COMMON TMMD_FIG_LEFT=0.40 TMMD_FIGA_MAIN=results/aos/main/fig_recovery_main.json TMMD_FIG_PANELS=a \
  TMMD_FIG_OUT=$OUT/fig_recovery_main.pdf $PY -I scripts/v3_fig_recovery.py
# S6: Gaussian ramp (stored) in panel (a), panel (b) with the analytic estimator under the new threshold
env $COMMON TMMD_FIGA_MAIN=results/aos/main/fig_recovery_gauss.json TMMD_FIG2_DATA=results/aos/main/fig2_data_aos.json \
  TMMD_FIG2_SMALL=results/aos/main/fig2_small_aos.json TMMD_FIG_OUT=$OUT/fig_recovery_gauss_s6.pdf \
  $PY -I scripts/v3_fig_recovery.py
# the main-text recovery figure (Figure 3 of the AoS draft) goes next to the paper's other figures
cp "$OUT/fig_recovery_main.pdf" aos/figs_aos/.fig_recovery_main.pdf.tmp
mv aos/figs_aos/.fig_recovery_main.pdf.tmp aos/figs_aos/fig_recovery_main.pdf
cmp "$OUT/fig_recovery_main.pdf" aos/figs_aos/fig_recovery_main.pdf
echo "copied $OUT/fig_recovery_main.pdf to aos/figs_aos/"
