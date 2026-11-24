#!/usr/bin/env bash
# Redraw the figures whose JMLR size would print below 8 pt at the imsart text width (406 pt = 5.62 in), through
# generator options whose defaults leave the JMLR figures unchanged (checked pixel for pixel on 2026-11-20).
# Writes aos/figs_aos/. Run from anywhere; the generator reads the raw replicates, which takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")/../.."
TMMD_FIGA_MATCHED=results/aos/baselines/summary.csv TMMD_FIG2_DATA=results/v3/figures/fig2_data_merged.json TMMD_FIG_OUT=aos/figs_aos/fig_recovery_aos.pdf TMMD_FIG_ONLY=1 TMMD_FIG_WIDTH=5.62 TMMD_FIG_MINPT=8 TMMD_FIG_LEFT=0.355 TMMD_FIG_RATIO=1.15 TMMD_FIG_ROWH=0.215 \
  python3 -I scripts/v3_fig_recovery.py
