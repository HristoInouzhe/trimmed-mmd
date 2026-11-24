#!/usr/bin/env bash
# Copy the generated tables, number macros and print-size figures of jmlr/v3 (which itself copies them from the
# generators' outputs in ../tables, ../figs_print and ../results, and derives compact copies with jmlr/v3/tools/derive.py).
# Never edit the copies here by hand; tools/derive.py makes AoS-specific derived copies with an _aos suffix.
# When results change: run ../jmlr/v3/build.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
# The public copy of the repository has no jmlr/: it keeps the copies already in tables/ and figs_print/.
if [ -d ../jmlr/v3 ]; then
  cp ../jmlr/v3/tables/*.tex tables/
  cp ../jmlr/v3/figs_print/*.pdf figs_print/
else
  echo "sync: no ../jmlr/v3 here; using the copies in tables/ and figs_print/"
fi
# AoS reruns (2026-11-20/21): cos/sin random features in Figure 2(b), batch-0 standardization; \renewcommand lines
# for the macros whose values change (scripts/aos_rerun_macros.py), loaded last in preamble.tex.
cp ../results/v3/tables/aos_rerun_macros.tex tables/
# AoS margin extension (2026-11-21): numbers of the numerical check in S4 (scripts/aos_margin_macros.py).
cp ../results/aos/tables/aos_margin_macros.tex tables/
# AoS real-data reruns (2026-11-21): converged RHC cohort, averaged e-values (scripts/aos_realdata_tables.py).
cp ../results/aos/tables/aos_realdata_macros.tex tables/
# AoS baselines (2026-11-21): Table 1 with the permutation references, macros and supplement tables
# (scripts/aos_baselines_tables.py).
cp ../results/aos/tables/aos_baselines_*.tex tables/
cp ../results/aos/tables/aos_extra_tests.tex ../results/aos/tables/aos_extra_tests_macros.tex tables/
# AoS excess of the population ramp (2026-11-21): the table and macros of Section S4 "A simulation on the line"
# (scripts/aos_excess_table.py).
cp ../results/aos/tables/aos_excess_mc.tex ../results/aos/tables/aos_excess_macros.tex tables/
# AoS cut-point rules and the Laplace kernel (2026-11-21): Section S6 "Cut points that cross, and the Laplace kernel"
# (scripts/aos_fallback_tables.py).
cp ../results/aos/tables/aos_fallback_macros.tex ../results/aos/tables/aos_kernel_macros.tex ../results/aos/tables/aos_kernel_realdata.tex ../results/aos/tables/aos_kernel_recovery.tex tables/
# AoS main runs (2026-11-22): the Laplace ramp with genuine quantile cuts, the analytic baseline under tau^P, the
# held-out replicates and the S6 tables (scripts/aos_main_tables.py, scripts/aos_main_s6.py; macros with prefix \mn);
# the recovery figure of Section 7 (scripts/aos_main_figs.sh).
cp ../results/aos/tables/aos_main_*.tex ../results/aos/tables/aos_s6_*.tex tables/
cp ../results/aos/main/figs/fig_recovery_main.pdf figs_aos/
# AoS referee runs (2026-11-22): RHC with all 45 covariates, averaged e-values, classifier labels and the
# propensity-score link on RHC, NHANES with whole sampling units to batches (scripts/aos_ref_realdata_tables.py;
# prefix \rr), and recovery against n with oracle references (scripts/aos_ref_recovery_tables.py,
# scripts/aos_ref_recovery_fig.py; prefix \rc).
cp ../results/aos/tables/aos_ref_*.tex tables/
# Cold read of 2026-11-24: the numbers of Figure 2 (scripts/aos_fig_excess.py; aos_main_extra_macros.tex comes with
# aos_main_*.tex above).
cp ../results/aos/tables/aos_fig_excess_macros.tex tables/
# Supplement cold read of 2026-11-24: the boundary calibration with tau^P (scripts/aos_supp_threshold.py; prefix \st).
cp ../results/aos/tables/aos_supp_threshold_macros.tex tables/
cp ../results/aos/main/figs/fig_recovery_curves.pdf figs_aos/
