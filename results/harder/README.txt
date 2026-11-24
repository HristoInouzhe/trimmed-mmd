Harder synthetic study: summary data used by the JMLR paper (Section 9.2 "Harder Designs" and Appendix E.7-E.10)
===================================================================================================================

Origin. The study was run in explore/harder_sims/ (not tracked by git): 18 designs with exact total variation
(D1-D8), 14 tests of H_{0,alpha} on the same data, n = m = 100-800 (1600 for D1_d2_a20 and D4_small; D5b: n = 100-400,
m = 5n), 200 replicates per design and size, delta = 0.05, plus four post hoc P = Q null designs (N0_*), a post hoc
soft-score classifier check and a 1200-replicate rerun of one cell. The protocol (DESIGN.md, copied here) was
pre-registered and frozen at 23:43 UTC on 2026-11-07, before any main-run replicate was drawn; its section 9 lists
every later change. Main runs: 2026-11-07/08; summaries written 2026-11-08 01:48-01:50 UTC by the study's analyze.py.

Copied on 2026-11-08 (repository HEAD 9b754c8) by

    python3 scripts/harder_collect.py

which copies the files below from explore/harder_sims/out/ and condenses two raw sources (no Monte Carlo work). The
paper reads only this folder: jmlr/tools/make_harder_tables.py writes jmlr/tables/harder_*.tex (macros and tables) and
scripts/fig_harder.py draws jmlr/figs_print/fig_harder*.pdf.

Files
  summary_long.csv     one row per design x n x method: replicates R, alpha_star (exact TV), power at alpha = 0,
                       alpha*/2, 0.75 alpha* (pow0, pow4, pow6), boundary rejections at alpha = alpha* (typeI, typeI_k,
                       typeI_cp = two-sided 95% Clopper-Pearson upper limit, typeI_p), mean lower bound alo_mean,
                       P(alo > TV) and P(alo >= TV). For the N0_* designs (TV = 0) typeI is the rejection rate of H_{0,0}.
  paired.csv, wtl.csv  paired comparisons on the same replicates (tier-1 grid, n <= 800, pre-registered designs);
                       a cell is informative if either method rejects in >= 10%, a win/loss is a paired difference
                       beyond two paired standard errors.
  n0_pooled.txt        P = Q nulls pooled over the 15 cells (3000 replicates per method).
  validity_logit.json  rerun of logistic betting at D1_d2_a20, n = 1600, with 1200 replicates, and the exact
                       population E_P g - E_Q g of every fitted half-space rule.
  popmmd.csv           population MMD at the analytic test's pilot bandwidth (Monte Carlo, fresh seeds), with
                       tau_{n,m}(0) and r_n + r_m at the largest size.
  thresholds.csv       r_n + r_m, the closed form of Theorem thm:plugin and tau_{n,m} at n = m on the levels used
                       (from out/thr_cache.json).
  soft_summary.csv     post hoc soft-score classifier bets: rejection counts and rates at grid levels j = 0, 4, 6, 8
                       per design x n x method, and how many regenerated replicates reproduced the stored hard-label
                       decisions (from out/posthoc_soft/*.jsonl; main replicates 0-199).
  runtime.csv          mean time per replicate (s, one core) by component, per design x n.
  pilot.json           pilot bandwidths and scalings; checks.txt: the study's unit checks against the package.
  DESIGN.md, prereg_sha256.txt   the protocol and the checksums recorded at freezing.

Post hoc alpha = 0 extension (DESIGN.md section 9, 2026-11-10; explore/harder_sims/alpha0.py, ladder_alpha0.py), added
because the author asked for the sample sizes at which power against H_{0,0} reaches 0.8 beyond the grid. Only the
tests of H_{0,0} (no QP), every (design, method) cell with power < 0.8 at its largest grid size continued at
n = m = 1600, 3200, 6400, 12800 (D5b_mixed: n = 800-12800, m = 5n) until power >= 0.8 at two consecutive sizes or
the cap 12800; 200 new replicates per size (rep 200000-200199).
  summary_n0ext.csv    one row per design x n x method (all 14 methods at alpha = 0, plus an_new_study, the study's
                       own minimum of thresholds): replicates R, rejections k0, rate pow0, cp0 (two-sided 95%
                       Clopper-Pearson upper limit, as typeI_cp in summary_long) and p0 (binomial p vs 0.05).
                       For the N0_* designs pow0 is the type-I rate of H_{0,0}.
  runtime_n0ext.csv    mean time per replicate (s, one core) by component, mean D_hat_0, tau_{n,m}(0) and the number
                       of replicates where tau_new and the study's minimum disagree.
  ladder_n0ext.json    final ladder state of each design x driving method (why it stopped).
  alpha0_check.json    reproduction check: the alpha = 0 mode recomputed stored replicates and matched every stored
                       alpha = 0 decision.

Note on the checksums. prereg_sha256.txt records the files as frozen. methods.py still matches (the alpha = 0
extension lives in alpha0.py; run.py gained an --alpha0 flag that leaves the default path unchanged). DESIGN.md, designs.py
and run.py differ from the recorded hashes because of the changes logged in DESIGN.md section 9 (the deviation log
appended to DESIGN.md; the post hoc P = Q designs added to designs.py; the pilot scaling for N0_D5b_mixed in run.py).
The frozen versions themselves were not kept.

Conventions in the paper differ from the study's REPORT.md in two places: rates are rounded half up on their decimal
value (so 0.985 prints as 0.99 and 0.105 as 0.11), and Clopper-Pearson limits are one-sided 95% upper limits, as
elsewhere in the paper.
