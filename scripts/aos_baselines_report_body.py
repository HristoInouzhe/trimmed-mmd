"""Body of results/aos/baselines/REPORT.md (read by scripts/aos_baselines_report.py). Numbers are {{macro}}
placeholders filled from results/aos/tables/aos_baselines_macros.tex; nothing numeric is typed here except the
fixed protocol constants that are also macros."""

BODY = r"""
## 1. What was asked, and what was reused or newly run

| referee request | what was run | reused or new |
|---|---|---|
| classifiers on the ramp's batch schedule | GBM and logistic regression refitted on the ramp's batches (fit on quarter 1, bet on quarter 2; refit on half 1, bet on half 2; one wealth, horizon fixed in advance): scikit-learn defaults (`*_m_*`) and tuned by 3-fold CV on each fit block (`*_mt_*`) | new |
| bet on probabilities, not only labels | each classifier bets either its label (`*_hard`) or its prior-corrected probability in [0, 1] (`*_soft`) | new (the study's post hoc soft bets used the single split only and no D9) |
| permutation MMD test as the H_{0,0} power reference | V-statistic MMD², Gaussian kernel at the analytic test's pilot bandwidth (`perm_pilot`; also at the pooled median, `perm_med`), {{abPermB}} relabellings | new |
| alpha = 0 witness with betting | `ramp0`: witness and cut points both fitted at alpha = 0, the same ramp bet at every level (the untrimmed ramp); `ramp0c`: alpha = 0 witness with the cut points of the tested level | new |
| nu-SVM with the same kernel as a betting score | `svm`: libsvm nu-SVM normal on the ramp's Gram matrices, nu by 3-fold CV, ramp with the cut points of the tested level, same schedule and priors as the ramp; `svm05` (nu = 0.5) and `svm_hard` (its labels) as variants | new |
| (extension a) MMDAgg | MMDAgg* of Schrab et al. (2023): {{abAggL}} Gaussian bandwidths from the inter-sample distances, uniform weights, wild bootstrap (n = m) or permutations (m = 5n), B1 = B2 = {{abAggB}}, B3 = 50; implemented here (no package installed) | new |
| (extension b) univariate KS comparison | KS statistic minus the two DKW-Massart radii on the 1-d designs: the stored oscillating designs (n = 100-800) and two new 1-d designs run with the study's own code: `U1_shift` (overlap shift, {p > q} a half-line, so KS = TV) and `U1_t3scale` (t3 scale, {p > q} an interval, so KS = TV/2) | new |
| (extension c) fixed-n empirical Bernstein for the ramp | the study's stored `ramp_b2s` (single split, ramp at the tested level fitted on half 1, two-sample Bernstein with Maurer-Pontil variance bounds, levels read upward) against single-split betting (`ramp_bet`) and the paper's sequential ramp (`seq_bet`) | reused (stored replicates) |
| (extension d) HPLB | Michel, Naef and Meinshausen (arXiv:2005.06006): lambda_bayes (their Prop. 3) and lambda_adapt (Prop. 6-7, bounding function (17), simulated hypergeometric band, S = {{abHplbS}}) on the out-of-sample scores of the paper's single-split GBM and logistic regression | new |

The paper's own columns (analytic test `an_new`, betting ramp `seq_bet`, GBM / logistic + betting `gbm_bet` /
`logit_bet`) are read from the stored replicates of explore/harder_sims, paired with the new tests on the same
replicates (same data, folds and fold bandwidths).

## 2. Protocol

* Designs, sizes, seeds and levels of the study: {{abNdesigns}} designs with exact TV at n = m = 800 (mixed types
  n = 400, m = 5n; Figure 2(a)'s size), {{abNnulls}} P = Q nulls at the same sizes, {{abReps}} replicates per cell
  (rep = 0-199, the stored main-run replicates), delta = 0.05, levels alpha_j = j alpha*/8, j = 0..10, lower bound =
  down-closed largest rejected level, recovered fraction = that level / alpha* (0 if none), as in Figure 2(a).
* Each replicate regenerates the stored data exactly (SeedSequence([crc32(design), n, rep]), same permutation into
  folds and batches); the study's fold bandwidths are taken from the stored replicate. New randomness (relabellings,
  MMDAgg bootstrap, HPLB tie-breaking) uses separate seed words, so it does not disturb the stored streams.
* Reproduction checks on every replicate: bandwidths {{abReproSigma}} identical; the untrimmed ramp at alpha = 0
  equals the stored ramp decision at alpha = 0 in {{abReproSeqZero}} replicates (at H_{0,0} the trimmed and
  untrimmed witnesses coincide); the recomputed single-split logistic decisions equal the stored ones in
  {{abReproHardLogit}}; for GBM in {{abReproHardGbm}} (the stored study ran on a server with another scikit-learn
  version, which changes a few boosting predictions; the paper's GBM columns are therefore always the stored ones and
  the new GBM variants use scikit-learn 1.9.1).
* Environment: Python 3.12, numpy 2.5.4 (OpenBLAS), scikit-learn 1.9.1, scipy 1.18.1 (~/.venvs/llm_book), 6 worker
  processes with one BLAS thread each.
* Calibration (calib.log; one replicate per cell, 6 concurrent workers): {{calMainMean}} s per main replicate on
  average over {{calMainN}} cells (at most {{calMainMax}} s; the first calibration, with the reference-BLAS numpy of
  ~/.venvs/reocert and an earlier, costlier tuning grid, took {{calRefMean}} s on average, calib_n800.log), {{calUnivSet}} s per 1-d replicate across the four sizes.
  That allowed the paper's 200 replicates in every cell within the budget, so no cell uses fewer.
* Replicates written (results/aos/baselines/raw): {{cntMain}} main (all new tests at n = 800, the study's
  replicates 0-199 of 19 designs and 4 nulls), {{cntH00}} more for the permutation test and MMDAgg at
  n = 100-400 (same replicates) and {{cntH00x}} at n = 1600 (curves, narrow bump; the replicates of the study's
  alpha = 0 extension), {{cntUniv}} 1-d replicates of the two new designs (all study methods + KS, four sizes),
  {{cntKs}} KS-only replicates of the stored oscillating designs below n = 800: {{cntAll}} in all, {{cpuHours}}
  CPU-hours, {{wallHours}} h of wall clock from the first launch to the last replicate (including two restarts,
  NOTES.md).

## 3. Validity

* **Boundary null H_{0,alpha*} (TV = alpha* exactly), every new betting or bound test, every design:**
  {{abBoundaryCells}} cells, red flags (exact one-sided binomial p < 0.01 against delta = 0.05): {{abBoundaryFlags}}.
  The largest rejection rate is {{abBoundaryMax}} ({{abBoundaryMaxK}}/200, one-sided 95% Clopper-Pearson upper limit
  {{abBoundaryMaxCP}}): {{abBoundaryMaxMethod}} on the {{abBoundaryMaxDesign}} design, whose mutually singular
  contaminations make the boundary nearly attainable by a half-space. Table: aos_baselines_validity.tex.
* **P = Q nulls (H_{0,0} true and tight for every [0,1]-valued score):** {{abNullCells}} cells (permutation MMD at both
  bandwidths, MMDAgg and every new betting or bound test, four null designs), red flags {{abNullFlags}}. Largest rate
  {{abNullMax}} ({{abNullMaxK}}/200, binomial p = {{abNullMaxP}}): {{abNullMaxMethod}} on the {{abNullMaxDesign}}
  null; the permutation test's largest rate is {{abNullPermMax}}, MMDAgg's {{abNullAggMax}}.
* **Implementation probes** (scripts/aos_baselines_probes.py, probes.json; fresh seeds): permutation p-values under
  P = Q, rejection rate {{abProbePermRate}} over {{abProbePermR}} draws (KS test of uniformity p = {{abProbePermKsP}});
  HPLB lambda_adapt, rate of lambda_hat > 0 under P = Q {{abProbeHplbNull}} and of lambda_hat > TV at a tight
  boundary {{abProbeHplbBoundary}} ({{abProbeHplbR}} draws); MMDAgg on the mixed-type null law at n = 100, m = 500,
  {{abProbeAggR}} draws: {{abProbeAggInter}} with its inter-sample (label-dependent) bandwidths and
  {{abProbeAggPooled}} with pooled-distance bandwidths (s.e. {{abProbeAggSe}}), so the mixed-type null rate above is
  within the Monte Carlo slack of B2 = 500 permutations.
* **KS bound:** at the boundary of the 1-d designs it rejected in {{abUnivKsBoundaryMaxK}} of 200 replicates in its
  worst cell (it is conservative by construction).
* HPLB lambda_bayes is only asymptotically valid (normal approximation); it is reported because the authors of HPLB
  present it as their simplest bound, and it holds its level here. lambda_adapt uses a simulated band, valid up to the
  simulation error of S = {{abHplbS}} draws.

## 4. Findings

### 4.1 Power against H_{0,0} (Table 1)

* At n = m = 800 the permutation MMD test has power {{abPowPermLo}}-{{abPowPermHi}} over the {{abNdesigns}} designs
  (median {{abPowPermMed}}), MMDAgg {{abPowAggLo}}-{{abPowAggHi}} (median {{abPowAggMed}}), the betting ramp
  {{abPowRampLo}}-{{abPowRampHi}} (median {{abPowRampMed}}), the analytic test {{abPowAnalyticLo}}-{{abPowAnalyticHi}}
  (median {{abPowAnalyticMed}}).
* Paired on the same replicates (win/tie/loss for the ramp at two paired s.e.): against the permutation test
  {{abWtlPowPerm}}, against MMDAgg {{abWtlPowAgg}}, against the analytic test {{abWtlPowAnalytic}}. The permutation test
  has larger power than the ramp in {{abPermAboveRampPow}} designs and smaller in {{abRampAbovePermPow}}; MMDAgg larger
  in {{abAggAboveRampPow}}, smaller in {{abRampAboveAggPow}}.
* In the sample-size form of Table 1 (aos_baselines_n80.tex; the paper's rule; the four paper columns reproduce all
  {{abNeightyReproduced}} cells of aos/tables/harder_n80_v2.tex), the permutation test needs the fewest observations,
  alone or tied, in {{abNeightyPermBest}} designs, MMDAgg in {{abNeightyAggBest}}, the ramp in {{abNeightyRampBest}};
  where both are interpolated ({{abNeightyRampOverPermN}} designs) the ramp needs {{abNeightyRampOverPermLo}} to
  {{abNeightyRampOverPermHi}} times the permutation test's n (median {{abNeightyRampOverPermMed}}). The permutation
  test reaches 0.8 at the smallest size (n = 100) in {{abNeightyPermLeHundred}} designs and not by the largest size run
  in {{abNeightyPermNotReached}}.
* Reading: as a test of P = Q, the split-sample ramp pays for splitting (it bets on about 3n/4 observations with a
  witness fitted on fewer) and the permutation test and MMDAgg are the stronger power references. They certify nothing
  about the size of TV(P, Q), so they do not compete with the lower bound; but the sentence "the ramp needs the fewest
  observations" holds only among the paper's four certificates.

### 4.2 Classifiers on the ramp's schedule, labels and probabilities (Figure 2(a) made like-for-like)

* Mean certified level / alpha* at n = 800 (aos_baselines_recovery.tex), median over designs: ramp {{abRecMedRamp}};
  GBM paper {{abRecMedGbmPaper}}, matched labels {{abRecMedGbmMatchedLabels}}, matched probabilities
  {{abRecMedGbmMatchedProb}}, tuned labels {{abRecMedGbmTunedLabels}}, tuned probabilities {{abRecMedGbmTunedProb}};
  logistic paper {{abRecMedLogitPaper}}, matched labels {{abRecMedLogitMatchedLabels}}, matched probabilities
  {{abRecMedLogitMatchedProb}}, tuned labels {{abRecMedLogitTunedLabels}}, tuned probabilities
  {{abRecMedLogitTunedProb}}.
* The paper's count is reproduced: the ramp recovers more than the better of the paper's two classifiers in
  {{abRampAbovePaperClf}} of {{abNdesigns}} designs. Against the better of the four matched default variants (GBM or
  logistic, labels or probabilities) it is {{abRampAboveBestMatched}} of {{abNdesigns}}; against the best of all eight
  matched variants (defaults and tuned) {{abRampAboveBestMatchedTuned}} of {{abNdesigns}} (a per-design maximum over
  eight variants, so biased toward the classifiers).
* Per variant, ramp win/tie/loss at two paired s.e.: logistic matched labels {{abWtlRecLogitMatchedLabels}}, logistic
  tuned labels {{abWtlRecLogitTunedLabels}}, GBM matched labels {{abWtlRecGbmMatchedLabels}}, GBM tuned labels
  {{abWtlRecGbmTunedLabels}}; probabilities: GBM matched {{abWtlRecGbmMatchedProb}}, logistic matched
  {{abWtlRecLogitMatchedProb}}, GBM tuned {{abWtlRecGbmTunedProb}}, logistic tuned {{abWtlRecLogitTunedProb}}.
  The single strongest classifier variant, logistic regression tuned by CV and betting its labels on the ramp's
  schedule, recovers more than the ramp (in mean) in {{abNAboveRampLogitTunedLabels}} designs; tuned GBM labels in
  {{abNAboveRampGbmTunedLabels}}.
* Where: tuned logistic labels gain most on the overlap, dense-shift and mixed-type designs and on both curve designs
  (a half-space is close to the optimal set {p > q}); tuned GBM labels on the t3 tails, the sparse shift and the
  oscillating designs. The ramp recovers more than every matched variant only on: {{abDesignsRampAboveBestMatchedTuned}};
  more than every matched default variant on: {{abDesignsRampAboveBestMatched}}.
* Betting probabilities instead of labels helps power against H_{0,0} in some designs (Table 1) but loses at
  alpha > 0 almost everywhere: a calibrated probability is pulled toward 1/2 on the shared core, so
  E_P g - E_Q g falls well below TV and the bet cannot clear alpha. This holds for defaults and for tuned models (whose
  better calibration makes it worse).

### 4.3 What trimming adds (alpha = 0 ramp)

* At H_{0,0} trimming plays no role: the untrimmed ramp reproduces the ramp's decision at alpha = 0 on every replicate
  ({{abReproSeqZero}}).
* The untrimmed ramp (witness and cut points fitted at alpha = 0, used at every level) certifies a median
  {{abRecMedRampZero}} of alpha* against the ramp's {{abRecMedRamp}}; ramp win/tie/loss {{abWtlRecRampZero}} in
  recovery and {{abWtlHalfRampZero}} in power against H_{0,alpha*/2} (aos_baselines_half.tex). So trimming matters,
  through the level-alpha cut points: clipping the witness at the alpha-quantiles of the fit block concentrates the bet
  on what the trimming leaves.
* Refitting the witness itself at each level adds little on these designs: the alpha = 0 witness with the cut points of
  the tested level certifies a median {{abRecMedRampZeroCut}} (ramp win/tie/loss {{abWtlRecRampZeroCut}} in recovery,
  {{abWtlHalfRampZeroCut}} at H_{0,alpha*/2}). It is slightly better on the dense and sparse shifts and the mixed
  types, equal on the overlap, RHC-shaped and small-contamination designs, and clearly worse on the t3 tails, both
  curve designs and the nu = 1 oscillation, where the trimmed witness moves away from the mean-embedding direction.

### 4.4 nu-SVM with the same kernel

* The nu-SVM normal (libsvm on the ramp's Gram matrices; nu by CV, median chosen nu {{abSvmNuMed}}, 10-90% range
  {{abSvmNuLoQ}}-{{abSvmNuHiQ}}; for n = m it is the trimmed witness at level 1 - nu) as a ramp with level-alpha cuts:
  median recovery {{abRecMedSvm}}; ramp win/tie/loss {{abWtlRecSvm}} in recovery and {{abWtlHalfSvm}} at
  H_{0,alpha*/2}. Its labels: median {{abRecMedSvmLabels}} (ramp {{abWtlRecSvmLabels}}). nu = 0.5 (alpha = 0.5
  trimming): median {{abRecMedSvmHalf}}. With m = 5n, libsvm's single cap forces nu <= 1/3, and the nu-SVM collapses
  on the mixed-type design.
* Against H_{0,0} it has power {{abPowSvmLo}}-{{abPowSvmHi}} (median {{abPowSvmMed}}). One cross-validated level is
  worse than fitting at the tested level: the classification-optimal nu (about 0.8, i.e. trimming at about 0.2 when
  n = m) trims more than the levels below alpha* at which the bound is decided.

### 4.5 HPLB (Michel, Naef and Meinshausen)

* On the paper's single-split scores: lambda_adapt median recovery {{abRecMedHplbAdaptGbm}} (GBM) and
  {{abRecMedHplbAdaptLogit}} (logistic); ramp win/tie/loss {{abWtlRecHplbAdaptGbm}} and {{abWtlRecHplbAdaptLogit}}.
  lambda_adapt with GBM is the best or tied-best method on the oscillating designs with nu = 4 and 16 (it searches all
  cutoffs of a score that splits [0,1] into intervals).
* lambda_bayes (asymptotic): median {{abRecMedHplbBayesGbm}} (GBM), {{abRecMedHplbBayesLogit}} (logistic); ramp
  {{abWtlRecHplbBayesGbm}} and {{abWtlRecHplbBayesLogit}}. The logistic lambda_bayes is a one-shot normal bound on the
  same half-space as the logistic labels, and is the strongest classifier bound on the overlap designs, but it has no
  finite-sample guarantee.

### 4.6 Fixed-n empirical Bernstein vs betting (aos_baselines_eb.tex; stored replicates)

* Same single split and ramp: empirical Bernstein (two-sample Bernstein with Maurer-Pontil variance bounds) median
  recovery {{abEbRecMed}}, single-split betting {{abSingleBetRecMed}}; betting recovers more in
  {{abSingleBetAboveEb}} designs and less in {{abEbAboveSingleBet}}; the paper's sequential ramp recovers more than
  the fixed-n bound in {{abSeqAboveEb}}. Power against H_{0,0}: median {{abEbPowMed}} (EB) vs {{abSingleBetPowMed}}
  (betting).

### 4.7 Univariate designs: KS lower bound (aos_baselines_univ.tex)

* KS statistic minus the two DKW-Massart radii, certified level / alpha* at n = 800: new shift design (KS = TV)
  {{abUnivShiftKs}} vs ramp {{abUnivShiftRamp}} vs analytic {{abUnivShiftAnalytic}}; new t3-scale design (KS = TV/2)
  {{abUnivScaleKs}} vs {{abUnivScaleRamp}} vs {{abUnivScaleAnalytic}}; oscillating nu = 1 {{abUnivOscOneKs}} vs
  {{abUnivOscOneRamp}} vs {{abUnivOscOneAnalytic}}; nu = 4 {{abUnivOscFourKs}} vs {{abUnivOscFourRamp}} vs
  {{abUnivOscFourAnalytic}}.
* Over all {{abUnivCells}} design-size cells, the KS bound recovers more than both kernel bounds in
  {{abUnivKsAboveKernel}} and less than the better of them in {{abUnivKernelAboveKs}}.

## 5. Against the paper's current claims

* "at the trimmed levels it certifies more of alpha* than classifiers calibrated in the same way in \recRampAboveClf
  of them" (Introduction; Section 7: "The ramp recovers more than the better of the two classifiers in
  \recRampAboveClf of the 19 designs"). Reproduced as {{abRampAbovePaperClf}} with the paper's classifiers, but those
  are not calibrated in the same way: on the ramp's own batch schedule the count falls to {{abRampAboveBestMatched}}
  (defaults, labels or probabilities) and to {{abRampAboveBestMatchedTuned}} if tuned variants are allowed; tuned
  logistic labels alone recover more than the ramp in {{abNAboveRampLogitTunedLabels}} designs. **This claim does not
  survive the fair comparison** and should be restated (e.g. the ramp is the best kernel bound and is competitive with
  matched classifiers, ahead where the signal is smooth at the kernel's scale, behind where a half-space or a tree is
  close to the optimal set).
* "logistic regression is close to the ramp on the Gaussian and mixed-type designs, where a linear rule is nearly
  optimal" (Section 7): on the matched schedule logistic regression is ahead of the ramp there, not just close.
* "In our simulations the ramp needs the fewest observations to detect a difference in \allRampBest of the 19 designs"
  (Introduction) and Table 1: true among the paper's four columns (reproduced), but the permutation MMD test, the
  natural power reference, needs fewer observations in every design where both sizes are interpolated (ramp / perm.
  ratio {{abNeightyRampOverPermLo}} to {{abNeightyRampOverPermHi}}) and is never less powerful at n = 800 (ramp
  win/tie/loss {{abWtlPowPerm}}). The sentence should say "among the certificates of TV > alpha", and Table 1 can
  carry the permutation test as the reference (aos_baselines_n80.tex).
* "fitting the witness at the level tested matters: a ramp of the untrimmed witness need not be consistent"
  (Introduction; Section 4): the theory (a constructed law) is untouched, but in the 19 designs the alpha = 0 witness
  with level-alpha cuts certifies as much as the trimmed witness overall ({{abWtlRecRampZeroCut}}); what the
  simulations show matters is the level-alpha clipping (untrimmed ramp {{abWtlRecRampZero}}). Worth one sentence of
  qualification.
* "No method claimed more contamination than there is" (Section 7): holds for every new test ({{abBoundaryFlags}}
  boundary flags in {{abBoundaryCells}} cells, {{abNullFlags}} null flags in {{abNullCells}}).
* Unaffected: the analytic test's role (it never certifies more than the split-sample bound; ramp vs analytic power
  {{abWtlPowAnalytic}}), the betting-vs-EB statements, and the classifiers' wins on the correlation flip and the
  oscillating designs (now with HPLB as well).

## 6. Files

* results/aos/tables/aos_baselines_table1.tex: power against H_{0,0} at n = 800, one row per design (analytic,
  permutation MMD, MMDAgg, ramp, nu-SVM; GBM and logistic: paper, matched labels, matched probabilities, matched +
  tuned labels, HPLB-adapt).
* results/aos/tables/aos_baselines_n80.tex: Table 1 itself (sample size for power 0.8) with the permutation test and
  MMDAgg added; its four paper columns are recomputed from the stored replicates and equal the paper's.
* results/aos/tables/aos_baselines_recovery.tex: mean certified level / alpha* (Figure 2(a)), with the alpha = 0 ramp,
  the alpha = 0 witness with level cuts, the nu-SVM and the same classifier columns.
* results/aos/tables/aos_baselines_half.tex: power against H_{0,alpha*/2} (where trimming can matter).
* results/aos/tables/aos_baselines_eb.tex: fixed-n empirical Bernstein vs betting.
* results/aos/tables/aos_baselines_univ.tex: 1-d designs, KS bound vs analytic vs split-sample, by n.
* results/aos/tables/aos_baselines_validity.tex: boundary type-I and P = Q nulls of every new test.
* results/aos/tables/aos_baselines_macros.tex: all summary numbers (prefix \ab).
* results/aos/baselines/summary.csv (every design x method: power at j = 0, 4, 6, boundary rate, CP limit, mean
  recovery), paired.csv (ramp minus each method, paired s.e., win/tie/loss), univariate.csv, probes.json,
  raw/*.jsonl (one line per replicate; raw/discarded_v1 holds records of a stopped first launch, not used),
  logs/*.log (run logs).
"""
