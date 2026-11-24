# Harder synthetic study: pre-registration

*Written 2026-11-07 and frozen at 23:43 UTC (hashes in out/prereg_sha256.txt), before any main-run replicate was drawn. Only
correctness checks (`checks.py`, out/checks.txt) and a timing pilot on seeds
disjoint from the main run (`timing.py`, rep indices ≥ 100000, only runtimes
inspected) were run before this file was frozen. Any change after the main runs
start is listed in §9 with its time and reason; designs added later are marked
**post hoc**.*

Code: `designs.py` (laws, exact TV), `methods.py` (all tests), `run.py`
(pilot, threshold cache, resumable runs), `analyze.py` (tables, figures).
Raw replicates: `out/<design>.jsonl` (one JSON line per replicate, appended
and fsynced as it lands).

## 1. Common construction and why TV is exact

Most designs use the contamination construction
$P=(1-a)R+aP_0$, $Q=(1-a)R+aQ_0$. Then $p-q=a(p_0-q_0)$ pointwise, so
$\mathrm{TV}(P,Q)=a\,\mathrm{TV}(P_0,Q_0)$ for every core $R$. If $P_0=A\otimes S$
and $Q_0=B\otimes S$ (the contaminations differ only in a block of coordinates
and share an independent factor $S$ on the rest), then
$\mathrm{TV}(P_0,Q_0)=\mathrm{TV}(A,B)$. Two Gaussians with a common covariance:
$\mathrm{TV}(N(\mu,\Sigma),N(-\mu,\Sigma))=2\Phi(\|\Sigma^{-1/2}\mu\|)-1$ (the
log-likelihood ratio is $2\mu'\Sigma^{-1}x$, normal with mean
$\pm2\|\Sigma^{-1/2}\mu\|^2$ and s.d. $2\|\Sigma^{-1/2}\mu\|$). Samplers draw the
number of contaminated points as Binomial$(n,a)$, i.e. exact i.i.d. draws from
the mixtures.

Every value below was also checked by Monte Carlo, $\mathbb E_{P_0}[(1-q_0/p_0)_+]$
with the exact density ratio of the differing block, $2\times10^6$ draws
(`python3 designs.py`): all 19 agree to within 3 MC standard errors
($\approx6\times10^{-5}$); D3b, at $z=2.96$ in that sweep, was re-checked with
three further $4\times10^6$ runs ($z=-0.6,-0.2,-0.9$) and by the orthant formula.

$\Phi^{-1}(0.9)=1.281552$, $\Phi^{-1}(0.7)=0.524401$.

| key | family | d | law | TV(P0,Q0) | a | **TV(P,Q)=α\*** |
|---|---|---|---|---|---|---|
| D1_d2_a10 | overlap | 2 | R=N(0,I); P0,Q0=N(±μ,I), μ=0.5244 e1 | 2Φ(0.5244)−1 = 0.4 | 0.25 | **0.1** |
| D1_d2_a20 | overlap | 2 | μ = 1.2816 e1 | 0.8 | 0.25 | **0.2** |
| D1_d10_a10 | overlap | 10 | μ = 0.5244 e1 | 0.4 | 0.25 | **0.1** |
| D1_d10_a20 | overlap | 10 | μ = 1.2816 e1 | 0.8 | 0.25 | **0.2** |
| D2_d20/50/100 | dense | 20/50/100 | μ = 1.2816·(1,…,1)/√d (per-coordinate shift 0.287/0.181/0.128) | 0.8 | 0.25 | **0.2** |
| D3a_loc | sparse | 50 | μ = 1.2816 (e1+e2)/√2 | 0.8 | 0.25 | **0.2** |
| D3b_corr | sparse | 50 | P0,Q0: (x1,x2) ~ N(0,[[1,±ρ],[±ρ,1]]), ρ=sin(0.4π)=0.95106, x3..x50 ~ N(0,1) | 2 arcsin(ρ)/π = 0.8 | 0.25 | **0.2** |
| D4_small | small | 2 | P0,Q0 = N(±2.5 e1, 0.25 I) | 2Φ(5)−1 = 0.99999943 | 0.05 | **0.0499999713** |
| D5b_mixed | imbalanced, mixed | 5 | see below; m = 5n | 0.736937 | 0.271394 | **0.2** |
| D5a_imb (tier 2) | imbalanced | 2 | D1_d2_a20 law, m = 5n | 0.8 | 0.25 | **0.2** |
| D6_t3 | heavy tails | 5 | R = t3^⊗5; P0,Q0: x1 ~ 0.5·t3 resp. 3·t3, x2..x5 ~ t3 | 0.623123 | 0.320964 | **0.2** |
| D7_wide | functional | 50 | Brownian motion on t_i=i/50; P0,Q0 = BM ± b, b = h·exp(−(t−½)²/(2w²)), w = 0.2, h = 0.6008 | 2Φ(‖b‖_CM)−1 = 0.8 | 0.25 | **0.2** |
| D7_narrow | functional | 50 | same, w = 0.05, h = 0.3075 | 0.8 | 0.25 | **0.2** |
| D8_N1/4/16/64 | oscillating | 1 | X ~ U[0,1]; Y density 1+s(Nx), s = ±1 square wave | — | — | **0.5** |

Derivations of the non-Gaussian rows:

* **D3b (sparse correlation flip).** Determinants agree, and
  $\log p_0/q_0=-\tfrac12x'(\Sigma_+^{-1}-\Sigma_-^{-1})x=2\rho x_1x_2/(1-\rho^2)$, so
  $\{p_0>q_0\}=\{x_1x_2>0\}$ and $\mathrm{TV}=P_+(x_1x_2>0)-P_-(x_1x_2>0)$. The
  orthant probability of a standard bivariate normal with correlation $r$ is
  $\tfrac14+\arcsin(r)/(2\pi)$, so $P_\pm(x_1x_2>0)=\tfrac12\pm\arcsin(\rho)/\pi$ and
  $\mathrm{TV}=2\arcsin(\rho)/\pi=0.8$ at $\rho=\sin(0.4\pi)$. Marginal means and
  variances of every coordinate are identical in $P$ and $Q$.
* **D5b (mixed types, ties).** Coordinates: $x_1=\mathrm{round}(c+1.5Z)$ (integer,
  "age"), $x_2$ binary, $x_3=0$ w.p. 0.85 else $e^{N(0,1)}$ ("capital gain": atom
  at 0), $x_4\in\{0,1,2,3\}$ with probabilities $(.4,.3,.2,.1)$ (ordinal code),
  $x_5\sim$Bern(0.4). $R$: $c=0$, $x_2\sim$Bern(.5); $P_0$: $c=1.5$,
  $x_2\sim$Bern(.75); $Q_0$: $c=-1.5$, $x_2\sim$Bern(.25); $x_3,x_4,x_5$ have one
  law under $R,P_0,Q_0$, independent of $(x_1,x_2)$. Hence
  $\mathrm{TV}(P_0,Q_0)=\tfrac12\sum_{k\in\mathbb Z,b\in\{0,1\}}|p_k\pi_b-q_k\pi'_b|$
  with $p_k=\Phi((k+\tfrac12-1.5)/1.5)-\Phi((k-\tfrac12-1.5)/1.5)$, $q_k$ likewise
  with $-1.5$; the sum over $|k|\le40$ is exact to $10^{-80}$: 0.736937. $a$ is
  set to $0.2/0.736937$. All methods see the data divided by the pooled pilot
  standard deviations (fixed in advance, `out/pilot.json`). $m=5n$.
* **D6 (t3 scale).** For $t_3$ densities with scales $s_P<s_Q$ the ratio is
  monotone in $x^2$, so the densities cross once at $\pm x^\*$ and
  $\mathrm{TV}=2(F_3(x^\*/s_P)-F_3(x^\*/s_Q))$; $x^\*=1.080040$ by root finding,
  TV$=0.623123$, agreeing with direct quadrature of $\tfrac12\int|p-q|$ to $10^{-9}$.
* **D7 (Brownian bump).** On the grid the BM covariance is $K=LL'$ with
  $L=\sqrt{\Delta t}\times$(lower-triangular ones), so
  $\|K^{-1/2}b\|^2=\sum_i(b_i-b_{i-1})^2/\Delta t$ ($b_0=0$), the discrete
  Cameron–Martin norm; $h$ is chosen to make it $\Phi^{-1}(0.9)$. TV is set by
  the Cameron–Martin (derivative) geometry while a Gaussian kernel on raw curves
  sees the $L^2$ size of the bump: $\|b\|_{L^2}=0.358$ (wide) and $0.092$
  (narrow), against a typical BM path norm of about 0.7.
* **D8 (paper's impossibility example).** $q_N=2\cdot\mathbf 1_{A_N}$ with
  $A_N=\{x:\mathrm{frac}(Nx)<\tfrac12\}$, $|A_N|=\tfrac12$, so
  $\mathrm{TV}=\tfrac12\int|s(Nx)|dx=\tfrac12$ for every $N$, while
  $\mu_{Q_N}-\mu_{P_u}\to0$ in the RKHS (Riemann–Lebesgue).

## 2. Hypotheses tested and grid

Every replicate is drawn at TV(P,Q) = α\*. On the same data we test
$H_{0,\alpha}$ at the grid $\alpha_j=j\,\alpha^\*/8$, $j=0,\dots,10$, and report

* **type-I at the boundary**: rejection rate of $H_{0,\alpha^\*}$ ($j=8$, a true
  null, TV = α exactly);
* **power**: rejection rates of $H_{0,0}$, $H_{0,\alpha^\*/2}$, $H_{0,0.75\alpha^\*}$
  ($j=0,4,6$; false nulls);
* **lower confidence bound** $\hat\alpha_{\rm lo}=\alpha_{j_{\rm lo}}$ with
  $j_{\rm lo}=\max\{j:H_{0,\alpha_i}\text{ rejected for all }i\le j\}$ (down-closed
  rule eq:downclosed; $\hat\alpha_{\rm lo}=0$ if $H_{0,0}$ is not rejected),
  reported as its mean, $P(\hat\alpha_{\rm lo}>\mathrm{TV})=P(j_{\rm lo}\ge9)$ (the
  paper's coverage statement) and $P(\hat\alpha_{\rm lo}\ge\mathrm{TV})=P(j_{\rm lo}\ge8)$
  (a false certificate "TV > α\*", also a true-null rejection).
* $\delta=0.05$ throughout.

## 3. Methods (all on the same data per replicate)

Folds: one random permutation of each sample per replicate; fold one = first
$\lfloor n/2\rfloor$, $\lfloor m/2\rfloor$ rows, fold two = the rest. All split
methods use these folds.

| id | method | data | bandwidth | guarantee |
|---|---|---|---|---|
| an_old | analytic test, paper τ_{n,m} (thm:concentration) | all n+m | fixed per design from a pilot | finite-sample, uniform |
| an_newcf | analytic, Theorem P closed form (NOTE §3.2), Gaussian constants κ=1, D_φ=ρ̄=√2, v²=1 | all | pilot | finite-sample, uniform |
| an_new | analytic, τ_new = min(τ_A, τ_P′, τ_P′^cpl) (NOTE Cor. C; numerically inverted, coupled constants) | all | pilot | finite-sample, uniform |
| prop17 | paper Prop. 17 (split + DKW), trimmed-mean statistic of the fold-one witness | folds | fold-one median | finite-sample |
| boot | paper Thm 16 percentile bootstrap (1000 resamples) | folds | fold-one median | asymptotic, per pair |
| ramp_b2s | sacert Cor. 1, empirical-Bernstein closed form on the [0,1] ramp, δ₁ = 0.8δ | folds | fold-one median | finite-sample |
| ramp_bet | sacert Thm 1, betting on the ramp, c = ½, WSR plug-in λ, prior weight 20, fold-one prior variance | folds | fold-one median | finite-sample |
| seq_bet | sacert §2.4 sequential batches e = (0, ¼, ½, 1): batch 1 bet with a witness fit on the first quarter, batch 2 (fold two) with the fold-one witness; one wealth | quarters | median of the batches already seen | finite-sample |
| gbm_hoef / gbm_bet | HistGradientBoostingClassifier (defaults) on fold one; TPR−FPR on fold two; Hoeffding / same betting test (neutral prior variance 0.5) | folds | — | finite-sample |
| logit_hoef / logit_bet | standardize + logistic regression (C = 1) | folds | — | finite-sample |
| knn_hoef / knn_bet | kNN, k = ⌊√(n₁+m₁)⌋ odd | folds | — | finite-sample |

Implementation notes fixed in advance:

* **Pilot bandwidth.** For each design one pilot draw (1000 + 1000 points;
  400 + 2000 for m = 5n), seed `SeedSequence([crc32(key), 424242])`, never
  reused; σ_fix = median pairwise distance of the pooled pilot
  (`trimmedmmd.kernels.median_heuristic`). Values in `out/pilot.json`
  (e.g. D1_d2: 1.72–1.79, D2_d100: 14.1, D8: 0.24–0.30). The analytic tests'
  level is therefore covered by the theorems (kernel fixed before the data).
* **Analytic statistic, certified.** The QP is solved by FISTA; every 10 steps
  the feasible iterate gives an upper bound $U\ge\widehat D_\alpha^2$ and its
  Frank–Wolfe gap a lower bound. A threshold is accepted when $U\le\tau^2$,
  rejected when $U-\text{gap}>\tau^2$; undecided after 4000 steps counts as
  *not rejected* (conservative) and is counted. Because $\widehat D_\alpha$ is
  nonincreasing and every τ nondecreasing in α, the rejection set is an initial
  segment of the grid, found by bisection. (Checked against
  `trimmedmmd.solver.solve_gram`: out/checks.txt.)
* **Split witness.** Trimmed-MMD witness at the tested level, fit on fold one
  by FISTA (≤ 1000 steps, warm-started along the grid); cut points are the
  fold-one self-excluded quantiles (sacert §2.3). For $\hat\alpha_{\rm lo}$ the
  levels are visited upward; a method stops at its first non-rejection, and
  levels 0, 4, 6, 8 are always computed.
* **Deviation from `seqcert.py`, on purpose.** `seq_ramp_bet` sets the horizon
  $N$ in $\lambda_k$ to the number of bets in batches with a valid cut, which
  depends on batch-1 data through batch 2's cut points; we fix $N$ in advance
  (total pairs in batches 1–2) so that $\lambda_k$ is predictable. When no batch
  is skipped the two agree (20/20 decisions in out/checks.txt).
* **Classifier threshold.** Label "X" iff $\hat p(X\mid z)>n_1/(n_1+m_1)$ (the
  Bayes rule for $\{p>q\}$ under the training mix; = `predict` when balanced).
  Any fold-one rule is valid; this one is fair to classifiers when m = 5n.

## 4. Sample sizes, replicates, seeds

* **Tier 1** (all designs except D5): n = m ∈ {100, 200, 400, 800}, exact Gram
  matrices. D5b: n ∈ {100, 200, 400}, m = 5n (exact Gram, up to 2400 points).
* **Tier 2**, run after tier 1 completes, in this order, as far as the time
  budget allows (rule: stop launching new cells when less than 60 min of the
  wall-time budget remains; a cell is reported only if all its 200 replicates
  finished): the designs D1_d2_a20, D4_small, D1_d10_a20, D8_N4, D6_t3, D3b_corr, D7_wide, D2_d20, D1_d2_a10 in that order at n = m = 1600 (exact Gram; about 35 s per replicate, i.e. about 7.5 min of wall time per design on 16 workers), then D5a_imb
  (n ∈ {100, 200, 400}, m = 5n).
* **Replicates: 200 per (design, n)** cell. Each replicate gives the boundary
  type-I and all power rows (same data), so every type-I row has 200 replicates
  (MC s.e. ≤ 0.0154 at a rate of 0.05; ≤ 0.0354 for any rate).
* Seeds: replicate (design, n, rep) draws everything from
  `SeedSequence([crc32(design), n, rep])` (data, permutation, bootstrap,
  classifier seed). Main runs use rep = 0..199.
* Threads: 16 single-threaded worker processes (OMP/BLAS = 1).

## 5. Metrics

Per (design, n, method): boundary type-I with exact Clopper–Pearson 95% upper
limit; power at j = 0, 4, 6 with MC s.e.; smallest n on the grid with power ≥
0.8 (">max n" if none); mean $\hat\alpha_{\rm lo}$, $P(\hat\alpha_{\rm lo}>\mathrm{TV})$,
$P(\hat\alpha_{\rm lo}\ge\mathrm{TV})$; mean runtime per replicate by component
(analytic incl. distances; split-kernel witness fits shared by
prop17/boot/ramp_b2s/ramp_bet; bootstrap and sequential extras; each
classifier). One figure per design (power vs n, three panels j = 0, 4, 6, plus
type-I) and one summary table.

**Red-flag rule (validity).** For a finite-sample method, a boundary rejection
count $k$ out of 200 with exact one-sided binomial p-value < 0.01 against δ =
0.05 (k ≥ 19) is a red flag and is investigated (bug vs theory). For the
bootstrap (asymptotic) the same rule marks a reportable level failure. The
same rule applies to $P(\hat\alpha_{\rm lo}\ge\mathrm{TV})$.

## 6. Pre-registered expectations

Written before any main-run result. "Power" means rejection of $H_{0,0}$ unless
stated.

1. **Validity.** No finite-sample method exceeds δ beyond MC error in any
   design. In overlapping designs (D1, D2, D3, D6, D7) the fold-one ramp is far
   from the optimal set, so $\mathbb E_Pg-\mathbb E_Qg<\mathrm{TV}$ strictly and boundary
   type-I is ≈ 0 for every method, the bootstrap included; the analytic tests
   have type-I ≈ 0 everywhere. Where the boundary is nearly attainable by a
   fold-one rule (D8_N1: Q is U[0,½], P∧Q = ½U[0,½]; D4: separated small
   contamination; D5b with atoms) the bootstrap may exceed δ.
2. **Analytic, old vs new.** an_new ≥ an_newcf ≥ an_old in power everywhere
   (they share $\widehat D_\alpha$). an_old has essentially no power at n ≤ 800
   anywhere except D8_N1. an_new has power ≥ 0.8 for $H_{0,0}$ by n = 800 in
   D1_d2_a20 and D8_N1 (MMD at the pilot bandwidth ≈ 0.19 and larger), probably
   D5b and D6; little or none in D1_*_a10, D1_d10, D2, D3, D4, D7_narrow,
   D8_N16/64. Against $H_{0,\alpha^\*/2}$ the analytic tests have little power
   except in D8_N1.
3. **Split kernel certificates.** In power: boot ≥ ramp_bet ≥ ramp_b2s ≫
   prop17 (prop17's DKW constant 2κβ makes it the weakest). seq_bet ≥ ramp_bet
   for n ≥ 400 and similar or worse at n = 100–200. The finite-sample price of
   ramp_bet relative to boot is the 1.75× sample-size factor of sacert Prop. 4
   or less. ramp_bet beats an_new for α > 0 in every design where either has
   power; at α = 0 they are closer.
4. **Kernel vs classifiers.**
   * D1 (low-d, linear Bayes boundary): logit_bet ≥ ramp_bet ≈ gbm_bet; the
     kernel methods competitive.
   * D2 (dense, d up to 100): logistic regression wins clearly (linear signal);
     kernel methods degrade with d at fixed TV; GBM weak.
   * D3a (sparse, axis-aligned): GBM and logistic beat the kernel (the kernel
     cannot tell D3a from D2_d50, which is a rotation of it).
   * D3b (correlation flip): GBM is the only method with real power; kernel and
     logistic ≈ 0.
   * D4 (small, separated): variance-adaptive tests (ramp_bet, seq_bet,
     *_bet) beat every Hoeffding/DKW/analytic test; $H_{0,\alpha^\*/2}$ is hard for
     all.
   * D5b (mixed, m = 5n): ramp_bet and GBM both have power; the bootstrap is the
     method most at risk of level problems (ties).
   * D6 (t3 scale): kernel tests competitive with GBM; logistic ≈ 0 (no mean
     shift).
   * D7_wide: kernel tests have some power; D7_narrow: kernels ≈ 0, logistic
     (linear in the increments) is the only method with a chance.
   * D8: everything works at N = 1; at N = 4 kernels weaken and GBM (interval
     splits) wins; at N = 16 kernels ≈ 0 and GBM partial; at N = 64 everything
     ≈ 0 at n ≤ 800.
5. **Lower bound.** $P(\hat\alpha_{\rm lo}\ge\mathrm{TV})\le\delta$ for every
   finite-sample method in every design. Mean $\hat\alpha_{\rm lo}$ is far below
   α\* for all methods at n ≤ 800 except D8_N1.
6. **Runtime.** Analytic tests are cheap given early stopping (QP decided in
   tens of iterations); the split kernel methods cost a fold-one QP per grid
   level; classifiers are cheapest except GBM in high d.

## 7. What would change the conclusions

A finite-sample method flagged by §5 in any design (would indicate a bug or a
theorem failure); the bootstrap flagged at the boundary (would support the
NOTE's caveat (iv)); kernel methods beating classifiers in D3 (would contradict
expectation 4).

## 8. Budget

Total wall time ≤ 4.5 h from 23:21 UTC (16 threads). Tier-1 estimate from
the timing pilot: about 1.6 h on 16 workers (per replicate: 1.5/3/6/14 s at
n = 100/200/400/800; D5b 24 s at n = 400, m = 2000). If tier 1 runs long, n = 800 cells
are completed design by design in catalogue order and an incomplete cell is
reported as such (no partial cells in the tables).

## 9. Changes after pre-registration

* 23:47 UTC (no results inspected): tier 2 (`tier2.sh`) runs with 15 instead of
  16 workers, to leave one thread for analysis inside the 16-thread cap. No
  effect on any result (all randomness is per replicate).
* `popmmd.py` (population MMD at the pilot bandwidth, fresh seeds) added as a
  descriptive diagnostic; it does not enter any test.
* **Post hoc, 00:09 UTC, after seeing the n <= 200 results** (kernel ramp
  bets beat the hard-label classifier bets even on linear designs):
  `posthoc_soft.py` adds `gbm_soft_bet`, `logit_soft_bet`, `knn_soft_bet` —
  the same betting test on the prior-corrected classifier probability
  g = p̃(X|z) ∈ [0,1] (an estimate of p/(p+q); = p̂ when n = m), on the same
  replicates (data, folds and classifier seeds regenerated; the regenerated
  hard-label decisions must match the stored ones, field `hard_match`). It is
  reported separately and marked post hoc; it does not change any
  pre-registered method or design. Run after tier 2 if time allows.
* **Post hoc, 00:27 UTC, after seeing the n <= 400 results** (boundary type-I
  ≈ 0 in every overlapping design, so the boundary rows are a weak validity
  check): four P = Q validity nulls `N0_D5b_mixed` (m = 5n, ties),
  `N0_D2_d100`, `N0_D6_t3`, `N0_D7_narrow` (X and Y both from the base
  design's P-law; TV = 0; H_{0,0} is tight for every [0,1]-valued g), same
  sizes and 200 replicates, run after tier 2 (`posthoc.sh`). `run.py`'s pilot
  now applies the D5b scaling to any key containing "D5b" (needed for
  N0_D5b_mixed; no effect on the pre-registered designs).
* **01:07 UTC, tier-2 reordering (budget).** The first tier-2 cell
  (D1_d2_a20, n = 1600) takes about 86 s per replicate, not the ~35 s of the
  timing pilot, because high power makes the α̂_lo grid walk visit many levels.
  That is about 25 min per cell, so only 4–5 of the 9 listed cells can finish
  before the 02:45 launch cutoff. To protect the post hoc validity nulls and
  the soft-classifier check, the chain was reordered (`chain3.sh`): finish
  D1_d2_a20, run the post hoc jobs, then resume tier 2 **in the
  pre-registered order** (D4_small, D1_d10_a20, D8_N4, …) under the same
  02:45 launch cutoff, with 15 workers. D5a_imb (last in tier 2) will not
  run. No design, method, seed or replicate count changed.
* **01:21 UTC, investigation (not a design change).** In the first tier-2
  cell, D1_d2_a20 at n = 1600, logit_bet rejected the boundary null in 13/200
  replicates (0.065, binomial p = 0.24, below the k ≥ 19 rule).
  `validity_logit.py` reruns the logistic certificate on 1200 replicates of
  that cell (0–199 must reproduce the stored decisions) and computes the
  exact population Δ of each fitted half-space rule.
* Development note on the soft-score variant: a first version bet on the raw
  p̂(X|z). It was tried on 6 replicates of D5b_mixed (m = 5n), where p̂ sits
  near the 1/6 training prior, and was replaced by the prior-corrected p̃
  before any full run. Those test records were deleted; all reported soft
  results use p̃. `posthoc.sh` was superseded by `chain3.sh` before it ran.
* **01:48 UTC, tier 2 stopped early.** The session was restarted and the
  coordinator asked for the results to be wrapped up on the shared machine.
  `chain3.sh` was stopped after the running cell (D4_small, n = 1600). Tier 2
  therefore has 2 completed cells (D1_d2_a20, D4_small), both in the
  pre-registered order; D1_d10_a20, D8_N4, D6_t3, D3b_corr, D7_wide, D2_d20,
  D1_d2_a10 and D5a_imb at their tier-2 sizes were not run.

* **Post hoc addition, 2026-11-10 (launched 10:13 UTC), after all results had been seen: α = 0
  extension to n = m = 12,800.** Reason: the author asked for the actual sample sizes at which power
  against $H_{0,0}$ reaches 0.8 ("find the actual numbers, with some reasonable upper bound for n"),
  where the grid of §4 ends below it (the ">800" cells of the n80 tables). Nothing in §1–§8 changed,
  and no stored replicate was touched. What was added:
  * `alpha0.py`: the tests of $H_{0,0}$ only, with no QP. At α = 0 the analytic statistic is the
    MMD between the empirical embeddings (the α = 0 point of the certified QP, whose feasible set is
    the single point $w=1/n$, $v=1/m$), at the pilot bandwidth of `out/pilot.json`, compared with
    $\tau_{n,m}(0,\delta)$ = `trimmedmmd.threshold.tau_new` with the Gaussian constants of §3
    (κ = 1, $D_\phi=\bar\rho=\sqrt2$, $v^2=1$); also `an_old`, `an_newcf` and the study's own
    minimum (`an_new_study`; `tau_new` also takes the closed form into the minimum, 1e-6 lower at
    n ≤ 3200, identical at 6400 and 12,800). The split witnesses at α = 0 are the mean-embedding
    differences of the fit block (fold one at the fold-one median bandwidth; batch 0 of the
    sequential certificate at the batch-0 median bandwidth), with the same cut points, ramp, priors,
    horizon and bets; the classifiers are trained as before (scikit-learn defaults; note that
    `HistGradientBoostingClassifier` switches on its default early stopping when the training fold
    has more than 10,000 points, i.e. at n = m = 12,800). All 14 methods' α = 0 decisions are
    recorded on every replicate. Kernel sums are blocked (512 × 512), so nothing of size
    $(n+m)^2$ is formed. Data, permutation, bandwidth subsamples, bootstrap stream and classifier
    seed come from `SeedSequence([crc32(design), n, rep])` in the order of `methods.run_replicate`
    (`methods.py` is unchanged, still at its pre-registered hash).
  * Reproduction check (`python3 alpha0.py check`, `out/alpha0_check.json`): recomputing 40
    stored replicates of every design at its largest size (and of four smaller cells) in α = 0 mode
    reproduced all 15,600 stored α = 0 decisions (14 methods + the study's minimum); statistics
    agree to 4e-12.
  * Ladder (`ladder_alpha0.py`; `run.py --alpha0` runs single cells): for every (design, method)
    cell whose power against $H_{0,0}$ at the largest pre-registered size is below 0.8, the sizes
    n = m ∈ {1600, 3200, 6400, 12800} above that size (for D5b_mixed, m = 5n, also n = 800 so that
    the ladder keeps doubling from its grid end 400), 200 replicates per size, stopping a cell's
    ladder once its power is ≥ 0.8 at two consecutive new sizes or at the cap 12,800. Driving
    methods: an_new, seq_bet, gbm_bet, logit_bet (Table 1 of the 50-page version) and an_old,
    ramp_bet, boot, knn_bet (the other columns of the long version's n80 table). A design is run at
    a size when one of its cells needs it. Replicates rep = 200000–200199 (no earlier run used rep
    ≥ 200000; the timing pilot of this extension used rep ≥ 300000, runtimes only). Raw records:
    `out/<design>_n0ext.jsonl`; log `out/run_n0ext.log`; ladder state `out/ladder_n0ext.json`;
    summary `analyze.py ext` → `out/summary_n0ext.csv`, `out/runtime_n0ext.csv`.
  * Budget: timing pilot (16 workers) 12–17 s per replicate at n = m = 12,800 (kernel sums
    ≈ 10 s, any d), so the full ladder fits well inside 6 h at 16 processes and all sizes keep 200
    replicates. Runs use 15 workers plus the driver (16 processes, shared machine).
  * Optional: the P = Q nulls N0_D2_d100, N0_D6_t3, N0_D7_narrow at n = 1600–12,800 and
    N0_D5b_mixed at n = 800–3200 (m = 5n), same mode and seeds scheme, for the α = 0 type-I.
  * Reporting (paper): the n80 tables now use every size run and interpolate the crossing of 0.8
    linearly in log n between the two bracketing sizes (two significant figures; ">12,800" when not
    reached at the cap). This replaces the "smallest grid size" rule of §5 for those tables only.

## 10. POST HOC addition (2026-11-17): the RHC-shaped design under this protocol (D9_rhc)

Decided after every result of this study had been seen, for the JMLR v3 revision (revision plan v1.1, decision D1),
so that the paper's RHC-shaped design of its Section 6.1 enters the same table and figure as the other designs.
Design `D9_rhc` (designs.py:RHCBox): core = smoothed resample of the pooled, standardized RHC covariates (33 dims,
Gaussian jitter 0.25), contaminations = the core with coordinates 1-4 replaced by Uniform boxes at -(3..4) (P) and
+(3..4) (Q) pooled SDs, a = 0.2, so TV(P, Q) = 0.2 exactly (mutually singular contaminations); this is
scripts/export_fig2_data.py:rhc_scenario(a=0.2, sep=3) with this study's seeds. Same methods, levels (j alpha*/8),
sizes (100-800), 200 replicates per size, pilot bandwidth and thresholds as every other design. Run locally (not on
the server) with scikit-learn 1.9.1, numpy 2.3.5, scipy 1.16.3 (~/.venvs/reocert); the system python lacks sklearn,
and a first launch that failed on the import wrote no replicate. Log: out/run_d9_rhc.log.
