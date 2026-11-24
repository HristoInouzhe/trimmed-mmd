# trimmedmmd

Code and data for *Certifying total-variation contamination by betting on trimmed kernel witnesses* by Hristo Inouzhe,
submitted to the Annals of Statistics. The repository holds the Python package `trimmedmmd`, the paper's sources, and
the scripts and records behind every table and figure. Each release is archived on Zenodo under
[10.5281/zenodo.24507577](https://doi.org/10.5281/zenodo.24507577). The Lean 4 formalization of the paper's numbered
results is at [github.com/HristoInouzhe/trimmed-mmd-lean](https://github.com/HristoInouzhe/trimmed-mmd-lean).

An α-trimming of a law discards a fraction α of its mass and renormalizes what is left. Two laws P and Q have a common
α-trimming exactly when α is at least their total variation distance α* = TV(P, Q), the fraction of each population
with no counterpart in the other. From two samples, the package computes a lower confidence bound on α* that holds at
every sample size. Optionally, it also returns the trimmed samples at that level.

## Contents

- `src/trimmedmmd/`: the package. `tests/` holds its test suite.
- `aos/`: the paper. The main text is `aos/main.tex` with its sections in `aos/sections/`, and the supplement is
  `aos/supplement.tex` with its sections in `aos/supplement/`. The built PDFs are `aos/main.pdf` and
  `aos/supplement.pdf`. `bash aos/build.sh` rebuilds them from the generated tables in `aos/tables/`.
- `scripts/`: one script for each table and figure. Section S6 of the supplement says which script makes which;
  `scripts/fetch_data.py` downloads the data.
- `results/`: the records and tables the scripts write and read; the paper's runs are in `results/aos/`.
- `explore/harder_sims/`: the simulation study whose per-replicate records (`out/`) the paper's tables read.
  `explore/split_adaptive/` and `explore/sharper_threshold/` hold the code it imports.

## Install

The dependencies are numpy, scipy, matplotlib and pandas. The quadratic program is solved by a self-contained
accelerated projected-gradient method, so no external solver is needed. The transport geometry in `trimmings` uses
[POT](https://pythonot.github.io/) when it is installed (`pip install -e '.[ot]'`). Otherwise it solves the same problem
as a HiGHS linear program; the two agree to machine precision, and the tests check this.

```bash
pip install -e .                          # or run from the checkout with PYTHONPATH=src
python3 -m unittest discover -s tests     # the test suite
```

## From a trimming level to a trimmed sample

Most of the package answers *how much* contamination there is; `trimmings` answers *which observations*, and returns
the trimmed samples.

```python
from trimmedmmd import bisect_alpha_hat, optimal_trimming, trim_at

# 1. evidence on the level
alpha_hat = bisect_alpha_hat(lambda a: d_hat_at(a), n=n, m=m, delta=0.05)

# 2. turn that number into data
t = trim_at(X, Y, alpha_hat)            # kernel geometry
t = trim_at(X, Y, alpha_hat, geometry="wasserstein")   # transport geometry

t.kept_x, t.dropped("x")                # indices retained / discarded
Xt, wt = t.apply(X, "x")                # the trimmed sample, with weights
t.hard("x")                             # a plain subsample, no weights
t.resample("x", size=1000, rng=rng)     # an i.i.d. draw from the trimmed law
t.summary()                             # value, ESS, budget used, diagnostics
```

Below the contact level the trimming is a hard subsample (Proposition 2.5). Every observation is kept at full weight
`1/(n(1-α))` or dropped outright, with at most one fractional exception, and the rule is a threshold on the witness
function. `t.fractional_count("x")` reports how many weights are interior, so you can check this on your own data. At
or above the contact level the optimal trimming is not unique. Since `α̂` is at the crossing by construction, this is
the usual case for `trim_at`, and then `t.unique` is `False`. The default `select="least"` returns the optimal trimming
closest to the untrimmed weights, the one that trims least.

## Unequal trimming fractions

Trimming the law P of X by `α_x` and the law Q of Y by `α_y` can leave a common remainder exactly when
`∫ min{p/(1-α_x), q/(1-α_y)} ≥ 1`. With `ζ = (1-α_x)/(1-α_y)`, this is `E_ζ(P‖Q) = sup_A {P(A) - ζ Q(A)} ≤ α_x`, the
hockey-stick divergence. Equal fractions give `ζ = 1` and `E_1 = TV(P, Q) = α*`. Section 8 of the paper discusses this
extension.

The certificate works as follows:
- it fits the witness with caps `1/(n(1-α_x))` and `1/(m(1-α_y))`;
- it ramps the witness between the `α_y`-quantile on Y and the `(1-α_x)`-quantile on X;
- it bets on `ψ(X) - ζ ψ(Y) - α_x`, with fractions below `c/(ζ + α_x)`.

The practitioner's question is the frontier `α_y*(α_x)`: if the treated arm loses a fraction `α_x`, how much of the
control arm must go before the two covariate distributions can coincide? `frontier_lower_bound` reads a grid of `α_y`
upward on one random order and stops at the first level it cannot certify. The null only gets easier as `α_y` grows, so
the grid needs no correction.

```python
import numpy as np
from trimmedmmd import frontier_lower_bound, hockey_stick_lower_bound, pair_certificate

# X: covariates of the treated, Y: of the controls (one row per unit); here a synthetic example
rng = np.random.default_rng(0)
X = rng.normal(0.5, 1.0, size=(1000, 1))
Y = rng.normal(-0.5, 1.0, size=(1000, 1))

# keep 90% of the treated arm: how much of the controls must go?
bound, rows = frontier_lower_bound(X, Y, alpha_x=0.10, alphas_y=np.arange(0.0, 0.96, 0.05),
                                   split_seed=1000)
# with probability >= 0.95: whichever 10% of the treated is dropped, the controls must
# lose more than `bound` before the two laws can coincide (None: nothing certified)

pc = pair_certificate(X, Y, alpha_x=0.10, alpha_y=0.40, split_seed=1000)   # one pair
pc.certified, pc.zeta, pc.log_max_wealth, pc.cuts

lo, rows = hockey_stick_lower_bound(X, Y, zeta=2.0, alphas_x=np.arange(0.0, 0.96, 0.05),
                                    split_seed=1000)
# with probability >= 0.95, E_2(P‖Q) > lo
```

On this example (`X ~ N(0.5, 1)` and `Y ~ N(-0.5, 1)`, 1000 draws each from `np.random.default_rng(0)`, X drawn first),
the frontier bound at `α_x = 0.10` is 0.50, against a true frontier of 0.718. The bound on `E_2` is 0.05, against the
true 0.191.

The defaults follow the paper's method:
- the Laplace kernel, with the median bandwidth computed on batch 0;
- batches `(0, 1/4, 1/2, 1)`;
- cut points at genuine empirical quantiles.

`standardize_batch0=True` standardizes the data by batch 0, as in the RHC study. With `alpha_y = alpha_x`, the
certificate is the paper's, number for number.

Validity holds for every pair `(α_x, α_y)` by Ville's inequality, as in Theorem 4.2, provided the bandwidth and any
standardization depend on the fitting data only. The defaults and `standardize_batch0=True` satisfy this. A `sigma` you
pass must be fixed without looking at the data, and X and Y must not be standardized beforehand with full-sample
moments. The paper's rates, and so its power guarantees, are proved for equal fractions only.

## Reproducing the paper

```bash
python3 scripts/fetch_data.py      # the RHC study and the NHANES accelerometry files, into the git-ignored data_raw/
bash aos/build.sh                  # main.pdf and supplement.pdf from the generated tables
```

The scripts that make each table and figure are listed in Section S6 of the supplement. They write to
`results/aos/`, and `aos/tools/sync.sh` (run by `aos/build.sh`) copies their tables into `aos/tables/`. The simulated
data are drawn from fixed seeds, and the runs also read the stored records in `explore/harder_sims/out/`.

## License

MIT (see `LICENSE`).
