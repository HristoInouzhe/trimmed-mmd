"""Exact/T0-T1 falsification checks from CLAIMS.md, block ids n1, n2, n3,
n4, n5, n6, n7. These are cheap (seconds), and a violation is a conclusive
finding -- either a bug or a wrong theorem, not a calibration nuance. See
CLAIMS.md for the falsification criterion each test encodes and why it was
chosen.

n8 (nestedness frequency) is a T2 block (experiments/run.py's e12); it
needs many replicates to report a meaningful frequency, not a standalone
identity.
"""

from __future__ import annotations

import os
import sys
import unittest

# Pin BLAS threads before numpy is imported: this box is shared (other
# tenants routinely run at load average > 20), and an unpinned dense matmul
# will otherwise spawn one thread per core and oversubscribe it. A single
# T0/T1 identity check has no business claiming the whole machine.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
from scipy.optimize import linprog, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trimmedmmd.kernels import gaussian_kernel  # noqa: E402
from trimmedmmd.solver import solve_gram  # noqa: E402
from trimmedmmd.splitting import (evaluate_direction, trimmed_mean,  # noqa: E402
                                  witness_direction)


# --------------------------------------------------------------------------
# shared exact primitives -- these test the PACKAGE's own closed form
# (lem:trimmedmean, splitting.trimmed_mean), not a parallel reimplementation, so
# a pass here is evidence about the actual codebase.
# --------------------------------------------------------------------------

def top_k_mean(v, alpha):
    return trimmed_mean(v, alpha, upper=True)


def bottom_k_mean(v, alpha):
    return trimmed_mean(v, alpha, upper=False)


# --------------------------------------------------------------------------
# n1: lem:cvar (Rockafellar-Uryasev duality) + lem:tcompact (localization)
# --------------------------------------------------------------------------

class TestCVaRDuality(unittest.TestCase):
    """CLAIMS.md n1: h_C(f) = sup_g E[gf] = inf_t{t + beta E[(f-t)+]}."""

    def test_dual_matches_closed_form_primal(self):
        rng = np.random.default_rng(0)
        for n, alpha in [(7, 0.0), (30, 0.2), (200, 0.4), (500, 0.75)]:
            v = rng.standard_normal(n) * rng.uniform(0.5, 3.0)
            primal = top_k_mean(v, alpha)
            beta = 1.0 / (1.0 - alpha)

            def dual_obj(t, v=v, beta=beta):
                return t + beta * np.mean(np.clip(v - t, 0.0, None))

            res = minimize_scalar(dual_obj, bounds=(v.min() - 1.0, v.max() + 1.0),
                                  method="bounded", options={"xatol": 1e-13})
            self.assertLess(abs(res.fun - primal), 1e-6,
                            f"n={n} alpha={alpha}: primal={primal} dual={res.fun}")

    def test_localizes_to_kappa_ball_for_rkhs_witness(self):
        # Build f as an actual unit-norm RKHS witness: f(x) = <phi(x), xi>
        # for xi a random combination of feature vectors, normalized so
        # ||xi||_H = 1. With the paper's normalization k(x,x)=1, kappa=1,
        # so lem:tcompact predicts the CVaR-dual optimal t* lands in [-1, 1].
        rng = np.random.default_rng(1)
        n, d = 60, 3
        X = rng.standard_normal((n, d))
        K = gaussian_kernel(X, X, sigma=1.1)
        c = rng.standard_normal(n)
        norm = float(np.sqrt(c @ K @ c))
        c /= norm  # ||xi||_H = 1
        f_vals = K @ c  # f(X_i) = <phi(X_i), xi>
        kappa = 1.0  # gaussian_kernel(x,x)=1 by construction

        for alpha in (0.0, 0.15, 0.5, 0.9):
            beta = 1.0 / (1.0 - alpha)

            def dual_obj(t, f=f_vals, beta=beta):
                return t + beta * np.mean(np.clip(f - t, 0.0, None))

            res = minimize_scalar(dual_obj, bounds=(-5 * kappa, 5 * kappa),
                                  method="bounded", options={"xatol": 1e-12})
            if alpha > 0:
                # Strictly decreasing left tail, strictly increasing right
                # tail (lem:tcompact's proof) => a genuinely interior minimizer,
                # so the argmin itself must lie in [-kappa, kappa].
                self.assertGreaterEqual(res.x, -kappa - 1e-6,
                                        f"alpha={alpha}: t*={res.x} escapes [-kappa,kappa]")
                self.assertLessEqual(res.x, kappa + 1e-6,
                                     f"alpha={alpha}: t*={res.x} escapes [-kappa,kappa]")
            else:
                # alpha=0: lem:tcompact's own proof says the left tail is
                # *constant* (slope -alpha/(1-alpha)=0), so the argmin need
                # not be unique or in-range -- only that t=-kappa attains
                # the SAME (flat, minimal) value as the unconstrained search.
                self.assertLess(abs(dual_obj(-kappa) - res.fun), 1e-6,
                                f"alpha=0: value at t=-kappa ({dual_obj(-kappa)}) "
                                f"does not match the unconstrained optimum "
                                f"({res.fun}) -- lem:tcompact claims -kappa attains it")
            # And the dual value must still match the exact primal (lem:cvar)
            # regardless of which point on a possible plateau was returned.
            primal = top_k_mean(f_vals, alpha)
            self.assertLess(abs(res.fun - primal), 1e-6)


# --------------------------------------------------------------------------
# n4: lem:trimmedmean (extreme trimmed means), cross-checked against an
# independent generic LP solver, not just the dual used in n1
# --------------------------------------------------------------------------

class TestExtremeTrimmedMean(unittest.TestCase):
    """CLAIMS.md n4: closed-form order-statistic formula vs brute-force LP."""

    def test_matches_independent_lp_solver(self):
        rng = np.random.default_rng(2)
        for n in (4, 5, 11, 37, 100):
            for alpha in (0.0, 0.1, 0.3, 0.6, 0.9):
                if n * (1 - alpha) < 1:
                    continue  # infeasible cap, not this lemma's regime
                v = rng.standard_normal(n)
                cap = 1.0 / (n * (1 - alpha))
                res = linprog(c=-v, A_eq=[np.ones(n)], b_eq=[1.0],
                             bounds=[(0, cap)] * n, method="highs")
                self.assertTrue(res.success, f"n={n} alpha={alpha}")
                lp_val = -res.fun
                closed = top_k_mean(v, alpha)
                self.assertLess(abs(lp_val - closed), 1e-7,
                                f"n={n} alpha={alpha}: lp={lp_val} closed={closed}")

    def test_top_and_bottom_bracket_every_feasible_point(self):
        # Sanity invariant used elsewhere in the codebase (splitting.py):
        # for ANY feasible w, bottom <= sum w_i v_i <= top.
        rng = np.random.default_rng(3)
        n, alpha = 25, 0.3
        v = rng.standard_normal(n)
        top, bot = top_k_mean(v, alpha), bottom_k_mean(v, alpha)
        self.assertLessEqual(bot, top + 1e-12)
        cap = 1.0 / (n * (1 - alpha))
        for _ in range(200):
            w = rng.dirichlet(np.ones(n))
            w = np.clip(w, 0, cap)
            w = w / w.sum()
            if w.max() > cap + 1e-9:
                continue  # renormalization pushed a coordinate over cap; skip
            val = w @ v
            self.assertLessEqual(val, top + 1e-8)
            self.assertGreaterEqual(val, bot - 1e-8)


# --------------------------------------------------------------------------
# n5: lem:qpstability -- constant 4 is both a valid bound AND sharp
# --------------------------------------------------------------------------

class TestConsistencySchedule(unittest.TestCase):
    """CLAIMS.md n12 (cor:consistency), exact/T1 half: the threshold formula
    itself, under the paper's own admissible vs. flagged-inadmissible
    delta-schedules. No sampling needed -- tau(alpha*, delta_n) is a closed
    form, so this checks the deterministic claim "tau does/doesn't shrink"
    directly. The empirical consequence for alpha-hat is the T2 half
    (experiments/run.py's e14).
    """

    def test_admissible_schedule_shrinks_tau(self):
        from trimmedmmd.threshold import tau as tau_fn
        alpha_star = 0.3
        ns = [100, 1000, 10_000, 100_000]
        taus = [tau_fn(n, n, alpha_star, 1.0 / n) for n in ns]
        for a, b in zip(taus, taus[1:]):
            self.assertLess(b, a, "tau should strictly shrink under delta_n=1/n")
        self.assertLess(taus[-1], 0.1)

    def test_inadmissible_schedule_does_not_shrink_tau(self):
        from trimmedmmd.threshold import tau as tau_fn
        alpha_star = 0.3
        # exp(-n) underflows to exactly 0 in float64 for n >~ 745, which
        # would make log(2/delta) spuriously infinite -- stay well inside
        # the representable range, since the point here is the asymptotic
        # TREND (tau approaching a strictly positive constant), already
        # visible at these n.
        ns = [10, 100, 300, 700]
        taus = [tau_fn(n, n, alpha_star, max(np.exp(-n), 1e-300)) for n in ns]
        # log(2/delta) = log(2) + n, so sqrt(2*(log2+n)/n) -> sqrt(2), a
        # STRICTLY POSITIVE constant -- tau must NOT shrink toward 0.
        self.assertGreater(taus[-1], 0.5,
                           "tau under delta_n=exp(-n) should stay bounded "
                           "away from zero (paper's own stated counterexample), "
                           f"not shrink -- got {taus[-1]}")
        # And it should be converging to a stable plateau, not still falling
        # at the same rate it would under a shrinking schedule.
        self.assertLess(abs(taus[-1] - taus[-2]) / taus[-2], 0.1)


class TestQPStability(unittest.TestCase):
    """CLAIMS.md n5: |val(K) - val(K~)| <= 4 eps under ||K - K~||_max <= eps."""

    def _problem(self, seed, n=25, m=20, d=2):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d)) + 0.5
        return (gaussian_kernel(X, X, 1.0), gaussian_kernel(Y, Y, 1.0),
                gaussian_kernel(X, Y, 1.0))

    def test_random_perturbations_respect_the_bound(self):
        rng = np.random.default_rng(4)
        for seed in range(6):
            Kxx, Kyy, Kxy = self._problem(seed)
            n, m = Kxx.shape[0], Kyy.shape[0]
            alpha = 0.25
            base = solve_gram(Kxx, Kyy, Kxy, alpha, max_iter=8000, tol=1e-13)
            eps = 10 ** rng.uniform(-4, -1)
            dKxx = rng.uniform(-eps, eps, Kxx.shape)
            dKyy = rng.uniform(-eps, eps, Kyy.shape)
            dKxy = rng.uniform(-eps, eps, Kxy.shape)
            true_eps = max(np.abs(dKxx).max(), np.abs(dKyy).max(), np.abs(dKxy).max())
            pert = solve_gram(Kxx + dKxx, Kyy + dKyy, Kxy + dKxy, alpha,
                              max_iter=8000, tol=1e-13)
            gap = abs(base.value - pert.value)
            # small slack for FISTA's own residual tolerance, not the lemma
            self.assertLessEqual(gap, 4.0 * true_eps + 1e-6,
                                 f"seed={seed} eps={true_eps}: gap={gap} > 4*eps")

    def test_bound_is_sharp_by_explicit_construction(self):
        # Exact analytic construction: for ANY feasible (w,v) with
        # sum(w)=sum(v)=1, setting dKxx=+eps*J, dKyy=+eps*J, dKxy=-eps*J
        # (J = all-ones) gives Delta-objective = eps(sum w)^2 + eps(sum v)^2
        # + 2*eps*(sum w)(sum v) = 4*eps EXACTLY, independent of w, v -- so
        # the bound is not just valid, it is attained. This needs no solver.
        rng = np.random.default_rng(5)
        for n, m in [(10, 8), (50, 3), (200, 199)]:
            w = rng.dirichlet(np.ones(n))
            v = rng.dirichlet(np.ones(m))
            eps = rng.uniform(1e-6, 1.0)
            dKxx = eps * np.ones((n, n))
            dKyy = eps * np.ones((m, m))
            dKxy = -eps * np.ones((n, m))
            delta_obj = (w @ dKxx @ w + v @ dKyy @ v - 2 * w @ dKxy @ v)
            self.assertAlmostEqual(delta_obj, 4.0 * eps, places=9)


# --------------------------------------------------------------------------
# n6: lem:continuity -- local Lipschitz continuity of alpha -> D_alpha,k
# --------------------------------------------------------------------------

class TestAlphaLipschitz(unittest.TestCase):
    """CLAIMS.md n6: |D_a - D_a'| <= (4 kappa / eta^2) |a - a'| on [0, 1-eta]."""

    def test_pairwise_slopes_respect_the_bound(self):
        # Large-n empirical D_alpha as a proxy for the population value D
        # (the inequality is about D itself; sampling noise is suppressed
        # by using n large relative to the alpha-grid spacing). Kept modest
        # (n=1200, 4 BLAS threads) so a T1 identity check doesn't compete
        # for cores with other tenants on a shared, already-loaded box.
        rng = np.random.default_rng(6)
        n = 1200
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((n, 2)) + 0.8
        Kxx = gaussian_kernel(X, X, 1.0)
        Kyy = gaussian_kernel(Y, Y, 1.0)
        Kxy = gaussian_kernel(X, Y, 1.0)
        eta = 0.2
        grid = np.linspace(0.0, 1.0 - eta, 7)
        kappa = 1.0
        from trimmedmmd.solver import solve_path

        def solve_one(a, w0, v0):
            return solve_gram(Kxx, Kyy, Kxy, a, w0=w0, v0=v0, max_iter=4000, tol=1e-11)

        res = solve_path(solve_one, grid)
        D = np.array([r.discrepancy for r in res])
        bound_const = 4.0 * kappa / (eta ** 2)
        for i in range(len(grid)):
            for j in range(i + 1, len(grid)):
                da = abs(grid[j] - grid[i])
                slope = abs(D[j] - D[i]) / da
                self.assertLessEqual(
                    slope, bound_const * 1.1,  # slack for finite-n noise
                    f"alpha={grid[i]:.2f}->{grid[j]:.2f}: slope={slope:.3f} "
                    f"vs bound {bound_const:.3f}")


# --------------------------------------------------------------------------
# n2: lem:nullbound -- exact 1-D realization via interval regions
# --------------------------------------------------------------------------

class TestDirectionalSeparation(unittest.TestCase):
    """CLAIMS.md n3: lem:separation.

    sep(f; A, B) = inf_{a in A}<f,a> - sup_{b in B}<f,b> <= dist(A,B) for
    every unit f, with equality at the optimal witness f* = (a*-b*)/norm.
    Both sides computed on the SAME sample (in-sample, X1=X2=X), since the
    lemma is a pure geometric identity about fixed sets and does not involve
    sample splitting -- splitting is a downstream design choice used
    elsewhere for statistical independence, not part of this claim.
    """

    def _problem(self, seed, n=30, m=25, d=2):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d)) + 0.6
        return X, Y, rng

    def test_random_directions_never_exceed_the_distance(self):
        X, Y, rng = self._problem(0)
        alpha = 0.25
        sigma = 1.0
        dist_AB = solve_gram(gaussian_kernel(X, X, sigma),
                             gaussian_kernel(Y, Y, sigma),
                             gaussian_kernel(X, Y, sigma),
                             alpha, max_iter=6000, tol=1e-12).discrepancy

        n, m = len(X), len(Y)
        for trial in range(30):
            # An arbitrary (not necessarily optimal) unit direction in the
            # RKHS: random coefficients on the training atoms, normalized by
            # the induced RKHS norm via the same quadratic form solve_gram
            # itself uses.
            w = rng.standard_normal(n)
            v = rng.standard_normal(m)
            Kxx, Kyy, Kxy = (gaussian_kernel(X, X, sigma), gaussian_kernel(Y, Y, sigma),
                            gaussian_kernel(X, Y, sigma))
            sq_norm = w @ Kxx @ w + v @ Kyy @ v - 2 * w @ Kxy @ v
            if sq_norm <= 1e-12:
                continue
            norm = float(np.sqrt(sq_norm))
            fX = evaluate_direction(w, v, X, Y, X, sigma, norm)
            fY = evaluate_direction(w, v, X, Y, Y, sigma, norm)
            sep = (trimmed_mean(fX, alpha, upper=False)
                  - trimmed_mean(fY, alpha, upper=True))
            self.assertLessEqual(sep, dist_AB + 1e-6,
                                 f"trial={trial}: sep={sep} > dist(A,B)={dist_AB}")

    def test_optimal_witness_attains_the_distance(self):
        X, Y, rng = self._problem(1)
        alpha = 0.25
        sigma = 1.0
        w, v, norm = witness_direction(X, Y, alpha, sigma, rng=rng,
                                       max_iter=6000, tol=1e-12)
        fX = evaluate_direction(w, v, X, Y, X, sigma, norm)
        fY = evaluate_direction(w, v, X, Y, Y, sigma, norm)
        sep_star = (trimmed_mean(fX, alpha, upper=False)
                   - trimmed_mean(fY, alpha, upper=True))
        self.assertLess(abs(sep_star - norm), 1e-5,
                        f"sep(f*)={sep_star} should equal dist(A,B)={norm}")


class TestNullBoundExactIntervals(unittest.TestCase):
    """CLAIMS.md n2. Uses the identity kernel/feature map phi(x) = x on

    Uniform[0,1], where C_alpha(P) is exactly the interval
    [(1-alpha)/2, (1+alpha)/2] in closed form (derivable from lem:cvar / lem:trimmedmean:
    the top-(1-alpha) mean of U[0,1] is (1+alpha)/2). lem:nullbound
    does not need characteristicness, so this linear-feature special case is
    a legitimate, fully closed-form realization -- no QP solver needed.
    """

    @staticmethod
    def _pop_interval(alpha):
        return (1.0 - alpha) / 2.0, (1.0 + alpha) / 2.0

    @staticmethod
    def _interval_hausdorff(a, b):
        (a1, a2), (b1, b2) = a, b
        return max(abs(a1 - b1), abs(a2 - b2))

    @staticmethod
    def _interval_dist(a, b):
        (a1, a2), (b1, b2) = a, b
        return max(0.0, b1 - a2, a1 - b2)

    def test_nullbound_holds_surely_under_P_equals_Q(self):
        rng = np.random.default_rng(7)
        violations = 0
        trials = 0
        for alpha in (0.0, 0.1, 0.3, 0.6, 0.85):
            amin_pop, amax_pop = self._pop_interval(alpha)
            for n, m in [(20, 25), (100, 90), (500, 480)]:
                for _ in range(30):
                    trials += 1
                    X = rng.uniform(0, 1, n)
                    Y = rng.uniform(0, 1, m)
                    Xn = (bottom_k_mean(X, alpha), top_k_mean(X, alpha))
                    Ym = (bottom_k_mean(Y, alpha), top_k_mean(Y, alpha))
                    d_hat = self._interval_dist(Xn, Ym)
                    dH_X = self._interval_hausdorff(Xn, (amin_pop, amax_pop))
                    dH_Y = self._interval_hausdorff(Ym, (amin_pop, amax_pop))
                    if d_hat > dH_X + dH_Y + 1e-9:
                        violations += 1
        self.assertEqual(violations, 0,
                         f"{violations}/{trials} trials violated the null bound "
                         f"(lem:nullbound), which the paper states holds SURELY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
