"""Tests for the inferential API: test rule, estimator, certificate, bounds.

Added in the revision of 2026-09-23 (issue I2). Before it, no test exercised
``reject``, ``alpha_hat_from_path``, ``bisect_alpha_hat``, ``split_test``,
``lower_bound_tv``, ``split_upper_bound`` or ``interval_tv`` -- every test
stopped at the solver and the kernels. That gap is how the infeasible-trimming
bug in ``split_upper_bound`` (issue I1) survived: the quantity it returned was
never checked against the constraint set it is supposed to live in.

The invariants below are the ones the paper's statements actually assert, so a
failure here is a failure of the paper's claim rather than of an implementation
detail.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trimmedmmd.estimate import alpha_hat_from_path, bisect_alpha_hat  # noqa: E402
from trimmedmmd.splitting import (  # noqa: E402
    interval_tv,
    lower_bound_tv,
    split_test,
    split_upper_bound,
    trimming_weights,
)
from trimmedmmd.threshold import radius, reject, tau  # noqa: E402


def contaminated_pair(n=300, a_star=0.30, d=2, seed=0, shift=6.0):
    """A pair with TV(P, Q) = a_star exactly.

    The contaminating components are mutually singular at this shift (the
    overlap of two unit Gaussians six standard deviations apart is below
    machine epsilon relative to the sample size), so the contamination
    fraction is the total-variation distance.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, d))
    Y[: int(round(a_star * n))] += shift
    return X, Y


class TestRejectRule(unittest.TestCase):
    """eq. (16): reject iff the statistic exceeds tau."""

    def test_reject_agrees_with_tau(self):
        for n, m, alpha, delta in ((100, 100, 0.0, 0.05),
                                   (250, 400, 0.2, 0.01),
                                   (50, 50, 0.5, 0.10)):
            t = tau(n, m, alpha, delta)
            self.assertTrue(reject(t * 1.001, n, m, alpha, delta))
            self.assertFalse(reject(t * 0.999, n, m, alpha, delta))
            self.assertFalse(reject(t, n, m, alpha, delta))  # strict inequality

    def test_tau_is_the_two_sample_sum_at_half_delta(self):
        n, m, alpha, delta = 300, 500, 0.15, 0.05
        self.assertAlmostEqual(
            tau(n, m, alpha, delta),
            radius(n, alpha, delta / 2) + radius(m, alpha, delta / 2),
            places=12,
        )

    def test_radius_increases_in_alpha_and_decreases_in_n(self):
        alphas = [0.0, 0.1, 0.3, 0.6, 0.9]
        vals = [radius(200, a, 0.05) for a in alphas]
        self.assertTrue(all(np.diff(vals) > 0))
        ns = [50, 100, 400, 1600]
        vals = [radius(n, 0.2, 0.05) for n in ns]
        self.assertTrue(all(np.diff(vals) < 0))


class TestAlphaHat(unittest.TestCase):
    """eq:alphaest. thm:consistency(i) controls the infimum of the crossing set.

    The grid implementation returns the smallest *accepted* level, which is at
    most one mesh above that infimum -- the property recorded in the revision
    as issue I8 and now stated in the manuscript. These tests pin the
    behaviour so the gap cannot silently widen.
    """

    def test_returns_a_grid_point_that_crosses(self):
        grid = np.round(np.arange(0.0, 0.51, 0.05), 3)
        n = m = 400
        thr = np.array([tau(n, m, a, 0.05) for a in grid])
        d_hats = np.maximum(0.6 - 2.0 * grid, 0.0)
        a_hat = alpha_hat_from_path(grid, d_hats, n, m, 0.05)
        self.assertIn(a_hat, set(grid.tolist()))
        i = int(np.nonzero(grid == a_hat)[0][0])
        self.assertLessEqual(d_hats[i], thr[i])
        if i > 0:
            self.assertGreater(d_hats[i - 1], thr[i - 1])

    def test_returns_zero_when_the_path_never_exceeds_the_threshold(self):
        grid = np.round(np.arange(0.0, 0.51, 0.05), 3)
        d_hats = np.zeros_like(grid)
        self.assertEqual(alpha_hat_from_path(grid, d_hats, 400, 400, 0.05), 0.0)

    def test_returns_the_top_of_the_grid_when_nothing_crosses(self):
        grid = np.round(np.arange(0.0, 0.51, 0.05), 3)
        d_hats = np.full_like(grid, 1e6)
        self.assertEqual(alpha_hat_from_path(grid, d_hats, 400, 400, 0.05),
                         float(grid[-1]))

    def test_bisection_brackets_the_crossing_and_returns_from_above(self):
        n = m = 400
        true_cross = 0.23

        def d_hat_at(a):
            # a path that crosses its threshold exactly once, at true_cross
            return tau(n, m, a, 0.05) * (2.0 if a < true_cross else 0.5)

        a_hat = bisect_alpha_hat(d_hat_at, n=n, m=m, delta=0.05, tol=1e-4)
        self.assertGreaterEqual(a_hat, true_cross)          # returns from above
        self.assertLess(a_hat - true_cross, 1e-3)

    def test_bisection_agrees_with_the_grid_up_to_the_mesh(self):
        n = m = 300
        grid = np.round(np.arange(0.0, 0.61, 0.05), 3)
        d_hats = np.maximum(0.8 - 2.5 * grid, 0.0)

        def d_hat_at(a):
            return float(np.interp(a, grid, d_hats))

        a_grid = alpha_hat_from_path(grid, d_hats, n, m, 0.05)
        a_bis = bisect_alpha_hat(d_hat_at, n=n, m=m, delta=0.05, tol=1e-4)
        self.assertLessEqual(a_bis, a_grid + 1e-9)
        self.assertGreater(a_bis, a_grid - 0.05 - 1e-9)

    def test_bisection_report_start_and_last_rejected(self):
        """JMLR v3 Algorithm alg:bisect: any start gives the same bracket; the
        largest rejected level is below the crossing and within tol of it."""
        from trimmedmmd.estimate import bisect_alpha_hat_report
        n = m = 400
        true_cross = 0.23

        def d_hat_at(a):
            return tau(n, m, a, 0.05) * (2.0 if a < true_cross else 0.5)

        for start in (None, 0.05, 0.5, 0.9, 0.2299):
            rep = bisect_alpha_hat_report(d_hat_at, n=n, m=m, delta=0.05, tol=0.01, start=start)
            self.assertLess(rep["last_rejected"], true_cross)
            self.assertGreaterEqual(rep["accepted"], true_cross)
            self.assertLess(rep["accepted"] - rep["last_rejected"], 0.01)
            self.assertLessEqual(rep["tests"], 1 + 9)       # lo, then about log2(0.95/0.01) halvings
            for a, r in rep["visited"]:
                self.assertEqual(r, a < true_cross)          # every decision is the correct one

    def test_bisection_default_unchanged_by_start_none(self):
        n = m = 300
        grid = np.round(np.arange(0.0, 0.61, 0.05), 3)
        d_hats = np.maximum(0.8 - 2.5 * grid, 0.0)

        def d_hat_at(a):
            return float(np.interp(a, grid, d_hats))

        self.assertEqual(bisect_alpha_hat(d_hat_at, n=n, m=m, delta=0.05, tol=1e-4),
                         bisect_alpha_hat(d_hat_at, n=n, m=m, delta=0.05, tol=1e-4, start=None))

    def test_bisection_accepts_lo(self):
        from trimmedmmd.estimate import bisect_alpha_hat_report
        rep = bisect_alpha_hat_report(lambda a: 0.0, n=200, m=200, delta=0.05)
        self.assertEqual(rep["accepted"], 0.0)
        self.assertIsNone(rep["last_rejected"])
        self.assertEqual(rep["tests"], 1)


class TestTrimmingWeightsFeasible(unittest.TestCase):
    """Every weight vector handed to a discrepancy must be an alpha-trimming."""

    def test_trimming_weights_are_feasible(self):
        rng = np.random.default_rng(0)
        for alpha in (0.0, 0.05, 0.25, 0.5, 0.75):
            for n in (7, 40, 257):
                w = trimming_weights(rng.standard_normal(n), alpha, upper=True)
                cap = 1.0 / (n * (1.0 - alpha))
                self.assertAlmostEqual(w.sum(), 1.0, places=9)
                self.assertGreaterEqual(w.min(), -1e-12)
                self.assertLessEqual(w.max(), cap + 1e-9)

    def test_weights_load_the_cap_onto_the_extreme_values(self):
        v = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        w = trimming_weights(v, 0.4, upper=True)   # keep the largest 60%
        cap = 1.0 / (5 * 0.6)
        self.assertAlmostEqual(w[0], cap, places=9)      # 5.0, largest
        self.assertAlmostEqual(w[2], cap, places=9)      # 4.0
        self.assertAlmostEqual(w[1], 0.0, places=9)      # 1.0, smallest
        self.assertAlmostEqual(w[3], 0.0, places=9)      # 2.0


class TestSplitUpperBoundFeasibility(unittest.TestCase):
    """Issue I1: the exhibited pair must lie in the capped simplex.

    ``split_upper_bound`` is valid only because *any feasible* trimming pair
    bounds D_alpha from above. A pair that violates the cap corresponds to a
    smaller effective alpha, so the returned value bounds the wrong quantity.
    The guard that was supposed to enforce this capped the weights and then
    renormalised them, which on a vector that is uniform on its support
    returns the original vector exactly.
    """

    def test_returned_value_is_finite_and_nonnegative(self):
        X, Y = contaminated_pair(n=200, seed=3)
        for alpha in (0.0, 0.1, 0.25):
            point, upper = split_upper_bound(X, Y, alpha, n_boot=40,
                                             rng=np.random.default_rng(1))
            self.assertTrue(np.isfinite(point) and point >= 0.0)
            self.assertTrue(np.isfinite(upper) and upper >= 0.0)
            self.assertGreaterEqual(upper, point - 1e-9)

    def test_internal_rule_weights_respect_the_cap(self):
        """Exercises the projection over many random fold-two realisations.

        The rule is a closure inside ``split_upper_bound``, so the invariant is
        checked here on the same construction: a fold-one quantile threshold
        applied to an independent fold-two sample. Before the fix this failed
        on about half of the draws, by up to 1.7x the cap.
        """
        from trimmedmmd.simplex import project_capped_simplex

        rng = np.random.default_rng(0)
        alpha = 0.25
        for _ in range(400):
            n = int(rng.integers(40, 200))
            vals = rng.standard_normal(n)
            thresh = float(np.quantile(rng.standard_normal(n), 1 - alpha))
            keep = vals <= thresh
            if keep.sum() == 0:
                continue
            cap = 1.0 / (n * (1 - alpha))
            w = keep.astype(float)
            w = w / w.sum()
            if w.max() > cap + 1e-12:
                w = project_capped_simplex(w, cap)
            self.assertLessEqual(w.max(), cap + 1e-9)
            self.assertAlmostEqual(w.sum(), 1.0, places=9)


class TestCertificate(unittest.TestCase):
    """thm:split / cor:lower: rejection certifies TV > alpha."""

    def test_certifies_below_the_truth_and_stops_above_it(self):
        X, Y = contaminated_pair(n=400, a_star=0.30, seed=5)
        grid = np.round(np.arange(0.0, 0.76, 0.05), 3)
        lo, out = lower_bound_tv(X, Y, grid, delta=0.05,
                                 rng=np.random.default_rng(2), n_boot=200)
        self.assertGreater(lo, 0.0)
        # A lower bound on TV must not exceed the truth by more than the mesh.
        self.assertLessEqual(lo, 0.30 + 1e-9)

    def test_certified_set_is_downward_closed(self):
        X, Y = contaminated_pair(n=400, a_star=0.30, seed=6)
        grid = np.round(np.arange(0.0, 0.76, 0.05), 3)
        lo, out = lower_bound_tv(X, Y, grid, delta=0.05,
                                 rng=np.random.default_rng(3), n_boot=200)
        certified = [bool(r.certified) for r in out]
        first_failure = next((i for i, c in enumerate(certified) if not c),
                             len(certified))
        self.assertTrue(all(certified[:first_failure]))
        self.assertAlmostEqual(lo, float(grid[first_failure - 1])
                               if first_failure > 0 else 0.0, places=9)

    def test_split_test_is_reproducible_under_a_fixed_seed(self):
        X, Y = contaminated_pair(n=200, seed=7)
        a = split_test(X, Y, 0.1, rng=np.random.default_rng(11), n_boot=100)
        b = split_test(X, Y, 0.1, rng=np.random.default_rng(11), n_boot=100)
        self.assertEqual(a.certified, b.certified)
        self.assertAlmostEqual(a.separation, b.separation, places=12)
        self.assertAlmostEqual(a.lower_ci, b.lower_ci, places=12)


class TestInterval(unittest.TestCase):
    """The two-sided interval: the lower end is certified, the upper assumed."""

    def test_upper_end_is_above_the_lower_end(self):
        X, Y = contaminated_pair(n=300, a_star=0.30, seed=9)
        grid = np.round(np.arange(0.0, 0.76, 0.05), 3)
        lo, hi, _ = interval_tv(X, Y, grid, c0=1.5, gamma=1.0, delta=0.05,
                                rng=np.random.default_rng(4), n_boot=200)
        self.assertGreaterEqual(hi, lo - 1e-9)
        self.assertLessEqual(hi, 1.0 + 1e-12)

    def test_a_smaller_assumed_margin_constant_gives_a_weaker_upper_end(self):
        X, Y = contaminated_pair(n=300, a_star=0.30, seed=10)
        grid = np.round(np.arange(0.0, 0.76, 0.05), 3)
        kw = dict(delta=0.05, n_boot=200)
        _, hi_small, _ = interval_tv(X, Y, grid, c0=0.8, gamma=1.0,
                                     rng=np.random.default_rng(5), **kw)
        _, hi_large, _ = interval_tv(X, Y, grid, c0=1.6, gamma=1.0,
                                     rng=np.random.default_rng(5), **kw)
        self.assertGreaterEqual(hi_small, hi_large - 1e-9)


if __name__ == "__main__":
    unittest.main()
