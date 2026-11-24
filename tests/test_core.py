"""Correctness tests. Run with: python3 -m unittest discover -s tests -v

Uses stdlib unittest so the suite runs without extra dependencies. The
solver is checked against an independent general-purpose optimizer, and the
data-generating processes are checked against numerical integration, so
these are genuine cross-checks rather than restatements of the code.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trimmedmmd import dgp, threshold  # noqa: E402
from trimmedmmd.discrepancy import plain_mmd, trimmed_mmd_path  # noqa: E402
from trimmedmmd.kernels import RFF, gaussian_kernel, median_heuristic  # noqa: E402
from trimmedmmd.simplex import project_capped_simplex  # noqa: E402
from trimmedmmd.solver import solve_features, solve_gram  # noqa: E402


class TestCappedSimplexProjection(unittest.TestCase):
    def test_constraints_hold(self):
        rng = np.random.default_rng(0)
        for n in (5, 50, 500):
            cap = 1.0 / (n * (1 - 0.3))
            w = project_capped_simplex(rng.standard_normal(n), cap)
            self.assertAlmostEqual(w.sum(), 1.0, places=9)
            self.assertGreaterEqual(w.min(), -1e-12)
            self.assertLessEqual(w.max(), cap + 1e-12)

    def test_matches_generic_optimizer(self):
        rng = np.random.default_rng(1)
        n = 12
        cap = 1.0 / (n * (1 - 0.4))
        y = rng.standard_normal(n)
        w = project_capped_simplex(y, cap)
        res = minimize(
            lambda z: np.sum((z - y) ** 2), x0=np.full(n, 1.0 / n),
            bounds=[(0, cap)] * n,
            constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1.0}],
            method="SLSQP", options={"maxiter": 500, "ftol": 1e-12},
        )
        self.assertTrue(res.success)
        self.assertLess(abs(np.sum((w - y) ** 2) - res.fun), 1e-7)

    def test_alpha_zero_forces_uniform(self):
        # At alpha = 0 the cap is exactly 1/n, so the only feasible point is
        # the uniform vector -- a useful exact anchor for the solver tests.
        n = 20
        w = project_capped_simplex(np.random.default_rng(2).standard_normal(n),
                                   1.0 / n)
        np.testing.assert_allclose(w, np.full(n, 1.0 / n), atol=1e-9)

    def test_infeasible_raises(self):
        with self.assertRaises(ValueError):
            project_capped_simplex(np.zeros(4), cap=0.1)  # 4 * 0.1 < 1


class TestSolver(unittest.TestCase):
    def _problem(self, seed=0, n=14, m=11, d=2, sigma=1.0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d)) + 0.7
        Kxx = gaussian_kernel(X, X, sigma)
        Kyy = gaussian_kernel(Y, Y, sigma)
        Kxy = gaussian_kernel(X, Y, sigma)
        return X, Y, Kxx, Kyy, Kxy

    def test_matches_generic_optimizer(self):
        _, _, Kxx, Kyy, Kxy = self._problem()
        n, m, alpha = Kxx.shape[0], Kyy.shape[0], 0.25
        cw, cv = 1.0 / (n * (1 - alpha)), 1.0 / (m * (1 - alpha))

        def obj(z):
            w, v = z[:n], z[n:]
            return w @ Kxx @ w + v @ Kyy @ v - 2 * w @ Kxy @ v

        res = minimize(
            obj, x0=np.concatenate([np.full(n, 1 / n), np.full(m, 1 / m)]),
            bounds=[(0, cw)] * n + [(0, cv)] * m,
            constraints=[
                {"type": "eq", "fun": lambda z: z[:n].sum() - 1.0},
                {"type": "eq", "fun": lambda z: z[n:].sum() - 1.0},
            ],
            method="SLSQP", options={"maxiter": 2000, "ftol": 1e-14},
        )
        self.assertTrue(res.success)
        ours = solve_gram(Kxx, Kyy, Kxy, alpha, max_iter=20000, tol=1e-14)
        self.assertLess(abs(ours.value - res.fun), 1e-6)

    def test_alpha_zero_equals_plain_mmd(self):
        X, Y, Kxx, Kyy, Kxy = self._problem()
        got = solve_gram(Kxx, Kyy, Kxy, 0.0, max_iter=5000).discrepancy
        want = plain_mmd(X, Y, sigma=1.0)
        self.assertAlmostEqual(got, want, places=8)

    def test_monotone_nonincreasing_in_alpha(self):
        _, _, Kxx, Kyy, Kxy = self._problem(seed=3)
        vals = [solve_gram(Kxx, Kyy, Kxy, a, max_iter=8000).value
                for a in (0.0, 0.1, 0.2, 0.35, 0.5)]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(b, a + 1e-9)

    def test_warm_start_matches_cold_start(self):
        _, _, Kxx, Kyy, Kxy = self._problem(seed=5)
        cold = solve_gram(Kxx, Kyy, Kxy, 0.3, max_iter=20000, tol=1e-14)
        first = solve_gram(Kxx, Kyy, Kxy, 0.2, max_iter=20000, tol=1e-14)
        warm = solve_gram(Kxx, Kyy, Kxy, 0.3, w0=first.w, v0=first.v,
                          max_iter=20000, tol=1e-14)
        self.assertLess(abs(cold.value - warm.value), 1e-8)

    def test_feature_backend_matches_gram_backend(self):
        rng = np.random.default_rng(7)
        n, m, r = 25, 20, 40
        Zx = rng.standard_normal((n, r)) / np.sqrt(r)
        Zy = rng.standard_normal((m, r)) / np.sqrt(r)
        Kxx, Kyy, Kxy = Zx @ Zx.T, Zy @ Zy.T, Zx @ Zy.T
        a = 0.3
        g = solve_gram(Kxx, Kyy, Kxy, a, max_iter=20000, tol=1e-14)
        f = solve_features(Zx, Zy, a, max_iter=20000, tol=1e-14)
        self.assertLess(abs(g.value - f.value), 1e-8)

    def test_identical_samples_give_zero(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((15, 2))
        K = gaussian_kernel(X, X, 1.0)
        res = solve_gram(K, K, K, 0.2, max_iter=20000, tol=1e-14)
        self.assertLess(res.value, 1e-10)


class TestKernels(unittest.TestCase):
    def test_median_heuristic_excludes_diagonal(self):
        # With duplicated points, including the diagonal would drag the
        # median to zero; excluding it must not.
        X = np.repeat(np.array([[0.0], [1.0], [2.0]]), 4, axis=0)
        self.assertGreater(median_heuristic(X), 0.0)

    def test_rff_approximates_kernel(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((40, 3))
        sigma = 1.3
        rff = RFF(dim=3, n_features=20000, sigma=sigma, rng=rng)
        Z = rff.transform(X)
        err = np.abs(Z @ Z.T - gaussian_kernel(X, X, sigma)).max()
        self.assertLess(err, 0.05)

    def test_gaussian_kernel_unit_diagonal(self):
        X = np.random.default_rng(4).standard_normal((10, 2))
        np.testing.assert_allclose(np.diag(gaussian_kernel(X, X, 0.9)),
                                   np.ones(10), atol=1e-12)


class TestDGP(unittest.TestCase):
    def test_gaussian_tv_matches_numerical_integration(self):
        # TV(N(mu,1), N(-mu,1)) = 2 Phi(mu) - 1; check by integrating |p-q|/2.
        for mu in (0.3, 1.0, 2.0):
            closed = 2 * norm.cdf(mu) - 1
            num, _ = quad(lambda x: abs(norm.pdf(x, mu) - norm.pdf(x, -mu)),
                          -30, 30, limit=400)
            self.assertLess(abs(closed - num / 2), 1e-6)

    def test_scenario_alpha_star_values(self):
        s = dgp.disjoint_contamination(dim=2, a=0.2)
        self.assertAlmostEqual(s.alpha_star, 0.2)
        g = dgp.gaussian_contamination(dim=2, a=0.3, mu_norm=1.0)
        self.assertAlmostEqual(g.alpha_star, 0.3 * (2 * norm.cdf(1.0) - 1))
        self.assertLess(g.alpha_star, 0.3)  # contamination not singular

    def test_disjoint_supports_are_actually_disjoint(self):
        # TV(P,Q) = a relies on the contaminating laws being mutually
        # singular; verify the boxes really do not overlap.
        s = dgp.disjoint_contamination(dim=1, a=0.5, sep=8.0, width=1.0)
        rng = np.random.default_rng(0)
        X = s.sample(4000, rng, "P")
        Y = s.sample(4000, rng, "Q")
        self.assertGreater(np.sum(X < -7.9), 0)  # P does use its own box
        self.assertGreater(np.sum(Y > 7.9), 0)   # Q does use its own box
        self.assertEqual(np.sum(X > 7.9), 0)     # P never lands in Q's box
        self.assertEqual(np.sum(Y < -7.9), 0)    # and vice versa

    def test_mixture_weight_is_right(self):
        s = dgp.disjoint_contamination(dim=1, a=0.25, sep=8.0)
        rng = np.random.default_rng(1)
        X = s.sample(20000, rng, "P")
        frac = float(np.mean(X < -7.0))
        self.assertLess(abs(frac - 0.25), 0.02)


class TestThreshold(unittest.TestCase):
    def test_radius_formula(self):
        n, a, d = 500, 0.3, 0.05
        want = (1.0 / (1 - a)) * (4 / np.sqrt(n)
                                  + np.sqrt(2 * np.log(2 / d) / n))
        self.assertAlmostEqual(threshold.radius(n, a, d), want, places=12)

    def test_tau_is_sum_of_two_radii_at_half_delta(self):
        self.assertAlmostEqual(
            threshold.tau(100, 200, 0.2, 0.1),
            threshold.radius(100, 0.2, 0.05) + threshold.radius(200, 0.2, 0.05),
            places=12)

    def test_threshold_increases_in_alpha(self):
        vals = [threshold.tau(300, 300, a, 0.05) for a in (0.0, 0.2, 0.5, 0.8)]
        for x, y in zip(vals, vals[1:]):
            self.assertGreater(y, x)

    def test_leading_term_dominates_at_practical_n(self):
        # The 4/sqrt(n) complexity term, not the confidence term, is the
        # larger part of the threshold at practical sample sizes. Recorded
        # as a test because the conservativeness story turns on it: at
        # n = 1000 its share is about 0.57, and it grows as delta relaxes.
        self.assertGreater(threshold.leading_term_share(1000, 1000, 0.05), 0.5)
        self.assertGreater(threshold.leading_term_share(1000, 1000, 0.20), 0.6)


class TestPath(unittest.TestCase):
    def test_path_is_monotone_and_matches_pointwise(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((30, 2))
        Y = rng.standard_normal((30, 2)) + 1.0
        alphas = np.array([0.0, 0.1, 0.2, 0.3])
        a, d, _ = trimmed_mmd_path(X, Y, alphas, sigma=1.0, rng=rng,
                                   max_iter=20000, tol=1e-13)
        for x, y in zip(d, d[1:]):
            self.assertLessEqual(y, x + 1e-8)
        Kxx = gaussian_kernel(X, X, 1.0)
        Kyy = gaussian_kernel(Y, Y, 1.0)
        Kxy = gaussian_kernel(X, Y, 1.0)
        single = solve_gram(Kxx, Kyy, Kxy, 0.2, max_iter=20000,
                            tol=1e-13).discrepancy
        self.assertLess(abs(single - d[2]), 1e-6)


class TestCheckpointing(unittest.TestCase):
    """Resuming must reuse finished work and recompute only what is missing."""

    def setUp(self):
        import tempfile
        from trimmedmmd.checkpoint import ReplicateStore
        self.tmp = tempfile.mkdtemp()
        self.store = ReplicateStore(self.tmp)

    def test_roundtrip_and_resume(self):
        from trimmedmmd.checkpoint import map_replicates
        calls = []

        def fn(job):
            calls.append(job)
            return {"x": job[0] * 2}

        jobs = [(i, "seed", i) for i in range(5)]
        seeds = [j[2] for j in jobs]
        out1 = map_replicates(fn, jobs, seeds, workers=1,
                              store=self.store, key="k")
        self.assertEqual([o["x"] for o in out1], [0, 2, 4, 6, 8])
        self.assertEqual(len(calls), 5)

        # Second pass: everything is on disk, so nothing should recompute.
        calls.clear()
        out2 = map_replicates(fn, jobs, seeds, workers=1,
                              store=self.store, key="k")
        self.assertEqual(out2, out1)
        self.assertEqual(len(calls), 0)

        # Extending the job list recomputes only the new replicates.
        jobs2 = jobs + [(9, "seed", 9)]
        calls.clear()
        out3 = map_replicates(fn, jobs2, [j[2] for j in jobs2], workers=1,
                              store=self.store, key="k")
        self.assertEqual(len(calls), 1)
        self.assertEqual(out3[-1]["x"], 18)

    def test_partial_final_line_is_tolerated(self):
        # A process killed mid-write leaves a truncated line; that replicate
        # should be recomputed rather than crashing the resume.
        self.store.append("k", 1, {"x": 1})
        with (self.store._path("k")).open("a") as f:
            f.write('{"seed": 2, "value": {"x": ')  # truncated
        loaded = self.store.load("k")
        self.assertEqual(set(loaded), {1})

    def test_corrupt_bytes_are_tolerated(self):
        # A filesystem hiccup can leave undecodable bytes in the middle of the
        # file. That record should be dropped and the rest still read, rather
        # than the whole resume failing.
        self.store.append("k", 1, {"x": 1})
        with (self.store._path("k")).open("ab") as f:
            f.write(b'\x81\x82 not utf-8 at all\n')
        self.store.append("k", 3, {"x": 3})
        loaded = self.store.load("k")
        self.assertEqual(set(loaded), {1, 3})

    def test_key_changes_when_parameters_change(self):
        from trimmedmmd.checkpoint import cell_key
        a = cell_key("e", {"n": 100, "alpha": 0.2})
        b = cell_key("e", {"n": 200, "alpha": 0.2})
        c = cell_key("e", {"alpha": 0.2, "n": 100})  # order must not matter
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)

    def test_numpy_values_survive_roundtrip(self):
        self.store.append("k", 7, (np.float64(1.5), np.arange(3)))
        got = self.store.load("k")[7]
        self.assertEqual(got[0], 1.5)
        self.assertEqual(got[1], [0, 1, 2])

    def test_duplicate_seeds_are_rejected(self):
        # Guards the failure mode where a wrong seed-field index makes every
        # replicate share a key: the store would hold one value and hand it
        # back for all of them, silently destroying the Monte Carlo.
        from trimmedmmd.checkpoint import map_replicates
        jobs = [(i, 0.2) for i in range(4)]
        with self.assertRaises(ValueError):
            map_replicates(lambda j: j[0], jobs, [0.2] * 4, workers=1,
                           store=self.store, key="dup")


class TestRealDataHelpers(unittest.TestCase):
    """The hand-rolled pieces of the real-data analysis, cross-checked."""

    def test_logistic_fit_matches_an_independent_optimizer(self):
        # Newton's method here vs. a generic quasi-Newton minimiser of the
        # same penalised negative log-likelihood. Agreement to 1e-6 in the
        # coefficients is a real check: the two share the objective but
        # nothing of the search.
        from trimmedmmd.experiments.realdata_run import _logistic_fit
        rng = np.random.default_rng(3)
        n, d, l2 = 400, 5, 1e-2
        Z = rng.standard_normal((n, d))
        truth = rng.standard_normal(d + 1)
        p = 1.0 / (1.0 + np.exp(-(np.c_[np.ones(n), Z] @ truth)))
        y = (rng.random(n) < p).astype(float)

        beta, e = _logistic_fit(Z, y, l2=l2)

        Za = np.c_[np.ones(n), Z]

        def nll(b):
            z = Za @ b
            # log(1+exp(z)) computed stably
            return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * b @ b)

        ref = minimize(nll, np.zeros(d + 1), method="L-BFGS-B",
                       options={"ftol": 1e-15, "gtol": 1e-12})
        self.assertTrue(ref.success)
        np.testing.assert_allclose(beta, ref.x, atol=1e-6)
        # And the returned probabilities are the ones implied by beta.
        np.testing.assert_allclose(e, 1.0 / (1.0 + np.exp(-Za @ beta)), atol=1e-12)

    def test_logistic_fit_separable_data_stays_finite(self):
        # Perfect separation sends the unpenalised MLE to infinity; the ridge
        # term is what keeps the propensity scores inside (0,1). If this ever
        # returns exact 0 or 1 the Crump cutoff search divides by zero.
        from trimmedmmd.experiments.realdata_run import _logistic_fit
        Z = np.linspace(-3, 3, 60).reshape(-1, 1)
        y = (Z.ravel() > 0).astype(float)
        _, e = _logistic_fit(Z, y)
        self.assertTrue(np.all(np.isfinite(e)))
        self.assertTrue(np.all((e > 0) & (e < 1)))


class TestTrimmings(unittest.TestCase):
    """The structure claims of Section "Recovering the trimming".

    These are falsification tests, not restatements: each one would fail if
    the corresponding proposition were false, and two of them cross-check
    the solvers against independent implementations.
    """

    @staticmethod
    def _data(n=200, alpha_star=0.10, seed=7):
        rng = np.random.default_rng(seed)
        nc = int(round(alpha_star * n))
        X = np.concatenate([rng.normal(-1, 1, n - nc), rng.normal(2, 1, nc)])
        Y = np.concatenate([rng.normal(-1, 1, n - nc), rng.normal(-4, 1, nc)])
        return X[:, None], Y[:, None]

    def test_weights_are_feasible_trimmings(self):
        from trimmedmmd.trimmings import (optimal_trimming_mmd,
                                          optimal_trimming_wasserstein)
        X, Y = self._data()
        for t in (optimal_trimming_mmd(X, Y, 0.08),
                  optimal_trimming_wasserstein(X, Y, 0.08)):
            self.assertAlmostEqual(t.w.sum(), 1.0, places=8)
            self.assertAlmostEqual(t.v.sum(), 1.0, places=8)
            self.assertGreaterEqual(t.w.min(), -1e-12)
            self.assertLessEqual(t.w.max(), t.cap_x * (1 + 1e-8))
            self.assertLessEqual(t.v.max(), t.cap_y * (1 + 1e-8))

    def test_mmd_trimming_matches_witness_threshold(self):
        # Proposition (bang-bang): the kept set is exactly the k_n smallest
        # values of the witness. An off-by-one or a sign error in the
        # gradient would break this immediately.
        from trimmedmmd.trimmings import optimal_trimming_mmd
        X, Y = self._data()
        t = optimal_trimming_mmd(X, Y, 0.08)
        kept = set(t.kept_x.tolist())
        order = np.argsort(t.meta["witness_x"], kind="stable")
        predicted = set(order[:len(kept)].tolist())
        self.assertEqual(kept, predicted)

    def test_mmd_trimming_is_bang_bang_below_contact(self):
        from trimmedmmd.trimmings import optimal_trimming_mmd
        X, Y = self._data()
        t = optimal_trimming_mmd(X, Y, 0.05)
        self.assertGreater(t.value, 0.0)
        # Allow a couple of near-boundary weights to solver tolerance; the
        # claim is that essentially none are interior, not literally none.
        self.assertLessEqual(t.fractional_count("x"), 3)
        self.assertLessEqual(t.fractional_count("y"), 3)
        self.assertAlmostEqual(t.ess("x") / t.n, 0.95, places=2)

    def test_mmd_trimming_smears_above_contact(self):
        # Proposition (iii): above alpha-star no minimizer is distinguished,
        # so the weights must NOT stay bang-bang. If this test starts
        # passing trivially the uniqueness diagnostic has broken.
        from trimmedmmd.trimmings import optimal_trimming_mmd
        X, Y = self._data()
        t = optimal_trimming_mmd(X, Y, 0.30, select="none")
        self.assertGreater(t.fractional_count("x"), 20)

    def test_wasserstein_pot_matches_highs(self):
        # Two independent solvers on the same linear program.
        from trimmedmmd.trimmings import HAVE_POT, optimal_trimming_wasserstein
        if not HAVE_POT:
            self.skipTest("POT not installed")
        X, Y = self._data(n=120)
        for a in (0.05, 0.12):
            tp = optimal_trimming_wasserstein(X, Y, a, backend="pot")
            th = optimal_trimming_wasserstein(X, Y, a, backend="highs")
            self.assertAlmostEqual(tp.value, th.value, places=8)
            np.testing.assert_allclose(tp.w, th.w, atol=1e-8)

    def test_wasserstein_trimming_is_exactly_bang_bang(self):
        from trimmedmmd.trimmings import optimal_trimming_wasserstein
        X, Y = self._data()
        t = optimal_trimming_wasserstein(X, Y, 0.10)
        self.assertEqual(t.fractional_count("x"), 0)
        self.assertEqual(t.kept_x.size, int(round(t.n * 0.90)))

    def test_untrimmed_limit_agrees_with_plain_mmd(self):
        from trimmedmmd.discrepancy import plain_mmd
        from trimmedmmd.kernels import median_heuristic
        from trimmedmmd.trimmings import optimal_trimming_mmd
        X, Y = self._data(n=150)
        sigma = median_heuristic(X, Y, rng=np.random.default_rng(0))
        t = optimal_trimming_mmd(X, Y, 0.0, sigma=sigma)
        self.assertAlmostEqual(
            t.value, plain_mmd(X, Y, sigma=sigma), places=9)
        np.testing.assert_allclose(t.w, np.full(t.n, 1.0 / t.n), atol=1e-12)

    def test_asymmetric_levels_use_their_own_caps(self):
        from trimmedmmd.trimmings import (optimal_trimming_mmd,
                                          optimal_trimming_wasserstein)
        X, Y = self._data(n=150)
        for t in (optimal_trimming_mmd(X, Y, 0.20, alpha_y=0.02),
                  optimal_trimming_wasserstein(X, Y, 0.20, alpha_y=0.02)):
            self.assertLessEqual(t.w.max(), t.cap_x * (1 + 1e-8))
            self.assertLessEqual(t.v.max(), t.cap_y * (1 + 1e-8))
            self.assertGreater(t.kept_y.size, t.kept_x.size)

    def test_least_trimming_selection_trims_less(self):
        # The selection rule must not make things worse: it has to attain
        # the same objective value and be closer to the uniform weights.
        from trimmedmmd.trimmings import optimal_trimming_mmd
        X, Y = self._data()
        plain = optimal_trimming_mmd(X, Y, 0.30, select="none")
        least = optimal_trimming_mmd(X, Y, 0.30, select="least")
        self.assertFalse(least.unique)
        u = np.full(least.n, 1.0 / least.n)
        self.assertLessEqual(np.linalg.norm(least.w - u),
                             np.linalg.norm(plain.w - u) + 1e-9)
        self.assertLessEqual(least.value, plain.value + 1e-6)

    def test_resample_and_hard_subsample(self):
        from trimmedmmd.trimmings import optimal_trimming_mmd
        X, Y = self._data()
        t = optimal_trimming_mmd(X, Y, 0.10)
        rng = np.random.default_rng(3)
        idx = t.resample("x", size=500, rng=rng)
        self.assertEqual(idx.size, 500)
        # A resample never draws a discarded observation.
        self.assertEqual(len(set(idx.tolist()) & set(t.dropped("x").tolist())), 0)
        self.assertEqual(t.hard("x").size, int(np.ceil(t.n * 0.90)))
        Xt, wt = t.apply(X, "x")
        self.assertEqual(Xt.shape[0], t.kept_x.size)
        self.assertAlmostEqual(wt.sum(), 1.0, places=10)

    def test_rff_trimming_agrees_with_the_exact_kernel(self):
        # The random-feature backend is what makes the trimming computable
        # at large n, so it has to return the same trimming, not merely a
        # similar number. Agreement of the kept SET is the sharp check,
        # because the weights are bang-bang. Exact equality is NOT the right
        # assertion: the features perturb the witness by O(r^{-1/2}), so
        # observations sitting within that of the threshold can and do swap
        # sides. What must hold is that the disagreement is confined to that
        # boundary layer -- a handful of points, not a different answer.
        from trimmedmmd.trimmings import optimal_trimming_mmd, witness_at
        X, Y = self._data(n=250)
        exact = optimal_trimming_mmd(X, Y, 0.08, select="none")
        approx = optimal_trimming_mmd(X, Y, 0.08, select="none",
                                      use_rff=True, n_features=4096)
        self.assertFalse(exact.meta["use_rff"])
        self.assertTrue(approx.meta["use_rff"])
        self.assertAlmostEqual(exact.value, approx.value, places=2)
        ke, ka = set(exact.kept_x.tolist()), set(approx.kept_x.tolist())
        jac = len(ke & ka) / len(ke | ka)
        self.assertGreater(jac, 0.95, f"kept sets diverge: Jaccard {jac:.3f}")
        self.assertLessEqual(abs(len(ke) - len(ka)), 3)
        # And the out-of-sample witness follows the backend it was fitted with.
        Z = X[:20]
        we = witness_at(exact, X, Y, Z)
        wa = witness_at(approx, X, Y, Z)
        self.assertEqual(we.shape, wa.shape)
        np.testing.assert_allclose(we, wa, atol=5e-3)

    def test_removal_order_is_monotone_in_alpha(self):
        from trimmedmmd.trimmings import removal_order, trimming_path
        X, Y = self._data(n=120)
        path = trimming_path(X, Y, [0.02, 0.06, 0.10], geometry="wasserstein")
        rank = removal_order(path, "x")
        self.assertEqual(rank.size, X.shape[0])
        # Points dropped at the first level must be dropped at every later
        # one for the ranking to mean anything; check the counts increase.
        sizes = [p.dropped("x").size for p in path]
        self.assertTrue(sizes[0] <= sizes[1] <= sizes[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
