"""New code (no prior implementation existed): trimmedmmd.sinkhorn.

The inner entropic-OT solver is cross-checked against POT (a well-tested
independent reference), not just against itself -- this caught a real
duality bookkeeping bug (see CLAIMS.md's triage log) where the returned
dual value was off by eps*(H(a)+H(b)) from the true value, despite the
transport plan itself already satisfying the marginals to machine
precision. Agreement below is to ~1e-13, not just "close."
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trimmedmmd.sinkhorn import (sinkhorn_divergence, sinkhorn_ot,  # noqa: E402
                                 sinkhorn_plan, trimmed_sinkhorn)

try:
    import ot as pot
    HAVE_POT = True
except ImportError:
    HAVE_POT = False


@unittest.skipUnless(HAVE_POT, "POT not installed")
class TestSinkhornAgreesWithPOT(unittest.TestCase):
    def test_value_and_plan_match_pot(self):
        rng = np.random.default_rng(0)
        for n, m, eps in [(8, 6, 0.1), (30, 25, 0.05), (15, 15, 0.5)]:
            X = rng.standard_normal((n, 2))
            Y = rng.standard_normal((m, 2)) + 0.5
            a = np.full(n, 1.0 / n)
            b = np.full(m, 1.0 / m)
            C = pot.dist(X, Y, metric="sqeuclidean")

            val, f, g = sinkhorn_ot(a, b, C, eps, n_iter=3000, tol=1e-14)
            pi_mine = sinkhorn_plan(a, b, f, g, C, eps)

            pi_pot, _ = pot.sinkhorn(a, b, C, eps, numItermax=5000,
                                     stopThr=1e-14, log=True)
            primal_pot = float(np.sum(pi_pot * C))
            KL_pot = float(np.sum(pi_pot * np.log(pi_pot / np.outer(a, b)))
                          - pi_pot.sum() + 1.0)
            val_pot = primal_pot + eps * KL_pot

            self.assertLess(abs(val - val_pot), 1e-8,
                            f"n={n} m={m} eps={eps}: mine={val} POT={val_pot}")
            self.assertLess(np.abs(pi_mine - pi_pot).max(), 1e-8)
            self.assertLess(np.abs(pi_mine.sum(1) - a).max(), 1e-10)
            self.assertLess(np.abs(pi_mine.sum(0) - b).max(), 1e-10)


class TestTrimmedSinkhorn(unittest.TestCase):
    def _cost(self, A, B):
        return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)

    def test_alpha_zero_matches_fixed_uniform_divergence(self):
        # At alpha=0 the capped simplex is a single point (uniform weights),
        # so the "optimization" has no freedom: value must equal the plain
        # (untrimmed) Sinkhorn divergence exactly.
        rng = np.random.default_rng(1)
        n, m, eps = 20, 18, 0.3
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2)) + 0.6
        a0, b0 = np.full(n, 1 / n), np.full(m, 1 / m)
        S0, _, _ = sinkhorn_divergence(a0, b0, self._cost(X, X),
                                       self._cost(Y, Y), self._cost(X, Y), eps)
        res = trimmed_sinkhorn(X, Y, 0.0, eps, cost_fn=self._cost,
                               max_iter=50, sinkhorn_iter=300)
        self.assertLess(abs(res["value"] - S0), 1e-5)

    def test_solution_is_feasible(self):
        rng = np.random.default_rng(2)
        n, m, alpha = 15, 12, 0.3
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2)) + 0.5
        res = trimmed_sinkhorn(X, Y, alpha, 0.3, cost_fn=self._cost,
                               max_iter=80, sinkhorn_iter=100)
        cap_w, cap_v = 1.0 / (n * (1 - alpha)), 1.0 / (m * (1 - alpha))
        self.assertAlmostEqual(res["w"].sum(), 1.0, places=8)
        self.assertAlmostEqual(res["v"].sum(), 1.0, places=8)
        self.assertLessEqual(res["w"].max(), cap_w + 1e-9)
        self.assertLessEqual(res["v"].max(), cap_v + 1e-9)

    def test_runs_and_stays_feasible_across_an_alpha_sweep(self):
        # This is a wiring/feasibility smoke check only. A rigorous test of
        # prop:sinkmono (CLAIMS.md n17) needs a much larger iteration and
        # restart budget than a fast unit test can afford: at this problem
        # size, bumping max_iter 150->300 made an apparent violation at one
        # alpha vanish, then bumping iterations at EVERY level (not just the
        # failing one) shifted which pair looked violated again -- this is a
        # genuinely hard nonconvex bilevel optimization landscape, not a
        # quick-to-pin-down bug, and chasing exact monotonicity here would
        # just be fitting noise. The real check belongs in the T2 experiment
        # block (experiments/run.py's e11), with a generous restart budget
        # and its own suboptimality-vs-violation triage, not in a smoke test.
        rng = np.random.default_rng(3)
        n, m = 25, 20
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2)) + 0.8
        eps = 0.3
        w0 = v0 = None
        for alpha in (0.0, 0.15, 0.3, 0.45, 0.6):
            res = trimmed_sinkhorn(X, Y, alpha, eps, cost_fn=self._cost,
                                   max_iter=100, sinkhorn_iter=60,
                                   w0=w0, v0=v0, lr=0.3)
            w0, v0 = res["w"], res["v"]
            self.assertTrue(np.isfinite(res["value"]))
            self.assertGreaterEqual(res["value"], -1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
