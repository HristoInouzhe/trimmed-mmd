"""Tests for the revision-v2 methods: the plug-in threshold and the betting certificate.

Added for revision v2 (plan items V8-V9). The expected values are the research
code's outputs (``explore/sharper_threshold/thresholds.py`` and
``explore/split_adaptive/{sacert,fold1,seqcert}.py``), hard-coded here so that
the suite does not depend on the untracked research folder:

* thresholds: ``explore/sharper_threshold/out/tables.md`` (three decimals) and
  full-precision values printed by the research functions;
* certificate: replicates of ``explore/split_adaptive/out/synth_*.jsonl`` and
  RHC split 0 of ``out/real.jsonl`` (same seeds, same random streams).

When the research folder is present, two extra tests compare the package with
it directly on a grid. Set ``TRIMMEDMMD_SLOW=1`` to also run the RHC
reproduction (needs ``data_raw/``; about 1 min on 4 threads) and the boundary
Monte Carlo of the full pipeline.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd import dgp  # noqa: E402
from trimmedmmd import threshold as T  # noqa: E402
from trimmedmmd.estimate import alpha_hat_from_path, bisect_alpha_hat  # noqa: E402
from trimmedmmd.kernels import gaussian_kernel, median_heuristic  # noqa: E402
from trimmedmmd.splitting import (  # noqa: E402
    betting_certificate,
    betting_test,
    evaluate_direction,
    fold1_witness_values,
    lcb_bernstein_2s,
    lower_bound_betting,
    pair_statistics,
    ramp,
    ramp_cut_points,
    ramp_prior_var,
    lower_bound_sequential,
    sequential_betting_certificate,
    sequential_certificate,
    sequential_horizon,
    split_certificate,
    split_test,
    witness_direction,
)

SLOW = os.environ.get("TRIMMEDMMD_SLOW", "") not in ("", "0")
EXPLORE = ROOT / "explore"
HAVE_EXPLORE_T = (EXPLORE / "sharper_threshold" / "thresholds.py").exists()
HAVE_EXPLORE_S = (EXPLORE / "split_adaptive" / "sacert.py").exists()


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------

# explore/sharper_threshold/out/tables.md, n = m = 300, delta = 0.05, Gaussian
TABLE_300 = {
    # alpha: (paper, A, P_closed, P, P_coupled, new)
    0.00: (0.804, 0.489, 0.223, 0.223, 0.223, 0.223),
    0.05: (0.846, 0.514, 0.361, 0.325, 0.302, 0.302),
    0.10: (0.893, 0.543, 0.411, 0.370, 0.336, 0.336),
    0.20: (1.005, 0.611, 0.501, 0.440, 0.391, 0.391),
    0.30: (1.148, 0.698, 0.594, 0.510, 0.447, 0.447),
    0.50: (1.607, 0.977, 0.838, 0.672, 0.581, 0.581),
}


class TestPluginThresholdValues(unittest.TestCase):
    """Numerical equality with the research code."""

    def test_table_n300(self):
        for a, row in TABLE_300.items():
            got = (T.tau(300, 300, a, 0.05), T.tau_A(300, 300, a, 0.05),
                   T.tau_plugin_closed(300, 300, a, 0.05),
                   T.tau_plugin_numeric(300, 300, a, 0.05),
                   T.tau_plugin_coupled(300, 300, a, 0.05),
                   T.tau_plugin(300, 300, a, 0.05))
            for g, want in zip(got, row):
                self.assertAlmostEqual(round(float(g), 3), want, places=9, msg=f"alpha={a}")

    def test_full_precision_fixtures(self):
        # printed by explore/sharper_threshold/thresholds.py
        self.assertAlmostEqual(T.tau_plugin_numeric(300, 300, 0.2, 0.05), 0.4402273281346131, places=12)
        self.assertAlmostEqual(T.tau_plugin_coupled(300, 300, 0.2, 0.05), 0.39141928012970284, places=12)
        val, det = T.tau_plugin_numeric(300, 300, 0.2, 0.05, detail=True)
        self.assertEqual(det["split"], 0.6)
        self.assertAlmostEqual(det["t1"], 0.2732182939009827, places=12)
        self.assertAlmostEqual(det["t2"], 0.11809322062601633, places=12)
        # the real-data sample sizes (RHC, NHANES child / age / BMI)
        fx = {(2184, 3551): (0.07426150112537325, 0.13850713441880153, 0.15636607366328886),
              (1969, 2300): (0.08384444586341007, 0.15564015167270226, 0.1756864890158013),
              (2519, 2300): (0.07875874611929681, 0.1460332258099119, 0.16481898401763911),
              (1627, 2008): (0.09109072576895988, 0.16934289411057768, 0.19119443883348688)}
        for (n, m), (c0, c25, p25) in fx.items():
            self.assertAlmostEqual(T.tau_plugin_coupled(n, m, 0.0, 0.05), c0, places=12)
            self.assertAlmostEqual(T.tau_plugin_coupled(n, m, 0.25, 0.05), c25, places=12)
            self.assertAlmostEqual(T.tau_plugin_numeric(n, m, 0.25, 0.05), p25, places=12)

    def test_general_kernel_constants(self):
        # NOTE Section 4: general bounded kernel (Dphi = rho = 2, v2 = 1)
        kb = T.KernelConstants.bounded(1.0)
        self.assertAlmostEqual(T.tau_A(300, 300, 0.0, 0.05, kb), 0.5472670656844669, places=12)
        self.assertAlmostEqual(T.tau_A(300, 300, 0.2, 0.05, kb), 0.6840838321055837, places=12)
        self.assertAlmostEqual(T.tau_plugin_closed(300, 300, 0.0, 0.05, kb), 0.28150734991552906, places=12)
        self.assertAlmostEqual(T.tau_plugin_closed(300, 300, 0.2, 0.05, kb), 0.6703938100744119, places=12)
        self.assertAlmostEqual(T.tau_plugin_numeric(300, 300, 0.0, 0.05, kb), 0.22983236318844597, places=12)
        self.assertAlmostEqual(T.tau_plugin_numeric(300, 300, 0.2, 0.05, kb), 0.5132471874169018, places=12)
        # "auto" never uses the coupled constants for a kernel without them
        self.assertEqual(T.tau_plugin(300, 300, 0.2, 0.05, kb),
                         min(T.tau_A(300, 300, 0.2, 0.05, kb),
                             T.tau_plugin_closed(300, 300, 0.2, 0.05, kb),
                             T.tau_plugin_numeric(300, 300, 0.2, 0.05, kb)))
        with self.assertRaises(ValueError):
            T.tau_plugin_coupled(300, 300, 0.2, 0.05, kb)
        with self.assertRaises(ValueError):
            T.tau_plugin(300, 300, 0.2, 0.05, kb, variant="sharp_coupled")

    def test_closed_form_at_zero(self):
        # NOTE Section 4: tau_P(0) = sqrt(2/n) (1 + sqrt(log(1/delta))) at n = m (Gaussian)
        for n in (100, 300, 2000):
            want = math.sqrt(2.0 / n) * (1.0 + math.sqrt(math.log(1 / 0.05)))
            self.assertAlmostEqual(T.tau_plugin_closed(n, n, 0.0, 0.05), want, places=12)

    def test_twosided_and_comparisons(self):
        for n, m in ((300, 300), (200, 500), (2184, 3551)):
            for a in (0.0, 0.1, 0.3, 0.6):
                for d in (0.01, 0.05, 0.1):
                    tA = T.tau_A(n, m, a, d)
                    self.assertLessEqual(tA, T.tau(n, m, a, d))            # Theorem A <= paper
                    self.assertLessEqual(T.tau_A(n, m, a, d, "bounded"), T.tau(n, m, a, d))
                    self.assertEqual(T.tau_twosided(n, m, a, d), T.tau_A(n, m, a, d / 2))
                    self.assertLessEqual(T.tau_new(n, m, a, d), tA)
                    self.assertLessEqual(T.tau_new(n, m, a, d), T.tau_plugin_closed(n, m, a, d))
        # tau_A is linear in beta
        self.assertAlmostEqual(T.tau_A(300, 300, 0.5, 0.05), 2 * T.tau_A(300, 300, 0.0, 0.05), places=12)

    def test_rademacher_abs_mean_exact(self):
        for n in (1, 2, 3, 7, 10):
            # exact enumeration
            from itertools import product
            ex = np.mean([abs(sum(s)) for s in product((-1, 1), repeat=n)]) / n
            self.assertAlmostEqual(T.rademacher_abs_mean(n), ex, places=12)
        self.assertLessEqual(T.rademacher_abs_mean(301), 301 ** -0.5)


class TestPluginThresholdMonotone(unittest.TestCase):
    """Lemma 1 needs tau nondecreasing in alpha: one event for every alpha."""

    def test_monotone_grid(self):
        grid = np.round(np.linspace(0.0, 0.9, 46), 4)
        for (n, m), kc in (((300, 300), None), ((300, 300), "bounded"), ((2184, 3551), None)):
            for fn in (T.tau_new, T.tau_plugin_closed, T.tau_A,
                       lambda n_, m_, a_, d_, k_: T.tau_plugin_numeric(n_, m_, a_, d_, k_)):
                vals = np.array([fn(n, m, a, 0.05, kc) for a in grid])
                self.assertTrue(np.all(np.isfinite(vals)))
                self.assertTrue(np.all(np.diff(vals) >= -1e-12), msg=f"{fn} n={n} m={m} {kc}")


class TestThresholdInEstimator(unittest.TestCase):
    def test_default_unchanged_and_plugin(self):
        alphas = np.round(np.arange(0.0, 0.76, 0.05), 3)
        d = 0.45 * np.exp(-6 * alphas)
        n, m = 1969, 2300
        a_old = alpha_hat_from_path(alphas, d, n, m, 0.05)
        self.assertEqual(a_old, alpha_hat_from_path(alphas, d, n, m, 0.05, threshold="paper"))
        thr_old = np.array([T.tau(n, m, a, 0.05) for a in alphas])
        self.assertEqual(a_old, float(alphas[np.nonzero(d <= thr_old)[0][0]]))
        a_new = alpha_hat_from_path(alphas, d, n, m, 0.05, threshold="plugin")
        thr_new = np.array([T.tau_new(n, m, a, 0.05) for a in alphas])
        self.assertEqual(a_new, float(alphas[np.nonzero(d <= thr_new)[0][0]]))
        self.assertLess(a_old, a_new)
        # bisection agrees with the grid crossing for a monotone path
        f = lambda a: 0.45 * math.exp(-6 * a)  # noqa: E731
        ab = bisect_alpha_hat(f, n=n, m=m, delta=0.05, threshold="plugin", tol=1e-4)
        self.assertLessEqual(f(ab), T.tau_new(n, m, ab, 0.05) + 1e-12)
        self.assertGreater(f(ab - 2e-4), T.tau_new(n, m, ab - 2e-4, 0.05))
        # threshold_fn still wins
        self.assertEqual(alpha_hat_from_path(alphas, d, n, m, 0.05, threshold="plugin",
                                             threshold_fn=lambda a: 1.0), 0.0)

    def test_reject_plugin(self):
        t = T.tau_new(300, 300, 0.2, 0.05)
        self.assertTrue(T.reject_plugin(t + 1e-9, 300, 300, 0.2, 0.05))
        self.assertFalse(T.reject_plugin(t, 300, 300, 0.2, 0.05))
        f = T.threshold_function("plugin", 300, 300, 0.05)
        self.assertEqual(f(0.2), t)
        self.assertEqual(T.threshold_function("paper", 300, 300, 0.05)(0.2), T.tau(300, 300, 0.2, 0.05))
        self.assertEqual(T.threshold_function("twosided", 300, 300, 0.05)(0.2),
                         T.tau_A(300, 300, 0.2, 0.025))

    @unittest.skipUnless((ROOT / "results" / "tables" / "r1_paths.csv").exists(), "no r1_paths.csv")
    def test_realdata_paths(self):
        # explore/sharper_threshold/out/realdata_summary.csv (tau_new = coupled column):
        # alpha-hat and the largest rejected grid level on the published paths
        import pandas as pd
        paths = pd.read_csv(ROOT / "results" / "tables" / "r1_paths.csv")
        summ = pd.read_csv(ROOT / "results" / "tables" / "r1_summary.csv").set_index("dataset")
        want = {"rhc": (0.05, 0.0), "nhanes_child": (0.45, 0.40), "nhanes_age": (0.15, 0.10),
                "nhanes_bmi": (0.0, None)}
        for ds, (ahat, last) in want.items():
            g = paths[paths.dataset == ds]
            n, m = int(summ.loc[ds, "n"]), int(summ.loc[ds, "m"])
            al, d = g.alpha.to_numpy(), g.d_hat.to_numpy()
            self.assertAlmostEqual(alpha_hat_from_path(al, d, n, m, 0.05, threshold="plugin"), ahat)
            thr = np.array([T.tau_new(n, m, a, 0.05) for a in al])
            rej = al[d > thr]
            self.assertEqual(float(rej.max()) if rej.size else None, last)


@unittest.skipUnless(HAVE_EXPLORE_T, "research folder explore/ not present")
class TestAgainstResearchThresholds(unittest.TestCase):
    def test_grid(self):
        sys.path.insert(0, str(EXPLORE / "sharper_threshold"))
        import thresholds as R  # noqa: E402
        G = R.GAUSS
        for n, m in ((300, 300), (200, 500)):
            for a in (0.0, 0.05, 0.2, 0.5):
                self.assertEqual(T.tau_A(n, m, a, 0.05), R.tau_A(n, m, a, 0.05, **G))
                self.assertEqual(T.tau_plugin_closed(n, m, a, 0.05), R.tau_P_closed(n, m, a, 0.05, **G))
                self.assertEqual(T.tau_plugin_numeric(n, m, a, 0.05), R.tau_P(n, m, a, 0.05, **G))
                self.assertEqual(T.tau_plugin_coupled(n, m, a, 0.05),
                                 R.tau_P_coupled(n, m, a, 0.05, kappa=1.0))
                # Corollary C also includes the closed form, which equals the McDiarmid
                # inversion at alpha = 0 without the root finder's +1e-6 margin
                ref = min(R.tau_A(n, m, a, 0.05, **G), R.tau_P(n, m, a, 0.05, **G),
                          R.tau_P_coupled(n, m, a, 0.05, kappa=1.0))      # analyze.thr("new")
                self.assertAlmostEqual(T.tau_new(n, m, a, 0.05), ref, delta=1.01e-6)


# ---------------------------------------------------------------------------
# betting certificate
# ---------------------------------------------------------------------------

class TestBettingPrimitives(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.G = rng.beta(2, 3, size=137)
        self.H = rng.beta(1.5, 3, size=211)

    def test_fixtures_research(self):
        # printed by explore/split_adaptive/sacert.py on the same inputs
        Z0 = pair_statistics(self.G, self.H)
        self.assertAlmostEqual(Z0.mean(), 0.06991840122634439, places=14)
        self.assertAlmostEqual(pair_statistics(self.H[:100], self.G).mean(), -0.07808096594749422, places=14)
        r = betting_test(Z0, -1.0, 0.05, c=0.5, prior_var=0.07)
        self.assertTrue(r.certified)
        self.assertAlmostEqual(r.log_max_wealth, 3.6526588742302524, places=12)
        self.assertAlmostEqual(r.log_final_wealth, 3.6144112892172937, places=12)
        r = betting_test(Z0 - 0.1, -1.1, 0.05)
        self.assertFalse(r.certified)
        self.assertAlmostEqual(r.log_max_wealth, 0.2399797641553496, places=12)
        self.assertAlmostEqual(r.log_final_wealth, -2.8748277549945778, places=12)
        self.assertAlmostEqual(lcb_bernstein_2s(self.G, 1.0, -self.H, 1.0, 0.05), -0.06454776550824064,
                               places=12)

    def test_pairs_are_unbiased_blocks(self):
        Z = pair_statistics(self.G, self.H)
        self.assertEqual(Z.size, 137)
        blocks = np.array_split(self.H, 137)
        self.assertTrue(all(len(b) in (1, 2) for b in blocks))
        np.testing.assert_allclose(Z, self.G - np.array([b.mean() for b in blocks]), atol=1e-15)

    def test_ramp(self):
        f = np.array([-1.0, 0.0, 0.25, 0.5, 2.0])
        np.testing.assert_allclose(ramp(f, 0.5, 0.0), [0, 0, 0.5, 1, 1])

    def test_ville_bound_scalar_boundary(self):
        """Exact-boundary null: E Z = alpha. Level <= delta + 3 SE on 2000 replicates."""
        rng = np.random.default_rng(7)
        for a, N in ((0.0, 50), (0.3, 200)):
            rej = 0
            reps = 2000
            for _ in range(reps):
                G = (rng.random(N) < 0.5 + a / 2).astype(float)
                H = (rng.random(N) < 0.5 - a / 2).astype(float)
                rej += betting_test(pair_statistics(G, H) - a, -(1 + a), 0.05, prior_var=0.5).certified
            se = math.sqrt(0.05 * 0.95 / reps)
            self.assertLessEqual(rej / reps, 0.05 + 3 * se)

    def test_crossed_cut_points_never_reject(self):
        r = betting_certificate(np.ones(10), np.zeros(10), 0.1, s=0.0, sp=0.5)
        self.assertFalse(r.certified)


def _synth_first_level(scen, n, seed, a):
    """The random stream of explore/split_adaptive/synth.py up to the first level."""
    sc = dgp.make(scen)
    rng = np.random.default_rng(seed)
    X, Y = sc.sample(n, rng, "P"), sc.sample(n, rng, "Q")
    ix, iy = rng.permutation(n), rng.permutation(n)
    n1 = n // 2
    X1, X2, Y1, Y2 = X[ix[:n1]], X[ix[n1:]], Y[iy[:n1]], Y[iy[n1:]]
    sigma = median_heuristic(X1, Y1, rng=rng)
    w, v, norm = witness_direction(X1, Y1, a, sigma, rng=rng)
    u1, v1 = fold1_witness_values(w, v, X1, Y1, sigma, norm)
    s, sp = ramp_cut_points(u1, v1, a)
    u2 = evaluate_direction(w, v, X1, Y1, X2, sigma, norm)
    v2 = evaluate_direction(w, v, X1, Y1, Y2, sigma, norm)
    pv = ramp_prior_var(u1, v1, s, sp, len(X2), len(Y2))
    return (sigma, norm, betting_certificate(u2, v2, a, s, sp, 0.05, prior_var=pv),
            betting_certificate(u2, v2, a, s, sp, 0.05, method="bernstein"))


class TestCertificateReproducesResearch(unittest.TestCase):
    """Replicates of explore/split_adaptive/out/synth_disjoint_d2_a20{,_boundary}.jsonl."""

    # (seed, alpha, sigma, norm, s, sp, Delta_hat, log final wealth, betting, bernstein)
    CASES = [
        (70000, 0.2, 2.6834299223608564, 0.0704417315162715, 0.13664157899245016,
         -0.24793725032898853, 0.030864279197380273, -5.052092242733201, False, False),
        (70002, 0.2, 2.4249270583042866, 0.12483864815938145, 0.2476627447725063,
         -0.6189630956827276, 0.23696877859016263, 0.10093931889487498, False, False),
        (70000, 0.0, 2.6834299223608564, 0.2995655037847595, 0.5934804036819827,
         -0.6546819058803925, 0.1520112997067452, 3.2740779295270777, True, False),
        (70001, 0.0, 2.3255737709687954, 0.3346507094632849, 0.6497576624343023,
         -0.5868356152006649, 0.2098067295261492, 4.585372727198571, True, False),
        (70004, 0.0, 1.9801984105801251, 0.18158710234378037, 0.6466150754850907,
         -0.43087341761045883, 0.24701350546531647, 5.409656030250537, True, False),
    ]

    def test_replicates(self):
        for seed, a, sig, norm, s, sp, dh, logk, bet, ber in self.CASES:
            sigma, nrm, rb, rs = _synth_first_level("disjoint_d2_a20", 100, seed, a)
            tol = 1e-7      # bit-exact on one BLAS thread; FISTA may differ in the last digits otherwise
            self.assertAlmostEqual(sigma, sig, places=12)
            self.assertAlmostEqual(nrm, norm, delta=tol)
            self.assertAlmostEqual(rb.s, s, delta=tol)
            self.assertAlmostEqual(rb.sp, sp, delta=tol)
            self.assertAlmostEqual(rb.delta_hat, dh, delta=tol)
            self.assertAlmostEqual(rb.log_final_wealth, logk, delta=1e-5)
            self.assertEqual(rb.certified, bet)
            self.assertEqual(rs.certified, ber)


class TestSplitCertificateAPI(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 300
        self.X = rng.standard_normal((n, 2))
        self.Y = rng.standard_normal((n, 2))
        self.Y[: int(0.3 * n)] += 6.0          # TV = 0.3

    def test_same_stream_as_split_test(self):
        for a in (0.0, 0.1, 0.25):
            r_old = split_test(self.X, self.Y, a, rng=np.random.default_rng(11), n_boot=200)
            r_new = split_certificate(self.X, self.Y, a, rng=np.random.default_rng(11), n_boot=200)
            self.assertEqual(r_new.boot_certified, r_old.certified)
            self.assertEqual(r_new.boot_lower, r_old.lower_ci)
            self.assertEqual(r_new.separation, r_old.separation)

    def test_lower_bound_down_closed_and_valid(self):
        grid = np.round(np.arange(0.0, 0.5, 0.05), 3)
        b, rows = lower_bound_betting(self.X, self.Y, grid, rng=np.random.default_rng(5),
                                      methods=("betting", "bernstein", "bootstrap"), n_boot=200)
        for mth, lo in b.items():
            self.assertLessEqual(lo, 0.3 + 1e-9, msg=mth)            # truth 0.3 (lower bound)
            # down-closed: every level <= lo is certified
            for r in rows:
                if r["alpha"] <= lo:
                    self.assertTrue(r[mth], msg=mth)
        self.assertGreaterEqual(b["betting"], 0.1)
        b0, rows0 = lower_bound_betting(self.X, self.Y, grid, rng=np.random.default_rng(5),
                                        witness="alpha0")
        self.assertLessEqual(b0["betting"], 0.3 + 1e-9)
        # same split: the alpha = 0 level of both runs uses the same witness
        self.assertEqual(rows0[0]["betting"], rows[0]["betting"])
        self.assertEqual(rows0[0]["s"], rows[0]["s"])
        # none_value convention of the fairness scripts
        rng = np.random.default_rng(17)
        bn, _ = lower_bound_betting(rng.standard_normal((200, 2)), rng.standard_normal((200, 2)),
                                    [0.0, 0.05], rng=np.random.default_rng(5), none_value=-1.0)
        self.assertEqual(bn["betting"], -1.0)

    def test_null_validity_alpha0(self):
        """P = Q at alpha = 0: 200 replicates, rejection <= delta + 3 SE, for the
        betting certificate and the plug-in-threshold test (D-hat_0 = MMD)."""
        reps, n = 200, 100
        rej_bet = rej_tau = 0
        t0 = T.tau_new(n, n, 0.0, 0.05)
        for i in range(reps):
            rng = np.random.default_rng(90000 + i)
            X, Y = rng.standard_normal((n, 2)), rng.standard_normal((n, 2))
            rej_bet += split_certificate(X, Y, 0.0, rng=rng).certified
            sig = median_heuristic(X, Y, rng=rng)
            mmd2 = (gaussian_kernel(X, X, sig).mean() + gaussian_kernel(Y, Y, sig).mean()
                    - 2 * gaussian_kernel(X, Y, sig).mean())
            rej_tau += math.sqrt(max(mmd2, 0.0)) > t0
        lim = 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps)
        self.assertLessEqual(rej_bet / reps, lim)
        self.assertLessEqual(rej_tau / reps, lim)


class TestSequentialHorizon(unittest.TestCase):
    """The WSR horizon N of the sequential certificate is fixed in advance:
    a batch whose cut points cross bets lam = 0 instead of shrinking N."""

    EDGES = (0, 0.25, 0.5, 1)

    @staticmethod
    def _fit(Xf, Yf, a):
        f = lambda Z: np.tanh(np.asarray(Z)[:, 0])  # noqa: E731
        return f, f(Xf), f(Yf)

    def test_horizon_is_data_free(self):
        n, m = 203, 157
        bx = [round(e * n) for e in self.EDGES]
        by = [round(e * m) for e in self.EDGES]
        want = sum(min(bx[b + 1] - bx[b], by[b + 1] - by[b]) for b in (1, 2))
        self.assertEqual(sequential_horizon(n, m, self.EDGES), want)
        rng = np.random.default_rng(4)
        # (a) every batch has a valid cut: X shifted right of Y
        Xa, Ya = rng.normal(1.0, 1, (n, 1)), rng.normal(-1.0, 1, (m, 1))
        # (b) batch 0 has X LEFT of Y, so the cut points for batch 1 cross; later rows as (a)
        Xb, Yb = Xa.copy(), Ya.copy()
        Xb[:bx[1]] -= 4.0
        Yb[:by[1]] += 4.0
        ra = sequential_betting_certificate(Xa, Ya, 0.1, self.EDGES, 1.0, fit=self._fit,
                                            return_detail=True)
        rb = sequential_betting_certificate(Xb, Yb, 0.1, self.EDGES, 1.0, fit=self._fit,
                                            return_detail=True)
        self.assertEqual(ra[2]["skipped_batches"], [])
        self.assertEqual(rb[2]["skipped_batches"], [1])
        self.assertEqual(ra[2]["horizon"], want)
        self.assertEqual(rb[2]["horizon"], want)          # not the pairs actually bet on
        self.assertLess(rb[2]["pairs_bet"], want)
        # (b) by hand: only batch 2 bets, with the fixed horizon N = want
        f = lambda Z: np.tanh(Z[:, 0])  # noqa: E731
        u1, v1 = f(Xb[:bx[2]]), f(Yb[:by[2]])
        s_, sp_ = ramp_cut_points(u1, v1, 0.1)
        Xc, Yc = Xb[bx[2]:], Yb[by[2]:]
        Z = pair_statistics(ramp(f(Xc), s_, sp_), ramp(f(Yc), s_, sp_)) - 0.1
        pv = ramp_prior_var(u1, v1, s_, sp_, len(Xc), len(Yc))
        L = math.log(1 / 0.05)
        logK, ssd, cnt, mus = 0.0, 0.0, 0, 0.0
        for y in Z:
            mu = mus / (20.0 + cnt)
            sig2 = max((20.0 * pv + ssd) / (20.0 + cnt), 1e-8)
            logK += math.log1p(min(0.5 / 1.1, math.sqrt(2 * L / (want * sig2))) * y)
            ssd += (y - mu) ** 2
            mus += y
            cnt += 1
        self.assertAlmostEqual(rb[1], logK, places=12)


class TestPrintedSampleSizes(unittest.TestCase):
    """eq:samplesize and eq:sharpsize as printed in jmlr/sections/11_applications.tex."""

    def test_samplesize_closed_form(self):
        C0 = T.samplesize_constant(0.05)
        self.assertAlmostEqual(C0, 1 + math.sqrt(math.log(20)), places=14)
        self.assertAlmostEqual(round(C0, 2), 2.73)
        for n in (50, 300, 2000):
            # tau_{n,n}(0, delta) = C_0 sqrt(2/n) (the closed form binds at alpha = 0)
            self.assertAlmostEqual(T.tau_new(n, n, 0.0, 0.05), C0 * math.sqrt(2 / n), places=12)
        for d0 in (0.069, 0.111, 0.406):
            nf = T.fire_sample_size(d0, 0.05)
            self.assertAlmostEqual(nf, 2 * (C0 / d0) ** 2, places=9)
            k = math.ceil(nf)
            self.assertLess(T.tau_new(k, k, 0.0, 0.05), d0)          # fires above the bound
            self.assertGreaterEqual(T.tau_new(k - 1, k - 1, 0.0, 0.05), d0)

    def test_sharpsize_numerical(self):
        for d_a, a in ((0.0694, 0.10), (0.2097, 0.40)):
            na = T.sharp_sample_size(d_a, a, 0.05)
            self.assertLess(T.tau_new(na, na, a, 0.05), d_a)
            self.assertGreaterEqual(T.tau_new(na - 1, na - 1, a, 0.05), d_a)
        self.assertEqual(T.sharp_sample_size(0.0, 0.1, 0.05), -1)


class TestSequentialCertificate(unittest.TestCase):
    """The paper's main split certificate (Theorem thm:betting, e = (0, 1/4, 1/2, 1))."""

    def test_lower_bound_and_null(self):
        rng = np.random.default_rng(3)
        n = 300
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((n, 2))
        Y[:90] += 6.0                                   # TV = 0.3
        grid = np.round(np.arange(0.0, 0.5, 0.05), 3)
        b, rows = lower_bound_sequential(X, Y, grid, rng=np.random.default_rng(5))
        self.assertLessEqual(b, 0.3 + 1e-9)
        self.assertGreaterEqual(b, 0.1)
        self.assertTrue(all(r["sequential"] for r in rows if r["alpha"] <= b))
        self.assertEqual(rows[0]["seq_horizon"], sequential_horizon(n, n, (0, .25, .5, 1)))
        # same answer through lower_bound_betting, next to the single-split columns
        bb, _ = lower_bound_betting(X, Y, grid, rng=np.random.default_rng(5),
                                    methods=("sequential", "betting"))
        self.assertEqual(bb["sequential"], b)
        # P = Q at alpha = 0: 200 replicates, rejection <= delta + 3 SE
        rej = 0
        for i in range(200):
            r2 = np.random.default_rng(91000 + i)
            Xn, Yn = r2.standard_normal((100, 2)), r2.standard_normal((100, 2))
            rej += sequential_certificate(Xn, Yn, 0.0, rng=r2).certified
        self.assertLessEqual(rej / 200, 0.05 + 3 * math.sqrt(0.05 * 0.95 / 200))


@unittest.skipUnless(SLOW, "set TRIMMEDMMD_SLOW=1")
class TestRHCSplitZero(unittest.TestCase):
    """RHC split 0 of explore/split_adaptive/out/real.jsonl (seed default_rng(1000))."""

    # alpha: (T_hat, s, sp, Delta_hat, bootstrap, betting, bernstein LCB)
    WANT = {0.25: (0.015256727825879182, 0.19053205791271624, 0.03194381686396809,
                   0.3402737888667651, True, True, 0.29696738664490263),
            0.30: (0.0033961643109989836, 0.11691144807736613, 0.05815172613944978,
                   0.37938548571307334, False, True, 0.3317373267427577)}

    def test_levels(self):
        try:
            from trimmedmmd.realdata import load_rhc
            X, Y, _, _ = load_rhc()
        except Exception as e:  # pragma: no cover
            self.skipTest(f"RHC data not available: {e}")
        from trimmedmmd.splitting import ramp_certificate_on_fold, split_fold
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        seed = int(np.random.default_rng(1000).integers(0, 2**31 - 1))
        self.assertEqual(seed, 437485271)
        for a, (th, s, sp, dh, boot, bet, lcb) in self.WANT.items():
            f = split_fold(X, Y, a, rng=np.random.default_rng(seed), n_boot=1000)
            rb = ramp_certificate_on_fold(f, a, 0.05, "betting")
            rs = ramp_certificate_on_fold(f, a, 0.05, "bernstein")
            for got, want in ((f.T_hat, th), (rb.s, s), (rb.sp, sp), (rb.delta_hat, dh), (rs.lcb, lcb)):
                self.assertAlmostEqual(got, want, delta=1e-5)
            self.assertEqual(f.boot_certified, boot)
            self.assertEqual(rb.certified, bet)


if __name__ == "__main__":
    unittest.main()
