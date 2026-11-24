"""Tests for unequal trimming fractions (trimmedmmd.unequal).

The null H0(alpha_x, alpha_y), "trimming P by alpha_x and Q by alpha_y can leave a common remainder", is
E_zeta(P || Q) <= alpha_x with zeta = (1 - alpha_x) / (1 - alpha_y). The tests check that

* with alpha_y = alpha_x the certificate is the paper's AoS certificate number for number
  (scripts/aos_main_realdata.py::level_record, cut rule "order", and the aos_fallback functions it calls);
* the closed form E_zeta(N(mu, 1) || N(-mu, 1)) = Phi(mu - log(zeta)/(2 mu)) - zeta Phi(-mu - log(zeta)/(2 mu))
  agrees with quadrature (p > zeta q exactly when x > log(zeta)/(2 mu));
* the pair statistics stay in [-(zeta + alpha_x), 1 - alpha_x] and every wealth factor 1 + lambda Z stays positive
  with the fraction cap c / (zeta + alpha_x), while the symmetric cap c / (1 + alpha_x) can break it for zeta > 1;
* the grid bounds read upward on one order and stop at the first failure.

With TRIMMEDMMD_SLOW=1 a Monte Carlo checks the level at the true frontier and the power below it, in d = 1 with
P = N(mu, 1), Q = N(-mu, 1) (TRIMMEDMMD_MC_REPS replicates, default 1000; TRIMMEDMMD_MC_WORKERS processes, default 1;
about 2 s per replicate and process at n = m = 400).

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trimmedmmd import (frontier_lower_bound, hockey_stick_lower_bound,  # noqa: E402
                        pair_certificate)
from trimmedmmd.aos_fallback import bet, sequential_all_rules  # noqa: E402
from trimmedmmd.splitting import (cut_ranks, pair_statistics, pair_variance,  # noqa: E402
                                  ramp_cut_points, sequential_certificate, sequential_order)

SLOW = os.environ.get("TRIMMEDMMD_SLOW", "") not in ("", "0")
SCRIPT = ROOT / "scripts" / "aos_main_realdata.py"
FAST = {"max_iter": 300}     # solver iterations for the structural tests (any witness is valid)


def hockey_stick_gauss(mu: float, zeta: float) -> float:
    """E_zeta(N(mu, 1) || N(-mu, 1)), mu > 0: the likelihood ratio p/q = exp(2 mu x) exceeds zeta exactly when
    x > x0 = log(zeta) / (2 mu), so E_zeta = P(X > x0) - zeta Q(Y > x0)."""
    x0 = math.log(zeta) / (2.0 * mu)
    return float(norm.cdf(mu - x0) - zeta * norm.cdf(-mu - x0))


def hockey_stick_quad(mu: float, zeta: float) -> float:
    """The integral of (p - zeta q)_+ by adaptive quadrature, split at the crossing."""
    f = lambda x: max(norm.pdf(x, mu, 1.0) - zeta * norm.pdf(x, -mu, 1.0), 0.0)  # noqa: E731
    x0 = math.log(zeta) / (2.0 * mu)
    a, _ = quad(f, x0, np.inf, epsabs=1e-14, epsrel=1e-12, limit=200)
    b, _ = quad(f, -np.inf, x0, epsabs=1e-14, epsrel=1e-12, limit=200)
    return a + b


def frontier_gauss(mu: float, alpha_x: float):
    """(zeta*, alpha_y*) with E_zeta*(N(mu,1) || N(-mu,1)) = alpha_x and alpha_y* = 1 - (1 - alpha_x) / zeta*."""
    z = brentq(lambda z: hockey_stick_gauss(mu, z) - alpha_x, 1e-8, 1e8, xtol=1e-15, rtol=1e-15, maxiter=500)
    return z, 1.0 - (1.0 - alpha_x) / z


def load_level_record():
    """scripts/aos_main_realdata.py as a module (its data loader is bypassed through _DATA)."""
    spec = importlib.util.spec_from_file_location("_aos_main_realdata_for_tests", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toy_pair(n, m, d, shift, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((m, d))
    Y[:, 0] += shift
    return X, Y


class TestSymmetricReduction(unittest.TestCase):
    """(a) alpha_y = alpha_x: zeta = 1 exactly and the certificate is the paper's AoS certificate."""

    LEVELS = (0.0, 0.1, 0.2, 0.3, 0.5)

    def assert_same(self, pc, ref):
        self.assertEqual(pc.zeta, 1.0)
        self.assertEqual(pc.certified, ref["certified"])
        self.assertEqual(pc.log_final_wealth, ref["log_final"])     # exact equality: same numbers
        self.assertEqual(pc.log_max_wealth, ref["log_max"])
        self.assertEqual(pc.pairs_bet, ref["pairs_bet"])
        self.assertEqual(pc.cuts, ref["cuts"])

    @unittest.skipUnless(SCRIPT.exists(), "scripts/aos_main_realdata.py not present")
    def test_level_record(self):
        mod = load_level_record()
        seed = 1003
        split_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))   # as level_record
        for name, (n, m) in (("toy_mirror", (300, 260)), ("toy_blocks", (220, 300))):
            X, Y = toy_pair(n, m, 2, 0.9, seed=n + m)
            mod._DATA[name] = (X, Y, False)
            decisions = []
            for a in self.LEVELS:
                rec = mod.level_record(name, seed, a)
                pc = pair_certificate(X, Y, a, a, delta=mod.DELTA, kernel=mod.KERNEL, split_seed=split_seed,
                                      cut_rule="order")
                self.assertEqual(pc.sigma, rec["sigma"])
                self.assertEqual(pc.horizon, rec["horizon"])
                self.assertEqual([b["norm"] for b in pc.batches], rec["norms"])
                self.assert_same(pc, rec["order"])
                decisions.append(pc.certified)
            self.assertIn(True, decisions)      # the comparison covers both outcomes
            self.assertIn(False, decisions)

    def test_aos_fallback_path(self):
        # the functions level_record calls, through sequential_all_rules (rule "own", cut rule "order"), with
        # batch-0 standardization on
        X, Y = toy_pair(240, 280, 3, 0.8, seed=5)
        X = 3.0 * X + 1.0
        for a in self.LEVELS:
            order = sequential_order(X, Y, np.random.default_rng(77), standardize_batch0=True)
            ref = sequential_all_rules(order, a, kernel="laplace", rules=("own",), cut_rule="order")["own"]
            pc = pair_certificate(X, Y, a, a, split_seed=77, standardize_batch0=True)
            self.assert_same(pc, ref)

    def test_gaussian_matches_sequential_certificate(self):
        # with the Gaussian kernel the fits are the package's main path (splitting.sequential_certificate)
        X, Y = toy_pair(200, 240, 2, 1.0, seed=9)
        for a in (0.0, 0.2, 0.4):
            order = sequential_order(X, Y, np.random.default_rng(31))
            sc = sequential_certificate(X, Y, a, order=order, cut_rule="order")
            pc = pair_certificate(X, Y, a, a, kernel="gaussian", split_seed=31)
            self.assertEqual(pc.certified, sc.certified)
            self.assertEqual(pc.pairs_bet, sc.pairs_bet)
            if sc.pairs_bet:    # with no pair bet the package reports -inf, aos_fallback.bet 0.0
                self.assertEqual(pc.log_final_wealth, sc.log_final_wealth)

    def test_defaults_are_symmetric(self):
        rng = np.random.default_rng(3)
        u, v = rng.normal(size=57), rng.normal(size=81)
        G, H = rng.uniform(size=57), rng.uniform(size=81)
        for a in (0.0, 0.13, 0.5):
            for rule in ("linear", "order"):
                self.assertEqual(ramp_cut_points(u, v, a, rule=rule), ramp_cut_points(u, v, a, rule=rule, alpha_y=a))
            self.assertEqual(cut_ranks(57, 81, a), cut_ranks(57, 81, a, a))
        np.testing.assert_array_equal(pair_statistics(G, H), pair_statistics(G, H, zeta=1.0))
        np.testing.assert_array_equal(pair_statistics(H, G), pair_statistics(H, G, zeta=1.0))


class TestUnequalHelpers(unittest.TestCase):

    def test_cuts_at_alpha_y(self):
        rng = np.random.default_rng(4)
        u, v = rng.normal(size=100), rng.normal(size=80)
        ax, ay = 0.1, 0.55
        k_s, k_u = cut_ranks(100, 80, ax, ay)
        self.assertEqual((k_s, k_u), (90, 45))
        s, sp = ramp_cut_points(u, v, ax, rule="order", alpha_y=ay)
        self.assertEqual(s, np.sort(u)[k_s - 1])
        self.assertEqual(sp, np.sort(v)[k_u - 1])
        # genuine quantiles: at least a (1 - alpha_x) fraction of u at or below s, of v at or below sp at least alpha_y
        self.assertGreaterEqual(np.mean(u <= s), 1 - ax)
        self.assertGreaterEqual(np.mean(v <= sp), ay)
        self.assertLessEqual(np.mean(v < sp), ay)
        s2, sp2 = ramp_cut_points(u, v, ax, rule="linear", alpha_y=ay)
        self.assertEqual((s2, sp2), (float(np.quantile(u, 1 - ax)), float(np.quantile(v, ay))))

    def test_weighted_pairs_and_variance(self):
        rng = np.random.default_rng(8)
        zeta = 1.7
        G, H = rng.uniform(size=40), rng.uniform(size=120)       # equal blocks of 3
        Z = pair_statistics(G, H, zeta=zeta)
        np.testing.assert_allclose(Z, G - zeta * H.reshape(40, 3).mean(1), rtol=0, atol=1e-15)
        Zm = pair_statistics(H, G, zeta=zeta)                     # mirror
        np.testing.assert_allclose(Zm, H.reshape(40, 3).mean(1) - zeta * G, rtol=0, atol=1e-15)
        self.assertAlmostEqual(pair_variance(0.2, 0.1, 40, 120, zeta=zeta), 0.2 + zeta**2 * 0.1 / 3, places=15)
        self.assertAlmostEqual(pair_variance(0.2, 0.1, 120, 40, zeta=zeta), 0.2 / 3 + zeta**2 * 0.1, places=15)


class TestHockeyStickClosedForm(unittest.TestCase):
    """(b) E_zeta(N(mu,1) || N(-mu,1)) = Phi(mu - log(zeta)/(2mu)) - zeta Phi(-mu - log(zeta)/(2mu))."""

    def test_against_quadrature(self):
        worst = 0.0
        for mu in (0.1, 0.25, 0.5, 0.75, 1.5, 3.0):
            for zeta in (0.05, 0.3, 0.8, 1.0, 1.3, 1.92, 4.0, 25.0):
                e, q = hockey_stick_gauss(mu, zeta), hockey_stick_quad(mu, zeta)
                worst = max(worst, abs(e - q))
                self.assertAlmostEqual(e, q, delta=1e-10, msg=f"mu={mu}, zeta={zeta}")
                self.assertGreaterEqual(e, max(0.0, 1.0 - zeta) - 1e-15)      # E_zeta >= (1 - zeta)_+
        self.assertLess(worst, 1e-10)

    def test_equal_fractions_give_tv(self):
        for mu in (0.2, 0.5, 1.0):
            self.assertAlmostEqual(hockey_stick_gauss(mu, 1.0), 2 * norm.cdf(mu) - 1, places=14)

    def test_sup_attained_at_the_half_line(self):
        # E_zeta = sup_A P(A) - zeta Q(A); over half-lines (t, inf) the maximum is at t = x0
        mu, zeta = 0.5, 1.92
        x0 = math.log(zeta) / (2 * mu)
        ts = np.linspace(-4, 4, 4001)
        vals = norm.sf(ts, mu) - zeta * norm.sf(ts, -mu)
        self.assertLessEqual(vals.max(), hockey_stick_gauss(mu, zeta) + 1e-15)
        self.assertAlmostEqual(ts[vals.argmax()], x0, delta=2e-3)

    def test_frontier_is_the_common_remainder_boundary(self):
        # at (alpha_x, alpha_y*) the integral of min{p/(1-alpha_x), q/(1-alpha_y*)} is exactly 1
        for mu, ax in ((0.5, 0.2), (0.5, 0.1), (0.75, 0.3)):
            zeta, ay = frontier_gauss(mu, ax)
            self.assertTrue(0.0 <= ay < 1.0)
            for ay_, side in ((ay, 0), (ay - 0.05, -1), (ay + 0.05, 1)):
                f = lambda x: min(norm.pdf(x, mu) / (1 - ax), norm.pdf(x, -mu) / (1 - ay_))  # noqa: E731
                x0 = math.log((1 - ax) / (1 - ay_)) / (2 * mu)
                tot = quad(f, -np.inf, x0, epsabs=1e-14)[0] + quad(f, x0, np.inf, epsabs=1e-14)[0]
                if side == 0:
                    self.assertAlmostEqual(tot, 1.0, delta=1e-9)
                elif side < 0:
                    self.assertLess(tot, 1.0)            # below the frontier: no common remainder
                else:
                    self.assertGreater(tot, 1.0)


class TestBettingRange(unittest.TestCase):
    """(d) Z >= -(zeta + alpha_x), and 1 + lambda Z > 0 along the wealth path."""

    def test_pair_statistics_range(self):
        rng = np.random.default_rng(12)
        for zeta in (0.3, 1.0, 1.92, 5.0):
            for ax in (0.0, 0.2, 0.6):
                for n, m in ((50, 173), (173, 50), (64, 64)):
                    G, H = rng.uniform(size=n), rng.uniform(size=m)
                    G[: n // 4], H[: m // 4] = 0.0, 1.0           # the extreme corners appear
                    G[-(n // 4):], H[-(m // 4):] = 1.0, 0.0
                    Z = pair_statistics(G, H, zeta=zeta) - ax
                    self.assertGreaterEqual(Z.min(), -(zeta + ax) - 1e-12)
                    self.assertLessEqual(Z.max(), 1.0 - ax + 1e-12)
                    Zlo = pair_statistics(np.zeros(n), np.ones(m), zeta=zeta) - ax
                    np.testing.assert_allclose(Zlo, -(zeta + ax), rtol=0, atol=1e-12)

    def test_wealth_factors_positive(self):
        c = 0.5
        for zeta in (0.5, 1.0, 3.0, 10.0):
            for ax in (0.0, 0.2, 0.5):
                lo = -(zeta + ax)
                Zs = [np.full(30, lo), np.concatenate([np.full(20, 1 - ax), np.full(20, lo)])]
                r = bet(Zs, [1e-4, 1e-4], ax, 70, c=c, zeta=zeta, return_path=True)
                self.assertEqual(len(r["factors"]), 70)
                self.assertLessEqual(max(r["lams"]), c / (zeta + ax) + 1e-15)
                self.assertAlmostEqual(max(r["lams"]), c / (zeta + ax), places=12)   # the cap is reached
                self.assertGreater(min(r["factors"]), 0.0)
                self.assertGreaterEqual(min(r["factors"]), 1.0 - c - 1e-12)

    def test_symmetric_cap_breaks_for_large_zeta(self):
        # with zeta = 3 the symmetric cap c / (1 + alpha) lets 1 + lambda Z go negative: the log is undefined
        zeta, ax = 3.0, 0.2
        Zs = [np.full(10, -(zeta + ax))]
        with self.assertRaises(ValueError):
            bet(Zs, [1e-4], ax, 10)                     # zeta left at 1: cap 0.5/1.2, factor 1 - 3.2*0.417 < 0
        r = bet(Zs, [1e-4], ax, 10, zeta=zeta, return_path=True)
        self.assertAlmostEqual(min(r["factors"]), 0.5, places=12)

    def test_path_of_a_certificate(self):
        X, Y = toy_pair(240, 240, 1, 1.2, seed=21)
        for ax, ay in ((0.1, 0.6), (0.05, 0.0), (0.3, 0.2)):
            pc = pair_certificate(X, Y, ax, ay, split_seed=4, return_path=True, **FAST)
            zeta = (1 - ax) / (1 - ay)
            self.assertEqual(pc.zeta, zeta)
            self.assertEqual(len(pc.path["factors"]), pc.pairs_bet)
            for b in pc.batches:
                if "min_Z" in b:
                    self.assertGreaterEqual(b["min_Z"], -(zeta + ax) - 1e-12)
            if pc.pairs_bet:
                self.assertLessEqual(max(pc.path["lams"]), 0.5 / (zeta + ax) + 1e-15)
                self.assertGreater(min(pc.path["factors"]), 0.0)
                self.assertGreaterEqual(min(pc.path["factors"]), 0.5 - 1e-12)


class TestGridBounds(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(2)
        mu = 0.75
        self.mu = mu
        self.X = rng.normal(mu, 1.0, size=320)
        self.Y = rng.normal(-mu, 1.0, size=320)

    def test_frontier_reads_upward(self):
        ax = 0.2
        grid = np.round(np.arange(0.9, -0.01, -0.1), 2)           # given in decreasing order on purpose
        bound, rows = frontier_lower_bound(self.X, self.Y, ax, grid, split_seed=6, **FAST)
        levels = [r.alpha_y for r in rows]
        self.assertEqual(levels, sorted(levels))
        self.assertTrue(all(r.certified for r in rows[:-1]))
        certified = [r.alpha_y for r in rows if r.certified]
        self.assertEqual(bound, certified[-1] if certified else None)
        self.assertEqual(len({r.sigma for r in rows}), 1)             # one order and bandwidth for the grid
        self.assertTrue(all(r.split_seed == 6 for r in rows))
        _, ay_star = frontier_gauss(self.mu, ax)
        self.assertIsNotNone(bound)
        self.assertLess(bound, ay_star)
        # each row is the stand-alone certificate of its pair
        r0 = pair_certificate(self.X, self.Y, ax, rows[0].alpha_y, split_seed=6, **FAST)
        self.assertEqual(r0.log_final_wealth, rows[0].log_final_wealth)
        self.assertEqual(r0.cuts, rows[0].cuts)

    def test_hockey_stick_ray(self):
        zeta = 2.0
        bound, rows = hockey_stick_lower_bound(self.X, self.Y, zeta, np.arange(0.0, 0.9, 0.1), split_seed=6,
                                               **FAST)
        for r in rows:
            self.assertEqual(r.zeta, zeta)
            self.assertAlmostEqual(r.alpha_y, 1 - (1 - r.alpha_x) / zeta, places=14)
        self.assertTrue(all(r.certified for r in rows[:-1]))
        self.assertIsNotNone(bound)
        self.assertLess(bound, hockey_stick_gauss(self.mu, zeta))

    def test_ray_drops_negative_alpha_y(self):
        zeta = 0.7                                       # alpha_y >= 0 needs alpha_x >= 1 - zeta = 0.3
        bound, rows = hockey_stick_lower_bound(self.X, self.Y, zeta, np.arange(0.0, 0.9, 0.05), split_seed=6,
                                               **FAST)
        self.assertGreaterEqual(rows[0].alpha_x, 0.3 - 1e-12)
        self.assertTrue(all(0.0 <= r.alpha_y < 1.0 for r in rows))
        self.assertEqual(rows[0].alpha_y, 0.0)          # alpha_x = 0.3 sits on the ray at alpha_y = 0

    def test_ray_at_zeta_one_is_the_symmetric_bound(self):
        grid = np.arange(0.0, 0.9, 0.1)
        bound, rows = hockey_stick_lower_bound(self.X, self.Y, 1.0, grid, split_seed=6, **FAST)
        for r in rows:
            pc = pair_certificate(self.X, self.Y, r.alpha_x, r.alpha_x, split_seed=6, **FAST)
            self.assertEqual(r.alpha_y, r.alpha_x)
            self.assertEqual(r.log_final_wealth, pc.log_final_wealth)
        self.assertLess(bound, 2 * norm.cdf(self.mu) - 1)

    def test_input_checks(self):
        with self.assertRaises(ValueError):
            pair_certificate(self.X, self.Y, 0.1, 1.0)
        with self.assertRaises(ValueError):
            pair_certificate(self.X, self.Y, -0.1, 0.2)
        with self.assertRaises(ValueError):
            pair_certificate(self.X, self.Y, 0.1, 0.2, kernel="cauchy")
        with self.assertRaises(ValueError):
            hockey_stick_lower_bound(self.X, self.Y, 0.0, [0.1])
        with self.assertRaises(ValueError):
            frontier_lower_bound(self.X, self.Y, 0.1, [0.2, 1.2])


# ---------------------------------------------------------------------------
# (c) Monte Carlo at the frontier (TRIMMEDMMD_SLOW=1)
# ---------------------------------------------------------------------------

MC_MU, MC_AX, MC_N, MC_DELTA = 0.5, 0.2, 400, 0.05
MC_POWER_AY = 0.30           # well below the frontier alpha_y* = 0.5838 (E_zeta = 0.342 there, against 0.2)
# Run of 2026-11-24 (1000 + 250 replicates, 3 workers, 13 min): 12/1000 = 0.012 (SE 0.0034) at the frontier,
# against delta + 3 SE = 0.0707, a batch bet on in 60.9% of the replicates; 215/250 = 0.86 (SE 0.022) at 0.30.


def _mc_one(args):
    """One replicate: fresh data and split, the certificate of H0(MC_AX, alpha_y)."""
    r, alpha_y, seed0 = args
    rng = np.random.default_rng([seed0, r])
    X = rng.normal(MC_MU, 1.0, size=MC_N)
    Y = rng.normal(-MC_MU, 1.0, size=MC_N)
    split_seed = int(rng.integers(0, 2**31 - 1))
    pc = pair_certificate(X, Y, MC_AX, alpha_y, delta=MC_DELTA, split_seed=split_seed)
    return bool(pc.certified), pc.pairs_bet


def mc_rejections(alpha_y, reps, seed0, workers=1):
    jobs = [(r, alpha_y, seed0) for r in range(reps)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_mc_one, jobs, chunksize=4)
    else:
        out = [_mc_one(j) for j in jobs]
    rej = np.array([o[0] for o in out])
    bet_on = np.array([o[1] > 0 for o in out])
    return rej, bet_on


@unittest.skipUnless(SLOW, "set TRIMMEDMMD_SLOW=1")
class TestFrontierMonteCarlo(unittest.TestCase):

    def test_level_at_frontier_and_power(self):
        reps = int(os.environ.get("TRIMMEDMMD_MC_REPS", "1000"))
        workers = int(os.environ.get("TRIMMEDMMD_MC_WORKERS", "1"))
        zeta, ay = frontier_gauss(MC_MU, MC_AX)
        self.assertLessEqual(hockey_stick_gauss(MC_MU, (1 - MC_AX) / (1 - ay)), MC_AX + 1e-12)   # on H0
        rej, bet_on = mc_rejections(ay, reps, seed0=20261124, workers=workers)
        rate = rej.mean()
        se0 = math.sqrt(MC_DELTA * (1 - MC_DELTA) / reps)
        print(f"\n[unequal MC] frontier mu={MC_MU} alpha_x={MC_AX} zeta*={zeta:.5f} alpha_y*={ay:.5f} n=m={MC_N}: "
              f"rejection {rej.sum()}/{reps} = {rate:.4f} (SE {math.sqrt(max(rate * (1 - rate), 1e-12) / reps):.4f};"
              f" bound delta + 3 SE = {MC_DELTA + 3 * se0:.4f}); replicates with a batch bet on: {bet_on.mean():.3f}",
              file=sys.stderr)
        self.assertLessEqual(rate, MC_DELTA + 3 * se0)
        reps_p = max(reps // 4, 50)
        rej_p, _ = mc_rejections(MC_POWER_AY, reps_p, seed0=20261125, workers=workers)
        zeta_p = (1 - MC_AX) / (1 - MC_POWER_AY)
        print(f"[unequal MC] power at alpha_y={MC_POWER_AY} (zeta={zeta_p:.4f}, "
              f"E_zeta={hockey_stick_gauss(MC_MU, zeta_p):.4f} > alpha_x={MC_AX}): rejection {rej_p.sum()}/{reps_p} = "
              f"{rej_p.mean():.4f} (SE {math.sqrt(max(rej_p.mean() * (1 - rej_p.mean()), 1e-12) / reps_p):.4f})",
              file=sys.stderr)
        self.assertGreaterEqual(rej_p.mean(), 0.8)


if __name__ == "__main__":
    unittest.main()
