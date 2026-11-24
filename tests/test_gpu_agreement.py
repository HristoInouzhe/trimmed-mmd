"""Numerical agreement between the GPU (torch) backend and the numpy
reference implementation it is meant to accelerate.

Per the theory-stress-suite's Phase 6 (performance/robustness): a faster
reimplementation of a hot path must be proved to agree with the reference
before it is trusted for any result, not just benchmarked. Skipped outright
if torch or a CUDA device is unavailable -- this is an agreement check, not
a portability requirement.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trimmedmmd import gpu  # noqa: E402
from trimmedmmd.kernels import gaussian_kernel  # noqa: E402
from trimmedmmd.simplex import project_capped_simplex  # noqa: E402
from trimmedmmd.solver import solve_features, solve_gram  # noqa: E402

SKIP = not gpu.available()
SKIP_REASON = "no CUDA device visible to torch"


@unittest.skipIf(SKIP, SKIP_REASON)
class TestProjectionAgreement(unittest.TestCase):
    def test_matches_numpy_reference(self):
        import torch
        rng = np.random.default_rng(0)
        for n in (5, 50, 733, 4001):
            cap = 1.0 / (n * (1 - 0.3))
            y = rng.standard_normal(n)
            w_cpu = project_capped_simplex(y, cap)
            w_gpu = gpu.project_capped_simplex_t(
                torch.as_tensor(y, dtype=torch.float64, device="cuda"), cap
            ).cpu().numpy()
            np.testing.assert_allclose(w_gpu, w_cpu, atol=1e-8)


@unittest.skipIf(SKIP, SKIP_REASON)
class TestSolverAgreement(unittest.TestCase):
    def _problem(self, seed, n, m, d=3, sigma=1.0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d)) + 0.6
        Kxx = gaussian_kernel(X, X, sigma)
        Kyy = gaussian_kernel(Y, Y, sigma)
        Kxy = gaussian_kernel(X, Y, sigma)
        return Kxx, Kyy, Kxy

    def test_gram_backend_matches_cpu_reference(self):
        for n, m, alpha in [(20, 15, 0.2), (150, 140, 0.35), (900, 850, 0.1)]:
            Kxx, Kyy, Kxy = self._problem(seed=n, n=n, m=m)
            cpu = solve_gram(Kxx, Kyy, Kxy, alpha, max_iter=4000, tol=1e-12)
            g = gpu.solve_gram_gpu(Kxx, Kyy, Kxy, alpha, max_iter=4000,
                                    tol=1e-12, device="cuda")
            self.assertLess(abs(cpu.value - g.value), 1e-6,
                            f"n={n} m={m} alpha={alpha}: "
                            f"cpu={cpu.value} gpu={g.value}")
            self.assertLess(abs(cpu.discrepancy - g.discrepancy), 1e-6)

    def test_feature_backend_matches_cpu_reference(self):
        rng = np.random.default_rng(42)
        n, m, r = 400, 350, 256
        Zx = rng.standard_normal((n, r)) / np.sqrt(r)
        Zy = rng.standard_normal((m, r)) / np.sqrt(r)
        alpha = 0.25
        cpu = solve_features(Zx, Zy, alpha, max_iter=4000, tol=1e-12)
        g = gpu.solve_features_gpu(Zx, Zy, alpha, max_iter=4000, tol=1e-12,
                                    device="cuda")
        self.assertLess(abs(cpu.value - g.value), 1e-6)

    def test_gpu_solution_is_itself_feasible(self):
        # Agreement in objective value is not enough: the GPU projection must
        # also produce a genuinely feasible point (cap and simplex both).
        Kxx, Kyy, Kxy = self._problem(seed=7, n=300, m=280)
        alpha = 0.3
        g = gpu.solve_gram_gpu(Kxx, Kyy, Kxy, alpha, max_iter=4000,
                                tol=1e-12, device="cuda")
        n, m = Kxx.shape[0], Kyy.shape[0]
        cap_w, cap_v = 1.0 / (n * (1 - alpha)), 1.0 / (m * (1 - alpha))
        self.assertAlmostEqual(g.w.sum(), 1.0, places=8)
        self.assertAlmostEqual(g.v.sum(), 1.0, places=8)
        self.assertGreaterEqual(g.w.min(), -1e-9)
        self.assertLessEqual(g.w.max(), cap_w + 1e-9)
        self.assertGreaterEqual(g.v.min(), -1e-9)
        self.assertLessEqual(g.v.max(), cap_v + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
