"""Fold-1 objects: witness, cut points, betting priors (all fold-1 only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from trimmedmmd.kernels import gaussian_kernel  # noqa: E402
from trimmedmmd.splitting import witness_direction  # noqa: E402

from sacert import psi_vars, ramp  # noqa: E402


class Witness:
    """Unit-norm trimmed-MMD witness fitted on fold 1 (eq:witness)."""

    def __init__(self, X1, Y1, alpha, sigma, rng, **kw):
        kw.setdefault("max_iter", int(os.environ.get("SA_MAX_ITER", "2000")))
        kw.setdefault("tol", 1e-10)
        self.X1, self.Y1, self.sigma, self.alpha = X1, Y1, sigma, alpha
        self.w, self.v, self.norm = witness_direction(X1, Y1, alpha, sigma, rng=rng, **kw)

    def __call__(self, Z):
        if self.norm <= 0:
            return np.zeros(len(Z))
        return (gaussian_kernel(Z, self.X1, self.sigma) @ self.w
                - gaussian_kernel(Z, self.Y1, self.sigma) @ self.v) / self.norm

    def fold1_values(self):
        """Self-excluded fold-1 values: drop each point's own kernel term.

        f(X1_i) contains w_i k(X1_i, X1_i)/norm = w_i/norm; removing it is a
        cheap approximate leave-one-out that reduces the in-sample bias of the
        fold-1 quantiles. Uses fold 1 only, so validity is unaffected.
        """
        if self.norm <= 0:
            return np.zeros(len(self.X1)), np.zeros(len(self.Y1))
        u1 = self(self.X1) - self.w / self.norm
        v1 = self(self.Y1) + self.v / self.norm
        return u1, v1


def cut_points(u1, v1, alpha):
    """s = (1-alpha)-quantile of fold-1 U values, sp = alpha-quantile of fold-1 V values."""
    s = float(np.quantile(u1, 1.0 - alpha))
    sp = float(np.quantile(v1, alpha))
    return s, sp


def pair_var(VA, VB, n2, m2):
    """Variance of one pair statistic A_i + mean(B over a block) (or mirror)."""
    if n2 <= m2:
        return VA + VB * n2 / m2
    return VA * m2 / n2 + VB


def priors(u1, v1, s, sp, alpha, n2, m2):
    """Fold-1 guesses of the pair-statistic variance, for the betting fractions."""
    pr = {}
    if s > sp:
        G, H = ramp(u1, s, sp), ramp(v1, s, sp)
        pr["ramp"] = max(pair_var(G.var(), H.var(), n2, m2), 1e-4)
        A, B = psi_vars(u1, v1, s, sp, alpha)
        pr["psi"] = max(pair_var(A.var(), B.var(), n2, m2), 1e-4)
    return pr
