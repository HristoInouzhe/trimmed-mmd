"""Checks for trimmedmmd.aos_fallback before the runs (results/aos/fallback/NOTES.md).

1. Gaussian: the OWN rule of aos_fallback.sequential_all_rules equals the package's
   splitting.sequential_certificate (same order, same rng) -- decision, log K_N, pairs bet and skipped batches --
   on a few (contrast, split, level) cells, and its fit values equal splitting.fold1_witness_values.
2. Laplace: the fit Gram matrices have k(x, x) = 1 exactly, and the own-term values equal the witness recomputed
   with the point's own term deleted from the sum (leave-own-term-out), at random points.

    PYTHONPATH=src python -B scripts/aos_fallback_verify.py
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trimmedmmd.aos_fallback import fit_batch, kernel_matrix, sequential_all_rules  # noqa: E402
from trimmedmmd.splitting import (SEQ_EDGES, fold1_witness_values, sequential_certificate,  # noqa: E402
                                  sequential_order, witness_direction)
from aos_fallback_realdata import data  # noqa: E402


def check_package(name, seed, alpha):
    X, Y, std0 = data(name)
    ss = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
    o1 = sequential_order(X, Y, np.random.default_rng(ss), edges=SEQ_EDGES, standardize_batch0=std0)
    ref = sequential_certificate(X, Y, alpha, delta=0.05, edges=SEQ_EDGES, order=o1)
    o2 = sequential_order(X, Y, np.random.default_rng(ss), edges=SEQ_EDGES, standardize_batch0=std0)
    new = sequential_all_rules(o2, alpha, kernel="gaussian")
    own = new["own"]
    skipped = [b + 1 for b, s in enumerate(own["source"]) if s == "skip"]
    same = (ref.certified == own["certified"] and ref.pairs_bet == own["pairs_bet"]
            and list(ref.skipped_batches) == skipped
            and (ref.pairs_bet == 0 or ref.log_final_wealth == own["log_final"]))
    print(f"{name} seed={seed} alpha={alpha}: package cert={ref.certified} logK_N={ref.log_final_wealth:.12g} "
          f"skipped={ref.skipped_batches} | OWN cert={own['certified']} logK_N={own['log_final']:.12g} "
          f"skipped={skipped} -> {'IDENTICAL' if same else 'DIFFERENT'}; fallback cert={new['fallback']['certified']} "
          f"plain cert={new['plain']['certified']} crossings={[(b['own_crossed'], b['plain_crossed']) for b in new['batches']]}")
    return same


def check_values(name, seed, alpha):
    """fit_batch's own/plain values against witness_direction + fold1_witness_values (Gaussian)."""
    X, Y, std0 = data(name)
    ss = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
    o = sequential_order(X, Y, np.random.default_rng(ss), edges=SEQ_EDGES, standardize_batch0=std0)
    n, m = len(o.X), len(o.Y)
    Xf, Yf = o.X[: n // 4], o.Y[: m // 4]
    st = o.rng.bit_generator.state
    w, v, norm = witness_direction(Xf, Yf, alpha, o.sigma, rng=o.rng)
    u1, v1 = fold1_witness_values(w, v, Xf, Yf, o.sigma, norm)
    u0, v0 = fold1_witness_values(w, v, Xf, Yf, o.sigma, norm, self_exclude=False)
    o.rng.bit_generator.state = st
    bf = fit_batch(Xf, Yf, o.X[:2], o.Y[:2], alpha, o.sigma, o.rng)
    ok = (np.array_equal(u1, bf.u_own) and np.array_equal(v1, bf.v_own) and np.array_equal(u0, bf.u_plain)
          and np.array_equal(v0, bf.v_plain))
    print(f"fit values (Gaussian, {name}, alpha={alpha}): {'IDENTICAL' if ok else 'DIFFERENT'} to the package")
    return ok


def check_laplace(name, seed, alpha):
    X, Y, std0 = data(name)
    ss = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
    o = sequential_order(X, Y, np.random.default_rng(ss), edges=SEQ_EDGES, standardize_batch0=std0)
    n, m = len(o.X), len(o.Y)
    Xf, Yf = o.X[: n // 4], o.Y[: m // 4]
    K = kernel_matrix(Xf, Xf, o.sigma, "laplace", same=True)
    diag_ok = bool(np.all(np.diag(K) == 1.0))
    st = o.rng.bit_generator.state
    bf = fit_batch(Xf, Yf, o.X[:2], o.Y[:2], alpha, o.sigma, o.rng, kernel="laplace")
    # recover the weights and recompute leave-own-term-out values directly
    o.rng.bit_generator.state = st
    from trimmedmmd.solver import solve_gram
    Kyy = kernel_matrix(Yf, Yf, o.sigma, "laplace", same=True)
    Kxy = kernel_matrix(Xf, Yf, o.sigma, "laplace")
    res = solve_gram(K, Kyy, Kxy, alpha, rng=o.rng)
    w, v, D = res.w, res.v, res.discrepancy
    rng = np.random.default_rng(1)
    err = 0.0
    for i in rng.choice(len(Xf), 20, replace=False):
        mask = np.ones(len(Xf), bool)
        mask[i] = False
        val = (K[i, mask] @ w[mask] - Kxy[i] @ v) / D
        err = max(err, abs(val - bf.u_own[i]))
    for j in rng.choice(len(Yf), 20, replace=False):
        mask = np.ones(len(Yf), bool)
        mask[j] = False
        val = (Kxy[:, j] @ w - Kyy[j, mask] @ v[mask]) / D
        err = max(err, abs(val - bf.v_own[j]))
    print(f"Laplace ({name}, alpha={alpha}, h={o.sigma:.4g}): diag k(x,x) == 1: {diag_ok}; "
          f"max |own-term value - leave-own-term-out value| over 40 points = {err:.2e}; witness norm {D:.4g}")
    return diag_ok and err < 1e-10


if __name__ == "__main__":
    ok = True
    for name, seed, alpha in [("rhc", 1000, 0.3), ("rhc", 1001, 0.35), ("nhanes_bmi", 1002, 0.05),
                              ("nhanes_age", 1003, 0.4)]:
        ok &= check_package(name, seed, alpha)
    ok &= check_values("nhanes_child", 1004, 0.5)
    ok &= check_laplace("rhc", 1000, 0.2)
    ok &= check_laplace("nhanes_age", 1001, 0.3)
    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
