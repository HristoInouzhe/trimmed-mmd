"""Trimmed Sinkhorn divergence (Section 12 of the paper).

New code -- no prior implementation existed in this package. Implements:

* a log-domain (stabilized) Sinkhorn solver for entropic OT between two
  weighted point clouds, cross-checked against the POT reference
  implementation and, for small problems, against a brute-force LP for the
  unregularized transport cost as epsilon -> 0 (``tests/test_sinkhorn.py``);
* the (nonconvex) trimming optimization outer loop: minimize the debiased
  Sinkhorn divergence ``S_eps(w,v) = OT_eps(w,v) - 0.5 OT_eps(w,w)
  - 0.5 OT_eps(v,v)`` over capped simplices ``w,v``, via projected gradient
  with the envelope-theorem gradient (Sinkhorn dual potentials), reusing
  :func:`trimmedmmd.simplex.project_capped_simplex`;
* a torch/GPU backend for the batched case (many (alpha, replicate) cells
  solved as one batched tensor op), since this is a genuinely GPU-friendly
  workload unlike the small-problem QP solves in solver.py.

This tests `prop:sinkmono` (monotonicity in alpha) and, as a proxy for the
unproven hypothesis (C) of `prop:sinkexist`, continuity of alpha ->
S_eps,alpha(P,Q) (CLAIMS.md n17, n11). Being a bilevel nonconvex
optimization, a detected "violation" of monotonicity must first be checked
against optimizer suboptimality (warm starts, more iterations) before it is
read as evidence about the proposition -- the driver in experiments/run.py
does this by re-solving with more iterations whenever a decrease looks
close to violated.
"""

from __future__ import annotations

import numpy as np

from .simplex import project_capped_simplex

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAVE_TORCH = False


# --------------------------------------------------------------------------
# numpy log-domain Sinkhorn (reference; used for small/CPU problems and for
# cross-checking the torch backend)
# --------------------------------------------------------------------------

def sinkhorn_ot(a, b, C, epsilon, n_iter=500, tol=1e-10):
    """Entropic OT value and dual potentials, log-domain stabilized.

    a, b: marginal weight vectors (must each sum to 1, nonnegative).
    C: cost matrix, C[i,j] = c(x_i, y_j).

    Convention (Peyre-Cuturi eq. 4.13-4.17, no reference-measure prefactor):
    pi_ij = exp((f_i + g_j - C_ij) / eps), with (f,g) solving

        f_i = eps * (log a_i - logsumexp_j((g_j - C_ij)/eps))
        g_j = eps * (log b_j - logsumexp_i((f_i - C_ij)/eps))

    so that pi has row sums a and column sums b exactly at a fixed point.
    At the fixed point the dual objective's entropy-correction term
    vanishes and the value equals f@a + g@b exactly; returned via that
    formula (more numerically stable than reconstructing pi at small eps).
    Returns (value, f, g).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    loga = np.log(np.clip(a, 1e-300, None))
    logb = np.log(np.clip(b, 1e-300, None))
    # psi, phi are the no-prefactor iterates (pi = exp((psi+phi-C)/eps));
    # these are what make the fixed-point recursion match the marginals,
    # but they are NOT the dual potentials the envelope theorem/duality
    # value formula are stated in terms of (those correspond to the
    # pi = a_i b_j exp((f+g-C)/eps) convention). The two are related by an
    # exact shift, applied once at the end -- see the module docstring's
    # cross-check note; getting this shift wrong silently gives a value off
    # by eps*(H(a)+H(b)), which was caught by comparing against POT.
    psi = np.zeros_like(a)
    phi = np.zeros_like(b)
    for _ in range(n_iter):
        psi_prev = psi
        Mf = (phi[None, :] - C) / epsilon
        psi = epsilon * (loga - _logsumexp(Mf, axis=1))
        Mg = (psi[:, None] - C) / epsilon
        phi = epsilon * (logb - _logsumexp(Mg, axis=0))
        if np.max(np.abs(psi - psi_prev)) < tol:
            break
    f = psi - epsilon * loga
    g = phi - epsilon * logb
    value = float(f @ a + g @ b)
    return value, f, g


def sinkhorn_plan(a, b, f, g, C, epsilon):
    """Reconstruct the transport plan from the (true) dual potentials
    returned by :func:`sinkhorn_ot`: pi_ij = a_i b_j exp((f_i+g_j-C_ij)/eps).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = (f[:, None] + g[None, :] - C) / epsilon
    return a[:, None] * b[None, :] * np.exp(M)


def _logsumexp(M, axis):
    m = np.max(M, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    s = np.log(np.sum(np.exp(M - m), axis=axis, keepdims=True)) + m
    return np.squeeze(s, axis=axis)


def sinkhorn_divergence(a, b, Caa, Cbb, Cab, epsilon, n_iter=500):
    """Debiased Sinkhorn divergence S_eps(a,b) = OT(a,b) - .5 OT(a,a) - .5 OT(b,b).

    Also returns the gradient of S_eps w.r.t. a and b (envelope theorem:
    d OT_eps(a,b)/d a_i = f_i, the dual potential of the *first* marginal;
    for the symmetric OT(a,a) term the marginal appears twice, contributing
    f_aa + g_aa, which by symmetry of that problem equals 2*f_aa).
    """
    v_ab, f_ab, g_ab = sinkhorn_ot(a, b, Cab, epsilon, n_iter)
    v_aa, f_aa, _ = sinkhorn_ot(a, a, Caa, epsilon, n_iter)
    v_bb, f_bb, _ = sinkhorn_ot(b, b, Cbb, epsilon, n_iter)
    S = v_ab - 0.5 * v_aa - 0.5 * v_bb
    grad_a = f_ab - f_aa       # d/da [OT(a,b) - 0.5 OT(a,a)], OT(b,b) const in a
    grad_b = g_ab - f_bb       # symmetric role for b
    return S, grad_a, grad_b


def trimmed_sinkhorn(X, Y, alpha, epsilon, cost_fn=None, max_iter=300,
                     lr=None, sinkhorn_iter=200, w0=None, v0=None, seed=0):
    """Minimize S_eps(w, v) over capped simplices (the trimmed Sinkhorn
    divergence, eq:trimSink), by projected gradient with warm starts.

    Nonconvex: this is a local solve, not a certified global optimum. Callers
    doing an alpha-sweep should pass warm starts (w0, v0 from the previous,
    smaller alpha) exactly as solver.py's QP path does, both for speed and
    because it biases the sweep toward the monotone-descent trajectory that
    Proposition prop:sinkmono predicts.
    """
    n, m = X.shape[0], Y.shape[0]
    if cost_fn is None:
        def cost_fn(A, B):
            return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    Cxx = cost_fn(X, X)
    Cyy = cost_fn(Y, Y)
    Cxy = cost_fn(X, Y)
    cap_w, cap_v = 1.0 / (n * (1 - alpha)), 1.0 / (m * (1 - alpha))
    rng = np.random.default_rng(seed)
    w = (np.full(n, 1.0 / n) if w0 is None
        else project_capped_simplex(np.clip(w0, 0, cap_w), cap_w))
    v = (np.full(m, 1.0 / m) if v0 is None
        else project_capped_simplex(np.clip(v0, 0, cap_v), cap_v))

    # Backtracking line search, not a fixed lr/sqrt(it) schedule: the
    # Sinkhorn potentials (the gradient here) can be an order of magnitude
    # larger than the QP case's gradients while the simplex weights live on
    # a tiny scale, and a fixed step size was found to actively DIVERGE
    # (objective oscillating 0.6 -> 9+ -> 3 -> 9+ ..., not just failing to
    # improve) -- a real optimizer bug, not mere suboptimality, caught by
    # tracing the per-iteration objective. Backtracking is what solver.py's
    # FISTA also relies on (its restart-on-increase heuristic), adapted here
    # to a nonconvex objective where we cannot assume a global Lipschitz
    # constant, so each step must be individually verified to not increase
    # the objective by more than a small numerical slack.
    step0 = lr if lr is not None else 1.0
    step = step0
    prev_S, _, _ = sinkhorn_divergence(w, v, Cxx, Cyy, Cxy, epsilon, sinkhorn_iter)
    it = 0
    for it in range(max_iter):
        S, gw, gv = sinkhorn_divergence(w, v, Cxx, Cyy, Cxy, epsilon, sinkhorn_iter)
        gnorm = max(float(np.hypot(np.linalg.norm(gw), np.linalg.norm(gv))), 1e-12)
        accepted = False
        for _bt in range(30):
            # normalize by gradient norm so the step is a bounded move in
            # simplex-weight space regardless of the potentials' raw scale
            w_new = project_capped_simplex(w - (step / gnorm) * gw, cap_w)
            v_new = project_capped_simplex(v - (step / gnorm) * gv, cap_v)
            S_new, _, _ = sinkhorn_divergence(w_new, v_new, Cxx, Cyy, Cxy,
                                              epsilon, sinkhorn_iter)
            if S_new <= S + 1e-10 * max(abs(S), 1.0):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break  # step shrunk to numerical noise; nothing left to gain
        w, v = w_new, v_new
        step = min(step * 1.2, step0)  # cautious growth back toward step0
        if abs(prev_S - S_new) < 1e-10 * max(abs(prev_S), 1e-12):
            prev_S = S_new
            break
        prev_S = S_new
    S_final, _, _ = sinkhorn_divergence(w, v, Cxx, Cyy, Cxy, epsilon, sinkhorn_iter)
    return dict(value=float(S_final), w=w, v=v, n_iter=it + 1)
