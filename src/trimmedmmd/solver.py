"""Convex QP solver for the empirical trimmed MMD.

The empirical problem is

    min_{w,v}  w' Kxx w + v' Kyy v - 2 w' Kxy v

over two capped simplices. As noted in the paper (Remark 2) the objective is
``|| sum_i w_i phi(X_i) - sum_j v_j phi(Y_j) ||^2``, a positive-semidefinite
quadratic form over a polytope, so this is a convex QP and accelerated
projected gradient (FISTA) applies.

Two backends share one driver:

* :func:`solve_gram` works from kernel matrices, at O((n+m)^2) per iteration;
* :func:`solve_features` works from explicit feature matrices (e.g. random
  Fourier features), at O((n+m) r) per iteration and O((n+m) r) memory, which
  is what makes large-n runs feasible at all.

Both accept warm starts, which is what makes a sweep along the nested
alpha-path cheap (milestone M3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simplex import project_capped_simplex


@dataclass
class QPResult:
    value: float          # squared discrepancy (the QP objective)
    w: np.ndarray
    v: np.ndarray
    n_iter: int
    converged: bool
    gap: float = float("nan")   # Frank-Wolfe duality gap at the returned point

    @property
    def discrepancy(self) -> float:
        """Unsquared discrepancy, i.e. what the paper calls D-hat."""
        return float(np.sqrt(max(self.value, 0.0)))

    @property
    def value_lower(self) -> float:
        """Certified lower bound on the true minimum of the QP.

        ``converged`` reports that the objective stopped moving, which is a
        statement about the iterates rather than about the distance to the
        optimum. The Frank-Wolfe gap is the real certificate: for a convex
        objective, ``F(w, v) - F* <= gap``, so the optimum lies in
        ``[value - gap, value]``.

        This matters in one direction in particular. A feasible but
        unconverged iterate has ``F(w, v) >= F*``, so it *overstates* the
        discrepancy -- which is anti-conservative for the analytic test of
        eq. (16), where the statistic is compared against a fixed threshold.
        For the split-sample certificate the same slack costs only power.
        """
        return float(max(self.value - max(self.gap, 0.0), 0.0))

    @property
    def discrepancy_lower(self) -> float:
        """Certified lower bound on D-hat, i.e. ``sqrt(value_lower)``."""
        return float(np.sqrt(self.value_lower))


def _linear_min_over_capped_simplex(g: np.ndarray, cap: float) -> float:
    """``min_u <g, u>`` over ``{0 <= u <= cap, sum u = 1}``.

    The minimum of a linear functional over a capped simplex loads the cap
    onto the smallest coordinates and zero onto the largest, with one
    coordinate carrying the remainder -- the same bang-bang structure the
    paper's Lemma on extreme trimmed means describes.
    """
    order = np.argsort(g)
    take = np.minimum(cap, np.maximum(
        0.0, 1.0 - cap * np.arange(order.size, dtype=float)))
    return float(np.dot(g[order], take))


def _fista(grad, objective, w0, v0, cap_w, cap_v, lipschitz,
           max_iter=2000, tol=1e-10):
    w, v = w0.copy(), v0.copy()
    yw, yv = w.copy(), v.copy()
    t = 1.0
    step = 1.0 / lipschitz
    prev_obj = None          # not inf: |inf - obj| <= tol*inf is vacuously true
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        gw, gv = grad(yw, yv)
        w_new = project_capped_simplex(yw - step * gw, cap_w)
        v_new = project_capped_simplex(yv - step * gv, cap_v)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_new
        yw = w_new + beta * (w_new - w)
        yv = v_new + beta * (v_new - v)
        w, v, t = w_new, v_new, t_new
        if it % 10 == 0 or it == max_iter:
            obj = objective(w, v)
            # Restart heuristic: FISTA is not monotone, so reset the momentum
            # whenever the objective has gone up.
            if prev_obj is not None and obj > prev_obj:
                yw, yv, t = w.copy(), v.copy(), 1.0
            # A relative test on the objective itself. Using
            # max(1.0, |prev_obj|) would make this an ABSOLUTE tolerance,
            # because the objective is a squared discrepancy and is almost
            # always far below 1; the error on the reported square root then
            # grows like tol/(2D) as D -> 0, which is exactly the regime near
            # the null boundary that matters most here.
            if prev_obj is not None and \
                    abs(prev_obj - obj) <= tol * max(abs(prev_obj), 1e-300):
                converged = True
                prev_obj = obj
                break
            prev_obj = obj
    return w, v, objective(w, v), it, converged


def _power_iteration(matvec, dim, rng, iters=60):
    x = rng.standard_normal(dim)
    x /= np.linalg.norm(x)
    lam = 1.0
    for _ in range(iters):
        y = matvec(x)
        ny = np.linalg.norm(y)
        if ny <= 0:
            return 0.0
        x = y / ny
        lam = ny
    return float(lam)


def solve_gram(Kxx, Kyy, Kxy, alpha, w0=None, v0=None,
               max_iter=2000, tol=1e-10, rng=None,
               alpha_y=None, ridge=0.0) -> QPResult:
    """Solve the trimmed-MMD QP from kernel matrices.

    ``alpha_y`` trims the second sample at its own level (default: the same
    ``alpha``). ``ridge`` adds ``ridge * (||w - 1/n||^2 + ||v - 1/m||^2)``,
    which makes the problem strictly convex and so picks out a single
    minimizer when the unregularized one is not unique; see
    :func:`trimmedmmd.trimmings.optimal_trimming_mmd`. The reported
    ``value`` is always the unregularized objective.
    """
    n, m = Kxx.shape[0], Kyy.shape[0]
    alpha_y = alpha if alpha_y is None else alpha_y
    cap_w, cap_v = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha_y))
    rng = np.random.default_rng(0) if rng is None else rng
    ref_w, ref_v = np.full(n, 1.0 / n), np.full(m, 1.0 / m)

    def grad(w, v):
        gw = 2.0 * (Kxx @ w - Kxy @ v)
        gv = 2.0 * (Kyy @ v - Kxy.T @ w)
        if ridge:
            gw = gw + 2.0 * ridge * (w - ref_w)
            gv = gv + 2.0 * ridge * (v - ref_v)
        return (gw, gv)

    def value_of(w, v):
        return float(w @ (Kxx @ w) + v @ (Kyy @ v) - 2.0 * w @ (Kxy @ v))

    def objective(w, v):
        val = value_of(w, v)
        if ridge:
            dw, dv = w - ref_w, v - ref_v
            val += ridge * (float(dw @ dw) + float(dv @ dv))
        return val

    def matvec(z):
        w, v = z[:n], z[n:]
        return np.concatenate([Kxx @ w - Kxy @ v, Kyy @ v - Kxy.T @ w])

    lam = _power_iteration(matvec, n + m, rng)
    lipschitz = max(2.0 * lam + 2.0 * ridge, 1e-12)

    w0 = np.full(n, 1.0 / n) if w0 is None else np.clip(w0, 0, cap_w)
    v0 = np.full(m, 1.0 / m) if v0 is None else np.clip(v0, 0, cap_v)
    w0 = project_capped_simplex(w0, cap_w)
    v0 = project_capped_simplex(v0, cap_v)

    w, v, _obj, it, conv = _fista(grad, objective, w0, v0, cap_w, cap_v,
                                  lipschitz, max_iter, tol)
    # Frank-Wolfe gap at the returned point. With a ridge it bounds the
    # suboptimality of the *regularized* objective, which is the one being
    # minimized; with the default ridge = 0 the two coincide.
    gw, gv = grad(w, v)
    gap = (float(np.dot(gw, w)) - _linear_min_over_capped_simplex(gw, cap_w)
           + float(np.dot(gv, v)) - _linear_min_over_capped_simplex(gv, cap_v))
    return QPResult(value=max(value_of(w, v), 0.0), w=w, v=v,
                    n_iter=it, converged=conv, gap=max(gap, 0.0))


def solve_features(Zx, Zy, alpha, w0=None, v0=None,
                   max_iter=2000, tol=1e-10, rng=None,
                   alpha_y=None, ridge=0.0) -> QPResult:
    """Solve the trimmed-MMD QP from feature matrices, objective
    ``|| Zx' w - Zy' v ||_2^2``. Never forms a dense Gram matrix (M6).

    ``alpha_y`` and ``ridge`` are as in :func:`solve_gram`."""
    n, m = Zx.shape[0], Zy.shape[0]
    alpha_y = alpha if alpha_y is None else alpha_y
    cap_w, cap_v = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha_y))
    rng = np.random.default_rng(0) if rng is None else rng
    ref_w, ref_v = np.full(n, 1.0 / n), np.full(m, 1.0 / m)

    def resid(w, v):
        return Zx.T @ w - Zy.T @ v

    def grad(w, v):
        d = resid(w, v)
        gw, gv = 2.0 * (Zx @ d), -2.0 * (Zy @ d)
        if ridge:
            gw = gw + 2.0 * ridge * (w - ref_w)
            gv = gv + 2.0 * ridge * (v - ref_v)
        return (gw, gv)

    def value_of(w, v):
        d = resid(w, v)
        return float(d @ d)

    def objective(w, v):
        val = value_of(w, v)
        if ridge:
            dw, dv = w - ref_w, v - ref_v
            val += ridge * (float(dw @ dw) + float(dv @ dv))
        return val

    def matvec(z):
        w, v = z[:n], z[n:]
        d = Zx.T @ w - Zy.T @ v
        return np.concatenate([Zx @ d, -(Zy @ d)])

    lam = _power_iteration(matvec, n + m, rng)
    lipschitz = max(2.0 * lam + 2.0 * ridge, 1e-12)

    w0 = np.full(n, 1.0 / n) if w0 is None else np.clip(w0, 0, cap_w)
    v0 = np.full(m, 1.0 / m) if v0 is None else np.clip(v0, 0, cap_v)
    w0 = project_capped_simplex(w0, cap_w)
    v0 = project_capped_simplex(v0, cap_v)

    w, v, _obj, it, conv = _fista(grad, objective, w0, v0, cap_w, cap_v,
                                  lipschitz, max_iter, tol)
    gw, gv = grad(w, v)
    gap = (float(np.dot(gw, w)) - _linear_min_over_capped_simplex(gw, cap_w)
           + float(np.dot(gv, v)) - _linear_min_over_capped_simplex(gv, cap_v))
    return QPResult(value=max(value_of(w, v), 0.0), w=w, v=v,
                    n_iter=it, converged=conv, gap=max(gap, 0.0))


def solve_path(solve_one, alphas, **kwargs):
    """Sweep a monotone alpha-path with warm starts (milestone M3).

    ``solve_one(alpha, w0, v0)`` should return a :class:`QPResult`. The
    feasible sets are nested in alpha, so the solution at one level is a
    feasible, and usually excellent, starting point for the next.
    """
    results, w0, v0 = [], None, None
    for a in alphas:
        res = solve_one(a, w0, v0)
        results.append(res)
        w0, v0 = res.w, res.v
    return results
