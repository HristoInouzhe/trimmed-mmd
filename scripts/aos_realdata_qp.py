"""AoS revision: an exact solver for the empirical trimmed-MMD program (2.10), with a certified duality gap.

    min_{w, v}  F(w, v) = || sum_i w_i phi(X_i) - sum_j v_j phi(Y_j) ||^2
    s.t.        0 <= w_i <= 1/(n(1-alpha)),  sum_i w_i = 1,   0 <= v_j <= 1/(m(1-alpha)),  sum_j v_j = 1.

With z = (w, v) and S = diag(1_n, -1_m), F(z) = z' H z with H = S G S, G the Gram matrix of (X, Y). No penalty of any
kind is added. Three stages:

1. accelerated projected gradient (the package's trimmedmmd.solver.solve_gram, ridge = 0) as a warm start;
2. sequential minimal optimization (two-coordinate exact line searches inside one arm, which keep both sums fixed),
   with the second-order working-set selection of Fan, Chen and Lin (2005, JMLR 6), compiled with numba, until the
   maximal KKT violation in each arm is below ``eps``; the gradient is recomputed from scratch every
   ``refresh`` steps so rounding does not accumulate;
3. a polish: with the sets at the cap and at zero fixed, the stationarity conditions on the free coordinates and the
   two sums are a square linear system; its solution is kept only if it is feasible, satisfies the KKT sign
   conditions on the bounded coordinates and does not increase F.

The certificate is the Frank-Wolfe gap at the returned point, gap = <grad F(z), z - s>, s the linear minimizer over
the polytope (cap on the smallest gradient entries of each arm). For convex F, F(z) - F* <= gap, so
F* lies in [F(z) - gap, F(z)] and D-hat = sqrt(F*) in [sqrt(F(z) - gap), sqrt(F(z))]. F and the gap are evaluated
twice, in float64 and in extended precision (numpy longdouble), from a fresh product H z.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from trimmedmmd.solver import solve_gram


@njit(cache=True)
def _block_extremes(z, g, lo, hi, cap):
    """min g over coordinates that can increase, max g over those that can decrease (within [lo, hi))."""
    gmin, imin = np.inf, -1
    gmax, imax = -np.inf, -1
    for k in range(lo, hi):
        if z[k] < cap and g[k] < gmin:
            gmin, imin = g[k], k
        if z[k] > 0.0 and g[k] > gmax:
            gmax, imax = g[k], k
    return gmin, imin, gmax, imax


@njit(cache=True)
def _smo(H, z, g, n, caps, max_steps, eps, refresh):
    N = z.size
    steps = 0
    viol = np.zeros(2)
    while steps < max_steps:
        # violations of the two arms
        best_b, best_v = -1, -1.0
        for b in range(2):
            lo = 0 if b == 0 else n
            hi = n if b == 0 else N
            gmin, imin, gmax, imax = _block_extremes(z, g, lo, hi, caps[b])
            v = gmax - gmin if (imin >= 0 and imax >= 0) else 0.0
            viol[b] = v
            if v > best_v:
                best_v, best_b = v, b
        if best_v <= eps:
            break
        b = best_b
        lo = 0 if b == 0 else n
        hi = n if b == 0 else N
        cap = caps[b]
        gmin, i, gmax, jmax = _block_extremes(z, g, lo, hi, cap)
        # second-order choice of j among the coordinates that can decrease and have g_j > g_i
        best, j = -1.0, -1
        for k in range(lo, hi):
            if z[k] > 0.0 and g[k] > gmin:
                a = H[i, i] + H[k, k] - 2.0 * H[i, k]
                if a < 1e-15:
                    a = 1e-15
                d = g[k] - gmin
                score = d * d / a
                if score > best:
                    best, j = score, k
        if j < 0:
            break
        a = H[i, i] + H[j, j] - 2.0 * H[i, j]
        if a < 1e-15:
            a = 1e-15
        t = (g[j] - g[i]) / (2.0 * a)
        room_i = cap - z[i]
        room_j = z[j]
        if t >= room_i and room_i <= room_j:
            t = room_i
            z[i] = cap
            z[j] = z[j] - t
            if z[j] < 0.0:
                z[j] = 0.0
        elif t >= room_j:
            t = room_j
            z[j] = 0.0
            z[i] = z[i] + t
            if z[i] > cap:
                z[i] = cap
        else:
            z[i] = z[i] + t
            z[j] = z[j] - t
        for k in range(N):
            g[k] += 2.0 * t * (H[i, k] - H[j, k])
        steps += 1
        if steps % refresh == 0:
            for k in range(N):
                s = 0.0
                for l in range(N):
                    s += H[k, l] * z[l]
                g[k] = 2.0 * s
    return steps, viol


def linear_min(g, cap):
    """min <g, u> over {0 <= u <= cap, sum u = 1} (cap on the smallest entries, remainder on the next)."""
    order = np.argsort(g, kind="stable")
    take = np.minimum(cap, np.maximum(0.0, 1.0 - cap * np.arange(order.size, dtype=float)))
    return float(np.dot(g[order], take))


def certificate(H, z, n, caps, dtype=np.float64):
    """F(z), the Frank-Wolfe gap and the per-arm gaps, from a fresh product H z in ``dtype``."""
    zd = z.astype(dtype)
    if dtype is np.float64:
        Hz = H @ z
    else:
        Hz = np.zeros(z.size, dtype=dtype)
        for k0 in range(0, z.size, 256):
            Hz[k0:k0 + 256] = (H[k0:k0 + 256].astype(dtype) * zd[None, :]).sum(axis=1)
    F = zd @ Hz
    g = 2.0 * Hz
    gaps = []
    for b, (lo, hi) in enumerate(((0, n), (n, z.size))):
        gb, zb = g[lo:hi], zd[lo:hi]
        order = np.argsort(gb.astype(np.float64), kind="stable")
        take = np.minimum(dtype(caps[b]), np.maximum(dtype(0.0), dtype(1.0) - dtype(caps[b])
                                                     * np.arange(order.size, dtype=dtype)))
        gaps.append(gb @ zb - gb[order] @ take)
    return float(F), float(gaps[0] + gaps[1]), [float(x) for x in gaps], Hz


def snap(z, n, caps, rel=1e-12):
    """Put coordinates within rel*cap of a bound exactly on it, restoring each sum on a free coordinate."""
    z = z.copy()
    for b, (lo, hi) in enumerate(((0, n), (n, z.size))):
        cap = caps[b]
        zb = z[lo:hi]
        zb[zb < rel * cap] = 0.0
        zb[zb > cap * (1 - rel)] = cap
        resid = 1.0 - zb.sum()
        free = np.where((zb > 0) & (zb < cap))[0]
        if free.size:
            k = free[np.argmax(np.minimum(zb[free], cap - zb[free]))]
            zb[k] += resid
        z[lo:hi] = np.clip(zb, 0.0, cap)
    return z


def polish(H, z, n, caps, tol_rel=1e-10):
    """Exact solve on the current face (free coordinates), kept only if feasible, KKT-consistent and not worse."""
    N = z.size
    free, fixed_val = [], z.copy()
    blk = np.zeros(N, dtype=int)
    blk[n:] = 1
    for b, (lo, hi) in enumerate(((0, n), (n, N))):
        cap = caps[b]
        zb = z[lo:hi]
        fr = lo + np.where((zb > tol_rel * cap) & (zb < cap * (1 - tol_rel)))[0]
        free.append(fr)
    F_idx = np.concatenate(free)
    k = F_idx.size
    info = {"free": int(k)}
    if k == 0:
        info["kept"] = False
        return z, info
    zB = fixed_val.copy()
    zB[F_idx] = 0.0
    # 2 H_FF z_F + 2 H_FB z_B - lam_b = 0 on F;  sum_{F_b} z_F = 1 - sum_{B_b} z_B
    A = np.zeros((k + 2, k + 2))
    rhs = np.zeros(k + 2)
    A[:k, :k] = 2.0 * H[np.ix_(F_idx, F_idx)]
    rhs[:k] = -2.0 * (H[F_idx] @ zB)
    for r, idx in enumerate(F_idx):
        A[r, k + blk[idx]] = -1.0
        A[k + blk[idx], r] = 1.0
    for b, (lo, hi) in enumerate(((0, n), (n, N))):
        rhs[k + b] = 1.0 - zB[lo:hi].sum()
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        info["kept"] = False
        return z, info
    zF, lam = sol[:k], sol[k:]
    cand = zB.copy()
    cand[F_idx] = zF
    capv = np.where(blk == 0, caps[0], caps[1])
    feasible = bool(np.all(zF > 0) and np.all(zF < capv[F_idx]))
    info.update(feasible=feasible, cond=float(np.linalg.cond(A)) if k <= 3000 else float("nan"),
                min_dist_over_cap=float(np.min(np.minimum(zF, capv[F_idx] - zF) / capv[F_idx])))
    if not feasible:
        info["kept"] = False
        return z, info
    F_old = float(z @ (H @ z))
    F_new = float(cand @ (H @ cand))
    info.update(F_old=F_old, F_new=F_new, lam=[float(lam[0]), float(lam[1])])
    keep = F_new <= F_old
    info["kept"] = bool(keep)
    return (cand if keep else z), info


def solve_exact(Kxx, Kyy, Kxy, alpha, eps=1e-14, max_steps=20_000_000, refresh=20_000, fista_iter=2000,
                z0=None, log=print):
    """Stages 1-3 above. Returns (w, v, info)."""
    n, m = Kxx.shape[0], Kyy.shape[0]
    N = n + m
    caps = np.array([1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))])
    t0 = time.time()
    H = np.empty((N, N))
    H[:n, :n] = Kxx
    H[n:, n:] = Kyy
    H[:n, n:] = -Kxy
    H[n:, :n] = -Kxy.T
    info = {"n": n, "m": m, "alpha": float(alpha), "cap_x": float(caps[0]), "cap_y": float(caps[1]),
            "eps": eps, "refresh": refresh}
    if z0 is None:
        r = solve_gram(Kxx, Kyy, Kxy, alpha, max_iter=fista_iter, tol=1e-10, rng=np.random.default_rng(0))
        z = np.concatenate([r.w, r.v])
        info["fista"] = {"iters": int(r.n_iter), "F": float(r.value), "gap": float(r.gap),
                         "seconds": round(time.time() - t0, 1)}
        log(f"  fista: {r.n_iter} it, F={r.value:.6e}, gap={r.gap:.3e}")
    else:
        z = np.asarray(z0, float).copy()
    z = snap(z, n, caps)
    g = 2.0 * (H @ z)
    t1 = time.time()
    total, rounds = 0, 0
    while True:
        steps, viol = _smo(H, z, g, n, caps, max_steps, eps, refresh)
        total += steps
        rounds += 1
        g = 2.0 * (H @ z)                     # fresh gradient
        F, gap, gaps, _ = certificate(H, z, n, caps)
        log(f"  smo round {rounds}: {steps} steps, viol={viol}, F={F:.12e}, gap={gap:.3e}")
        if steps < max_steps or rounds >= 3:
            break
    info["smo"] = {"steps": int(total), "rounds": rounds, "viol_end": [float(x) for x in viol],
                   "seconds": round(time.time() - t1, 1)}
    z_s = z.copy()
    zp, pinfo = polish(H, z, n, caps)
    info["polish"] = pinfo
    if pinfo.get("kept"):
        Fp, gapp, _, _ = certificate(H, zp, n, caps)
        Fs, gaps_, _, _ = certificate(H, z_s, n, caps)
        if gapp <= gaps_:
            z = zp
        else:
            info["polish"]["kept"] = False
            z = z_s
        log(f"  polish: free={pinfo['free']}, F {Fs:.12e} -> {Fp:.12e}, gap {gaps_:.3e} -> {gapp:.3e}")
    else:
        log(f"  polish not kept: {pinfo}")
    F, gap, gaps, Hz = certificate(H, z, n, caps)
    Fl, gapl, gapsl, _ = certificate(H, z, n, caps, dtype=np.longdouble)
    info.update(F=F, gap=gap, gaps=gaps, F_longdouble=Fl, gap_longdouble=gapl, gaps_longdouble=gapsl,
                D_hat=float(np.sqrt(max(F, 0.0))),
                D_hat_lower=float(np.sqrt(max(F - max(gap, gapl, 0.0), 0.0))),
                seconds_total=round(time.time() - t0, 1))
    w, v = z[:n].copy(), z[n:].copy()
    # witness values: d(X_i) = (H z)_i, d(Y_j) = -(H z)_{n+j}
    info["_witness_x"] = Hz[:n].copy()
    info["_witness_y"] = -Hz[n:].copy()
    del H
    return w, v, info
