"""Optional GPU (torch) backend for the hot path: Gram-matrix formation and
the FISTA solve.

This module is imported lazily and must never be a hard dependency: every
public entry point in :mod:`trimmedmmd.discrepancy` still works with plain
numpy if torch is absent or no GPU is visible. The device policy is
conservative on purpose (see :func:`pick_device`) because this machine's GPU
is typically shared with other tenants (theory-stress-suite non-negotiable:
"share the machine").

Numerical agreement with the numpy reference (:mod:`trimmedmmd.solver`,
:mod:`trimmedmmd.kernels`) is checked in ``tests/test_gpu_agreement.py``,
not asserted here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - environment without torch
    torch = None
    _HAVE_TORCH = False


def available() -> bool:
    """True if torch is importable and a CUDA device is visible."""
    return _HAVE_TORCH and torch.cuda.is_available()


def gpu_free_gb() -> float | None:
    if not available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def pick_device(requested: str | None = "auto", min_free_gb: float = 1.5) -> str:
    """Resolve a device string.

    ``requested`` is "cpu", "cuda", or "auto" (use cuda only if it looks
    genuinely free enough to be worth it). Never raises: unavailable or
    insufficiently free GPU silently falls back to "cpu", because a stress
    suite must not crash because another tenant is using the card.
    """
    if requested == "cpu":
        return "cpu"
    if not available():
        return "cpu"
    if requested == "cuda":
        return "cuda"
    # auto
    free = gpu_free_gb()
    return "cuda" if (free is not None and free >= min_free_gb) else "cpu"


def _require_torch():
    if not _HAVE_TORCH:
        raise RuntimeError(
            "torch is not installed; install it (see the project README) "
            "or pass device='cpu'"
        )


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

def sqdist_t(X, Y):
    xx = (X * X).sum(dim=1, keepdim=True)
    yy = (Y * Y).sum(dim=1, keepdim=True).T
    d = xx + yy - 2.0 * (X @ Y.T)
    return d.clamp_(min=0.0)


def gaussian_kernel_t(X, Y, sigma: float):
    return torch.exp(-sqdist_t(X, Y) / (2.0 * sigma * sigma))


# --------------------------------------------------------------------------
# capped-simplex projection, fully vectorized bisection (no host syncs
# inside the loop -- essential for GPU throughput; see simplex.py for the
# numpy reference this must agree with)
# --------------------------------------------------------------------------

def project_capped_simplex_t(y, cap: float, total: float = 1.0, iters: int = 100):
    n = y.numel()
    if cap * n < total - 1e-12:
        raise ValueError(f"infeasible: cap*n = {cap * n:.6g} < total = {total:.6g}")
    lo = y.min() - cap - 1.0
    hi = y.max() + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = torch.clamp(y - mid, 0.0, cap).sum()
        # s is nonincreasing in mid: if s > total, mid is too small (raise lo)
        lo = torch.where(s > total, mid, lo)
        hi = torch.where(s > total, hi, mid)
    w = torch.clamp(y - 0.5 * (lo + hi), 0.0, cap)
    ssum = w.sum()
    if ssum > 0:
        w = torch.clamp(w * (total / ssum), 0.0, cap)
    return w


# --------------------------------------------------------------------------
# FISTA QP solve (mirrors solver.py exactly, tensor ops only)
# --------------------------------------------------------------------------

@dataclass
class QPResultGPU:
    value: float
    w: "object"    # numpy array (converted back on return)
    v: "object"
    n_iter: int
    converged: bool

    @property
    def discrepancy(self) -> float:
        return float(max(self.value, 0.0)) ** 0.5


def _power_iteration_t(matvec, dim, device, dtype, iters=60, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(dim, generator=g, dtype=dtype).to(device)
    x = x / x.norm()
    lam = torch.tensor(1.0, device=device, dtype=dtype)
    for _ in range(iters):
        y = matvec(x)
        ny = y.norm()
        if float(ny) <= 0:
            return 0.0
        x = y / ny
        lam = ny
    return float(lam)


def _fista_t(grad, objective, w0, v0, cap_w, cap_v, lipschitz,
             max_iter=2000, tol=1e-10):
    w, v = w0.clone(), v0.clone()
    yw, yv = w.clone(), v.clone()
    t = 1.0
    step = 1.0 / lipschitz
    prev_obj = None
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        gw, gv = grad(yw, yv)
        w_new = project_capped_simplex_t(yw - step * gw, cap_w)
        v_new = project_capped_simplex_t(yv - step * gv, cap_v)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        beta = (t - 1.0) / t_new
        yw = w_new + beta * (w_new - w)
        yv = v_new + beta * (v_new - v)
        w, v, t = w_new, v_new, t_new
        if it % 10 == 0 or it == max_iter:
            obj = float(objective(w, v))
            if prev_obj is not None and obj > prev_obj:
                yw, yv, t = w.clone(), v.clone(), 1.0
            if prev_obj is not None and \
                    abs(prev_obj - obj) <= tol * max(abs(prev_obj), 1e-300):
                converged = True
                prev_obj = obj
                break
            prev_obj = obj
    return w, v, float(objective(w, v)), it, converged


def solve_gram_gpu(Kxx, Kyy, Kxy, alpha, w0=None, v0=None,
                    max_iter=2000, tol=1e-10, device="cuda", seed=0):
    """GPU-backed twin of :func:`trimmedmmd.solver.solve_gram`.

    Accepts and returns numpy arrays so callers do not need to know the
    backend. ``Kxx``/``Kyy``/``Kxy`` may be numpy arrays or torch tensors
    already on ``device``.
    """
    _require_torch()
    import numpy as np
    dtype = torch.float64

    def to_t(a):
        if isinstance(a, torch.Tensor):
            return a.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)

    Kxx, Kyy, Kxy = to_t(Kxx), to_t(Kyy), to_t(Kxy)
    n, m = Kxx.shape[0], Kyy.shape[0]
    cap_w, cap_v = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))

    def grad(w, v):
        return (2.0 * (Kxx @ w - Kxy @ v), 2.0 * (Kyy @ v - Kxy.T @ w))

    def objective(w, v):
        return w @ (Kxx @ w) + v @ (Kyy @ v) - 2.0 * (w @ (Kxy @ v))

    def matvec(z):
        w, v = z[:n], z[n:]
        return torch.cat([Kxx @ w - Kxy @ v, Kyy @ v - Kxy.T @ w])

    lam = _power_iteration_t(matvec, n + m, device, dtype, seed=seed)
    lipschitz = max(2.0 * lam, 1e-12)

    w0_t = (torch.full((n,), 1.0 / n, device=device, dtype=dtype) if w0 is None
            else torch.clamp(to_t(w0), 0, cap_w))
    v0_t = (torch.full((m,), 1.0 / m, device=device, dtype=dtype) if v0 is None
            else torch.clamp(to_t(v0), 0, cap_v))
    w0_t = project_capped_simplex_t(w0_t, cap_w)
    v0_t = project_capped_simplex_t(v0_t, cap_v)

    w, v, val, it, conv = _fista_t(grad, objective, w0_t, v0_t, cap_w, cap_v,
                                    lipschitz, max_iter, tol)
    return QPResultGPU(value=max(val, 0.0), w=w.cpu().numpy(),
                        v=v.cpu().numpy(), n_iter=it, converged=conv)


def solve_features_gpu(Zx, Zy, alpha, w0=None, v0=None,
                        max_iter=2000, tol=1e-10, device="cuda", seed=0):
    """GPU-backed twin of :func:`trimmedmmd.solver.solve_features`."""
    _require_torch()
    import numpy as np
    dtype = torch.float64

    def to_t(a):
        if isinstance(a, torch.Tensor):
            return a.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)

    Zx, Zy = to_t(Zx), to_t(Zy)
    n, m = Zx.shape[0], Zy.shape[0]
    cap_w, cap_v = 1.0 / (n * (1.0 - alpha)), 1.0 / (m * (1.0 - alpha))

    def resid(w, v):
        return Zx.T @ w - Zy.T @ v

    def grad(w, v):
        d = resid(w, v)
        return (2.0 * (Zx @ d), -2.0 * (Zy @ d))

    def objective(w, v):
        d = resid(w, v)
        return d @ d

    def matvec(z):
        w, v = z[:n], z[n:]
        d = Zx.T @ w - Zy.T @ v
        return torch.cat([Zx @ d, -(Zy @ d)])

    lam = _power_iteration_t(matvec, n + m, device, dtype, seed=seed)
    lipschitz = max(2.0 * lam, 1e-12)

    w0_t = (torch.full((n,), 1.0 / n, device=device, dtype=dtype) if w0 is None
            else torch.clamp(to_t(w0), 0, cap_w))
    v0_t = (torch.full((m,), 1.0 / m, device=device, dtype=dtype) if v0 is None
            else torch.clamp(to_t(v0), 0, cap_v))
    w0_t = project_capped_simplex_t(w0_t, cap_w)
    v0_t = project_capped_simplex_t(v0_t, cap_v)

    w, v, val, it, conv = _fista_t(grad, objective, w0_t, v0_t, cap_w, cap_v,
                                    lipschitz, max_iter, tol)
    return QPResultGPU(value=max(val, 0.0), w=w.cpu().numpy(),
                        v=v.cpu().numpy(), n_iter=it, converged=conv)


def gpu_memory_cap_bytes(max_gb: float | None = None, headroom_gb: float = 2.0) -> int:
    """A conservative byte budget for a single allocation on the shared card.

    Defaults to leaving ``headroom_gb`` for other tenants beyond whatever is
    currently reported free, and never assumes the whole card is ours.
    """
    free = gpu_free_gb()
    if free is None:
        return 0
    budget_gb = max(0.0, free - headroom_gb)
    if max_gb is not None:
        budget_gb = min(budget_gb, max_gb)
    return int(budget_gb * (1024 ** 3))


if __name__ == "__main__":
    print("torch available:", _HAVE_TORCH)
    print("cuda available:", available())
    if available():
        print("free GB:", round(gpu_free_gb(), 2))
        print("picked device (auto):", pick_device("auto"))
