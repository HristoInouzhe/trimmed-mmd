"""Shared helpers for R6: the Bohman band-limited density K (K>=0, K^ supported in [-1,1], m2=pi^2),
its CDF, the prior discretizations, and the bounds of RESULTS.md R2-R5 evaluated on discrete laws."""
import numpy as np
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid

def K(t):
    t = np.asarray(t, float)
    out = np.empty_like(t)
    near = np.abs(np.abs(t) - np.pi) < 1e-6
    tt = t[~near]
    out[~near] = 4*np.pi*np.cos(tt/2)**2/(np.pi**2-tt**2)**2
    out[near] = 1/(4*np.pi)   # limit at |t|=pi: 4pi*(1/4)(t-pi)^2/((2pi)^2 (t-pi)^2) = 1/(4 pi)
    return out

def Khat(xi):
    a = np.minimum(np.abs(xi), 1.0)
    return np.where(np.abs(xi) <= 1, (1-a)*np.cos(np.pi*a)+np.sin(np.pi*a)/np.pi, 0.0)

# CDF H(t) of K on a fine grid; analytic tail 2pi/(3t^3) (average of cos^2 is 1/2) beyond T
_T = 400.0
_tg = np.linspace(0, _T, 800001)
_kg = K(_tg)
_cum = cumulative_trapezoid(_kg, _tg, initial=0.0)
_tail_T = 0.5 - _cum[-1]          # mass beyond T computed from normalisation

def H(t):
    t = np.asarray(t, float)
    a = np.abs(t)
    inside = np.interp(np.minimum(a, _T), _tg, _cum)
    tail_beyond = np.where(a > _T, _tail_T*(_T/np.maximum(a, _T))**3, 0.0)
    val = np.where(a <= _T, inside, 0.5 - tail_beyond)
    return 0.5 + np.sign(t)*val

def mixture_grid(mu, a=0.2, L=9.5, h=0.01):
    cP = lambda t: (1-a)*norm.cdf(t)+a*norm.cdf(t-mu)
    cQ = lambda t: (1-a)*norm.cdf(t)+a*norm.cdf(t+mu)
    edges = np.arange(-L, L+h/2, h)
    x = 0.5*(edges[1:]+edges[:-1]); p = np.diff(cP(edges)); q = np.diff(cQ(edges))
    p = np.clip(p, 0, None); q = np.clip(q, 0, None); p /= p.sum(); q /= q.sum()
    return x, p, q

def lap_grid(mu, hf, hc, w0, a=0.2, L=9.5):
    cP = lambda t: (1-a)*norm.cdf(t)+a*norm.cdf(t-mu)
    cQ = lambda t: (1-a)*norm.cdf(t)+a*norm.cdf(t+mu)
    inner = np.arange(0, w0+hf/2, hf); outer = np.arange(w0+hc, L+hc/2, hc)
    pos = np.r_[inner, outer]; edges = np.r_[-pos[::-1], pos[1:]]
    x = 0.5*(edges[1:]+edges[:-1]); p = np.diff(cP(edges)); q = np.diff(cQ(edges)); p /= p.sum(); q /= q.sum()
    return x, p, q

def bathtub_sigma(fv, p, q, m):
    """sigma(f;alpha) = min_{0<=h<=1, sum h p = m} sum h p f - max_{...} sum h q f (exact dual value)."""
    o = np.argsort(fv, kind='stable')
    def low(w, vals, order):
        cw = np.cumsum(w[order]); k = np.searchsorted(cw, m); k = min(k, len(w)-1)
        prev = cw[k-1] if k > 0 else 0.0
        return np.sum(w[order[:k]]*vals[order[:k]]) + (m-prev)*vals[order[k]]
    lowP = low(p, fv, o)
    highQ = -low(q, -fv, o[::-1])
    return lowP - highQ

def eta(fv, x, p, q):
    A = (x > 0).astype(float)
    return np.sum(np.abs(A-fv)*np.abs(p-q))
