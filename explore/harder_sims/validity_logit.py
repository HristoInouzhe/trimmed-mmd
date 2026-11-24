"""Investigation (post hoc): logit_bet rejected the boundary null of D1_d2_a20 at
n = 1600 in 13/200 replicates (0.065; binomial p = 0.24, not flagged). In D1 the
Bayes set {p > q} is a half-space, so a well-fit logistic rule makes the
boundary nearly TIGHT (Delta = E_P g - E_Q g close to TV), the regime where a
level-delta test should reject close to delta. This script
  (1) regenerates replicates 0..199 exactly (must reproduce the 13 rejections),
  (2) adds replicates 200..R-1 (fresh seeds of the same SeedSequence scheme),
  (3) computes for every fitted rule the exact population Delta: for a
      half-space g = 1{w'x + b > 0} and a Gaussian mixture,
      E_P g - E_Q g = a [Phi((w'mu + b)/|w|) - Phi((-w'mu + b)/|w|)] <= TV.
Output: out/validity_logit.json
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import math
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import methods as M  # noqa: E402
import designs as DS  # noqa: E402
import run as R  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm, binom, beta as B  # noqa: E402
import sacert as S  # noqa: E402


def one(key, n, rep):
    dz = DS.catalogue()[key]
    m = n
    ss = np.random.SeedSequence([R.seed_key(key), n, rep])
    rng = np.random.default_rng(ss)
    X = dz.sample(n, rng, "P")
    Y = dz.sample(m, rng, "Q")
    px, py = rng.permutation(n), rng.permutation(m)
    X, Y = X[px], Y[py]
    n1, m1 = n // 2, m // 2
    X1, X2, Y1, Y2 = X[:n1], X[n1:], Y[:m1], Y[m1:]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(np.vstack([X1, Y1]), np.r_[np.ones(n1, int), np.zeros(m1, int)])
        c = list(clf.classes_).index(1)
        G = (clf.predict_proba(X2)[:, c] > 0.5).astype(float)
        H = (clf.predict_proba(Y2)[:, c] > 0.5).astype(float)
    a = dz.alpha_star
    rej = S.betting_test(S.pair_up(G, -H) - a, -(1.0 + a), M.DELTA, c=0.5, prior_var=0.5)[0]
    sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    sign = 1.0 if lr.classes_[1] == 1 else -1.0
    w = sign * lr.coef_[0] / sc.scale_
    b = sign * (lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))
    nw = np.linalg.norm(w)
    mu = dz.mu
    delta_pop = dz.a * (norm.cdf((w @ mu + b) / (nw * dz.s)) - norm.cdf((-w @ mu + b) / (nw * dz.s)))
    return bool(rej), float(delta_pop), float(G.mean() - H.mean())


if __name__ == "__main__":
    key, n, Rtot = "D1_d2_a20", 1600, int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    stored = {}
    for line in (HERE / "out" / f"{key}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["n"] == n:
            stored[r["rep"]] = r["dec"]["logit_bet"][3]
    res = [one(key, n, r) for r in range(Rtot)]
    rej = np.array([x[0] for x in res])
    dpop = np.array([x[1] for x in res])
    match = all(int(rej[r]) == stored[r] for r in range(min(200, Rtot)) if r in stored)
    k_all, k_new = int(rej.sum()), int(rej[200:].sum())
    out = {"design": key, "n": n, "alpha": DS.catalogue()[key].alpha_star, "reps": Rtot,
           "reproduces_stored_0_199": match, "rej_0_199": int(rej[:200].sum()),
           "rej_200_end": k_new, "rate_200_end": k_new / (Rtot - 200),
           "rej_all": k_all, "rate_all": k_all / Rtot,
           "cp95_upper_all": float(B.ppf(0.975, k_all + 1, Rtot - k_all)),
           "binom_p_all_vs_0.05": float(binom.sf(k_all - 1, Rtot, 0.05)),
           "delta_pop_mean": float(dpop.mean()), "delta_pop_max": float(dpop.max()),
           "delta_pop_q10_q50_q90": [float(x) for x in np.quantile(dpop, [0.1, 0.5, 0.9])],
           "max_delta_pop_minus_TV": float(dpop.max() - DS.catalogue()[key].alpha_star)}
    (HERE / "out" / "validity_logit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
