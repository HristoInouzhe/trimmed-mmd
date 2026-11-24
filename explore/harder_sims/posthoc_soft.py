"""POST HOC (added after the n <= 200 results were seen; see DESIGN.md §9):
soft-score classifier certificates on the SAME replicates.

For a classifier fit on fold one, g(z) = the prior-corrected p_hat(X | z) (an estimate of p/(p+q); = p_hat when
n = m) is a fixed [0,1]-valued
function on fold two, so E_P g - E_Q g <= TV(P,Q) and the betting test of
sacert Thm 1 applies verbatim (same c = 1/2, WSR plug-in, prior weight 20,
neutral prior variance 0.5 as for the hard-label bets). This separates "kernel
vs classifier" from "soft score vs hard label".

Data, permutation and classifier seeds are regenerated exactly as in
methods.run_replicate; the hard-label decisions are recomputed and compared
with the stored ones (reproducibility check, field `hard_match`).

    PYTHONPATH=src python3 -I explore/harder_sims/posthoc_soft.py --workers 16
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import methods as M  # noqa: E402
import designs as DS  # noqa: E402
import run as R  # noqa: E402

import numpy as np  # noqa: E402
import sacert as S  # noqa: E402

OUTD = HERE / "out" / "posthoc_soft"
OUTD.mkdir(exist_ok=True, parents=True)
CL = ("gbm", "logit", "knn")


def probs(name, X1, Y1, X2, Y2, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n1, m1 = len(X1), len(Y1)
    clf = {"gbm": lambda: HistGradientBoostingClassifier(random_state=seed),
           "logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
           "knn": lambda: KNeighborsClassifier(n_neighbors=int(math.sqrt(n1 + m1)) | 1)}[name]()
    Z1 = np.vstack([X1, Y1])
    lab = np.r_[np.ones(n1, int), np.zeros(m1, int)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Z1, lab)
        c = list(clf.classes_).index(1)
        return clf.predict_proba(X2)[:, c], clf.predict_proba(Y2)[:, c]


def balanced(p, pi1):
    """Prior-corrected posterior for equal class weights, an estimate of
    p/(p+q); equals p when pi1 = 1/2. Any fold-one map into [0,1] is valid."""
    a = p / pi1
    b = (1.0 - p) / (1.0 - pi1)
    return a / np.maximum(a + b, 1e-300)


def soft_tests(G, H, levels, delta=M.DELTA):
    Zb = S.pair_up(G, -H)
    dec, alive, jlo = {}, True, -1
    for j, a in enumerate(levels):
        if not alive and j not in M.REPORT_J:
            continue
        r = S.betting_test(Zb - a, -(1.0 + a), delta, c=0.5, prior_var=0.5)[0]
        dec[j] = r
        if alive and r:
            jlo = j
        elif alive:
            alive = False
    return [int(dec[j]) for j in M.REPORT_J], jlo


def one(args):
    key, n, rep, stored = args
    C = DS.catalogue()
    dz = C[key]
    pil = json.loads(R.PILOT.read_text())[key]
    m = dz.m_ratio * n
    ss = np.random.SeedSequence([R.seed_key(key), n, rep])
    rng = np.random.default_rng(ss)
    X = dz.sample(n, rng, "P")
    Y = dz.sample(m, rng, "Q")
    if pil["scale"] is not None:
        s = np.array(pil["scale"])
        X, Y = X / s, Y / s
    px, py = rng.permutation(n), rng.permutation(m)
    X, Y = X[px], Y[py]
    n1, m1 = n // 2, m // 2
    X1, X2, Y1, Y2 = X[:n1], X[n1:], Y[:m1], Y[m1:]
    cseed = int(ss.generate_state(1)[0] % (2 ** 31 - 1))
    levels = dz.levels()
    rec = {"design": key, "n": n, "m": m, "rep": rep, "dec": {}, "jlo": {}, "hard_match": {}, "time": {}}
    thr = n1 / (n1 + m1)
    for c in CL:
        t0 = time.perf_counter()
        pX, pY = probs(c, X1, Y1, X2, Y2, cseed)
        dec, jlo = soft_tests(balanced(pX, thr), balanced(pY, thr), levels)
        rec["dec"][f"{c}_soft_bet"] = dec
        rec["jlo"][f"{c}_soft_bet"] = jlo
        hd, hj, _ = M.classifier_tests((pX > thr).astype(float), (pY > thr).astype(float), levels)
        rec["hard_match"][c] = bool(hd["hoef"] == stored["dec"][f"{c}_hoef"] and hd["bet"] == stored["dec"][f"{c}_bet"]
                                    and hj["hoef"] == stored["jlo"][f"{c}_hoef"] and hj["bet"] == stored["jlo"][f"{c}_bet"])
        rec["time"][c] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--designs", nargs="+", default=["all"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    keys = list(DS.catalogue()) if a.designs == ["all"] else a.designs
    jobs = []
    for key in keys:
        f = HERE / "out" / f"{key}.jsonl"
        if not f.exists():
            continue
        done = set()
        g = OUTD / f"{key}.jsonl"
        if g.exists():
            for line in g.read_text().splitlines():
                r = json.loads(line)
                done.add((r["n"], r["rep"]))
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["rep"] < 200 and (r["n"], r["rep"]) not in done:
                jobs.append((key, r["n"], r["rep"], {"dec": r["dec"], "jlo": r["jlo"]}))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    handles = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(one, jobs, chunksize=8)):
            f = handles.get(rec["design"])
            if f is None:
                f = handles[rec["design"]] = open(OUTD / f"{rec['design']}.jsonl", "a")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    for f in handles.values():
        f.close()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
