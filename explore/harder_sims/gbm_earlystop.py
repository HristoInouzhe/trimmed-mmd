"""POST HOC diagnostic (DESIGN.md section 9, 2026-11-10; not a method of the study): on the stored alpha = 0 replicates of D3b_corr, refit gradient boosting with the
default early stopping and with early_stopping=False, and compare held-out TPR - FPR and the betting decision.
Why gradient boosting's power on the flip drops between n = 6400 and 12800: scikit-learn's HistGradientBoostingClassifier
switches its default early stopping on above 10000 training points. Writes out/gbm_earlystop.json.

    PYTHONPATH=src python3 -I explore/harder_sims/gbm_earlystop.py 60
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, math, warnings
sys.path.insert(0, "/nvme0n1-disk/trimmed-mmd/explore/harder_sims")
import numpy as np
import run as R, designs as DS, methods as M, sacert as S
from concurrent.futures import ProcessPoolExecutor

def one(args):
    key, n, rep = args
    from sklearn.ensemble import HistGradientBoostingClassifier
    dz = DS.catalogue()[key]
    m = dz.m_ratio * n
    ss = np.random.SeedSequence([R.seed_key(key), n, rep])
    rng = np.random.default_rng(ss)
    X = dz.sample(n, rng, "P"); Y = dz.sample(m, rng, "Q")
    px, py = rng.permutation(n), rng.permutation(m)
    X, Y = X[px], Y[py]
    n1, m1 = n // 2, m // 2
    ss.spawn(1); ss.spawn(2); ss.spawn(3)
    cseed = int(ss.generate_state(1)[0] % (2 ** 31 - 1))
    out = {}
    for es in ("auto", False):
        clf = HistGradientBoostingClassifier(random_state=cseed, early_stopping=es)
        Z1 = np.vstack([X[:n1], Y[:m1]]); lab = np.r_[np.ones(n1, int), np.zeros(m1, int)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Z1, lab)
            px2 = clf.predict_proba(X[n1:])[:, 1]; py2 = clf.predict_proba(Y[m1:])[:, 1]
        G, H = (px2 > 0.5).astype(float), (py2 > 0.5).astype(float)
        dec = S.betting_test(S.pair_up(G, -H), -1.0, 0.05, c=0.5, prior_var=0.5)[0]
        out[str(es)] = dict(diff=float(G.mean() - H.mean()), dec=int(dec), iters=int(clf.n_iter_))
    return key, n, rep, out

if __name__ == "__main__":
    jobs = [("D3b_corr", n, 200000 + r) for n in (6400, 12800) for r in range(int(sys.argv[1]))]
    stored = {}
    for line in open("/nvme0n1-disk/trimmed-mmd/explore/harder_sims/out/D3b_corr_n0ext.jsonl"):
        r = json.loads(line); stored[(r["n"], r["rep"])] = r
    with ProcessPoolExecutor(15) as ex:
        res = list(ex.map(one, jobs))
    summary = {}
    for n in (6400, 12800):
        rr = [x for x in res if x[1] == n]
        match = sum(x[3]["auto"]["dec"] == stored[(n, x[2])]["dec0"]["gbm_bet"] for x in rr)
        summary[str(n)] = {"replicates": len(rr), "default_reproduces_stored": match}
        for es in ("auto", "False"):
            summary[str(n)][f"early_stopping_{es}"] = {
                "power": float(np.mean([x[3][es]["dec"] for x in rr])),
                "mean_diff": float(np.mean([x[3][es]["diff"] for x in rr])),
                "mean_iters": float(np.mean([x[3][es]["iters"] for x in rr]))}
    print(json.dumps(summary, indent=1))
    (R.OUT / "gbm_earlystop.json").write_text(json.dumps(summary, indent=1) + "\n")
