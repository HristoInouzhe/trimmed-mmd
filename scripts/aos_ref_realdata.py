"""AoS referees (aos/reviews/referee2_methods.md M6, M7, M8 and MISMATCH 3; aos/reviews/referee1_theory.md): the
real-data computations they ask for. The method throughout is the paper's split-sample bound: Laplace kernel,
sigma = the median distance on batch 0, genuine-quantile cut points (cut rule "order"), batches (0, 1/4, 1/2, 1),
eight splits (seeds 1000-1007), grid mesh 0.05, delta = 0.05 (scripts/aos_main_realdata.py, whose level_record is
reused unchanged). Nothing stored is rewritten; every record is appended and fsynced on local disk
($AOS_REF_RD_WORK, default /home/hristouam/aos_main_work/ref_realdata) and `sync` copies it atomically to
results/aos/main/raw/ref_realdata_<kind>/. A killed run resumes (finished keys are skipped).

Commands

  evalues     M6. Extends the Laplace chains of results/aos/main/raw/realdata_laplace (rhc, the three NHANES contrasts,
              and rhc_full, the 45-covariate RHC run of scripts/aos_main_realdata.py --contrasts rhc_full) level by
              level, all eight splits at once, while one of the averaged e-value tests is still rejecting: the
              average over the splits of the wealth stopped at its first crossing of 1/delta (or at N), and of the
              final wealth K_N, each against 1/delta (level delta) and 1/(2 delta) (level 2 delta). The levels a chain
              already holds are read, the others computed by aos_main_realdata.level_record (kind ext).
  cluster     M8. NHANES with whole sampling units assigned to batches. The public files have no household
              identifier, so the cluster is the masked variance unit (SDMVPSU within SDMVSTRA; strata are numbered
              apart in cycles G and H, 61 units). Per split seed, the units are permuted and each goes to the batch in
              which the midpoint of its cumulative share of the contrast's participants falls (edges 1/4, 1/2), then
              the participants of a batch are permuted within it, the bandwidth is the median distance on batch 0,
              and the witnesses, cut points, pairs and bets are those of level_record (the horizon is the sum over the
              betting batches of min(n_b, m_b), fixed by the unit assignment before any curve is seen). Read upward
              to the first failure of both cut rules (kind cluster).
  classifiers M7. RHC (33 and 45 covariates): gradient boosting and logistic regression fitted on the earlier batches
              of the ramp's schedule (quarter 1 -> quarter 2, half 1 -> half 2, one wealth, horizon fixed), the
              classifiers and bets of the synthetic study (scripts/aos_baselines_run.py: make_clf, fit_predict,
              balanced, tuned_probs, stream_decisions; hard labels 1{p > training share of X}, the prior-corrected
              probability, and the CV-tuned versions), on the split order of the bound (batch-0 standardization),
              every grid level (kind clf).
  propensity  M7.1. The propensity-model estimate of alpha* = TV(law of e(X) | catheterized, law of e(X) | others):
              ridge logistic regression of Table S7 (experiments.realdata_run._logistic_fit, pooled standardization)
              on the 33 and on the 45 covariates; 1 - OVL (the 40-bin overlap of Table S7) and the plug-in
              1 - mean_i min{e_i / pi, (1 - e_i) / (1 - pi)}; 95% percentile intervals from 2,000 arm-stratified
              bootstrap resamples with the model refitted; five-fold cross-fitted values as a check.
  estimand    The five comparison rows of Table tab:cohort (all patients, matching, overlap, energy and kernel
              balancing) on the 45 covariates, exactly as scripts/v3_rhc_estimand.py (same propensity model,
              standardization, bandwidth, solvers, bootstrap stream); --covariates paper recomputes the stored rows
              as a check (kernel balancing skipped with --skip-kernel).
  check       The generic batch-level code of `cluster` on the participant-level order reproduces stored records of
              results/aos/main/raw/realdata_laplace exactly.
  sync        Copy the local records to results/aos/main/raw/ref_realdata_<kind>/.

    PYTHONPATH=src python -B scripts/aos_ref_realdata.py evalues --workers 3 --threads 2
    PYTHONPATH=src python -B scripts/aos_ref_realdata.py cluster --workers 3 --threads 2
    PYTHONPATH=src python -B scripts/aos_ref_realdata.py classifiers
    PYTHONPATH=src python -B scripts/aos_ref_realdata.py propensity
    PYTHONPATH=src python -B scripts/aos_ref_realdata.py estimand --covariates full --threads 3

Tables, macros (\\rr...) and results/aos/main/REFEREE_RUNS.md: scripts/aos_ref_realdata_tables.py.
"""
from __future__ import annotations

import os
import sys


def _arg(name, default):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_T = _arg("--threads", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import aos_main_realdata as AM  # noqa: E402
from trimmedmmd.aos_fallback import bet, choose_cuts, fit_batch  # noqa: E402
from trimmedmmd.kernels import median_heuristic  # noqa: E402
from trimmedmmd.splitting import (SEQ_EDGES, pair_statistics, ramp, ramp_prior_var,  # noqa: E402
                                  sequential_horizon, sequential_order)

WORK = Path(os.environ.get("AOS_REF_RD_WORK", "/home/hristouam/aos_main_work/ref_realdata"))
RAW = ROOT / "results" / "aos" / "main" / "raw"
STORED = RAW / "realdata_laplace"
OUTD = ROOT / "results" / "aos" / "main" / "ref_realdata"
SEEDS = AM.SEEDS
DELTA = 0.05
EV_CONTRASTS = ["rhc", "nhanes_child", "nhanes_age", "nhanes_bmi", "rhc_full"]
NH_CONTRASTS = ["nhanes_child", "nhanes_age", "nhanes_bmi"]
EV_THRESH = {"d": 1.0 / DELTA, "2d": 1.0 / (2.0 * DELTA)}       # level delta, level 2 delta
EV_KINDS = ("log_stopped", "log_final")
CLF_SETS = {"rhc": "paper", "rhc_full": "full"}
PROP_B, PROP_SEED, PROP_FOLDS = 2000, 20261122, 5


def grid_for(name):
    return AM.grid_for(name)


def split_seed_of(seed):
    return int(np.random.default_rng(seed).integers(0, 2**31 - 1))       # as lower_bound_betting / level_record


# ----------------------------------------------------------------------------------------------------------------
# records on local disk
# ----------------------------------------------------------------------------------------------------------------

def kind_dir(kind, local=True):
    return (WORK if local else RAW) / f"ref_realdata_{kind}"


def append(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path):
    out = []
    if Path(path).exists():
        for line in Path(path).read_bytes().split(b"\n"):
            if not line.strip() or b"\x00" in line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def read_kind(kind, fname):
    """Records of one file of a kind: the local copy if it exists, else the synced one."""
    loc = kind_dir(kind) / fname
    return read_jsonl(loc if loc.exists() else kind_dir(kind, local=False) / fname)


def sync(log=print):
    nf = nr = 0
    for d in sorted(WORK.glob("ref_realdata_*")):
        od = RAW / d.name
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.jsonl")):
            recs = read_jsonl(f)
            txt = "".join(json.dumps(r) + "\n" for r in recs)
            tmp = od / (f.name + ".tmp")
            tmp.write_text(txt)
            os.replace(tmp, od / f.name)
            if (od / f.name).read_text() != txt:
                log(f"SYNC MISMATCH {f}")
            nf += 1
            nr += len(recs)
    for f in sorted(WORK.glob("*.json")):
        txt = f.read_text()
        OUTD.mkdir(parents=True, exist_ok=True)
        tmp = OUTD / (f.name + ".tmp")
        tmp.write_text(txt)
        os.replace(tmp, OUTD / f.name)
        nf += 1
    log(f"synced {nf} files ({nr} records) to {RAW} and {OUTD}")


# ----------------------------------------------------------------------------------------------------------------
# averaged e-values (M6)
# ----------------------------------------------------------------------------------------------------------------

def chain(name, seed):
    """{alpha: record} of one split: the stored chain (realdata_laplace) and the extension levels (kind ext)."""
    recs = {round(r["alpha"], 6): r for r in read_jsonl(STORED / f"{name}_{seed}.jsonl")}
    for r in read_kind("ext", f"{name}_{seed}.jsonl"):
        k = round(r["alpha"], 6)
        assert k not in recs or recs[k]["order"] == r["order"], (name, seed, k)
        recs.setdefault(k, r)
    return recs


def ev_status(name, chains):
    """Walk the grid upward; for each averaged test the last level it certifies (None: not even alpha = 0) and whether
    it is still alive after the last level with all eight records. Returns (status, k_complete)."""
    grid = grid_for(name)
    alive = {(kd, t): True for kd in EV_KINDS for t in EV_THRESH}
    last = {key: None for key in alive}
    k_done = -1
    for k, a in enumerate(grid):
        key_a = round(float(a), 6)
        if not all(key_a in chains[s] for s in SEEDS):
            break
        k_done = k
        for kd, t in alive:
            if not alive[(kd, t)]:
                continue
            avg = float(np.mean([math.exp(chains[s][key_a]["order"][kd]) for s in SEEDS]))
            if avg >= EV_THRESH[t]:
                last[(kd, t)] = float(a)
            else:
                alive[(kd, t)] = False
    return alive, last, k_done


def ev_job(args):
    name, seed, alpha = args
    rec = AM.level_record(name, seed, alpha)
    rec["ref_kind"] = "ext"
    return rec


def run_evalues(contrasts, workers, log):
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    for name in contrasts:
        for s in SEEDS:
            assert (STORED / f"{name}_{s}.jsonl").exists(), f"missing stored chain {name}_{s}"
    with ProcessPoolExecutor(workers) as ex:
        active = set(contrasts)
        pending = {}
        while active or pending:
            for name in list(active):
                chains = {s: chain(name, s) for s in SEEDS}
                alive, last, k_done = ev_status(name, chains)
                grid = grid_for(name)
                if not any(alive.values()) or k_done + 1 >= len(grid):
                    active.discard(name)
                    log(f"{name}: averaged tests settled; last certified {last}")
                    continue
                a = float(grid[k_done + 1])
                for s in SEEDS:
                    if round(a, 6) not in chains[s] and not any(v == (name, s, a) for v in pending.values()):
                        pending[ex.submit(ev_job, (name, s, a))] = (name, s, a)
                if any(v[0] == name for v in pending.values()):
                    active.discard(name)            # re-activated when its jobs finish
            if not pending:
                continue
            fin, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fu in fin:
                name, s, a = pending.pop(fu)
                rec = fu.result()
                append(kind_dir("ext") / f"{name}_{s}.jsonl", rec)
                log(f"ext {name} seed={s} a={a:.2f} order cert={rec['order']['certified']} "
                    f"stop={rec['order']['log_stopped']:.3f} final={rec['order']['log_final']:.3f} {rec['seconds']}s")
                if not any(v[0] == name for v in pending.values()):
                    active.add(name)


# ----------------------------------------------------------------------------------------------------------------
# a level of the certificate on given batches (shared by `cluster` and `check`)
# ----------------------------------------------------------------------------------------------------------------

def level_on_batches(Xo, Yo, bx, by, sigma, rng, alpha, kernel=AM.KERNEL, cuts=AM.CUTS):
    """aos_main_realdata.level_record on explicit batch boundaries bx, by (lists [0, b1, b2, n]): the same fits (in
    the same order on the same stream), the same cut rules, pairs, variance guesses and bets."""
    nb_ = len(bx) - 1
    N = int(sum(min(bx[b + 1] - bx[b], by[b + 1] - by[b]) for b in range(1, nb_)))
    fits = []
    for b in range(1, nb_):
        Xf, Yf = Xo[:bx[b]], Yo[:by[b]]
        Xb, Yb = Xo[bx[b]:bx[b + 1]], Yo[by[b]:by[b + 1]]
        fits.append((fit_batch(Xf, Yf, Xb, Yb, float(alpha), sigma, rng, kernel=kernel), len(Xb), len(Yb)))
    rec = {"horizon": N, "norms": [float(bf.norm) for bf, _, _ in fits]}
    for c in cuts:
        Zs, pri, src, cts = [], [], [], []
        for bf, nb, mb in fits:
            s, sp, u, v, pick, _ = choose_cuts(bf.u_own, bf.v_own, bf.u_plain, bf.v_plain, float(alpha), "own",
                                               cut_rule=c)
            src.append(pick)
            if pick == "skip":
                cts.append(None)
                continue
            cts.append([float(s), float(sp)])
            G, H = ramp(bf.fXb, s, sp), ramp(bf.fYb, s, sp)
            Zs.append(pair_statistics(G, H) - float(alpha))
            pri.append(ramp_prior_var(u, v, s, sp, nb, mb))
        r = bet(Zs, pri, float(alpha), N, delta=AM.DELTA)
        r.update(source=src, cuts=cts)
        rec[c] = r
    return rec


def check_generic(log):
    """level_on_batches with the participant-level order of sequential_order reproduces stored records."""
    cells = [("nhanes_bmi", 1000, 0.05), ("nhanes_bmi", 1003, 0.10), ("nhanes_age", 1001, 0.40),
             ("rhc", 1003, 0.35), ("nhanes_child", 1002, 0.75)]
    out = []
    for name, seed, a in cells:
        X, Y, std0 = AM.data(name)
        ss = split_seed_of(seed)
        order = sequential_order(X, Y, np.random.default_rng(ss), edges=SEQ_EDGES, sigma=None,
                                 standardize_batch0=std0)
        n, m = len(order.X), len(order.Y)
        bx = [int(round(e * n)) for e in SEQ_EDGES]
        by = [int(round(e * m)) for e in SEQ_EDGES]
        mine = level_on_batches(order.X, order.Y, bx, by, order.sigma, order.rng, a)
        st = chain(name, seed)[round(a, 6)]
        same = all(mine[c] == st[c] for c in AM.CUTS) and mine["horizon"] == st["horizon"] \
            and mine["norms"] == st["norms"]
        dmax = max(abs(mine[c][k] - st[c][k]) for c in AM.CUTS for k in ("log_final", "log_max", "log_stopped"))
        out.append({"contrast": name, "seed": seed, "alpha": a, "identical": bool(same),
                    "decision_equal": all(mine[c]["certified"] == st[c]["certified"] for c in AM.CUTS),
                    "max_abs_log_wealth_diff": dmax})
        log(json.dumps(out[-1]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# NHANES, whole sampling units to batches (M8)
# ----------------------------------------------------------------------------------------------------------------

_NH = {}


def nhanes_units():
    """(curves, cov with unit, stratum, psu, weight) aligned with realdata.load_nhanes_activity."""
    if "units" not in _NH:
        import pandas as pd
        from trimmedmmd.realdata import data_root, load_nhanes_activity
        curves, cov, _ = load_nhanes_activity()
        demo = pd.concat([pd.read_sas(data_root() / "nhanes" / f"DEMO_{c}.xpt")[["SEQN", "SDMVSTRA", "SDMVPSU",
                                                                                  "WTMEC2YR"]] for c in "GH"])
        assert not demo.SEQN.duplicated().any()
        d = demo.set_index("SEQN").reindex(cov.SEQN.to_numpy())
        assert d.notna().all().all(), "participants without design variables"
        cov = cov.copy()
        cov["stratum"] = d.SDMVSTRA.to_numpy().astype(int)
        cov["psu"] = d.SDMVPSU.to_numpy().astype(int)
        cov["wtmec"] = d.WTMEC2YR.to_numpy()
        keys = sorted(set(zip(cov.stratum, cov.psu)))
        idx = {k: i for i, k in enumerate(keys)}
        cov["unit"] = [idx[k] for k in zip(cov.stratum, cov.psu)]
        _NH["units"] = (curves, cov, keys)
    return _NH["units"]


def nhanes_contrast_units(name):
    """X, Y as realdata.nhanes_groups (checked equal) and the unit of each row."""
    key = ("contrast", name)
    if key not in _NH:
        from trimmedmmd.realdata import nhanes_groups
        curves, cov, keys = nhanes_units()
        age, bmi = cov.age.to_numpy(), cov.bmi.to_numpy()
        by = name.split("_", 1)[1]
        if by == "child":
            a, b = (age >= 6) & (age < 12), (age >= 60) & (age < 80)
        elif by == "age":
            a, b = (age >= 20) & (age < 40), (age >= 60) & (age < 80)
        else:
            mid = (age >= 20) & (age < 60) & np.isfinite(bmi)
            a, b = mid & (bmi < 25), mid & (bmi >= 30)
        X0, Y0, _, _ = nhanes_groups(curves, cov, by=by)
        assert np.array_equal(curves[a], X0) and np.array_equal(curves[b], Y0)
        Xs, Ys = AM.data(name)[:2]
        assert np.array_equal(np.asarray(X0, float), Xs) and np.array_equal(np.asarray(Y0, float), Ys)
        u = cov.unit.to_numpy()
        _NH[key] = (Xs, Ys, u[a], u[b], keys)
    return _NH[key]


def cluster_order(name, seed, edges=SEQ_EDGES):
    """The unit-level split of one seed: (Xo, Yo, bx, by, sigma, rng, info)."""
    X, Y, ux, uy, keys = nhanes_contrast_units(name)
    rng = np.random.default_rng(split_seed_of(seed))
    K = len(keys)
    size = np.bincount(ux, minlength=K) + np.bincount(uy, minlength=K)
    units = np.nonzero(size > 0)[0]
    perm = units[rng.permutation(units.size)]
    cum = np.cumsum(size[perm])
    mid = (cum - size[perm] / 2.0) / cum[-1]
    batch_of = np.full(K, -1)
    batch_of[perm] = np.searchsorted(np.asarray(edges[1:-1]), mid, side="right")
    bxs, bys, ix, iy = [], [], [], []
    for b in range(len(edges) - 1):
        rx = np.nonzero(batch_of[ux] == b)[0]
        ry = np.nonzero(batch_of[uy] == b)[0]
        ix.append(rx[rng.permutation(rx.size)])
        iy.append(ry[rng.permutation(ry.size)])
        bxs.append(rx.size)
        bys.append(ry.size)
    ix, iy = np.concatenate(ix), np.concatenate(iy)
    Xo, Yo = X[ix], Y[iy]
    bx = [0] + list(np.cumsum(bxs).astype(int))
    by = [0] + list(np.cumsum(bys).astype(int))
    sigma = float(median_heuristic(Xo[:bx[1]], Yo[:by[1]], rng=rng))
    info = {"units_per_batch": [int(np.sum(batch_of[units] == b)) for b in range(len(edges) - 1)],
            "n_batches": bxs, "m_batches": bys, "units": int(units.size),
            "x_share_batch0": bxs[0] / len(X), "y_share_batch0": bys[0] / len(Y)}
    return Xo, Yo, [int(v) for v in bx], [int(v) for v in by], sigma, rng, info


def cluster_level(args):
    name, seed, alpha = args
    t0 = time.time()
    Xo, Yo, bx, by, sigma, rng, info = cluster_order(name, seed)
    rec = level_on_batches(Xo, Yo, bx, by, sigma, rng, alpha)
    rec.update(contrast=name, seed=int(seed), split_seed=split_seed_of(seed), alpha=float(alpha),
               kernel=AM.KERNEL, split="unit (SDMVPSU within SDMVSTRA)", sigma=sigma, bx=bx, by=by, **info,
               seconds=round(time.time() - t0, 2))
    return rec


def cluster_chain(args):
    name, seed = args
    path = kind_dir("cluster") / f"{name}_{seed}.jsonl"
    recs = {round(r["alpha"], 6): r for r in read_jsonl(path)}
    alive = set(AM.CUTS)
    for a in grid_for(name):
        key = round(float(a), 6)
        if key not in recs:
            rec = cluster_level((name, seed, float(a)))
            append(path, rec)
            recs[key] = rec
        alive = {c for c in alive if recs[key][c]["certified"]}
        if not alive:
            break
    return name, seed, len(recs)


def run_cluster(contrasts, workers, log):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    for name in contrasts:                       # unit summary, once
        X, Y, ux, uy, keys = nhanes_contrast_units(name)
        log(f"{name}: n={len(X)} m={len(Y)} units with participants {len(set(ux) | set(uy))} of {len(keys)}; "
            f"participants per unit {np.bincount(np.r_[ux, uy]).max()} max")
    jobs = [(nm, s) for nm in contrasts for s in SEEDS]
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(cluster_chain, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            nm, s, nl = fu.result()
            log(f"[{i}/{len(jobs)}] cluster {nm} seed={s}: {nl} levels")


# ----------------------------------------------------------------------------------------------------------------
# RHC classifiers on the batches of the ramp (M7)
# ----------------------------------------------------------------------------------------------------------------

CLF_METHODS = ("gbm_m_hard", "gbm_m_soft", "logit_m_hard", "logit_m_soft", "gbm_mt_hard", "gbm_mt_soft",
               "logit_mt_hard", "logit_mt_soft")


def clf_split(args):
    """One split of the RHC classifier bounds: decisions at every grid level for every classifier variant."""
    name, seed = args
    import aos_baselines_run as B          # the synthetic study's classifiers and bets
    t0 = time.time()
    X, Y, std0 = AM.data(name)
    ss = split_seed_of(seed)
    order = sequential_order(X, Y, np.random.default_rng(ss), edges=SEQ_EDGES, sigma=None, standardize_batch0=std0)
    Xo, Yo = order.X, order.Y
    n, m = len(Xo), len(Yo)
    bx = [int(round(e * n)) for e in SEQ_EDGES]
    by = [int(round(e * m)) for e in SEQ_EDGES]
    N = sequential_horizon(n, m, SEQ_EDGES)
    blocks = [(Xo[:bx[b]], Yo[:by[b]], Xo[bx[b]:bx[b + 1]], Yo[by[b]:by[b + 1]]) for b in (1, 2)]
    levels = [float(a) for a in grid_for(name)]
    cseed = int(ss % (2 ** 31 - 1))
    rec = {"contrast": name, "covariates": CLF_SETS[name], "seed": int(seed), "split_seed": ss, "levels": levels,
           "horizon": int(N), "dec_all": {}, "jlo": {}, "diag": {}}
    for c in B.CLF:
        bh, bs = [], []
        for Xf, Yf, Xe, Ye in blocks:
            nf, mf = len(Xf), len(Yf)
            thr = nf / (nf + mf)
            p = B.fit_predict(B.make_clf(c, cseed), np.vstack([Xf, Yf]), np.r_[np.ones(nf, int), np.zeros(mf, int)],
                              np.vstack([Xe, Ye]))
            px_, py_ = p[: len(Xe)], p[len(Xe):]
            bh.append(((px_ > thr).astype(float), (py_ > thr).astype(float), 0.5))
            bs.append((B.balanced(px_, thr), B.balanced(py_, thr), 0.5))
        rec["dec_all"][f"{c}_m_hard"], rec["jlo"][f"{c}_m_hard"] = B.stream_decisions(lambda j, a: bh, levels, N)
        rec["dec_all"][f"{c}_m_soft"], rec["jlo"][f"{c}_m_soft"] = B.stream_decisions(lambda j, a: bs, levels, N)
        rec["diag"][f"{c}_m_hard_meandiff"] = [float(g.mean() - h.mean()) for g, h, _ in bh]
        rec["diag"][f"{c}_m_soft_meandiff"] = [float(g.mean() - h.mean()) for g, h, _ in bs]
        th, ts, sel = [], [], []
        for Xf, Yf, Xe, Ye in blocks:
            nf, mf = len(Xf), len(Yf)
            thr = nf / (nf + mf)
            px_, py_, ox, oy, s_ = B.tuned_probs(c, Xf, Yf, Xe, Ye, cseed)
            nex, ney = len(Xe), len(Ye)
            gh, hh = (ox > thr).astype(float), (oy > thr).astype(float)
            gs, hs = B.balanced(ox, thr), B.balanced(oy, thr)
            th.append(((px_ > thr).astype(float), (py_ > thr).astype(float),
                       max(B.pair_var(gh.var(), hh.var(), nex, ney), 1e-4)))
            ts.append((B.balanced(px_, thr), B.balanced(py_, thr),
                       max(B.pair_var(gs.var(), hs.var(), nex, ney), 1e-4)))
            sel.append(s_)
        rec["dec_all"][f"{c}_mt_hard"], rec["jlo"][f"{c}_mt_hard"] = B.stream_decisions(lambda j, a: th, levels, N)
        rec["dec_all"][f"{c}_mt_soft"], rec["jlo"][f"{c}_mt_soft"] = B.stream_decisions(lambda j, a: ts, levels, N)
        rec["diag"][f"{c}_tuned"] = sel
        rec["diag"][f"{c}_mt_hard_meandiff"] = [float(g.mean() - h.mean()) for g, h, _ in th]
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def run_classifiers(names, log):
    for name in names:
        path = kind_dir("clf") / f"{name}.jsonl"
        done = {r["seed"] for r in read_jsonl(path)}
        for s in SEEDS:
            if s in done:
                continue
            rec = clf_split((name, s))
            append(path, rec)
            lev = {k: (rec["levels"][j] if j >= 0 else None) for k, j in rec["jlo"].items()}
            log(f"clf {name} seed={s}: {lev} ({rec['seconds']}s)")


# ----------------------------------------------------------------------------------------------------------------
# propensity-score estimate of alpha* (M7.1)
# ----------------------------------------------------------------------------------------------------------------

def _prop_stats(ex, ey, pi):
    from trimmedmmd.experiments.realdata_run import _overlap
    e = np.r_[ex, ey]
    plug = 1.0 - float(np.mean(np.minimum(e / pi, (1.0 - e) / (1.0 - pi))))
    return {"one_minus_ovl": 1.0 - _overlap(ex, ey), "plugin": plug}


def run_propensity(log):
    from trimmedmmd.experiments.realdata_run import _logistic_fit
    from trimmedmmd.realdata import load_rhc
    out = {"model": "ridge logistic regression (experiments.realdata_run._logistic_fit, l2 = 1e-2, Newton), "
                    "covariates standardized by the pooled moments (realdata.load_rhc), as Table S7",
           "estimands": {"one_minus_ovl": "1 - overlapping coefficient of the 40-bin histograms of e-hat on [0, 1] "
                                          "in the two arms (Table S7's overlap)",
                         "plugin": "1 - (1/N) sum_i min{e_i / pi, (1 - e_i) / (1 - pi)}, pi = n / (n + m)"},
           "bootstrap": f"{PROP_B} resamples, stratified by arm, model refitted, seed {PROP_SEED}; 95% percentile "
                        "interval and the 5% quantile (one-sided 95% lower limit)",
           "crossfit": f"{PROP_FOLDS}-fold, folds stratified by arm, numpy default_rng({PROP_SEED})", "sets": {}}
    for cs in ("paper", "full"):
        t0 = time.time()
        X, Y, names, _ = load_rhc(covariates=cs)
        n, m = len(X), len(Y)
        Z = np.vstack([X, Y])
        lab = np.r_[np.ones(n), np.zeros(m)]
        pi = n / (n + m)
        _, prop = _logistic_fit(Z, lab)
        est = _prop_stats(prop[:n], prop[n:], pi)
        rng = np.random.default_rng(PROP_SEED)
        boot = {k: [] for k in est}
        for _ in range(PROP_B):
            i = rng.integers(0, n, n)
            j = n + rng.integers(0, m, m)
            ij = np.r_[i, j]
            _, p_b = _logistic_fit(Z[ij], lab[ij])
            s_b = _prop_stats(p_b[:n], p_b[n:], pi)
            for k in boot:
                boot[k].append(s_b[k])
        # cross-fitted
        frng = np.random.default_rng(PROP_SEED)
        fold = np.empty(n + m, int)
        fold[:n] = frng.permutation(np.arange(n) % PROP_FOLDS)
        fold[n:] = frng.permutation(np.arange(m) % PROP_FOLDS)
        oof = np.empty(n + m)
        for f in range(PROP_FOLDS):
            tr = fold != f
            beta, _ = _logistic_fit(Z[tr], lab[tr])
            oof[~tr] = 1.0 / (1.0 + np.exp(-(np.c_[np.ones((~tr).sum()), Z[~tr]] @ beta)))
        cf = _prop_stats(oof[:n], oof[n:], pi)
        rec = {"d": len(names), "n": n, "m": m, "pi": pi, "estimate": est, "crossfit": cf,
               "e_range_x": [float(prop[:n].min()), float(prop[:n].max())],
               "e_range_y": [float(prop[n:].min()), float(prop[n:].max())],
               "share_e_outside_0.1_0.9": float(np.mean((prop < 0.1) | (prop > 0.9))),
               "seconds": round(time.time() - t0, 1)}
        for k, v in boot.items():
            v = np.asarray(v)
            rec[f"{k}_ci95"] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            rec[f"{k}_lcl95"] = float(np.quantile(v, 0.05))
            rec[f"{k}_boot_mean"] = float(v.mean())
            rec[f"{k}_boot_sd"] = float(v.std(ddof=1))
        out["sets"][cs] = rec
        log(f"propensity {cs} (d={len(names)}): {json.dumps({k: rec[k] for k in rec if k != 'seconds'})}")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "propensity.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------------------------------------------
# the comparison rows of Table tab:cohort on a covariate set
# ----------------------------------------------------------------------------------------------------------------

def run_estimand(cs, skip_kernel, log):
    import pandas as pd
    from scipy.spatial.distance import cdist

    import exp_rhc_balancing as EB
    from trimmedmmd.experiments.realdata_run import _logistic_fit
    from trimmedmmd.kernels import gaussian_kernel
    from trimmedmmd.realdata import load_rhc
    from v3_rhc_estimand import BOOT_SEED, SEED, boot_ci, rhc_outcome, risk_difference
    t_start = time.time()
    X, Y, names, _ = load_rhc(covariates=cs)
    yx, yy = rhc_outcome()
    assert len(yx) == len(X) and len(yy) == len(Y)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y])
    _, prop = _logistic_fit(Z, np.r_[np.ones(n), np.zeros(m)])
    ex, ey = prop[:n], prop[n:]
    sigma = float(median_heuristic(X, Y, rng=np.random.default_rng(SEED)))
    Kfull = gaussian_kernel(Z, Z, sigma)
    Dfull = cdist(Z, Z)
    weights, timing = {}, {}
    weights["unweighted"] = (np.ones(n), np.ones(m), "observed arms (no reweighting)")
    weights["trimming"] = None              # not recomputed here; its bootstrap draws are consumed below
    a_all = Dfull.mean(axis=1)
    c = float(Dfull.mean())
    t0 = time.time()
    w_e, _ = EB.energy_per_arm(Dfull[:n, :n], a_all[:n], c)
    v_e, _ = EB.energy_per_arm(Dfull[n:, n:], a_all[n:], c)
    timing["energy"] = time.time() - t0
    weights["energy"] = (w_e, v_e, "pooled population (ATE)")
    log(f"estimand {cs}: energy {timing['energy']:.0f}s")
    if skip_kernel:
        weights["kernel"] = None
    else:
        b_all = Kfull.mean(axis=1)
        cK = float(Kfull.mean())
        t0 = time.time()
        w_k, _ = EB.kernel_per_arm(Kfull[:n, :n], b_all[:n], cK)
        v_k, _ = EB.kernel_per_arm(Kfull[n:, n:], b_all[n:], cK)
        timing["kernel"] = time.time() - t0
        weights["kernel"] = (w_k, v_k, "pooled population (ATE)")
        log(f"estimand {cs}: kernel {timing['kernel']:.0f}s")
    weights["ato"] = (1.0 - ex, ey, "overlap population, tilting e(1-e) (ATO)")
    mx, my = np.minimum(ex, 1 - ex), np.minimum(ey, 1 - ey)
    weights["atm"] = (mx / ex, my / (1 - ey), "overlap population, tilting min(e,1-e) (ATM)")
    labels = {"unweighted": "all patients", "energy": "energy balancing", "kernel": "kernel balancing",
              "ato": "overlap weights", "atm": "matching weights"}
    rows = []
    brng = np.random.default_rng(BOOT_SEED)
    for key, wv in weights.items():         # insertion order = scripts/v3_rhc_estimand.py
        if wv is None:
            boot_ci(yx, yy, np.ones(n), np.ones(m), brng)     # the same resamples as there
            continue
        w, v, target = wv
        r, _ = EB.describe(labels[key], X, Y, names, w, v, prop, Kfull, Dfull)
        rd = risk_difference(yx, yy, w, v)
        ci = boot_ci(yx, yy, np.asarray(w, float), np.asarray(v, float), brng)
        r.update({"key": key, "target": target, "rd_dth30": rd, "rd_lo": ci[0], "rd_hi": ci[1]})
        rows.append(r)
    out = pd.DataFrame(rows)
    WORK.mkdir(parents=True, exist_ok=True)
    tag = "full" if cs == "full" else "paper_check"
    out.to_csv(WORK / f"rhc_estimand_{tag}.csv", index=False)
    meta = {"covariates": cs, "d": len(names), "names": names, "sigma": sigma, "timing_seconds": timing,
            "seconds_total": round(time.time() - t_start, 1), "skip_kernel": bool(skip_kernel)}
    if cs == "paper":
        st = pd.read_csv(ROOT / "results" / "v3" / "tables" / "rhc_estimand.csv").set_index("key")
        cols = ["max_abs_smd", "mean_abs_smd", "ess_frac_x", "ess_frac_y", "prop_overlap", "rd_dth30", "rd_lo",
                "rd_hi"]
        diff = {r["key"]: max(abs(float(r[c_]) - float(st.loc[r["key"], c_])) for c_ in cols) for r in rows}
        meta["check_vs_results_v3_rhc_estimand_csv_max_abs_diff"] = diff
        log(f"estimand paper check: {diff}")
    (WORK / f"rhc_estimand_{tag}.json").write_text(json.dumps(meta, indent=1))
    od = OUTD
    od.mkdir(parents=True, exist_ok=True)
    for suf in ("csv", "json"):
        src = WORK / f"rhc_estimand_{tag}.{suf}"
        tmp = od / (src.name + ".tmp")
        tmp.write_text(src.read_text())
        os.replace(tmp, od / src.name)


# ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evalues", "cluster", "classifiers", "propensity", "estimand", "check", "sync"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--contrasts", nargs="*", default=None)
    ap.add_argument("--covariates", choices=["paper", "full"], default="full")
    ap.add_argument("--skip-kernel", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    logf = open(WORK / f"{a.cmd}.log", "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        print(line, file=logf, flush=True)
    t0 = time.time()
    log(f"start {a.cmd} {' '.join(sys.argv[1:])}")
    if a.cmd == "sync":
        sync(log)
    elif a.cmd == "evalues":
        run_evalues(a.contrasts or EV_CONTRASTS, a.workers, log)
        sync(log)
    elif a.cmd == "cluster":
        run_cluster(a.contrasts or NH_CONTRASTS, a.workers, log)
        sync(log)
    elif a.cmd == "classifiers":
        run_classifiers(a.contrasts or list(CLF_SETS), log)
        sync(log)
    elif a.cmd == "propensity":
        run_propensity(log)
        sync(log)
    elif a.cmd == "estimand":
        run_estimand(a.covariates, a.skip_kernel, log)
    elif a.cmd == "check":
        res = check_generic(log)
        (WORK / "check_generic.json").write_text(json.dumps(res, indent=1))
        sync(log)
    log(f"done {a.cmd} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
