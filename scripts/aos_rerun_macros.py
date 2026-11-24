r"""AoS (2026-11-20): the macros of the two reruns, as \renewcommand lines for exactly the macros whose values change.

The reruns: the RHC-shaped run of Appendix E.2 with cos/sin random features from n = 2,000 (diagonal exactly 1, so the
plug-in threshold is evaluated at kappa = 1; scripts/export_fig2_data.py --rff-map cossin), and the sequential
certificate on the RHC subsamples with the covariates standardized by the moments of its batch 0
(scripts/export_fig1_regimes.py --std batch0).

Inputs: results/v3/figures/fig2_data_cossin.json (n >= 2,000), results/v2/figures/fig2_data.json (its rows below
2,000), results/v2/figures/fig2_small.json and fig2_null.json (unchanged), results/v3/figures/fig1_regimes_std0.json.
Writes results/v3/figures/fig2_data_merged.json (v2 rows below 2,000, cos/sin rows from there: the input of
TMMD_FIG2_DATA for scripts/v3_fig_recovery.py) and results/v3/tables/aos_rerun_macros.tex.

Each macro is computed with the code of its generator, on the v2 inputs (checked against the definitions in
aos/tables/*.tex) and on the new ones: scripts/fig2.py (_macros, imported), jmlr/tools/make_tables.py and
jmlr/tools/make_v2_text.py (scripts without functions: their blocks are repeated here with their own formatting
functions, read from their source), jmlr/v3/tools/derive.py (regime(), imported and run on the new rows) and
scripts/v3_fig_recovery.py (load_regime, size_reached, sig2). Only macros defined in aos/tables are written. The
claims those generators assert are checked and reported, not enforced.

    python3 -I scripts/aos_rerun_macros.py
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")

V2F, V3F = ROOT / "results" / "v2" / "figures", ROOT / "results" / "v3" / "figures"
COSSIN, STD0, MERGED = V3F / "fig2_data_cossin.json", V3F / "fig1_regimes_std0.json", V3F / "fig2_data_merged.json"
OUT = ROOT / "results" / "v3" / "tables" / "aos_rerun_macros.tex"
AOS = ROOT / "aos"
NOTES = []        # failed claims and remarks, printed at the end


def note(msg):
    NOTES.append(msg)


def load(p):
    return json.loads(Path(p).read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fns(path, *names):
    """Function definitions of a generator script, executed alone (the script itself writes into jmlr/)."""
    body = [f for f in ast.parse(Path(path).read_text()).body if isinstance(f, ast.FunctionDef) and f.name in names]
    assert sorted(f.name for f in body) == sorted(names), (path, names)
    ns = {"math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return [ns[n] for n in names]


def constant(path, name):
    for f in ast.parse(Path(path).read_text()).body:
        if isinstance(f, ast.Assign) and getattr(f.targets[0], "id", None) == name:
            return ast.literal_eval(f.value)
    raise KeyError(name)


def parse_macros(text):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))


F2 = module("fig2_gen", ROOT / "scripts" / "fig2.py")
VR = module("v3_fig_recovery_gen", ROOT / "scripts" / "v3_fig_recovery.py")
DV = module("derive_gen", ROOT / "jmlr" / "v3" / "tools" / "derive.py")
(MT_thousands,) = fns(ROOT / "jmlr" / "tools" / "make_tables.py", "thousands")
VT_f2, VT_pct = fns(ROOT / "jmlr" / "tools" / "make_v2_text.py", "f2", "pct")
RFF_FROM_N = constant(ROOT / "scripts" / "export_fig2_data.py", "RFF_FROM_N")


def merge_fig2():
    """v2 rows below RFF_FROM_N + the cos/sin rows from there (same cells, replicates, data and bandwidths)."""
    D, C = load(V2F / "fig2_data.json"), load(COSSIN)
    for k in ("levels", "grid", "delta", "sep", "threshold", "certificate"):
        assert C[k] == D[k], f"fig2_data_cossin.json differs from the v2 run in {k}"
    assert C.get("rff_map") == "cossin"
    key = lambda r: (r["alpha_star"], r["n"], r["rep"])  # noqa: E731
    old = {key(r): r for r in D["rows"] if r["n"] >= RFF_FROM_N}
    assert Counter(map(key, C["rows"])) == Counter(map(key, (r for r in D["rows"] if r["n"] >= RFF_FROM_N))), \
        "cos/sin run: not the v2 cells and replicates"
    for r in C["rows"]:     # the data and the bandwidth are drawn before the features: identical to v2
        s = old[key(r)]["sigma"]
        assert abs(r["sigma"] - s) <= 1e-9 * max(1.0, abs(s)), f"bandwidth differs at {key(r)}"
        assert r["kappa_rff"] == 1.0 and abs(r["kappa_realized"] - 1.0) < 1e-9
    meta = {k: v for k, v in D.items() if k != "rows"}
    meta.update({"rff_map": "cossin", "kernel_constants": C["kernel_constants"],
                 "merged_from": {f"n < {RFF_FROM_N}": "results/v2/figures/fig2_data.json",
                                 f"n >= {RFF_FROM_N}": "results/v3/figures/fig2_data_cossin.json"},
                 "seconds_cossin": C["seconds"]})
    M = {**meta, "rows": [r for r in D["rows"] if r["n"] < RFF_FROM_N] + C["rows"]}
    MERGED.write_text(json.dumps(M, indent=1))
    return M


def regime_dir(tmp, fig2):
    """A results root whose figures/ hold this fig2_data.json and the v2 fig2_small.json and fig2_null.json."""
    F = Path(tmp) / "figures"
    F.mkdir(parents=True)
    (F / "fig2_data.json").write_text(json.dumps(fig2))
    for name in ("fig2_small.json", "fig2_null.json"):
        (F / name).symlink_to(V2F / name)
    return Path(tmp)


def from_fig2_py(root):
    """scripts/fig2.py: regime_macros.tex (as build() calls _macros)."""
    with contextlib.redirect_stdout(io.StringIO()):
        D, rows, thr = F2.load_rows(root / "figures", None, require_null=True)
    levels, sizes = sorted({r["alpha_star"] for r in rows}), sorted({r["n"] for r in rows})
    d0_per_a = float(np.mean([r["d_hat_0"] / r["alpha_star"] for r in rows if r["alpha_star"] > 0]))
    with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
        F2._macros(D, rows, levels, sizes, d0_per_a, F2.closed_C(D["delta"], thr), D["delta"], thr, tables=t)
        text = (Path(t) / "regime_macros.tex").read_text()
    for _ in range(2):      # jmlr/tools/sync.sh: thousands separators as {,} in the paper's copy
        text = re.sub(r"([0-9]),([0-9]{3})", r"\1{,}\2", text)
    return parse_macros(text)


def from_make_tables(fig2, small):
    """jmlr/tools/make_tables.py, the regime-study block (Section 9.1)."""
    M = {}
    reg_rows = list(fig2["rows"]) + small["rows"]
    rb = {int(k): v for k, v in fig2["reps_by_n"].items()}
    rb.update({int(k): v for k, v in small["reps_by_n"].items()})
    contam = [r for r in reg_rows if r["alpha_star"] > 0]
    at_boundary = [r for r in contam if r["certified"] >= r["alpha_star"] - 1e-9]
    over = [r for r in contam if r["certified"] > r["alpha_star"] + 1e-9]
    rg_delta, n_c = float(fig2["delta"]), len(contam)
    rg_lim = rg_delta + 3 * math.sqrt(rg_delta * (1 - rg_delta) / n_c)
    err_rate = len(at_boundary) / n_c
    if not err_rate <= rg_lim:
        note(f"make_tables claim fails: certificate reaches alpha* in {err_rate:.3f} > delta + 3 SE = {rg_lim:.3f}")
    M["rgContamReps"] = str(n_c)
    M["rgCertBoundary"] = str(len(at_boundary))
    M["rgCertBoundaryPct"] = f"{100 * err_rate:.1f}"
    M["rgCertOver"] = str(len(over))
    vb = [r for r in contam if r.get("certified_bootstrap") is not None]
    assert len(vb) == n_c
    M["rgCertBootBoundaryPct"] = f"{100 * sum(r['certified_bootstrap'] >= r['alpha_star'] - 1e-9 for r in vb) / n_c:.1f}"
    M["rgRepsMin"] = str(min(rb.values()))
    M["rgRepsMax"] = str(max(rb.values()))
    return M


def rhc_subsamples(reg):
    """jmlr/tools/make_tables.py, \\rhcCertSubLo/Hi (fig1_regimes.json)."""
    assert reg.get("threshold") == "plugin" and reg.get("certificate") in ("sequential", "all")
    first = {}
    for rep in sorted({r["rep"] for r in reg["rows"]}):
        rr = sorted((r for r in reg["rows"] if r["rep"] == rep), key=lambda r: r["n"])
        first[rep] = next((r["n"] for r in rr if r["certified"] > 0), None)
    hits = [v for v in first.values() if v is not None]
    assert hits, "no RHC subsample certifies a positive level"
    return {"rhcCertSubLo": MT_thousands(min(hits)),
            "rhcCertSubHi": (MT_thousands(max(hits)) if None not in first.values()
                             else ">" + MT_thousands(max(reg["sizes"])))}, first


def from_make_v2_text(fd, fs, fn):
    """jmlr/tools/make_v2_text.py, the regime-study block (rgFireAboveMin/BelowMax) and rgKappaSqLo/Hi."""
    from trimmedmmd.threshold import samplesize_constant
    M = {}
    C0 = samplesize_constant(float(fd["delta"]))
    allrows = fd["rows"] + fs["rows"]
    d0_per_a = sum(r["d_hat_0"] / r["alpha_star"] for r in allrows if r["alpha_star"] > 0) / \
        sum(1 for r in allrows if r["alpha_star"] > 0)
    above, below = [], []
    for a in sorted({r["alpha_star"] for r in allrows if r["alpha_star"] > 0}):
        nfire = 2 * (C0 / (d0_per_a * a)) ** 2
        rate = {}
        for r in allrows:
            if r["alpha_star"] == a:
                rate.setdefault(r["n"], []).append(bool(r["fires"]))
        up = [n for n in rate if n > nfire]
        lo = [n for n in rate if n < nfire]
        if up:
            above.append(sum(rate[min(up)]) / len(rate[min(up)]))
        if lo:
            below.append(sum(rate[max(lo)]) / len(rate[max(lo)]))
    M["rgFireAboveMin"] = VT_pct(min(above))
    M["rgFireBelowMax"] = VT_pct(max(below))
    kap = [r["kappa_rff"] for D in (fd, fn) for r in D["rows"] if r.get("kappa_rff") is not None]
    M["rgKappaSqLo"], M["rgKappaSqHi"] = VT_f2(min(kap) ** 2), VT_f2(max(kap) ** 2)
    return M


def from_derive(root, rm):
    """jmlr/v3/tools/derive.py regime(): its own code on ``root``/figures, with ``rm`` as the macros it checks."""
    M = {}
    DV.RES, DV.macros = root, (lambda *files: dict(rm))
    try:
        DV.regime(M)
    except AssertionError as e:
        import traceback
        line = traceback.extract_tb(e.__traceback__)[-1]
        note(f"derive.regime claim fails at line {line.lineno}: {line.line}")
    return M


def recovery(fig2_path):
    """scripts/v3_fig_recovery.py, panel (b): the dominance assertion and recGap*."""
    levels, sizes, r_ana, r_cert, _ = VR.load_regime(fig2_path)
    bad = [(levels[i], int(sizes[j]), round(float(r_ana[i, j]), 4), round(float(r_cert[i, j]), 4))
           for i in range(len(levels)) for j in range(len(sizes)) if r_ana[i, j] > r_cert[i, j] + 1e-12]
    factors = []
    for N in VR.GAP_SIZES:
        j = int(np.where(sizes == N)[0][0])
        for i in range(len(levels)):
            n_split = VR.size_reached(sizes, r_cert[i], r_ana[i, j])
            assert n_split is not None
            factors.append(N / n_split)
    M = {"recGapLo": VR.sig2(min(factors)), "recGapHi": VR.sig2(max(factors)),
         "recGapNlo": VR.sig2(VR.GAP_SIZES[0]), "recGapNhi": VR.sig2(VR.GAP_SIZES[1])}
    table = {a: [(int(n), round(float(x), 3), round(float(c), 3)) for n, x, c in zip(sizes, r_ana[i], r_cert[i])]
             for i, a in enumerate(levels)}
    return M, bad, factors, table


def compute(fd, regime_json, fig2_path, tmp, aos_defined):
    fs, fn = load(V2F / "fig2_small.json"), load(V2F / "fig2_null.json")
    root = regime_dir(tmp, fd)
    M = from_fig2_py(root)
    M.update(from_make_tables(fd, fs))
    M.update(from_make_v2_text(fd, fs, fn))
    rhc, first = rhc_subsamples(regime_json)
    M.update(rhc)
    M.update(from_derive(root, {**aos_defined, **M}))
    rec, bad, factors, table = recovery(fig2_path)
    M.update(rec)
    return M, {"first": first, "bad": bad, "factors": factors, "table": table}


def used_in_aos(name):
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
    files = list((AOS / "sections").glob("*.tex")) + list((AOS / "supplement").glob("*.tex")) + \
        [AOS / "main.tex", AOS / "supplement.tex"]
    return sorted({f.relative_to(AOS).as_posix() for f in files if f.exists() and pat.search(f.read_text())})


def main():
    aos_defined = {}
    for f in sorted((AOS / "tables").glob("*.tex")):
        aos_defined.update(parse_macros(f.read_text()))
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        old, old_x = compute(load(V2F / "fig2_data.json"), load(V2F / "fig1_regimes.json"),
                             V2F / "fig2_data.json", t1, aos_defined)
        n_notes_v2 = len(NOTES)
        fm = merge_fig2()
        new, new_x = compute(fm, load(STD0), MERGED, t2, aos_defined)
    mism = {k: (v, aos_defined.get(k)) for k, v in old.items() if k in aos_defined and aos_defined[k] != v}
    print(f"v2 inputs: {len(old)} macros recomputed, {len(mism)} differ from aos/tables: {mism}")
    assert n_notes_v2 == 0 and not old_x["bad"], f"v2 inputs fail the generators' claims: {NOTES}"
    changed = {k: (aos_defined[k], v) for k, v in new.items() if k in aos_defined and aos_defined[k] != v}
    kappa = {k: changed.pop(k) for k in ("rgKappaSqLo", "rgKappaSqHi") if k in changed}
    lines = ["% Generated by scripts/aos_rerun_macros.py -- do not edit. The macros of aos/tables whose values change",
             "% with cos/sin random features in the RHC-shaped run (n >= 2,000; kappa = 1) and batch-0 standardization",
             "% of the RHC subsamples (results/v3/figures/fig2_data_cossin.json, fig1_regimes_std0.json). Load after",
             "% the tables it overrides. Comment: the old value."]
    lines += [f"\\renewcommand{{\\{k}}}{{{v}}}  % was {o}" for k, (o, v) in changed.items()]
    if kappa:
        lines += ["% rgKappaSqLo/Hi describe the realized kappa of the phased map, which now only fig2_null.json uses",
                  "% (the main run has kappa = 1 exactly); the generator's formula over both runs would give:"]
        lines += [f"% \\renewcommand{{\\{k}}}{{{v}}}  % was {o}" for k, (o, v) in kappa.items()]
        kn = [r["kappa_rff"] for r in load(V2F / "fig2_null.json")["rows"] if r.get("kappa_rff") is not None]
        lines += [f"% the null run alone (phased map): kappa^2 from {VT_f2(min(kn) ** 2)} to {VT_f2(max(kn) ** 2)}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(changed)} macros) and {MERGED}")
    for k, (o, v) in {**changed, **kappa}.items():
        print(f"  {k}: {o} -> {v}   used in aos: {', '.join(used_in_aos(k)) or '-'}")
    unchanged = sorted(k for k in new if k in aos_defined and k not in changed and k not in kappa)
    print(f"unchanged ({len(unchanged)}): {' '.join(unchanged)}")
    print("RHC subsamples, first size with a nonzero bound per subsample: v2", old_x["first"], "batch 0", new_x["first"])
    print("RHC-shaped run, analytic/split mean fraction by size (new):")
    for a, row in new_x["table"].items():
        print(f"  alpha*={a}: " + " ".join(f"{n}:{x:.2f}/{c:.2f}" for n, x, c in row))
    print("sample-size factors (new):", " ".join(f"{f:.1f}" for f in new_x["factors"]))
    if new_x["bad"]:
        note(f"v3_fig_recovery assertion fails: analytic above the split-sample bound in cells "
             f"(alpha*, n, analytic, split) {new_x['bad']}")
    print("claims:", "all hold" if not NOTES else "")
    for m in NOTES:
        print("  " + m)


if __name__ == "__main__":
    main()
