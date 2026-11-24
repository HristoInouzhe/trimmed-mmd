r"""AoS revision (2026-11-21, plan v1 T1): the number macros of Section S4 "A numerical check" of the margin results.

Inputs: results/aos/margin_check/*.json, written by the independent check of the margin results
(scripts/aos_margin_check/run_laplace.py, run_gauss.py; analysis as in scripts/aos_margin_check/analyse.py). Laws:
the mixtures P = 0.8 N(0,1) + 0.2 N(mu,1), Q = 0.8 N(0,1) + 0.2 N(-mu,1) ("mix") and the shift N(mu,1), N(-mu,1)
("shift"); kernels exp(-|x-y|) (Laplace) and exp(-(x-y)^2/2) (Gaussian). Each row has the level eps = alpha* - alpha,
continuum certificates lo_c <= u <= up_c of the rescaled path u = (1-alpha) D_alpha, the lower bound of the
approximation margin for the optimized function of the Sobolev or Gaussian construction ("direct_lower"; for the
Laplace kernel the single-width function of Theorem thm:newsobolev, "direct_lower_asym" two widths), and, for the
Laplace kernel, the exact norm of the explicit trimming of Proposition prop:newupperone ("upperone_trim_exact").

Rows whose continuum certificate failed (lo_c <= 0 or up_c >= 1.1 lo_c) are dropped, as in the check. Asserts that no
lower bound exceeds the upper certificate and no upper bound falls below the lower certificate. The collapse constant
is the least-squares fit of log u = A + B log eps - c / eps over the rows with WIN_LO <= u <= WIN_HI, reported as
c / l0, l0 the slope of p - q at the crossing.

    python3 -I scripts/aos_margin_macros.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "aos" / "margin_check"
OUT = ROOT / "results" / "aos" / "tables" / "aos_margin_macros.tex"
WIN_LO, WIN_HI = 1e-8, 1e-3


def rows(name):
    o = json.loads((IN / name).read_text())
    rs = list(o["rows"])
    ext = IN / name.replace(".json", "_ext.json")
    if ext.exists():
        rs += json.loads(ext.read_text())["rows"]
    rs = [r for r in rs if r["lo_c"] > 0 and r["up_c"] < 1.1 * r["lo_c"]]
    return o, sorted(rs, key=lambda r: -r["eps"])


def sci(x, digits=2):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits - 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{digits - 1}f}\\times10^{{{e}}}"


def window_fit(o, rs, key):
    sel = [r for r in rs if WIN_LO <= r["up_c"] <= WIN_HI]
    e = np.array([r["eps"] for r in sel])
    y = np.log([key(r) for r in sel])
    X = np.column_stack([np.ones_like(e), np.log(e), -1.0 / e])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[2] / o["l0"]


def main():
    n_rows, viol = 0, []
    names = sorted(p.name for p in IN.glob("*.json") if not p.name.endswith("_ext.json")
                   and (p.name.startswith("laplace_") or p.name.startswith("gauss_")))
    for name in names:
        o, rs = rows(name)
        for r in rs:
            n_rows += 1
            for k in ("direct_lower", "direct_lower_asym", "thm_lower"):
                if k in r and r[k] is not None and r[k] > r["up_c"] * (1 + 1e-9):
                    viol.append((name, r["eps"], k))
            if "upperone_trim_exact" in r and r["upperone_trim_exact"] < r["lo_c"] * (1 - 1e-9):
                viol.append((name, r["eps"], "upper"))
    assert not viol, viol

    _, lap = rows("laplace_mix_3.0.json")
    lr = min(lap, key=lambda r: abs(r["eps"] - 3e-4))
    _, shift = rows("laplace_shift_1.0.json")
    s_last = shift[-1]
    s_prev = shift[-2]
    lap_slope = math.log(s_last["up_c"] / s_prev["up_c"]) / math.log(s_last["eps"] / s_prev["eps"])

    og, g3 = rows("gauss_mix_3.0.json")
    gfirst, glast = g3[0], g3[-1]
    fits_path, fits_lb = [], []
    for mu in ("3.0", "2.0", "1.5"):
        o, rs = rows(f"gauss_mix_{mu}.json")
        fits_path.append(window_fit(o, rs, lambda r: r["up_c"]))
        fits_lb.append(window_fit(o, rs, lambda r: r["direct_lower"]))
    os_, rs_ = rows("gauss_shift_1.0.json")
    fit_shift = window_fit(os_, rs_, lambda r: r["up_c"])

    M = {
        "mgRows": f"{n_rows}",
        "mgLapEps": sci(lr["eps"], 1), "mgLapPathLo": sci(lr["lo_c"], 4), "mgLapPathHi": sci(lr["up_c"], 4),
        "mgLapLB": sci(lr["direct_lower"], 3), "mgLapLBtwo": sci(lr["direct_lower_asym"], 3),
        "mgLapUB": sci(lr["upperone_trim_exact"], 3),
        "mgLapShiftEps": sci(s_last["eps"], 1), "mgLapShiftSlope": f"{lap_slope:.2f}",
        "mgGaussEpsHi": sci(gfirst["eps"], 2), "mgGaussEpsLo": sci(glast["eps"], 2),
        "mgGaussPathHi": sci(gfirst["up_c"], 2), "mgGaussPathLo": sci(glast["up_c"], 3),
        "mgGaussLBHi": sci(gfirst["direct_lower"], 2), "mgGaussLBLo": sci(glast["direct_lower"], 2),
        "mgFitPathLo": f"{min(fits_path):.2f}", "mgFitPathHi": f"{max(fits_path):.2f}",
        "mgFitLBLo": f"{min(fits_lb):.2f}", "mgFitLBHi": f"{max(fits_lb):.2f}",
        "mgFitShift": f"{fit_shift:.2f}", "mgPiEight": f"{math.pi ** 2 / 8:.2f}",
        "mgWinLo": f"10^{{{round(math.log10(WIN_LO))}}}", "mgWinHi": f"10^{{{round(math.log10(WIN_HI))}}}",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    head = ("% Generated by scripts/aos_margin_macros.py -- do not edit. Numerical check of the margin results\n"
            "% (results/aos/margin_check/*.json, the independent check); no lower bound above an upper certificate.\n")
    OUT.write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    print(f"wrote {OUT}: {n_rows} certified rows, no violation")
    for k, v in M.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
