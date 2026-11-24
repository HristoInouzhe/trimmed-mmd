"""Assemble NOTE.md from head.md, theory.md, lit_lean.md, results.md, reco.md and tables/*.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def computed_snippets():
    import numpy as np
    import pandas as pd
    import analyze_synth as A
    r = pd.read_csv(HERE / "out" / "rates_disjoint_d2_a20_all.csv")
    parts = []
    for a in (0.1, 0.15):
        vals = {}
        for m in ("boot", "ramp_bet", "dir0_ramp_bet", "prop17", "gbm_hoef"):
            v = A.n_at_power(r[~r[m].isna()], m, a)
            vals[m] = ("≤100" if v < 0 else (f"> {int(r[~r[m].isna()].n.max())}" if np.isnan(v) else f"{v:.0f}"))
        parts.append(f"α = {a:g}: bootstrap {vals['boot']}, new {vals['ramp_bet']} (α = 0 witness "
                     f"{vals['dir0_ramp_bet']}), Prop 17 {vals['prop17']}, GBM classifier + Hoeffding {vals['gbm_hoef']}")
    n80 = "; ".join(parts) + " (for Prop 17 and the classifier the closed-form median-power sizes are ≈ 4,000 / 19,500 and ≈ 1,340 / 4,900)"
    s = pd.read_csv(HERE / "out" / "real_summary.csv")
    names = {"rhc": "RHC", "nhanes_child": "NHANES 6–11 vs 60–79", "nhanes_age": "NHANES 20–39 vs 60–79",
             "nhanes_bmi": "NHANES BMI", "adult": "Adult", "acs": "ACS"}
    f = lambda v: "none" if v < 0 else f"{v:.2f}"
    items = []
    for d, lab in names.items():
        b = s[(s.dataset == d) & (s.method == "boot")]
        n = s[(s.dataset == d) & (s.method == "ramp_bet")]
        z = s[(s.dataset == d) & (s.method == "dir0_ramp_bet")]
        if len(b) and len(n):
            items.append(f"{lab}: bootstrap {f(b['min'].iloc[0])} → finite-sample {f(n['min'].iloc[0])}"
                         + (f" (α = 0 witness {f(z['min'].iloc[0])})" if len(z) else ""))
    import json
    fd = HERE / "out" / "real_dir0.jsonl"
    if fd.exists():
        recs = [json.loads(l) for l in fd.read_text().splitlines()]
        root = HERE.parents[1]
        for d, lab in (("adult", "Adult"), ("acs", "ACS")):
            rr = [r for r in recs if r["dataset"] == d]
            if not rr:
                continue
            b = min(r["lower"]["boot_dir0"] for r in rr)
            nw = min(r["lower"]["ramp_bet"] for r in rr)
            q = pd.read_csv(root / "results" / "fair" / f"{d}_split_cert.csv")
            pub = q[q.method == "original"].certified_level.min()
            items.append(f"{lab} (one α = 0 witness for both): bootstrap {f(b)} → finite-sample {f(nw)}; "
                         f"published per-level-witness bootstrap {f(pub)}")
    realshort = "; ".join(items)
    return n80, realshort


def main():
    head = (HERE / "head.md").read_text()
    n80, realshort = computed_snippets()
    head = head.replace("@@N80LINE@@", n80).replace("@@REALSHORT@@", realshort)
    theory = (HERE / "theory.md").read_text()
    lit_lean = (HERE / "lit_lean.md").read_text()
    lit, lean = lit_lean.split("## 5. Lean", 1)
    lean = "## 5. Lean" + lean
    res = (HERE / "results.md").read_text()
    reco = (HERE / "reco.md").read_text()
    reco = reco.replace("@@REALSHORT@@", realshort)
    for k in ("SEQ", "REALTEXT", "ADULTTOP", "REAL2"):
        f = HERE / f"{k.lower()}.md"
        val = f.read_text().strip() if f.exists() else f"({k.lower()} pending)"
        res = res.replace(f"@@{k}@@", val)
        reco = reco.replace(f"@@{k}@@", val)

    def table(m):
        f = HERE / "tables" / f"{m.group(1)}.md"
        return f.read_text().strip() if f.exists() else f"(table {m.group(1)} pending)"

    res = re.sub(r"@@TABLE:(\w+)@@", table, res)
    figs = ("\n\n![power](figs/power.png)\n\n*Figure 1: rejection rates on `disjoint_d2_a20`, matched splits.*\n\n"
            "![alpha_lo](figs/alpha_lo.png)\n\n*Figure 2: mean certified lower bound α̂_lo (down-closed rule).*\n\n"
            "![real](figs/real.png)\n\n*Figure 3: certified TV lower bounds on the real contrasts (min over 8 "
            "splits).*\n")
    files = (HERE / "files.md").read_text() if (HERE / "files.md").exists() else ""
    note = "\n\n".join([head.strip(), theory.strip(), lit.strip(), res.strip() + figs, lean.strip(),
                        reco.strip(), files.strip()]) + "\n"
    (HERE / "NOTE.md").write_text(note)
    print("NOTE.md", len(note.split()), "words")


if __name__ == "__main__":
    main()
