"""Fill the power / boundary-ratio placeholders of NOTE.md from out/power_reps.jsonl."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import analyze as An

DELTA = 0.05
p, first = An.table_power()
b = An.table_boundary_ratio()
ALPH = [0.0, 0.05, 0.1, 0.15]

lines = ["### 8.2 Power on the main design (disjoint_d2_a20, $\\alpha^\\star=0.2$)", "",
         "Same draws as Table `tab:calibrations` and the classifier baseline (seed $70000+i$, pooled median bandwidth); "
         "replicates per size: " + ", ".join(f"{int(n)}: {int(r)}" for n, r in p.groupby('n').reps.first().items()) + ". "
         "Rejection rate of $H_{0,\\alpha}$ (power for $\\alpha<0.2$; type-I error at $\\alpha=0.2$). "
         "'boot' = split bootstrap certificate (Thm `thm:split`, percentile limit, asymptotic), "
         "'P17' = finite-sample split certificate (Prop. `prop:splitfs`), 'clf' = split gradient-boosting classifier with Hoeffding term "
         "(from `results/tables/e6b_classifier_baseline_replicates.csv`, 100 replicates).", "",
         "| $n=m$ | $\\alpha$ | mean $\\widehat D_\\alpha$ | paper $\\tau$ | rate | $\\tau_A$ | rate | $\\tau_P$ closed | rate | $\\tau_{\\rm new}$ | rate | P17 | boot | clf |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in p.iterrows():
    lines.append(f"| {int(r.n)} | {r.alpha:.2f} | {r.mean_dhat:.3f} | {r.tau_old:.3f} | {r.old:.2f} | {r.tau_A:.3f} | {r.A:.2f} | "
                 f"{r.tau_P_closed:.3f} | {r.P_closed:.2f} | {r.tau_new:.3f} | **{r.new:.2f}** | {r.prop17:.2f} | {r.bootstrap:.2f} | "
                 + (f"{r.classifier_gbm:.2f}" if not np.isnan(r.classifier_gbm) else "–") + " |")
lines += ["", "**Smallest $n=m$ on the grid $\\{200,400,800,1600,3200\\}$ with power $\\ge0.8$**, and the crossing size "
          "$n^\\ast$ where the threshold equals the mean statistic (extrapolated from the largest $n$ with $\\tau\\propto n^{-1/2}$):", "",
          "| $\\alpha$ | paper $\\tau$ | $\\tau_A$ | $\\tau_P$ closed | $\\tau_{\\rm new}$ | P17 | boot | clf | $n^\\ast$ paper | $n^\\ast$ new |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in first.iterrows():
    lines.append(f"| {r.alpha:.2f} | {r.old} | {r.A} | {r.P_closed} | **{r.new}** | {r.prop17} | {r.bootstrap} | {r.classifier_gbm} | {r.n_cross_old} | {r.n_cross_new} |")
bt = p[np.isclose(p.alpha, 0.2)]
lines += ["", "At the boundary $\\alpha=0.2$ no analytic threshold and no Prop. 17 test rejected at any $n$; the bootstrap certificate "
          "rejected at rates " + ", ".join(f"{x:.2f}" for x in bt.bootstrap) + " for $n=" + ",".join(str(int(x)) for x in bt.n) +
          "$ (nominal 0.05; see the caveat in Section 9). " +
          ("The decisions of $\\tau_{\\rm new}$ are identical in every replicate when the certified lower bound of the QP value is used instead of the iterate."
           if bool(p.new_lower_same.all()) else
           "Using the certified lower bound of the QP value instead of the iterate changes some decisions; see `out/tables.md` (column new_certified_lower).")]
power_md = "\n".join(lines)

bl = ["**Table 8.1d — threshold / true 95% quantile at the boundary of the main design across $n$** "
      "(median bandwidth; quantile from the $\\alpha=0.2$ point of the power paths, so it is noisy at $n\\ge1600$):", "",
      "| $n=m$ | reps | $q_{0.95}$ | paper | $\\tau_A$ | $\\tau_{P'}$ | $\\tau_{\\rm new}$ |", "|---|---|---|---|---|---|---|"]
for _, r in b.iterrows():
    bl.append(f"| {int(r.n)} | {int(r.reps)} | {r.q95:.3f} | {r.ratio_old:.1f} | {r.ratio_A:.1f} | {r.ratio_P:.1f} | **{r.ratio_new:.1f}** |")
s = An.table_slack()
bl += ["", "On the published slack design (Table `tab:slack`: disjoint, fixed wide bandwidth, 160 replicates; the statistic there uses "
       "256 cos/sin random features of the same kernel), threshold / quantile:", "",
       "| $\\alpha$ | $n$ | $q_{0.95}$ | paper | $\\tau_A$ | $\\tau_{P'}$ | $\\tau_{\\rm new}$ |", "|---|---|---|---|---|---|---|"]
for _, r in s.iterrows():
    bl.append(f"| {r.alpha:.2f} | {int(r.n)} | {r.q95:.3f} | {r.ratio_old:.0f} | {r.ratio_A:.0f} | {r.ratio_P:.1f} | **{r.ratio_new:.1f}** |")
bl += ["", "At $\\alpha=0$ on that design the bandwidth ($\\sigma\\approx11$, about five times the main design's median bandwidth) makes the feature variance tiny, so even "
       "the new threshold is 12–14 times the quantile; this is the regime where only variance information would help (Section 5)."]
ratio_md = "\n".join(bl) + "\n"

note = open("NOTE.md").read()
note = note.replace("TABLE_81D_PLACEHOLDER", ratio_md).replace("POWER_PLACEHOLDER", power_md)
open("NOTE.md", "w").write(note)
print(power_md); print(ratio_md)
