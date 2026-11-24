#!/usr/bin/env python3
"""AoS-specific derived copies of generated tables (never edit the outputs by hand). Reads only files that
tools/sync.sh copied into tables/, and writes files with an _aos suffix:
  tables/estimand_aos.tex  the RHC weighting table of jmlr/v3 (tables/estimand_v3.tex) for the imsart text width, with shorter header words (the
                           Kish fractions 'cath.'/'others', 'risk diff.', '95% int.').
Fails loudly when an expected pattern is missing. Run from anywhere: python3 -I tools/derive.py
"""
from pathlib import Path
T = Path(__file__).resolve().parents[1] / "tables"
src = (T / "estimand_v3.tex").read_text()
# Shorter header words so that the table fits the imsart text width at footnote size without scaling.
EDITS = [("& catheterized & others &", "& cath. & others &"),
         ("& risk difference \\\\", "& risk diff. \\\\"),
         ("& (95\\% interval) \\\\", "& (95\\% int.) \\\\")]
out = src
for a, b in EDITS:
    if a not in out:
        raise SystemExit(f"derive.py: pattern not found in estimand_v3.tex: {a!r}")
    out = out.replace(a, b)
(T / "estimand_aos.tex").write_text("% Derived by tools/derive.py from estimand_v3.tex; do not edit.\n" + out)
print("tables/estimand_aos.tex")

# ---------------------------------------------------------------------------------------------------------------------
# Supplement cuts approved by the author (2026-11-22, cold read 7): compact derived copies of three S6 tables. Each is a
# mechanical transformation of generated files (rows or columns selected, one column joined by design, bold recomputed
# as "every cell equal to the largest value of its row, when that value is positive", the rule of the generators).
# Generator changes that would make these copies unnecessary are listed in aos/restructure_log.md.
import re as _re

def _cells(line):
    body = line.rstrip()
    if not body.endswith("\\\\"):
        raise SystemExit(f"derive.py: not a table row: {line!r}")
    return [c.strip() for c in body[:-2].split("&")]

def _val(cell):
    m = _re.search(r"-?\d+\.\d+|-?\d+", cell.replace("{,}", ""))
    return float(m.group()) if m and "--" not in cell else None

def _plain(cell):
    return cell.replace("\\mathbf{", "").replace("}$", "$") if "\\mathbf{" in cell else cell

def _rebold(cells, first):
    cells = [_plain(c) for c in cells]
    vals = [_val(c) for c in cells[first:]]
    top = max([v for v in vals if v is not None], default=None)
    if top is not None and top > 0:
        for i, v in enumerate(vals):
            if v is not None and abs(v - top) < 1e-12:
                c = cells[first + i]
                inner = c[1:-1] if c.startswith("$") and c.endswith("$") else c
                cells[first + i] = "$\\mathbf{" + inner + "}$"
    return cells

def _rows(text):
    lines = text.splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("\\midrule")) + 1
    i1 = next(i for i, l in enumerate(lines) if l.startswith("\\bottomrule"))
    return lines, i0, i1

# (1) Univariate designs without the oscillating densities with nu >= 4 (every bound certifies almost nothing there).
src = (T / "aos_s6_univ.tex").read_text()
lines, i0, i1 = _rows(src)
body, keep = [], True
for l in lines[i0:i1]:
    if l.startswith("oscillating, $\\nu=") and not l.startswith("oscillating, $\\nu=1$"):
        keep = False
    elif l and not l.startswith(" ") and not l.startswith("\\addlinespace"):
        keep = True
    if keep:
        body.append(l)
while body and body[-1].startswith("\\addlinespace"):
    body.pop()
if any("nu=4" in l or "nu=16" in l or "nu=64" in l for l in body) or len(body) < 10:
    raise SystemExit("derive.py: aos_s6_univ row selection failed")
head = "\n".join(lines[1:i0]).replace("& $\\widehat\\alpha_{\\mathrm{an}}$ & $\\widehat\\alpha$ & betting & EB",
                                      "& $\\widehat\\alpha_{\\mathrm{an}}$ & ramp & ramp, single & EB")
if "ramp, single" not in head:
    raise SystemExit("derive.py: aos_s6_univ header pattern not found")
(T / "aos_s6_univ_aos.tex").write_text("% Derived by tools/derive.py from aos_s6_univ.tex (rows nu = 4, 16, 64 dropped); do not edit.\n"
                                        + head + "\n" + "\n".join(body) + "\n" + "\n".join(lines[i1:]) + "\n")
print("tables/aos_s6_univ_aos.tex")

# (2) Recovery with every method (Gaussian ramp) plus the Laplace ramp of Figure 3, joined by (design, alpha*).
rec = (T / "aos_s6_recovery.tex").read_text()
lap = (T / "aos_main_recovery_gauss.tex").read_text()
L, l0, l1 = _rows(lap)
lapv = {}
for l in L[l0:l1]:
    c = _cells(l)
    lapv[(c[0], c[1])] = _plain(c[2])
lines, i0, i1 = _rows(rec)
out = []
for l in lines[i0:i1]:
    c = _cells(l)
    key = (c[0], c[1])
    if key not in lapv:
        raise SystemExit(f"derive.py: no Laplace value for {key}")
    c = c[:3] + [lapv.pop(key)] + c[3:]
    out.append(" & ".join(_rebold(c, 2)) + " \\\\")
if lapv:
    raise SystemExit(f"derive.py: Laplace rows not used: {list(lapv)}")
head = "\n".join(lines[1:i0])
for a, b in [("\\begin{tabular}{lcccccccccccccccc}", "\\begin{tabular}{lccccccccccccccccc}"),
             ("\\multicolumn{5}{c}{kernel}", "\\multicolumn{6}{c}{kernel}"),
             ("\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\\cmidrule(lr){13-17}", "\\cmidrule(lr){3-8}\\cmidrule(lr){9-13}\\cmidrule(lr){14-18}"),
             ("& $\\widehat\\alpha_{\\mathrm{an}}$ & $\\widehat\\alpha$ & $\\alpha{=}0$", "& $\\widehat\\alpha_{\\mathrm{an}}$ & $\\widehat\\alpha$ & ramp & $\\alpha{=}0$"),
             ("design & $\\alpha^\\star$ & $\\widehat\\alpha_{\\mathrm{an}}$ & $\\widehat\\alpha$ & ramp", "design & $\\alpha^\\star$ & $\\widehat\\alpha_{\\mathrm{an}}$ & $\\widehat\\alpha$ & ramp"),
             ("\n &  &  &  & ramp & witness & ramp", "\n &  &  & Laplace & Gaussian & ramp & witness & ramp")]:
    if a not in head:
        raise SystemExit(f"derive.py: aos_s6_recovery header pattern not found: {a!r}")
    head = head.replace(a, b)
(T / "aos_s6_recovery_aos.tex").write_text("% Derived by tools/derive.py from aos_s6_recovery.tex and the Laplace column of aos_main_recovery_gauss.tex; do not edit.\n"
                                            + head + "\n" + "\n".join(out) + "\n" + "\n".join(lines[i1:]) + "\n")
print("tables/aos_s6_recovery_aos.tex")

# (3) Power against H_{0,0}: the Gaussian ramp, the nu-SVM normal and the two classifiers on the batches of the ramp.
pw = (T / "aos_s6_table1.tex").read_text()
lines, i0, i1 = _rows(pw)
hdr = _cells(next(l for l in lines if l.startswith("design &")))
want = {"$\\widehat\\alpha$": None, "$\\nu$-SVM": None}
idx = [hdr.index("$\\widehat\\alpha$"), hdr.index("$\\nu$-SVM")]
lab = [i for i, h in enumerate(hdr) if h == "labels"]
if len(lab) != 2:
    raise SystemExit("derive.py: aos_s6_table1 'labels' columns not found")
idx += lab
out = []
for l in lines[i0:i1]:
    c = _cells(l)
    out.append(" & ".join(_rebold([c[0], c[1]] + [c[i] for i in idx], 2)) + " \\\\")
head = ("\\begin{tabular}{lccccc}\n\\toprule\n"
        "design & $\\alpha^\\star$ & ramp & $\\nu$-SVM & GBM & logistic \\\\\n"
        " &  &  & ramp & labels & labels \\\\\n\\midrule")
(T / "aos_s6_table1_aos.tex").write_text("% Derived by tools/derive.py from aos_s6_table1.tex (columns: Gaussian ramp, nu-SVM ramp, GBM and logistic labels on the batches of the ramp); do not edit.\n"
                                          + head + "\n" + "\n".join(out) + "\n" + "\n".join(lines[i1:]) + "\n")
print("tables/aos_s6_table1_aos.tex")

# (4) Referee round (2026-11-22): counts from the derived recovery table (2) for Section 7.1 and S6.3, comparing the
# untrimmed Gaussian witness cut at the quantiles of the tested level (column "alpha=0 witness") with the Laplace ramp
# (column "alpha-hat Laplace"), at the two decimals printed; no paired standard errors, as the kernels differ.
rec2 = (T / "aos_s6_recovery_aos.tex").read_text()
lines, i0, i1 = _rows(rec2)
hdr1 = _cells(next(l for l in lines if l.startswith("design &")))
hdr2 = _cells(lines[lines.index(next(l for l in lines if l.startswith("design &"))) + 1])
cols = [a + "|" + b for a, b in zip(hdr1, hdr2)]
try:
    i_lap = cols.index("$\\widehat\\alpha$|Laplace")
    i_zc = cols.index("$\\alpha{=}0$|witness")
except ValueError:
    raise SystemExit(f"derive.py: recovery columns not found: {cols}")
above = below = equal = 0
for l in lines[i0:i1]:
    c = _cells(l)
    a, z = _val(c[i_lap]), _val(c[i_zc])
    if a is None or z is None:
        raise SystemExit(f"derive.py: missing value in row {c[0]}")
    if z > a + 1e-12:
        above += 1
    elif z < a - 1e-12:
        below += 1
    else:
        equal += 1
if above + below + equal != 19:
    raise SystemExit(f"derive.py: expected 19 designs, got {above + below + equal}")
(T / "derived_aos_macros.tex").write_text(
    "% Derived by tools/derive.py from aos_s6_recovery_aos.tex (and, below, aos_kernel_recovery.tex and aos_s6_univ.tex); do not edit.\n"
    f"\\newcommand{{\\dvZeroCutAboveLap}}{{{above}}}\n"
    f"\\newcommand{{\\dvZeroCutBelowLap}}{{{below}}}\n"
    f"\\newcommand{{\\dvZeroCutEqualLap}}{{{equal}}}\n")
print("tables/derived_aos_macros.tex", above, equal, below)

# ---------------------------------------------------------------------------------------------------------------------
# Readability pass (2026-11-22): Table S2 (kernel comparison, aos_kernel_recovery.tex) and Table S4 (univariate designs,
# aos_s6_univ.tex) are folded into sentences of Section S6; these counts, read from the generated tables, replace them.
# (5) Analytic estimator, Laplace against Gaussian kernel, at the two decimals of aos_kernel_recovery.tex (last two columns).
kr = (T / "aos_kernel_recovery.tex").read_text()
lines, i0, i1 = _rows(kr)
an_below = an_above = an_equal = 0
for l in lines[i0:i1]:
    if not l.strip() or l.startswith("\\midrule") or l.startswith("median"):
        continue
    c = _cells(l)
    g, la = _val(c[-2]), _val(c[-1])
    if g is None or la is None:
        raise SystemExit(f"derive.py: aos_kernel_recovery analytic columns missing in row {c[0]}")
    if la < g - 1e-12:
        an_below += 1
    elif la > g + 1e-12:
        an_above += 1
    else:
        an_equal += 1
if an_below + an_above + an_equal != 19:
    raise SystemExit(f"derive.py: aos_kernel_recovery: expected 19 designs, got {an_below + an_above + an_equal}")
# (6) Univariate designs: design and size cells of aos_s6_univ.tex in which the Kolmogorov--Smirnov bound exceeds the
# better of the analytic estimator and the Gaussian ramp (columns 4, 5, 6 of each row; design and alpha* only on the
# first row of a design).
uv = (T / "aos_s6_univ.tex").read_text()
lines, i0, i1 = _rows(uv)
ks_cells = ks_above = 0
for l in lines[i0:i1]:
    if not l.strip() or l.startswith("\\addlinespace"):
        continue
    c = _cells(l)
    ks, an, rp = _val(c[3]), _val(c[4]), _val(c[5])
    if ks is None or an is None or rp is None:
        raise SystemExit(f"derive.py: aos_s6_univ: missing value in row {l!r}")
    ks_cells += 1
    ks_above += int(ks > max(an, rp) + 1e-12)
if ks_cells < 20:
    raise SystemExit(f"derive.py: aos_s6_univ: only {ks_cells} cells read")
with open(T / "derived_aos_macros.tex", "a") as fh:
    fh.write(f"\\newcommand{{\\dvKerAnLapBelow}}{{{an_below}}}\n"
             f"\\newcommand{{\\dvKerAnLapAbove}}{{{an_above}}}\n"
             f"\\newcommand{{\\dvKsCells}}{{{ks_cells}}}\n"
             f"\\newcommand{{\\dvKsAboveKernel}}{{{ks_above}}}\n")
print("tables/derived_aos_macros.tex (S6 folds)", an_below, an_above, an_equal, ks_cells, ks_above)
# (7) Table S3 (aos_s6_recovery_aos.tex) is folded into the text of Section S6.3: the designs on which tuned gradient
# boosting and tuned logistic regression (labels, refitted on the batches of the ramp) recover more than the Laplace ramp,
# at the two decimals of the table, and the medians over the designs of these columns.
rec3 = (T / "aos_s6_recovery_aos.tex").read_text()
lines, i0, i1 = _rows(rec3)
h1 = _cells(next(l for l in lines if l.startswith("design &")))
h2 = _cells(lines[lines.index(next(l for l in lines if l.startswith("design &"))) + 1])
if len(h1) != 18 or h1[3] != "$\\widehat\\alpha$" or h2[3] != "Laplace" or h1[11] != "tuned" or h1[16] != "tuned":
    raise SystemExit(f"derive.py: aos_s6_recovery_aos header not as expected: {h1} / {h2}")
above = {11: [], 16: []}
for l in lines[i0:i1]:
    c = _cells(l)
    name = c[0] + (f" ($\\alpha^\\star={_plain(c[1]).strip('$')}$)" if _plain(c[1]) not in ("$0.2$",) else "")
    lapv = _val(c[3])
    for k in above:
        v = _val(c[k])
        if v is None or lapv is None:
            raise SystemExit(f"derive.py: missing value in row {c[0]}")
        if v > lapv + 1e-12:
            above[k].append(name)
with open(T / "derived_aos_macros.tex", "a") as fh:
    fh.write("\\newcommand{\\dvGbmTunedAboveLapN}{%d}\n" % len(above[11]))
    fh.write("\\newcommand{\\dvGbmTunedAboveLapDesigns}{%s}\n" % "; ".join(above[11]))
    fh.write("\\newcommand{\\dvLogitTunedAboveLapN}{%d}\n" % len(above[16]))
    fh.write("\\newcommand{\\dvLogitTunedAboveLapDesigns}{%s}\n" % "; ".join(above[16]))
print("tables/derived_aos_macros.tex (Table S3 fold)", len(above[11]), len(above[16]))
