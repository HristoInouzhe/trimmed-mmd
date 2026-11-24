#!/usr/bin/env python3
"""AoS version: write refs_main.tex (default) or refs_supp.tex (--supp), natbib author-year thebibliography, from the source's numbered list ../refs_all.tex.

Entries keep their text; each \\bibitem{key} gains the natbib label [Short(Year)Long], and entries are sorted by
first-author surname and year. Only keys cited in the JMLR sources are kept. Run from jmlr/.

jmlr/refs_extra.tex (same format as ../refs_all.tex, optional) is read after ../refs_all.tex: an entry there with a
key already in ../refs_all.tex replaces it (corrections for the JMLR version only), other keys are added.
../refs_all.tex itself is never modified, since it also serves the original manuscript.

  python3 -I tools/make_refs.py            write refs.tex
  python3 -I tools/make_refs.py --check    write nothing; list replaced/added keys and which of them are cited
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CHECK = "--check" in sys.argv[1:]
DOC = "supp" if "--supp" in sys.argv[1:] else "main"  # main.tex + sections/ or supplement.tex + supplement/

def parse(path):
    """(key, body) pairs of a thebibliography-style list; % comment lines are dropped."""
    text = "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("%"))
    text = re.sub(r"\\(begin|end)\{thebibliography\}(\{[^}]*\})?", "", text)
    return re.findall(r"\\bibitem\{([^}]+)\}\s*\n(.*?)(?=\n\s*\\bibitem|\Z)", text, flags=re.S)

entries = dict(parse(HERE.parent / "refs_all.tex"))  # aos -> repository root
base_keys = set(entries)
extra_path = HERE.parent / "jmlr" / "refs_extra.tex"  # shared with the JMLR versions; aos/refs_extra.tex adds AoS-only entries
extra = parse(extra_path) if extra_path.exists() else []
aos_extra = HERE / "refs_extra.tex"  # AoS-only entries (Crossref-checked), read last; they may correct jmlr entries
aos_list = parse(aos_extra) if aos_extra.exists() else []
for name, lst in (("jmlr/refs_extra.tex", extra), ("aos/refs_extra.tex", aos_list)):
    dup = sorted({k for k, _ in lst if [e for e, _ in lst].count(k) > 1})
    if dup:
        sys.exit(f"duplicate keys in {name}: " + ", ".join(dup))
aos_keys = {k for k, _ in aos_list}
extra = [(k, b) for k, b in extra if k not in aos_keys] + aos_list
replaced = sorted(k for k, _ in extra if k in base_keys)
added = sorted(k for k, _ in extra if k not in base_keys)
entries.update(extra)
entries = list(entries.items())

def surname(a):
    a = a.strip().rstrip(".")
    toks = a.split()
    while len(toks) > 1 and re.fullmatch(r"[A-Z{\\][^ ]*\.(-?[A-Z]\.)*|[A-Z]\.-[A-Z]\.?|[A-Z]\.~?[A-Z]\.?", toks[0]):
        toks = toks[1:]
    return " ".join(toks)

def authors(line):
    line = line.strip().rstrip(".")
    etal = bool(re.search(r",?\s*et al\.?$", line))
    line = re.sub(r",?\s*et al\.?$", "", line)
    parts = [p for p in re.split(r",\s*and\s+|\s+and\s+|,\s+", line) if p.strip()]
    return [surname(p) for p in parts], etal

cited = set()
for p in (list(HERE.glob("sections/*.tex")) + [HERE / "main.tex"]) if DOC == "main" else (list(HERE.glob("supplement/*.tex")) + [HERE / "supplement.tex"]):
    if p.exists():
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", p.read_text()):
            cited |= {k.strip() for k in m.group(1).split(",")}

out = []
for key, body in entries:
    if cited and key not in cited:
        continue
    first = body.strip().splitlines()[0]
    names, etal = authors(first)
    body_noarxiv = re.sub(r"arXiv[:\s][\d.v]+", "", body)
    years = re.findall(r"\b(19\d\d|20\d\d)\b", body_noarxiv)
    year = years[-1] if years else "n.d."
    if len(names) == 1 and not etal:
        short = names[0]
    elif len(names) == 2 and not etal:
        short = f"{names[0]} and {names[1]}"
    else:
        short = f"{names[0]} et~al."
    if len(names) <= 2 and not etal:
        long_ = short
    elif etal:
        long_ = ", ".join(names) + ", et~al."
    else:
        long_ = ", ".join(names[:-1]) + ", and " + names[-1]
    sortkey = re.sub(r"[^a-z]", "", re.sub(r"\\.|\{|\}", "", names[0].lower().replace("del ", "").replace("van der ", ""))) + year
    out.append((sortkey, key, short, year, long_, body.strip()))

out.sort()
# Same short label and year (e.g. two del Barrio et al. 2020 papers): suffix a, b, ... as natbib/plainnat does.
groups = {}
for i, (_, _, short, year, *_) in enumerate(out):
    groups.setdefault((short, year), []).append(i)
for idx in groups.values():
    if len(idx) > 1:
        for j, i in enumerate(idx):
            s, k, sh, y, lg, b = out[i]
            out[i] = (s, k, sh, f"{y}\\natexlab{{{chr(97 + j)}}}", lg, b)
missing = sorted(cited - {k for _, k, *_ in out})
if CHECK:
    for what, keys in (("replaced", replaced), ("added", added)):
        for k in keys:
            print(f"{what:8s} {k:28s} {'cited' if k in cited else 'not cited'}")
    if missing:
        print("cited but in neither refs_all.tex nor refs_extra.tex: " + ", ".join(missing))
    print(f"would write {len(out)} entries ({len(replaced)} replaced, {len(added)} added in refs_extra.tex)")
    sys.exit(0)
if missing:
    sys.exit("cited but in neither refs_all.tex nor refs_extra.tex: " + ", ".join(missing))
OUT = HERE / ("refs_main.tex" if DOC == "main" else "refs_supp.tex")
with open(OUT, "w") as f:
    f.write("% Generated by tools/make_refs.py from ../refs_all.tex and refs_extra.tex; do not edit by hand.\n")
    f.write("\\begin{thebibliography}{%d}\n\\providecommand{\\natexlab}[1]{#1}\n\n" % len(out))
    for _, key, short, year, long_, body in out:
        f.write(f"\\bibitem[{{{short}}}({year}){{{long_}}}]{{{key}}}\n{body}\n\n")
    f.write("\\end{thebibliography}\n")
print(f"{OUT.name}: {len(out)} entries")
