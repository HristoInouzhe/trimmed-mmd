#!/usr/bin/env python3
"""Write the single-file submission the Annals asks for ("Please do not use \\input"): submission/main.tex and
submission/supplement.tex with every \\input{...} and \\InputIfFileExists{...}{}{} inlined recursively (preamble,
sections, generated tables and number macros, label files, bibliography), comment-only lines dropped, and the figures
and class files copied next to them. Run after build.sh: python3 -I tools/flatten.py. Then compile submission/ to check.
References to objects of the other document (main <-> supplement) are written as plain text, as the Annals asks
("Do not use \\ref{} commands to cite objects from other files (including supplements)"); the label files
supp_refs.tex / main_refs.tex are not included.
"""
import re, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "submission"
PAT = re.compile(r"\\input\{([^}]+)\}|\\InputIfFileExists\{([^}]+)\}\{\}\{\}")

def resolve(name):
    p = HERE / name
    return p if p.suffix == ".tex" else p.with_suffix(".tex")

def inline(path, depth=0):
    assert depth < 10, f"input nesting too deep at {path}"
    out = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("%"):
            continue
        def sub(m):
            name = m.group(1) or m.group(2)
            if "#" in name:  # a macro argument, e.g. \fittable{#1}
                return m.group(0)
            if Path(name).stem in ("supp_refs", "main_refs"):
                return ""  # label files of the other document: cross-document references become plain text below
            p = resolve(name)
            if not p.exists():
                if m.group(2):
                    return ""
                raise SystemExit(f"flatten.py: missing input {name} in {path}")
            return "\n" + inline(p, depth + 1) + "\n"
        out.append(PAT.sub(sub, line))
    return "\n".join(out)

def clean(text):
    """Drop revision markup and trailing comments, which a submission should not carry: \\rev{...} keeps its argument,
    the revblock environment keeps its body, and an unescaped % ends the line (verbatim-free sources)."""
    text = re.sub(r"\\(begin|end)\{revblock\}\n?", "", text)
    out, i, key = [], 0, "\\rev{"
    while True:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k, depth = j + len(key), 1
        while depth:
            c = text[k]
            if c == "\\":
                k += 2
                continue
            depth += {"{": 1, "}": -1}.get(c, 0)
            k += 1
        text = text[:j] + text[j + len(key):k - 1] + text[k:]
        i = j
    text = "".join(out)
    lines = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        if m:
            line = line[:m.start()].rstrip() + ("%" if line[:m.start()].rstrip().endswith("}") is False and line[:m.start()].strip() == "" else "")
            if not line.strip():
                continue
        lines.append(line)
    return "\n".join(lines)

def aux_numbers(aux):
    """Printed number of every label in a .aux file (first group of \\newlabel; @cref records skipped)."""
    nums = {}
    for line in aux.read_text(errors="ignore").splitlines():
        m = re.match(r"\\newlabel\{([^}]+)\}\{(.*)\}$", line)
        if not m or m.group(1).endswith("@cref"):
            continue
        parts = re.findall(r"\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", m.group(2))
        if parts:
            nums[m.group(1)] = parts[0]
    return nums


def plain_cross_refs(text, foreign, doc):
    """The Annals asks that objects of another file (the supplement) be cited in plain text, not with \\ref: every
    \\ref / \\eqref to a label not defined in this document becomes its printed number, from the other document's .aux
    ((S3.3) for \\eqref). Labels of this document keep their \\ref."""
    own = set(re.findall(r"\\label\{([^}]+)\}", text))
    count = 0

    def sub(m):
        nonlocal count
        cmd, lab = m.group(1), m.group(2)
        if lab in own:
            return m.group(0)
        if lab not in foreign:
            raise SystemExit(f"flatten.py: {doc}: \\{cmd}{{{lab}}} is defined in neither document")
        count += 1
        return f"({foreign[lab]})" if cmd == "eqref" else foreign[lab]
    return re.sub(r"\\(ref|eqref)\{([^}]+)\}", sub, text), count


OUT.mkdir(exist_ok=True)
for doc, other in (("main", "supplement"), ("supplement", "main")):
    text, k = plain_cross_refs(clean(inline(HERE / f"{doc}.tex")), aux_numbers(HERE / f"{other}.aux"), doc)
    (OUT / f"{doc}.tex").write_text(text + "\n")
    print(f"submission/{doc}.tex: {k} references to the {other} written as plain text")
used = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", (OUT / "main.tex").read_text() + (OUT / "supplement.tex").read_text()))
for f in used:
    dst = OUT / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / f, dst)
for f in ("imsart.cls", "imsart.sty"):
    shutil.copy2(HERE / f, OUT / f)
assert not re.search(r"\\input\{[^#]", (OUT / "main.tex").read_text()), "an \\input survived in submission/main.tex"
print(f"submission/main.tex, submission/supplement.tex, {len(used)} figures")
