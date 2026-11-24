#!/usr/bin/env python3
"""Check that every decimal number in the rewritten text occurs in an allowed source: the original paper's text and
tables, or a generated table. Numbers typed by hand are the most common silent error of a rewrite.

Usage (from the target paper folder):
  python3 -I check_numbers.py --text sections supp --sources ../v4/sections ../v4/supp ../v4/tables tables \
      [--skip-line-with Opus GPT]
Prints every decimal (d.d form) in --text files not found verbatim in any --sources .tex file; exit 1 if any.
Lines starting with % are ignored; --skip-line-with ignores lines containing a word (e.g. model version names).
"""
import argparse, re, sys
from pathlib import Path

NUM = re.compile(r"(?<![\w.\\])\d+\.\d+(?![\w.])")
ap = argparse.ArgumentParser()
ap.add_argument("--text", nargs="+", default=["sections", "supp"])
ap.add_argument("--sources", nargs="+", required=True)
ap.add_argument("--skip-line-with", nargs="*", default=[])
a = ap.parse_args()
known = set()
for d in a.sources:
    for p in ([Path(d)] if Path(d).is_file() else Path(d).rglob("*.tex")):
        known |= set(NUM.findall(p.read_text(errors="ignore")))
missing = []
for d in a.text:
    for p in sorted(Path(d).rglob("*.tex")):
        for k, line in enumerate(p.read_text().splitlines(), 1):
            if line.lstrip().startswith("%") or any(w in line for w in a.skip_line_with):
                continue
            missing += [f"{p}:{k}: {x}" for x in NUM.findall(line) if x not in known]
if missing:
    print("numbers not found in the sources:\n" + "\n".join(missing))
    sys.exit(1)
print("check_numbers: every decimal number in the text occurs in a source or a generated table")
