#!/usr/bin/env bash
# Build the Annals of Statistics version: main.pdf and supplement.pdf.
# 1. copy generated tables, macros and figures (tools/sync.sh); 2. derived copies (tools/derive.py);
# 3. references cited in each document (tools/make_refs.py); 4. pdflatex passes of both documents, exchanging labels
# through tools/supp_refs.py (imsart forbids xr); 5. report errors, undefined references, overfull boxes and pages;
# 6. check that every decimal number comes from a generated or checked source.
set -euo pipefail
cd "$(dirname "$0")"
bash tools/sync.sh
python3 -I tools/derive.py
python3 -I tools/make_refs.py
python3 -I tools/make_refs.py --supp
run() { pdflatex -interaction=nonstopmode -halt-on-error "$1.tex" > "$1.buildlog" 2>&1 || { echo "pdflatex failed on $1; see $1.buildlog"; grep -m5 -A3 '^!' "$1.log"; exit 1; }; }
for i in 1 2; do run supplement; run main; python3 -I tools/supp_refs.py >/dev/null; python3 -I tools/supp_refs.py --reverse >/dev/null; done
run supplement; run main
for d in main supplement; do
  echo "$d: pages $(pdfinfo $d.pdf | awk '/Pages/{print $2}'), undefined references $(grep -c 'Reference.*undefined\|Citation.*undefined' $d.log || true), overfull boxes $(grep -c 'Overfull' $d.log || true)"
done
echo "main text ends on page: $(grep -oP '\\newlabel\{mainend\}\{\{[^}]*\}\{\K[0-9]+' main.aux || echo '?')"
if [ -d ../jmlr/v3 ]; then
  python3 -I tools/check_numbers.py --text sections supplement --sources ../jmlr/v3/sections ../jmlr/v3/appendix ../paper.tex ../supplement.tex ../results/v2/tables ../results/v3/tables ../results/aos/tables ../results/tables ../tables tables NEW_STATEMENTS.md --skip-line-with Opus GPT vandervaart1998asymptotic vandervaart1996weak steinwart2008svm wendland2005scattered parbox boucheron2013concentration pinelis1994optimum '\node' '0.5,0.6' '\draw' '\fill' 'coordinates' '{(0.3,' || true
else
  echo "check_numbers: skipped (it compares the text with the development sources, which the public copy does not hold)"
fi
