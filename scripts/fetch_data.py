#!/usr/bin/env python
"""Download the two public datasets used in the real-data section.

Both are open, unauthenticated, and in the public domain or freely
redistributed for research; neither needs an application or a login. The
raw files land in ``data_raw/`` (git-ignored) and everything downstream is
derived from them, so the repository stays small and the analysis stays
reproducible from a single command.

**RHC** -- the Right Heart Catheterization study (Connors et al., JAMA
1996), 5735 critically ill patients, 2184 of whom received a catheter.
This is the canonical limited-overlap dataset in the causal-inference
literature; it is the example Crump, Hotz, Imbens and Mitnik (2009) use to
motivate *trimming* the sample down to a region where treated and control
covariate distributions actually meet. That makes it the natural test of a
method whose output is an estimate of how much trimming is needed.
Mirrored by the Vanderbilt Department of Biostatistics dataset repository.

**NHANES** -- cycle G (2011-2012) wrist accelerometry, 6917 participants
with hourly activity summaries, plus demographics and body measurements
for every one of them. Gives genuinely functional observations (a daily
activity profile per participant) at a sample size where the analytic
threshold is usable. Served by the CDC's public file server.

Idempotent: an output file that already exists is left alone. Pass
``--only rhc`` or ``--only nhanes`` to fetch just one.

Usage: ``python scripts/fetch_data.py``
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RHC_URL = "https://hbiostat.org/data/repo/rhc.csv"

# Covertype: 581,012 forest sites, 54 attributes, 7 cover types. Backs an
# exploratory large-sample study (`realdata_run.r9_large_sample`) that is not
# reported in the paper; fetch it explicitly with `--only covertype` (11 MB).
COVERTYPE_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"

_NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles"
# PAXHR: hourly activity summaries (the bulk of the download, ~130-145 MB).
# PAXHD: one header row per participant, carrying the recording start time.
# DEMO/BMX: age, sex, income-to-poverty ratio, BMI.
NHANES_STEMS = ("PAXHR", "PAXHD", "DEMO", "BMX")
# Cycle letter -> the year that names the directory on the CDC server.
NHANES_CYCLES = {"G": "2011", "H": "2013"}


def _get(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip {dest.name} ({dest.stat().st_size / 1e6:.1f} MB already present)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get  {dest.name} ...", flush=True)
    # Download to a .part file and rename only on success, so an interrupted
    # fetch is retried rather than mistaken for a complete one next run.
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"       done ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data_raw"))
    ap.add_argument("--only", choices=["rhc", "nhanes", "covertype"], default=None)
    ap.add_argument("--cycles", nargs="+", default=["G", "H"],
                    choices=sorted(NHANES_CYCLES),
                    help="NHANES cycles to fetch (G=2011-12, H=2013-14). "
                         "Both are needed for the child-vs-older contrast, "
                         "which is the one where the analytic test has the "
                         "sample size to fire.")
    args = ap.parse_args()
    out = Path(args.out)

    if args.only in (None, "rhc"):
        print("[RHC: right heart catheterization, Connors et al. 1996]")
        _get(RHC_URL, out / "rhc" / "rhc.csv")

    # Covertype backs an exploratory large-sample study (realdata_run.r9) that
    # the paper does not report, so it is fetched only on request rather than
    # costing every user 11 MB for data no published number depends on.
    if args.only == "covertype":
        print("[Covertype: forest cover types, UCI -- not used by the paper]")
        _get(COVERTYPE_URL, out / "covertype" / "covertype.zip")

    if args.only in (None, "nhanes"):
        for cycle in args.cycles:
            year = NHANES_CYCLES[cycle]
            print(f"[NHANES cycle {cycle} ({year}-{int(year) + 1}): "
                  f"wrist accelerometry + covariates]")
            base = _NHANES_BASE.format(year=year)
            for stem in NHANES_STEMS:
                name = f"{stem}_{cycle}.xpt"
                _get(f"{base}/{name}", out / "nhanes" / name)

    print(f"\nraw data in {out}")


if __name__ == "__main__":
    main()
