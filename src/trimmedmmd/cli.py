"""Command line entry point."""
from __future__ import annotations

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="trimmedmmd")
    p.add_argument("command", choices=["run", "resources", "realdata", "figures"])
    p.add_argument("--quick", action="store_true",
                   help="smoke profile: tiny sizes, for checking the pipeline")
    p.add_argument("--profile", default=None,
                   help="override the detected resource profile")
    p.add_argument("--fresh", action="store_true",
                   help="discard checkpoints and recompute from scratch")
    p.add_argument("--reps", type=int, default=None,
                   help="realdata: splits per contrast (published design is 8)")
    # revision v2 (defaults reproduce v1 exactly)
    p.add_argument("--threshold", choices=["paper", "plugin"], default=None,
                   help="test threshold: 'paper' (v1 tau, default) or 'plugin' (v2 tau_new)")
    p.add_argument("--certificate", choices=["bootstrap", "betting", "both", "sequential", "all"],
                   default=None,
                   help="split certificate: 'bootstrap' (v1 Thm 16, default); 'sequential' (v2 "
                        "main: sequential betting, batches (0, 1/4, 1/2, 1)); 'all' (sequential "
                        "reported + single-split betting and bootstrap columns); 'betting' "
                        "(single-split betting); 'both' (single-split betting + bootstrap)")
    p.add_argument("--outdir", default=None,
                   help="write tables/figures/manifests under this directory instead of results/")
    p.add_argument("--only", default=None,
                   help="run: comma-separated experiment ids (e.g. E1,E1b,E6,E7,E8); "
                        "realdata: comma-separated datasets")
    p.add_argument("--workers", type=int, default=None,
                   help="run/realdata: worker processes (default: detected envelope)")
    a = p.parse_args(argv)
    v2 = {k: getattr(a, k) for k in ("threshold", "certificate", "outdir", "only", "workers")
          if getattr(a, k) is not None}
    if a.command == "resources":
        from .resources import detect
        print(detect(profile=a.profile).to_json())
        return 0
    if a.command == "figures":
        from .experiments.figures import all_figures
        all_figures(**({"outdir": a.outdir} if a.outdir is not None else {}))
        return 0
    if a.command == "realdata":
        from .experiments.realdata_run import run_all as realdata_run_all
        realdata_run_all(reps=a.reps, **v2)
        return 0
    from .experiments.run import CHECKPOINTS, run_all
    if a.fresh and CHECKPOINTS.exists():
        import shutil
        shutil.rmtree(CHECKPOINTS)
        print(f"cleared checkpoints at {CHECKPOINTS}")
    run_all(profile=a.profile, quick=a.quick, **v2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
