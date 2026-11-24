"""Shared plumbing for the revision-v2 reruns (plan items V8-V9).

Every experiment script that depends on the test threshold or on the split
certificate takes the same three flags, with defaults that reproduce v1
exactly:

    --threshold {paper,plugin}            default: paper   (v1 tau_{n,m})
    --certificate {bootstrap,betting,both,sequential,all}
                                          default: bootstrap (v1 Theorem 16)
        sequential: the paper's main certificate (Theorem thm:betting with
                    batches e = (0, 1/4, 1/2, 1)); all: sequential reported,
                    plus the single-split betting (``*_betting``) and the
                    bootstrap (``*_bootstrap``) as extra columns (v2 reruns);
        betting / both: single-split betting reported (+ bootstrap column).
    --outdir DIR                          default: the script's v1 location

Conventions (so that downstream generators run unchanged on v2 outputs):

* With ``--outdir D`` every output goes under ``D`` with the v1 layout
  (``D/tables/...``, ``D/figures/...``, ``D/fair/...``); inputs that are
  products of another v2 step are read from ``D`` as well and a missing one
  is an error (no silent fallback to v1 results). Inputs that do not depend
  on the threshold or the certificate (raw data, D-hat paths, solver caches)
  may be read from ``results/``.
* Columns that held the v1 threshold hold the selected threshold; a
  ``threshold`` column names it and, where cheap, ``tau_paper`` keeps the v1
  value. Columns that held the bootstrap certificate hold the selected
  certificate; with ``--certificate both`` the bootstrap is kept in
  ``*_bootstrap`` columns.
* Kernel constants of the plug-in threshold: exact Gaussian Gram matrices
  use :data:`trimmedmmd.threshold.GAUSSIAN` (coupled constants); random
  features cannot (the feature kernel can be negative) and use
  ``KernelConstants.bounded(kappa)`` with kappa = 1 for the cos/sin map and
  the realized kappa for the phased map.
"""

from __future__ import annotations

from pathlib import Path

from ..threshold import GAUSSIAN, KernelConstants, threshold_function

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
V2 = RESULTS / "v2"

THRESHOLD_CHOICES = ("paper", "plugin")
CERTIFICATE_CHOICES = ("bootstrap", "betting", "both", "sequential", "all")


def add_v2_args(parser, threshold=True, certificate=True, outdir=True):
    """Add the standard v2 flags to an argparse parser (defaults = v1)."""
    if threshold:
        parser.add_argument("--threshold", choices=THRESHOLD_CHOICES, default="paper",
                            help="test threshold: paper (v1 tau, default) or plugin (v2 tau_new)")
    if certificate:
        parser.add_argument("--certificate", choices=CERTIFICATE_CHOICES, default="bootstrap",
                            help="split certificate: bootstrap (v1, default); sequential (v2 main: "
                                 "sequential betting, e = (0, 1/4, 1/2, 1)); all (sequential + "
                                 "single-split betting + bootstrap columns); betting (single split); "
                                 "both (single-split betting + bootstrap)")
    if outdir:
        parser.add_argument("--outdir", default=None,
                            help="write outputs under this directory (v1 layout) instead of "
                                 "results/; e.g. results/v2")
    return parser


def out_root(outdir) -> Path:
    """``results/`` (v1) or the given directory, created."""
    p = RESULTS if outdir is None else Path(outdir)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def sub(outdir, name: str) -> Path:
    """``<root>/<name>`` (e.g. tables, figures, fair), created."""
    p = out_root(outdir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def certificate_methods(certificate: str):
    """Methods for :func:`trimmedmmd.splitting.lower_bound_betting`; the FIRST
    one is the reported certificate, the others are extra columns."""
    if certificate == "bootstrap":
        return ("bootstrap",)
    if certificate == "betting":
        return ("betting",)
    if certificate == "both":
        return ("betting", "bootstrap")
    if certificate == "sequential":
        return ("sequential",)
    if certificate == "all":
        return ("sequential", "betting", "bootstrap")
    raise ValueError(certificate)


def main_certificate(certificate: str) -> str:
    """The certificate a table reports under the v1 column names."""
    return certificate_methods(certificate)[0]


def kernel_consts_for(backend: str = "gram", kappa: float = 1.0) -> KernelConstants:
    """Constants of the plug-in threshold for the kernel the statistic uses.

    ``"gram"``: exact Gaussian (or Laplace/IMQ) Gram matrix, coupled constants.
    ``"cossin"``/``"rff"``/``"features"``: a random-feature kernel, bounded
    constants with the given kappa (1 for cos/sin features; pass
    ``kernels.realized_kappa(Z)`` for the phased map).
    """
    if backend == "gram":
        return GAUSSIAN if kappa == 1.0 else KernelConstants.gaussian(kappa)
    if backend in ("cossin", "rff", "features"):
        return KernelConstants.bounded(kappa)
    raise ValueError(backend)


def threshold_for(threshold: str, n: int, m: int, delta: float, backend: str = "gram",
                  kappa: float = 1.0):
    """``alpha -> threshold`` with the kernel constants of ``backend``.

    ``"paper"`` returns v1's tau(n, m, alpha, delta, kappa) whatever the backend.
    """
    if threshold == "paper":
        return threshold_function("paper", n, m, delta, kappa=kappa)
    return threshold_function(threshold, n, m, delta, kappa=kappa,
                              kernel_consts=kernel_consts_for(backend, kappa))
