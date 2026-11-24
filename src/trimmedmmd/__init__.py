"""Trimmed kernel mean embeddings: finite-sample inference for the total-variation null.

Companion package to `paper.tex`. Implements the empirical trimmed MMD as a
convex QP, the analytic threshold and test, the threshold-crossing estimator
of the contamination level, the recovery of the optimal trimming itself in
both the kernel and the transport geometry, and the experiments that
stress-test them.
"""

from .discrepancy import plain_mmd, trimmed_mmd_gram, trimmed_mmd_path
from .estimate import alpha_hat_from_path, bisect_alpha_hat
from .resources import detect
from .solver import solve_features, solve_gram, solve_path
from .threshold import radius, reject, tau
# revision v2: the plug-in threshold (Theorem P / Corollary C), Theorem A
from .threshold import (KernelConstants, kernel_constants, reject_plugin, sample_size_for,
                        tau_A, tau_new, tau_plugin, tau_twosided, threshold_function)
# revision v2: the finite-sample betting certificate on the witness ramp
from .splitting import (betting_certificate, lower_bound_betting, lower_bound_sequential,
                        ramp_cut_points, sequential_certificate,
                        sequential_betting_certificate, sequential_horizon,
                        split_certificate)
from .trimmings import (Trimming, optimal_trimming, optimal_trimming_mmd,
                        optimal_trimming_wasserstein, removal_order, trim_at,
                        trimming_path, witness, witness_at)
# unequal trimming fractions: the frontier alpha_y*(alpha_x) and the hockey-stick divergence E_zeta
from .unequal import (PairCertificate, frontier_lower_bound, hockey_stick_lower_bound,
                      pair_certificate)

__all__ = [
    "plain_mmd", "trimmed_mmd_gram", "trimmed_mmd_path",
    "alpha_hat_from_path", "bisect_alpha_hat",
    "detect", "solve_features", "solve_gram", "solve_path",
    "radius", "reject", "tau",
    "KernelConstants", "kernel_constants", "reject_plugin", "sample_size_for", "tau_A",
    "tau_new", "tau_plugin", "tau_twosided", "threshold_function",
    "betting_certificate", "lower_bound_betting", "lower_bound_sequential", "ramp_cut_points",
    "sequential_certificate",
    "sequential_betting_certificate", "sequential_horizon", "split_certificate",
    "Trimming", "optimal_trimming", "optimal_trimming_mmd",
    "optimal_trimming_wasserstein", "trimming_path", "removal_order",
    "trim_at", "witness", "witness_at",
    "PairCertificate", "frontier_lower_bound", "hockey_stick_lower_bound", "pair_certificate",
]
__version__ = "0.1.0"
