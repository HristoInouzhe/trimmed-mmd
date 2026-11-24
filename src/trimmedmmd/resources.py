"""Detect the machine's resource envelope and size experiments to it.

Experiment sizes are not hard-coded. They are derived from the CPU count,
available memory, and GPU visibility at run time, and the envelope actually
used is written into every result file, so a run on a laptop and a run on a
48-core shared workstation stay comparable and explainable after the fact.

Worker count and device policy mirror
``theory-stress-suite/scripts/probe_machine.py``: reserve 2 cores + 15%
headroom, and only offer the GPU when it looks genuinely free (>= 4 GB),
because this box's card is routinely shared with other tenants.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class Envelope:
    cpu_count: int
    ram_gb: float | None
    ram_tier: str
    max_workers: int
    platform: str
    profile: str          # "smoke" | "minimal" | "modest" | "comfortable" | "generous" | "huge"
    n_reps: int           # Monte Carlo repetitions for rep-heavy experiments
    n_mc_null: int        # repetitions for estimating null quantiles
    max_n_gram: int       # largest sample size solved with a dense Gram matrix
    max_n_rff: int        # largest sample size solved with random features
    n_features: int       # default number of random features
    device: str            # "cpu" | "cuda" | "auto" -- recommended, not forced
    gpu_free_gb: float | None
    gpu_tenants: int
    reserved_cores: int
    headroom_frac: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _ram_gb() -> tuple[float | None, float | None]:
    """Returns (total_gb, available_gb)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / (1024 ** 3), 2), round(vm.available / (1024 ** 3), 2)
    except Exception:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0])  # kB
        total = info["MemTotal"] / (1024 ** 2)
        avail = info.get("MemAvailable", info.get("MemFree", 0)) / (1024 ** 2)
        return round(total, 2), round(avail, 2)
    except Exception:
        pass
    if shutil.which("sysctl"):
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            total = int(out.strip()) / (1024 ** 3)
            return round(total, 2), None
        except Exception:
            pass
    return None, None


def _load_busy_cores(ncpu: int) -> float:
    try:
        one, _five, _fifteen = os.getloadavg()
        return min(one, float(ncpu))
    except Exception:
        return 0.0


def _gpu_state() -> tuple[float | None, int]:
    """(free_gb on the best GPU, number of tenant processes) or (None, 0)."""
    if not shutil.which("nvidia-smi"):
        return None, 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        frees = [float(x) / 1024.0 for x in out.splitlines() if x.strip()]
        best = max(frees) if frees else None
    except Exception:
        best = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        tenants = len([l for l in out.splitlines() if l.strip()])
    except Exception:
        tenants = 0
    return best, tenants


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def detect(profile: str | None = None, reserve_cores: int = 2,
           headroom_frac: float = 0.15) -> Envelope:
    cpu = _cpu_count()
    ram, ram_avail = _ram_gb()
    if ram is None:
        tier = "unknown"
    elif ram < 4:
        tier = "minimal"
    elif ram < 16:
        tier = "modest"
    elif ram < 64:
        tier = "comfortable"
    elif ram < 96:
        tier = "generous"
    else:
        tier = "huge"

    busy = _load_busy_cores(cpu)
    free_cores = max(1.0, cpu - busy - reserve_cores)
    workers = max(1, int(free_cores * (1.0 - headroom_frac)))
    prof = profile or tier

    gpu_free, gpu_tenants = _gpu_state()
    device = "cpu"
    if gpu_free is not None:
        device = "auto" if gpu_free >= 4.0 else "cpu"

    # A dense Gram for sample size n costs roughly 3 * 8 * n^2 bytes, and each
    # worker holds its own. Budget a quarter of *available* RAM across all
    # workers -- available, not total, so this degrades correctly when other
    # processes already hold memory.
    ram_for_budget = ram_avail if ram_avail is not None else ram
    if ram_for_budget is None:
        max_n_gram = 800
    else:
        budget_bytes = 0.25 * ram_for_budget * (1024 ** 3) / max(workers, 1)
        max_n_gram = int((budget_bytes / (3 * 8)) ** 0.5)
        max_n_gram = max(300, min(max_n_gram, 20000))

    sizes = {
        "smoke":       dict(n_reps=8,    n_mc_null=40,   max_n_rff=2000,   n_features=128),
        "minimal":     dict(n_reps=40,   n_mc_null=150,  max_n_rff=4000,   n_features=256),
        "modest":      dict(n_reps=200,  n_mc_null=400,  max_n_rff=20000,  n_features=512),
        "comfortable": dict(n_reps=400,  n_mc_null=800,  max_n_rff=50000,  n_features=1024),
        "generous":    dict(n_reps=800,  n_mc_null=2000, max_n_rff=100000, n_features=2048),
        # "huge": a 48-core/125GB-class shared server. Reps scale with actual
        # worker count (not a fixed constant) so the tier stays meaningful
        # across different big machines rather than hard-coding one box's
        # numbers; max_n_rff/n_features go further because RFF feature
        # matrices, not dense Grams, are what actually gets big here.
        "huge":        dict(n_reps=max(1600, 40 * workers),
                            n_mc_null=max(4000, 100 * workers),
                            max_n_rff=200000, n_features=4096),
        "unknown":     dict(n_reps=100,  n_mc_null=200,  max_n_rff=4000,   n_features=256),
    }[prof if prof in ("smoke", "minimal", "modest", "comfortable",
                       "generous", "huge", "unknown") else "modest"]

    if prof == "smoke":
        max_n_gram = min(max_n_gram, 400)

    return Envelope(
        cpu_count=cpu,
        ram_gb=ram,
        ram_tier=tier,
        max_workers=workers,
        platform=platform.platform(),
        profile=prof,
        max_n_gram=max_n_gram,
        device=device,
        gpu_free_gb=gpu_free,
        gpu_tenants=gpu_tenants,
        reserved_cores=reserve_cores,
        headroom_frac=headroom_frac,
        **sizes,
    )


if __name__ == "__main__":
    print(detect().to_json())
