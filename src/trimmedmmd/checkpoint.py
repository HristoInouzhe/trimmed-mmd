"""Replicate-level checkpointing, so an interrupted run costs minutes not hours.

A long simulation that writes its results only at the end is one power cut
away from being worthless. The unit of work here is a single replicate, and
each one is appended to disk as soon as it lands, keyed by the parameters
that produced it. Re-running then recomputes only what is missing.

Two design choices worth stating:

* **Keys hash the full parameter set.** If you change ``n``, ``delta``, the
  scenario, or anything else that affects the answer, the key changes and
  nothing stale is silently reused. Changing something that does *not*
  affect the answer (worker count, say) leaves the key alone, so the work
  is correctly reused.
* **Seeds identify replicates.** Results are stored per seed rather than
  per position, so a partially finished cell resumes deterministically and
  a completed replicate is never recomputed under a different index.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _jsonable(obj):
    """Convert numpy types to plain Python so results survive a round trip."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def cell_key(experiment: str, params: dict) -> str:
    """Stable identifier for one cell of an experiment."""
    blob = json.dumps(_jsonable(params), sort_keys=True)
    digest = hashlib.sha1(blob.encode()).hexdigest()[:12]
    return f"{experiment}__{digest}"


def _is_fuse(path: Path) -> bool:
    """Is ``path`` on a FUSE filesystem? Best effort, Linux only."""
    try:
        target = str(Path(path).resolve())
        best, fstype = "", ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt, typ = parts[1], parts[2]
                if target == mnt or target.startswith(mnt.rstrip("/") + "/"):
                    if len(mnt) > len(best):
                        best, fstype = mnt, typ
        return fstype.startswith("fuse")
    except OSError:
        return False


def default_root(preferred: Path) -> Path:
    """Where checkpoints should actually live.

    ``TRIMMEDMMD_CHECKPOINTS`` wins if set. Otherwise ``preferred`` is used
    unless it sits on a FUSE mount, in which case we fall back to a local
    cache directory.

    This is not paranoia. Checkpoints written into a *synced* FUSE mount
    (pCloud, Dropbox and friends) have been observed to disappear mid-run:
    the sync daemon reconciles the directory against the server copy and
    removes files it does not know about, with nothing in this code deleting
    anything. Checkpoints are run state, not results -- they are regenerable
    and machine-specific -- so keeping them on local disk costs nothing and
    removes a whole class of silent failure.
    """
    env = os.environ.get("TRIMMEDMMD_CHECKPOINTS")
    if env:
        return Path(env)
    preferred = Path(preferred)
    if _is_fuse(preferred.parent if not preferred.exists() else preferred):
        base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        return Path(base) / "trimmedmmd" / "checkpoints"
    return preferred


class ReplicateStore:
    """Append-only JSONL store, one file per cell."""

    def __init__(self, root: Path, verify: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._verified = not verify

    def _verify_durable(self, key: str) -> None:
        """Check once that what we append actually comes back.

        A store that silently drops writes is worse than no store at all:
        every replicate is recomputed, and a partially surviving file can
        make a resume look complete when it is not. Better to find out on
        the first record than after an hour.
        """
        self._verified = True
        probe = self.root / f".probe-{os.getpid()}"
        try:
            probe.write_text("x")
            ok = probe.exists() and probe.read_text() == "x"
            probe.unlink(missing_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"checkpoint directory {self.root} is not writable: {e}") from e
        if not ok:
            raise RuntimeError(
                f"checkpoint directory {self.root} does not reliably retain "
                f"files. If it is on a synced folder, set TRIMMEDMMD_CHECKPOINTS "
                f"to a local path.")

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.jsonl"

    def load(self, key: str) -> dict:
        """Map seed -> payload for everything already computed."""
        p = self._path(key)
        if not p.exists():
            return {}
        out = {}
        # Read bytes and decode per line: a process killed mid-write leaves a
        # truncated line, and a filesystem hiccup (FUSE mounts are a common
        # culprit) can leave undecodable bytes. Either way the right response
        # is to drop that one record and recompute it, never to fail the whole
        # resume -- losing one replicate is cheap, losing the run is not.
        with p.open("rb") as f:
            for raw in f:
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    out[int(rec["seed"])] = rec["value"]
                except (KeyError, TypeError, ValueError):
                    continue
        return out

    def append(self, key: str, seed: int, value) -> None:
        if not self._verified:
            self._verify_durable(key)
        with self._path(key).open("a") as f:
            f.write(json.dumps({"seed": int(seed),
                                "value": _jsonable(value)}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_meta(self, key: str, params: dict) -> None:
        (self.root / f"{key}.meta.json").write_text(
            json.dumps(_jsonable(params), indent=2, sort_keys=True))


def _pin_blas_threads() -> None:
    """Worker-process initializer: force single-threaded BLAS.

    Process parallelism (one worker per replicate) plus a BLAS that spawns
    one thread per core per worker is catastrophic oversubscription --
    ``workers`` processes each fanning out to every core turns a
    38-way-parallel run into a machine pinned at ``workers`` times the core
    count. This box is shared (theory-stress-suite non-negotiable #5), so
    each worker gets exactly one thread; parallelism comes entirely from the
    process pool, whose size is the envelope's own considered choice.

    Env vars alone are not reliable here: the default ``fork`` start method
    hands each worker a BLAS library that may already be initialized from
    the parent, and some implementations read the thread-count env var only
    once, at that initialization, not per call. ``threadpoolctl`` fixes this
    properly by calling into the already-loaded BLAS library's own C API at
    runtime; the env vars are set too, as a fallback for spawn-based pools
    and for any subprocess a worker itself launches.
    """
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def map_replicates(fn, jobs, seeds, workers, store: ReplicateStore | None = None,
                   key: str | None = None, progress=None):
    """Run ``fn`` over ``jobs``, resuming from and extending a checkpoint.

    Results are written as each replicate completes rather than at the end,
    so killing the process loses at most the replicates currently in flight.
    Returns values in the order of ``jobs``.
    """
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        # Without this guard a wrong seed field silently collapses every
        # replicate onto one key: the store holds a single value and the
        # resume hands that same value back for every replicate, quietly
        # destroying the Monte Carlo while looking like a fast success.
        raise ValueError(
            f"replicate seeds must be unique; got {len(seeds)} jobs but "
            f"{len(set(seeds))} distinct seeds -- check the seed field index")
    done = store.load(key) if store is not None and key is not None else {}
    todo = [(j, s) for j, s in zip(jobs, seeds) if s not in done]

    if todo:
        if workers <= 1:
            for j, s in todo:
                val = fn(j)
                done[s] = _jsonable(val)
                if store is not None:
                    store.append(key, s, val)
                if progress:
                    progress(len(done), len(seeds))
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_pin_blas_threads) as ex:
                futs = {ex.submit(fn, j): s for j, s in todo}
                for fut in as_completed(futs):
                    s = futs[fut]
                    val = fut.result()
                    done[s] = _jsonable(val)
                    if store is not None:
                        store.append(key, s, val)
                    if progress:
                        progress(len(done), len(seeds))
    return [done[s] for s in seeds]
