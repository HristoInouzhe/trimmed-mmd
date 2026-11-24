"""Loaders for the two real datasets used in Section~\\ref{sec:applications}.

Both are public and fetched by ``scripts/fetch_data.py``; nothing here
vendors data into the repository. Each loader returns plain arrays plus a
``meta`` dict recording provenance and every filtering decision, because on
real data the preprocessing is part of the result and has to travel with it.

The two applications exercise different parts of the method:

* :func:`load_rhc` gives a **multivariate** two-sample problem (33 baseline
  covariates, 2184 vs 3551 patients) where the estimand -- how much of each
  arm has to be discarded before the two look alike -- is the quantity the
  causal-inference literature currently sets by a rule of thumb.
* :func:`load_nhanes_activity` gives a **functional** one (a 24-point daily
  activity profile per participant, ~6900 participants), which is where the
  dimension-free guarantee is supposed to earn its keep.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Where the raw downloads live. ``TRIMMEDMMD_DATA`` overrides."""
    env = os.environ.get("TRIMMEDMMD_DATA")
    return Path(env) if env else _REPO / "data_raw"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python3 scripts/fetch_data.py` first "
            f"(or set TRIMMEDMMD_DATA to a directory that already has it).")
    return path


# --------------------------------------------------------------------------
# RHC: right heart catheterization (Connors et al., JAMA 1996)
# --------------------------------------------------------------------------

# Baseline physiology and demographics measured before the treatment
# decision. Deliberately excluded: anything dated or outcome-derived
# (``t3d30``, ``death``, ``dth30``, discharge/death dates), ``adld3p`` and
# ``urin1`` (75% and 53% missing), and the treatment indicator itself.
RHC_CONTINUOUS = [
    "age", "edu", "surv2md1", "das2d3pc", "aps1", "scoma1", "meanbp1",
    "wblc1", "hrt1", "resp1", "temp1", "pafi1", "alb1", "hema1", "bili1",
    "crea1", "sod1", "pot1", "paco21", "ph1", "wtkilo1",
]
RHC_BINARY = [
    "cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx",
    "liverhx", "gibledhx", "malighx", "immunhx", "transhx", "amihx",
]
# AoS referee runs (2026-11-22): the twelve further binary baseline covariates of rhc.csv, stored as strings and
# left out of every earlier run: sex, the do-not-resuscitate status on day 1 and the ten admission-diagnosis
# indicators. Encoded 0/1 by RHC_STRING_CODES (the "1" level of each); no value is missing. The multi-category
# variables cat1, cat2, ca, ninsclas, race and income stay out (they would need a one-hot coding).
RHC_STRING_BINARY = [
    "sex", "dnr1", "resp", "card", "neuro", "gastr", "renal", "meta", "hema",
    "seps", "trauma", "ortho",
]
RHC_STRING_CODES = {"sex": ("Female", "Male")} | {
    c: ("Yes", "No") for c in RHC_STRING_BINARY if c != "sex"}   # column -> (level coded 1, level coded 0)
RHC_COVARIATE_SETS = ("paper", "full")
RHC_COVARIATES_ENV = "TRIMMEDMMD_RHC_COVARIATES"


def rhc_covariate_names(covariates: str | None = None):
    """Covariate names of a set: "paper" (the 33 numeric and 0/1 columns of every earlier run, the default) or
    "full" (those 33 plus the 12 string-coded binaries, 45). ``None`` reads the environment variable
    TRIMMEDMMD_RHC_COVARIATES (unset: "paper"), so a script can switch every loader of a process at once."""
    covariates = covariates or os.environ.get(RHC_COVARIATES_ENV, "paper")
    if covariates == "paper":
        return RHC_CONTINUOUS + RHC_BINARY
    if covariates == "full":
        return RHC_CONTINUOUS + RHC_BINARY + RHC_STRING_BINARY
    raise ValueError(f"unknown RHC covariate set {covariates!r}; choose from {RHC_COVARIATE_SETS}")


def rhc_covariate_matrix(df, names):
    """The covariates ``names`` of the RHC data frame as floats; the string-coded binaries by RHC_STRING_CODES
    (an unexpected level raises)."""
    cols = []
    for c in names:
        if c in RHC_STRING_CODES:
            one, zero = RHC_STRING_CODES[c]
            s = df[c].to_numpy()
            ok = (s == one) | (s == zero)
            if not ok.all():
                raise ValueError(f"rhc.csv column {c}: values other than {one!r}/{zero!r}: "
                                 f"{sorted(set(map(str, s[~ok])))}")
            cols.append((s == one).astype(float))
        else:
            cols.append(df[c].to_numpy(dtype=float))
    return np.column_stack(cols)


def load_rhc(standardize: bool = True, covariates: str | None = None):
    """Baseline covariates of catheterized vs. non-catheterized patients.

    Returns ``(X, Y, names, meta)`` with ``X`` the RHC arm and ``Y`` the
    control arm. Standardizing uses the *pooled* mean and standard
    deviation, so it is a single invertible map applied to both samples and
    therefore cannot change the total-variation distance between them --
    it only puts the coordinates on a comparable scale before the median
    heuristic picks a bandwidth.

    ``covariates`` (AoS referee runs; default "paper", every earlier result unchanged): "full" adds the 12
    string-coded binaries (:func:`rhc_covariate_names`). They have no missing values, so the complete-case rows
    are the same 5,735 under both sets.
    """
    import pandas as pd

    path = _require(data_root() / "rhc" / "rhc.csv")
    df = pd.read_csv(path, low_memory=False)

    cov_set = covariates or os.environ.get(RHC_COVARIATES_ENV, "paper")
    names = rhc_covariate_names(cov_set)
    Z = rhc_covariate_matrix(df, names)
    treated = (df["swang1"].to_numpy() == "RHC")

    keep = np.isfinite(Z).all(axis=1)
    Z, treated = Z[keep], treated[keep]

    meta = {
        "source": "Connors et al. (1996), via the Vanderbilt Biostatistics "
                  "dataset repository (hbiostat.org/data/repo/rhc.csv)",
        "n_rows_raw": int(len(df)),
        "n_rows_complete": int(keep.sum()),
        "d": len(names),
        "n_treated": int(treated.sum()),
        "n_control": int((~treated).sum()),
        "excluded": "outcome and date fields; adld3p (75% missing); "
                    "urin1 (53% missing)",
        "caveat": "wtkilo1 records 515 zero weights, the dataset's known "
                  "missing-as-zero coding; kept as in the published analyses",
        "standardized": bool(standardize),
    }
    if cov_set != "paper":
        meta["covariates"] = cov_set
        meta["string_codes"] = {c: f"{RHC_STRING_CODES[c][0]} = 1, {RHC_STRING_CODES[c][1]} = 0"
                                for c in names if c in RHC_STRING_CODES}

    if standardize:
        mu, sd = Z.mean(axis=0), Z.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd

    return Z[treated], Z[~treated], names, meta


# --------------------------------------------------------------------------
# Covertype: forest cover types (UCI)
# --------------------------------------------------------------------------

# The ten quantitative attributes. The remaining 44 columns are one-hot
# wilderness-area and soil-type indicators, excluded because mixing binary
# indicators into a Gaussian kernel on standardized coordinates makes the
# bandwidth answer to the indicators rather than to the terrain.
COVERTYPE_FEATURES = [
    "elevation", "aspect", "slope",
    "horiz_dist_hydrology", "vert_dist_hydrology", "horiz_dist_roadways",
    "hillshade_9am", "hillshade_noon", "hillshade_3pm",
    "horiz_dist_fire_points",
]
COVERTYPE_CLASSES = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine"}


def load_covertype(classes=(1, 2), standardize: bool = True, cache: bool = True):
    """Quantitative terrain attributes for two forest cover types.

    Returns ``(X, Y, names, meta)``. Classes 1 and 2 (Spruce/Fir and Lodgepole
    Pine) have 211,840 and 283,301 sites, which is what makes this the one
    problem here large enough for the analytic estimator to be sharp rather
    than merely valid. Standardization is pooled, hence a single invertible
    map applied to both samples, and cannot change the total variation.
    """
    import zipfile

    root = data_root() / "covertype"
    npz = root / "covertype.npz"
    if cache and npz.exists():
        z = np.load(npz)
        data, y = z["data"], z["y"]
    else:
        _require(root / "covertype.zip")
        with zipfile.ZipFile(root / "covertype.zip") as zf:
            name = [n for n in zf.namelist() if n.endswith(".gz")][0]
            import gzip
            import io
            with zf.open(name) as fh:
                raw = np.loadtxt(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(fh.read()))),
                                 delimiter=",")
        data, y = raw[:, :10], raw[:, -1].astype(int)
        if cache:
            np.savez_compressed(npz, data=data, y=y)

    a, b = classes
    X, Y = data[y == a], data[y == b]
    meta = {
        "source": "UCI Machine Learning Repository, Covertype (Blackard & Dean), "
                  "Roosevelt National Forest, Colorado",
        "classes": f"{a} = {COVERTYPE_CLASSES.get(a, a)} vs "
                   f"{b} = {COVERTYPE_CLASSES.get(b, b)}",
        "n": int(len(X)), "m": int(len(Y)), "d": len(COVERTYPE_FEATURES),
        "excluded": "44 binary wilderness-area and soil-type indicators",
        "standardized": bool(standardize),
    }
    if standardize:
        Z = np.vstack([X, Y])
        mu, sd = Z.mean(0), Z.std(0)
        sd[sd == 0] = 1.0
        X, Y = (X - mu) / sd, (Y - mu) / sd
    return X, Y, list(COVERTYPE_FEATURES), meta


# --------------------------------------------------------------------------
# NHANES cycle G: wrist accelerometry
# --------------------------------------------------------------------------

_SAMPLES_PER_HOUR = 80 * 3600   # PAXSSNHP counts 80 Hz samples since start

# A day is usable if the device was worn (awake or asleep) for at least this
# many of its 1440 minutes. 1200 = 20 hours; below that the daily mean is
# dominated by which hours happened to be observed rather than by behaviour.
_MIN_WEAR_MIN = 1200
_MIN_VALID_DAYS = 4


def _nhanes_cache(cycles) -> Path:
    return data_root() / "nhanes" / f"activity_curves_{''.join(cycles)}.npz"


def _build_cycle(root: Path, cycle: str):
    """Reduce one NHANES cycle to (profiles DataFrame, covariates, counts).

    The clock alignment is the delicate step. ``PAXHR`` rows carry
    ``PAXSSNHP``, an elapsed count of 80 Hz samples since the device started;
    the start clock time is ``PAXFTIME`` in ``PAXHD``. A record opens with a
    partial hour running to the next clock hour, and every later delta is
    exactly 288000 samples = 1 hour, which is what makes
    ``floor(start + PAXSSNHP / 288000) mod 24`` the row's clock hour. That
    identity is checked against the data rather than assumed, per cycle.
    """
    import pandas as pd

    hr = pd.read_sas(_require(root / f"PAXHR_{cycle}.xpt"))
    hd = pd.read_sas(_require(root / f"PAXHD_{cycle}.xpt"))
    demo = pd.read_sas(_require(root / f"DEMO_{cycle}.xpt"))
    bmx = pd.read_sas(_require(root / f"BMX_{cycle}.xpt"))

    n_raw_rows, n_raw_people = len(hr), hr.SEQN.nunique()

    # SAS writes missing numerics as a denormal rather than a true zero.
    for c in ("PAXSSNHP", "PAXMTSH", "PAXWWMH", "PAXSWMH", "PAXQFH"):
        hr[c] = np.where(np.abs(hr[c]) < 1e-50, 0.0, hr[c])

    # Only records the survey itself marks as valid.
    ok = hd.loc[hd.PAXSTS == 1, ["SEQN", "PAXFTIME"]].copy()
    ftime = ok.PAXFTIME.str.decode("ascii")
    ok["start_h"] = (ftime.str.slice(0, 2).astype(float)
                     + ftime.str.slice(3, 5).astype(float) / 60.0
                     + ftime.str.slice(6, 8).astype(float) / 3600.0)
    hr = hr.merge(ok[["SEQN", "start_h"]], on="SEQN", how="inner")

    seqn = hr.SEQN.to_numpy()
    step = np.diff(hr.PAXSSNHP.to_numpy())
    same = seqn[1:] == seqn[:-1]
    opening = np.r_[True, seqn[1:] != seqn[:-1]][:-1]   # step leaves a first row
    interior = same & ~opening
    if not np.all(step[interior] == _SAMPLES_PER_HOUR):
        bad = int((step[interior] != _SAMPLES_PER_HOUR).sum())
        raise AssertionError(
            f"cycle {cycle}: PAXSSNHP is not a 1-hour grid in {bad} "
            "within-participant steps; the clock-hour mapping would be wrong")
    if not np.all(step[same & opening] <= _SAMPLES_PER_HOUR):
        raise AssertionError(f"cycle {cycle}: an opening partial hour exceeds one hour")
    hr["hour"] = np.floor(hr.start_h
                          + hr.PAXSSNHP / _SAMPLES_PER_HOUR).astype(int) % 24

    # Drop flagged hours, then keep days with enough wear time.
    hr = hr[hr.PAXQFH == 0]
    hr["wear"] = hr.PAXWWMH + hr.PAXSWMH
    day_wear = hr.groupby(["SEQN", "PAXDAYH"], observed=True)["wear"].sum()
    good_days = day_wear[day_wear >= _MIN_WEAR_MIN].index
    hr = hr.set_index(["SEQN", "PAXDAYH"]).loc[
        hr.set_index(["SEQN", "PAXDAYH"]).index.isin(good_days)].reset_index()

    n_days = hr.groupby("SEQN")["PAXDAYH"].nunique()
    keep_ids = n_days[n_days >= _MIN_VALID_DAYS].index
    hr = hr[hr.SEQN.isin(keep_ids)]

    # Average MIMS per clock hour across usable days, on the log scale --
    # hourly activity is heavily right-skewed and a handful of exercise
    # hours would otherwise set the kernel bandwidth for everybody.
    hr["y"] = np.log1p(hr.PAXMTSH)
    prof = (hr.groupby(["SEQN", "hour"], observed=True)["y"].mean()
              .unstack("hour").reindex(columns=range(24)))
    prof = prof.dropna()   # participants missing a whole clock hour

    d = demo.set_index("SEQN")
    b = bmx.set_index("SEQN")
    ids = prof.index.to_numpy()
    cov = pd.DataFrame({
        "SEQN": ids,
        "age": d.RIDAGEYR.reindex(ids).to_numpy(),
        "sex": d.RIAGENDR.reindex(ids).to_numpy(),
        "bmi": b.BMXBMI.reindex(ids).to_numpy(),
        "pir": d.INDFMPIR.reindex(ids).to_numpy(),
        "cycle": cycle,
    })
    return prof, cov, n_raw_rows, n_raw_people


def load_nhanes_activity(cycles=("G", "H"), rebuild: bool = False):
    """Clock-aligned 24-hour activity profiles, one per participant.

    Returns ``(curves, cov, meta)``: ``curves`` is ``(n, 24)`` of
    ``log(1 + MIMS)`` averaged over usable days, ``cov`` is a DataFrame with
    ``SEQN``, ``age``, ``sex``, ``bmi``, ``pir``, ``cycle``.

    Cycles G (2011--12) and H (2013--14) are structurally identical and are
    pooled by default. Pooling matters: the contrast that gives the analytic
    test enough sample size to fire only clears the threshold with both.
    ``SEQN`` is unique across cycles and that is checked, not assumed.

    The result is cached as an ``.npz`` beside the raw files, since parsing
    2.6 million rows on every notebook run is pure waste.
    """
    import pandas as pd

    cycles = tuple(cycles)
    cache = _nhanes_cache(cycles)
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        cov = pd.DataFrame({k: z[f"cov_{k}"] for k in
                            ("SEQN", "age", "sex", "bmi", "pir", "cycle")})
        return z["curves"], cov, dict(z["meta"].item())

    root = data_root() / "nhanes"
    profs, covs, rows, people = [], [], 0, 0
    for cycle in cycles:
        pr, cv, nr, np_ = _build_cycle(root, cycle)
        profs.append(pr)
        covs.append(cv)
        rows += nr
        people += np_

    cov = pd.concat(covs, ignore_index=True)
    if cov.SEQN.duplicated().any():
        # NHANES assigns disjoint SEQN ranges per cycle; if that ever stopped
        # being true, pooling would silently double-count participants.
        raise AssertionError("SEQN collides across cycles; pooling is unsafe")
    curves = np.vstack([p.to_numpy(dtype=float) for p in profs])

    meta = {
        "source": "NHANES physical activity monitor, CDC/NCHS public files "
                  + ", ".join(f"PAXHR_{c}/PAXHD_{c}/DEMO_{c}/BMX_{c}"
                              for c in cycles),
        "cycles": ", ".join(f"{c} ({'2011-2012' if c == 'G' else '2013-2014'})"
                            for c in cycles),
        "measure": "PAXMTSH (MIMS triaxial activity summed over the hour), "
                   "log(1+x), averaged over usable days per clock hour",
        "n_rows_raw": int(rows),
        "n_participants_raw": int(people),
        "n_participants_kept": int(len(curves)),
        "filters": f"PAXSTS==1; PAXQFH==0; >= {_MIN_WEAR_MIN} wear-minutes "
                   f"per day; >= {_MIN_VALID_DAYS} usable days; all 24 clock "
                   f"hours present",
        "dim": 24,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, curves=curves, meta=np.array(meta, dtype=object),
                        **{f"cov_{c}": cov[c].to_numpy() for c in cov.columns})
    return curves, cov, meta


def nhanes_groups(curves, cov, by: str = "age"):
    """Split the activity curves into two contrasting groups.

    ``by="child"`` contrasts 6-11 year olds with 60-79 year olds. This is
    the largest effect available here and, pooled over both cycles, the only
    contrast whose sample size clears the analytic threshold -- which is why
    it exists rather than being an afterthought.

    ``by="age"`` contrasts 20-39 with 60-79 year olds; ``by="bmi"``
    contrasts BMI < 25 with BMI >= 30 among 20-59 year olds, holding the
    age range roughly fixed so the contrast is not just age in disguise.
    Returns ``(X, Y, label_x, label_y)``.
    """
    age, bmi = cov.age.to_numpy(), cov.bmi.to_numpy()
    if by == "child":
        a = (age >= 6) & (age < 12)
        b = (age >= 60) & (age < 80)
        return curves[a], curves[b], "age 6-11", "age 60-79"
    if by == "age":
        a = (age >= 20) & (age < 40)
        b = (age >= 60) & (age < 80)
        return curves[a], curves[b], "age 20-39", "age 60-79"
    if by == "bmi":
        mid = (age >= 20) & (age < 60) & np.isfinite(bmi)
        a = mid & (bmi < 25)
        b = mid & (bmi >= 30)
        return curves[a], curves[b], "BMI < 25 (age 20-59)", "BMI >= 30 (age 20-59)"
    raise ValueError(f"unknown split: {by!r}")
