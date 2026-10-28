"""Loader for the Kamei et al. (2013) just-in-time change data, 6 projects (Fu & Menzies 2017 copy).
Local copies in data/raw/jit/, hashed in data/raw/MANIFEST.sha256.
"""
import hashlib, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

JIT = ["bugzilla", "columba", "jdt", "mozilla", "platform", "postgres"]
JIT_FEATS = ["ns", "nm", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "pd", "npt", "exp", "rexp", "sexp"]


def verify_manifest():
    bad = []
    for line in (RAW / "MANIFEST.sha256").read_text().splitlines():
        h, f = line.split(maxsplit=1)
        f = f.lstrip("*")
        if hashlib.sha256((RAW / f).read_bytes()).hexdigest() != h:
            bad.append(f)
    return bad


def clean(df, feats, label="__y", nonneg=None, drop_conflict=True, dedup=True):
    """Removes rows with missing or negative values, conflicting labels (all members) and duplicates.
    `nonneg`: columns that must be >= 0 (default all; signed metrics such as net churn are exempt).
    Returns the cleaned frame (index reset) and a report."""
    nonneg = feats if nonneg is None else nonneg
    rep = {"n_raw": len(df), "pos_raw": int(df[label].sum())}
    n0 = len(df); df = df.dropna(subset=feats + [label]); rep["missing_removed"] = n0 - len(df)
    n0 = len(df); df = df[(df[nonneg] >= 0).all(axis=1)]; rep["negative_removed"] = n0 - len(df)
    n0 = len(df)
    g = df.groupby(feats, dropna=False)[label].transform("nunique")
    rep["conflicting_rows"] = int((g > 1).sum()); rep["duplicate_rows"] = int(df.duplicated(subset=feats + [label]).sum())
    if drop_conflict:
        df = df[g == 1]
    rep["conflicting_removed"] = n0 - len(df)
    n0 = len(df)
    if dedup:
        df = df.drop_duplicates(subset=feats + [label])
    rep["duplicate_removed"] = n0 - len(df)
    df = df.reset_index(drop=True)
    rep["n_final"] = len(df); rep["pos_final"] = int(df[label].sum())
    return df, rep


def load_jit(project, return_report=False):
    """Commits sorted by time. Kamei's file stores LA, LD relative to LT and LT, NPT relative to NF;
    effort = absolute churn LA+LD reconstructed as in the authors' R scripts (core.r);
    where LT = 0 (only new files) LA and LD already hold absolute line counts, so churn = LA + LD there; commits are
    a stream, so identical commits are kept and conflicting ones are only removed from training data (runner)."""
    d = pd.read_csv(RAW / "jit" / f"{project}.csv")
    d["time"] = pd.to_datetime(d["commitdate"], format="%Y/%m/%d %H:%M")
    d["__y"] = d["bug"].astype(int)
    d["fix"] = d["fix"].astype(int)
    d, rep = clean(d, JIT_FEATS, drop_conflict=False, dedup=False)
    lt_abs = d["lt"] * d["nf"]
    d["churn"] = np.where(d["lt"] > 0, (d["la"] + d["ld"]) * lt_abs, d["la"] + d["ld"])
    d = d.sort_values("time", kind="stable").reset_index(drop=True)
    d["month"] = d["time"].dt.to_period("M")
    rep["dataset"] = project
    return (d, rep) if return_report else d

