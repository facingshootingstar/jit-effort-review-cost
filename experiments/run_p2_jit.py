"""Paper 2: effort-aware JIT defect prediction vs cost on the Kamei corpus, time-wise windows.
  python experiments/run_p2_jit.py perf    -> results/raw/p2jit_perf.csv   (parallel over windows)
  python experiments/run_p2_jit.py perf12  -> results/raw/p2jit_perf12.csv (12-month training window, exploratory)
  python experiments/run_p2_jit.py hybrid  -> results/raw/p2jit_hybrid.csv (review-cost model churn + c0, deviation 21)
  python experiments/run_p2_jit.py timing  -> results/raw/p2jit_cost.csv   (run ALONE, single-threaded, sequential)
Revision round 1 (deviation 31; all exploratory):
  info    -> p2jit_info.json   windows before/after the eligibility filter, zero-churn commits per project
  gap     -> p2jit_gap.csv     temporal buffer between training and test of 0, 6 and 12 months (2 is the main run)
  mask    -> p2jit_mask.csv    labels of the last 1, 2 or 3 training months hidden (defects not yet found)
  trees   -> p2jit_trees.csv   RF / LGBM / XGB with 50 and 100 trees or rounds (300 is the main run)
  seeds   -> p2jit_seeds.csv   stochastic learners refitted with two further seeds"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, pathlib, time, warnings, pickle, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from core.corpora import load_jit, JIT, JIT_FEATS
from core.folds import time_windows
from core.models import make, Prep, SEED, ALL
from core.metrics import rank_metrics, effort_metrics

warnings.filterwarnings("ignore")
RAW = ROOT / "results" / "raw"
TREES = ["RF", "LGBM", "XGB"]
SUB = ["LR", "RF", "XGB"]  # learners of the sensitivity runs


def windows(d, train=6, gap=2):
    y = d["__y"].to_numpy()
    return [w for w in time_windows(d["month"].astype(str).to_numpy(), train=train, gap=gap)
            if y[w[0]].sum() >= 20 and len(w[1]) >= 30 and y[w[1]].sum() >= 5]


def ifa(y, score, effort):
    """Initial false alarms: clean commits ranked before the first defect-inducing one (same tie rule as Popt:
    among equal scores the larger commit comes first)."""
    order = np.lexsort((-(effort + 1.0), -score))
    hits = np.flatnonzero(y[order] == 1)
    return int(hits[0]) if len(hits) else np.nan


def inspected(y, score, effort, budget=0.2):
    """Commits a reviewer opens within `budget` of total churn (PCI@20%, Huang et al.) and their precision,
    using the same ordering and +1 effort shift as core.metrics.effort_metrics."""
    e = effort + 1.0; order = np.lexsort((-e, -score))
    within = np.cumsum(e[order]) / e.sum() <= budget
    n = int(within.sum())
    return {"pci20": n / len(y), "n_open": n, "prec20": y[order][:n].mean() if n else np.nan}


def evaluate(y, score, effort):
    r = effort_metrics(y, score, effort)
    out = {"popt": r["popt"], "recall20": r["recall20"], "ifa": ifa(y, score, effort), **inspected(y, score, effort)}
    nz = effort > 0  # zero-churn sensitivity: the same ranking scored on the commits that change at least one line
    if nz.sum() >= 30 and y[nz].sum() >= 5:
        r = effort_metrics(y[nz], score[nz], effort[nz])
        out.update(popt_nz=r["popt"], recall20_nz=r["recall20"])
    return out


_CACHE = {}


def arrays(proj):
    """Load inside the worker: passing the large arrays through loky broke the pool on Windows."""
    if proj not in _CACHE:
        d = load_jit(proj)
        _CACHE[proj] = (d[JIT_FEATS].to_numpy(float), d["__y"].to_numpy(), d["churn"].to_numpy(),
                        d["month"].astype(str).to_numpy())
    return _CACHE[proj]


def train_rows(X, y, tr, mask_months=0, month=None):
    """Training labels as seen at the cut-off: optionally hide the defects of the last `mask_months` training months,
    then drop the commits whose metric vector occurs with both labels in this training block (test data untouched)."""
    yt = y[tr].copy()
    if mask_months:
        last = sorted(set(month[tr].tolist()))[-mask_months:]
        yt[np.isin(month[tr], last)] = 0
    key = pd.DataFrame(X[tr]).astype(str).agg("|".join, axis=1)
    ok = (pd.Series(yt).groupby(key.values).transform("nunique") == 1).to_numpy()
    return tr[ok], yt[ok]


def fit_predict(clf, X, tr, ytr, te, seed=SEED, n_est=None):
    prep = Prep(clf); Ztr = prep.fit_transform(X[tr]); Zte = prep.transform(X[te])
    m = make(clf, seed=seed, n_jobs=1)
    if n_est:
        m.set_params(n_estimators=n_est)
    return m.fit(Ztr, ytr).predict_proba(Zte)[:, 1]


def unsupervised(X, y, eff, tr, ytr, te):
    """ManualUp (smallest first), ManualDown (largest first) and CBS+-style: a class-balanced LR classifies, the commits
    predicted defective come first, each group in ascending churn (Huang et al. 2019)."""
    prep = Prep("LR"); Ztr = prep.fit_transform(X[tr]); Zte = prep.transform(X[te])
    s = LinearRegression().fit(Ztr, ytr / (eff[tr] + 1.0)).predict(Zte)
    p = make("LR", balanced=True).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
    return {"EALR": s, "ManualUp": -eff[te], "ManualDown": eff[te].astype(float),
            "CBS+": (p >= 0.5) + 1.0 / (eff[te] + 2.0)}


def perf_window(proj, tr, te, wid, k, clfs=ALL, mask=0, seed=SEED, n_est=None, extra=None, unsup=True):
    X, y, eff, month = arrays(proj)
    tr2, ytr = train_rows(X, y, tr, mask, month)
    rows = []
    base = {"project": proj, "window": wid, "w": k, "n_train": len(tr2), "n_test": len(te),
            "pos_train": int(ytr.sum()), "pos_test": int(y[te].sum()), "train_dropped": len(tr) - len(tr2), **(extra or {})}
    if ytr.sum() < 5:
        return rows
    for clf in clfs:
        p = fit_predict(clf, X, tr2, ytr, te, seed, n_est)
        auc = rank_metrics(y[te], p)["auc"]
        for rk, sc in (("prob", p), ("density", p / (eff[te] + 1.0))):
            rows.append({**base, "method": f"{clf}-{rk}", "clf": clf, "rank": rk, "auc": auc,
                         **evaluate(y[te], sc, eff[te])})
    if unsup:
        for m, s in unsupervised(X, y, eff, tr2, ytr, te).items():
            rows.append({**base, "method": m, "clf": m, "rank": "density",
                         "auc": rank_metrics(y[te], s)["auc"], **evaluate(y[te], s, eff[te])})
    return rows


def run(jobs, n_jobs):
    """n_jobs=1 runs in-process: loky's IPC repeatedly failed on this Windows host (WinError 6)."""
    if n_jobs == 1:
        return [f(*a, **kw) for f, a, kw in jobs]
    return Parallel(n_jobs=n_jobs, verbose=2)(jobs)


def save(res, out):
    pd.DataFrame([r for rows in res for r in rows]).to_csv(RAW / out, index=False)


def jobs_for(projs=JIT, train=6, gap=2, **kw):
    jobs = []
    for proj in projs:
        d = load_jit(proj)
        for k, (tr, te, wid) in enumerate(windows(d, train, gap)):
            jobs.append(delayed(perf_window)(proj, tr, te, wid, k, **kw))
    return jobs


def perf(train=6, out="p2jit_perf.csv", projs=JIT, n_jobs=10):
    save(run(jobs_for(projs, train), n_jobs), out)


def sensitivity(mode, n_jobs=10, projs=JIT, out=None):
    if mode == "gap":
        jobs = [j for g in (0, 6, 12) for j in jobs_for(projs, gap=g, clfs=SUB, extra={"gap": g})]
    elif mode == "mask":
        jobs = [j for m in (1, 2, 3) for j in jobs_for(projs, clfs=SUB, mask=m, extra={"mask": m})]
    elif mode == "trees":
        jobs = [j for n in (50, 100) for j in jobs_for(projs, clfs=TREES, n_est=n, unsup=False, extra={"n_est": n})]
    else:
        jobs = [j for s in (1, 2) for j in jobs_for(projs, clfs=["RF", "LGBM", "XGB", "MLP"], seed=SEED + s, unsup=False,
                                                    extra={"seed_offset": s})]
    save(run(jobs, n_jobs), out or f"p2jit_{mode}.csv")


def info():
    out = {}
    for proj in JIT:
        d, rep = load_jit(proj, return_report=True)
        y = d["__y"].to_numpy(); allw = time_windows(d["month"].astype(str).to_numpy())
        z = d["churn"] == 0
        out[proj] = {**{k: v for k, v in rep.items() if k != "dataset"}, "n_commits": len(d),
                     "windows_all": len(allw), "windows_kept": len(windows(d)),
                     "excluded_train_pos": sum(y[w[0]].sum() < 20 for w in allw),
                     "excluded_test": sum(y[w[0]].sum() >= 20 and (len(w[1]) < 30 or y[w[1]].sum() < 5) for w in allw),
                     "zero_churn_share": float(z.mean()), "zero_churn_defect_rate": float(y[z].mean()) if z.any() else None,
                     "nonzero_defect_rate": float(y[~z].mean()), "lt0_share": float((d["lt"] == 0).mean())}
    (RAW / "p2jit_info.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


C0 = [0, 10, 30, 100, np.inf]  # per-commit review overhead in LOC-equivalents; inf = count commits only


def hybrid_window(proj, tr, te, wid, k):
    """Exploratory (deviation 21): review cost = churn + c0 per commit. Each ranking is evaluated under
    every cost model; 'costaware' ranks by p / (churn + c0 + 1), i.e. density under the same cost model."""
    X, y, eff, month = arrays(proj)
    tr2, ytr = train_rows(X, y, tr)
    rows = []
    scores = {clf: fit_predict(clf, X, tr2, ytr, te) for clf in ALL}
    uns = unsupervised(X, y, eff, tr2, ytr, te)
    for c0 in C0:
        cost = np.ones(len(te)) if np.isinf(c0) else eff[te] + c0
        ranks = {"ManualUp": uns["ManualUp"], "ManualDown": uns["ManualDown"]}
        for clf, p in scores.items():
            ranks[f"{clf}-prob"] = p
            ranks[f"{clf}-density"] = p / (eff[te] + 1.0)
            ranks[f"{clf}-costaware"] = p if np.isinf(c0) else p / (cost + 1.0)
        for m, sc in ranks.items():
            rows.append({"project": proj, "window": wid, "w": k, "c0": c0, "method": m,
                         **evaluate(y[te], sc, cost)})
    return rows


def hybrid(out="p2jit_hybrid.csv", projs=JIT, n_jobs=10):
    jobs = []
    for proj in projs:
        d = load_jit(proj)
        for k, (tr, te, wid) in enumerate(windows(d)):
            jobs.append(delayed(hybrid_window)(proj, tr, te, wid, k))
    save(run(jobs, n_jobs), out)


def timing(every=4, reps=3):
    """Sequential, single-threaded: every `every`-th window of each project, `reps` fits per learner (and the 50/100
    tree variants), latency = median of 5 scoring passes."""
    out = RAW / "p2jit_cost.csv"
    done = set(pd.read_csv(out)[["project", "w"]].itertuples(index=False, name=None)) if out.exists() else set()
    variants = [(c, None) for c in ALL] + [(c, n) for c in TREES for n in (50, 100)]
    for proj in JIT:
        d = load_jit(proj); X, y, eff, month = arrays(proj)
        for k, (tr, te, wid) in enumerate(windows(d)):
            if k % every or (proj, k) in done:
                continue
            tr2, ytr = train_rows(X, y, tr)
            rows = []
            for clf, n in variants:
                prep = Prep(clf); Ztr = prep.fit_transform(X[tr2]); Zte = prep.transform(X[te])
                for r in range(reps):
                    m = make(clf, seed=SEED, n_jobs=1)
                    if n:
                        m.set_params(n_estimators=n)
                    s = time.perf_counter(); m.fit(Ztr, ytr); fit_s = time.perf_counter() - s
                    m.predict_proba(Zte)
                    lat = []
                    for _ in range(5):
                        s = time.perf_counter(); m.predict_proba(Zte); lat.append(time.perf_counter() - s)
                    rows.append({"project": proj, "w": k, "window": wid, "clf": clf, "n_est": n or "", "rep": r,
                                 "n_train": len(tr2), "n_test": len(te), "fit_s": fit_s,
                                 "latency_ms_per_1k": 1000 * np.median(lat) / len(te) * 1000,
                                 "size_kb": len(pickle.dumps(m)) / 1024})
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(proj, "done", flush=True)


if __name__ == "__main__":
    # mode [project]: with a project the run is in-process and writes p2jit_<mode>.<project>.part.csv (loky's pool
    # broke on this Windows host); `merge <mode>` concatenates the six part files
    mode = sys.argv[1]; one = sys.argv[2:3]
    if mode == "merge":
        m = sys.argv[2]
        pd.concat([pd.read_csv(RAW / f"p2jit_{m}.{p}.part.csv") for p in JIT]).to_csv(RAW / f"p2jit_{m}.csv", index=False)
    elif mode == "timing":
        timing()
    elif mode == "info":
        info()
    elif not one:
        sys.exit("give a project")
    else:
        p = one[0]; out = f"p2jit_{mode}.{p}.part.csv"
        if mode == "perf":
            perf(out=out, projs=[p], n_jobs=1)
        elif mode == "perf12":
            perf(12, out, [p], 1)
        elif mode == "hybrid":
            hybrid(out, [p], 1)
        else:
            sensitivity(mode, 1, [p], out)
