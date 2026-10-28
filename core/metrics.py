"""Ranking and effort-aware metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def rank_metrics(y, p):
    if len(set(y)) < 2:
        return {"auc": np.nan, "prauc": np.nan}
    return {"auc": roc_auc_score(y, p), "prauc": average_precision_score(y, p)}


def _order(score, loc):
    # descending score; ties broken pessimistically (larger modules first)
    return np.lexsort((-loc, -score))


def _curve(y, loc, order):
    loc = loc[order]; y = y[order]
    x = np.concatenate([[0], np.cumsum(loc) / loc.sum()])
    t = np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])
    return x, t


def _area(x, t):
    return float(np.trapezoid(t, x))


def effort_metrics(y, score, loc, budget=0.2):
    """Recall@budget (share of defective modules found within budget of total LOC) and Popt
    (Kamei et al. 2013; Mende & Koschke 2010). LOC is shifted by +1 to avoid zero-size modules."""
    y = np.asarray(y); loc = np.asarray(loc, float) + 1.0; score = np.asarray(score, float)
    xm, tm = _curve(y, loc, _order(score, loc))
    dens = y / loc
    xo, to = _curve(y, loc, np.lexsort((loc, -dens)))
    xw, tw = _curve(y, loc, np.lexsort((-loc, dens)))
    a_m, a_o, a_w = _area(xm, tm), _area(xo, to), _area(xw, tw)
    popt = 1 - (a_o - a_m) / (a_o - a_w) if a_o > a_w else np.nan
    rec = float(np.interp(budget, xm, tm))
    # effort (share of LOC) needed to find 20% of defective modules
    eff = float(np.interp(budget, tm, xm)) if y.sum() else np.nan
    return {"popt": popt, "recall20": rec, "effort20": eff}
