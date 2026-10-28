"""Time-wise validation windows."""
import numpy as np


def time_windows(month, train=6, gap=2, test=3, step=3):
    """Time-wise windows over calendar months. `month` is a Period/str array per commit,
    already sorted. Returns (train_idx, test_idx, window_id) with month ranges in the id."""
    month = np.asarray(month)
    months = sorted(set(month.tolist()))
    out = []
    for w, s in enumerate(range(0, len(months) - (train + gap + test) + 1, step)):
        tr_m = set(months[s:s + train]); te_m = set(months[s + train + gap:s + train + gap + test])
        tr = np.flatnonzero(np.isin(month, list(tr_m))); te = np.flatnonzero(np.isin(month, list(te_m)))
        out.append((tr, te, f"{months[s]}..{months[s + train + gap + test - 1]}"))
    return out

