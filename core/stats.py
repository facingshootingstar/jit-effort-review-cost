"""Statistical toolkit fixed in the pre-registered analysis plan."""
import numpy as np
import pandas as pd
from scipy import stats as ss

RNG_SEED = 20261018


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def cliff_label(d):
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"W": np.nan, "p": 1.0}
    r = ss.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"W": float(r.statistic), "p": float(r.pvalue)}


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def boot_ci(x, n=10000, seed=RNG_SEED, alpha=0.05, stat=np.mean):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = stat(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def fmt_p(p):
    if p is None or np.isnan(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def scott_knott_esd(wide, alpha=0.05, negligible=0.2):
    """Scott-Knott ESD (Tantithamthavorn et al. 2017). wide: units x treatments (higher = better).
    Recursively split the mean-sorted treatments where the between-group sum of squares is maximal,
    keep a split if a one-way ANOVA over the two groups' observations is significant, then merge adjacent
    groups whose Cohen's d is negligible (< 0.2). Returns Series treatment -> rank (1 = best)."""
    wide = wide.dropna()
    means = wide.mean().sort_values(ascending=False)
    names = list(means.index)

    def split(group):
        if len(group) < 2:
            return [group]
        m = means[group].values; best, cut = -1.0, None
        for i in range(1, len(group)):
            a, b = m[:i], m[i:]
            bss = len(a) * (a.mean() - m.mean()) ** 2 + len(b) * (b.mean() - m.mean()) ** 2
            if bss > best:
                best, cut = bss, i
        ga, gb = group[:cut], group[cut:]
        p = ss.f_oneway(wide[ga].values.ravel(), wide[gb].values.ravel()).pvalue
        return split(ga) + split(gb) if p < alpha else [group]

    groups = split(names)
    merged = [groups[0]]
    for g in groups[1:]:
        a = wide[merged[-1]].values.ravel(); b = wide[g].values.ravel()
        sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = abs(a.mean() - b.mean()) / sd if sd > 0 else np.inf
        if d < negligible:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    return pd.Series({n: r + 1 for r, g in enumerate(merged) for n in g})[names]


def mixed_diff(df, value, treat, a, b, group="project", nest=None):
    """Linear mixed model value ~ treat + (1 | group[/nest]) on the rows of treatments a and b (b = reference).
    With nest, a variance component for nest within group is added (e.g. window within project).
    Returns the fixed-effect estimate of a - b, its 95% CI and p-value."""
    import statsmodels.formula.api as smf
    d = df[df[treat].isin([a, b])].dropna(subset=[value]).copy()
    d["is_a"] = (d[treat] == a).astype(int)
    d = d.rename(columns={value: "v", group: "g"})
    vc = {"nest": f"0 + C({nest})"} if nest else None
    r = smf.mixedlm("v ~ is_a", d, groups=d["g"], re_formula="1", vc_formula=vc).fit(reml=True, method="lbfgs")
    lo, hi = r.conf_int().loc["is_a"]
    return {"est": float(r.params["is_a"]), "lo": float(lo), "hi": float(hi), "p": float(r.pvalues["is_a"])}
