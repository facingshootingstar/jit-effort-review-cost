"""Analysis: effort-aware JIT defect prediction on time-wise windows, value vs cost."""
from common import *
from scipy.stats import spearmanr
from core.stats import scott_knott_esd, mixed_diff
from core.models import ALL

out = {}; tex = []
PRIMARY = ["popt", "recall20"]
EXTRA = ["pci20", "prec20", "ifa", "auc"]
df = pd.read_csv(RAW / "p2jit_perf.csv")
df["unit"] = df.project + ":" + df.w.astype(str)
n_win = df.unit.nunique()
assert df.groupby("method").size().eq(n_win).all()
out["n_windows"] = n_win
out["windows_per_project"] = df.groupby("project").w.nunique().to_dict()
out["test_commits_median"] = float(df.drop_duplicates("unit").n_test.median())
out["test_defect_ratio_median"] = float((df.drop_duplicates("unit").pos_test / df.drop_duplicates("unit").n_test).median())


def wide(d, met):
    return d.pivot(index="unit", columns="method", values=met)


def ptest(a, b):
    """pair_test on the windows where both values are defined (Prec@20% and IFA are undefined when the
    ranking opens no commit within the budget or the window has no hit)."""
    ok = a.notna() & b.notna()
    return pair_test(a[ok], b[ok])


def proj_signs(d, a, b, met, higher=True):
    """Number of projects whose median paired difference a - b favours a (out of the projects)."""
    w = d[d.method.isin([a, b])].pivot_table(index=["project", "unit"], columns="method", values=met)
    m = (w[a] - w[b]).groupby("project").median()
    return int((m > 0).sum() if higher else (m < 0).sum()), int(len(m))


# ---- descriptives per method (median and IQR over windows)
desc = df.groupby("method")[PRIMARY + EXTRA].agg(["median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
desc.columns = [f"{a}|{['median', 'q1', 'q3'][i % 3]}" for i, a in enumerate([c[0] for c in desc.columns])]
out["descriptives"] = desc.reset_index()
med = df.groupby("method")[PRIMARY + EXTRA].median()

# ---- RQ2.1 density vs probability ranking, per classifier; Holm over 8 per metric
rows = []
for met in PRIMARY + ["pci20", "prec20", "ifa"]:
    W = wide(df, met)
    for clf in ALL:
        a, b = f"{clf}-density", f"{clf}-prob"
        r = ptest(W[a], W[b]); r.update(metric=met, clf=clf)
        r["proj_favour"], r["n_proj"] = proj_signs(df, a, b, met, higher=(met != "ifa"))
        rows.append(r)
rq1 = add_holm(pd.DataFrame(rows), by="metric")
mx = []
for clf in ALL:
    m = mixed_diff(df, "popt", "method", f"{clf}-density", f"{clf}-prob", nest="w"); m["clf"] = clf; mx.append(m)
out["rq21"] = rq1; out["rq21_mixed_popt"] = mx

# ---- RQ2.2 each method vs LR-density (9 comparisons), Holm per metric
REF = "LR-density"
comp = [f"{c}-density" for c in ALL if c != "LR"] + ["EALR", "ManualUp", "ManualDown", "CBS+"]
rows = []
for met in PRIMARY + ["pci20", "prec20"]:
    W = wide(df, met)
    for m in comp:
        r = ptest(W[m], W[REF]); r.update(metric=met, method=m)
        r["proj_favour"], r["n_proj"] = proj_signs(df, m, REF, met)
        if met in PRIMARY:
            mm = mixed_diff(df, met, "method", m, REF, nest="w")
            r.update(mixed_est=mm["est"], mixed_lo=mm["lo"], mixed_hi=mm["hi"], mixed_p=mm["p"])
        rows.append(r)
rq2 = add_holm(pd.DataFrame(rows), by="metric")
out["rq22"] = rq2
sk = {met: scott_knott_esd(wide(df, met)) for met in PRIMARY + ["prec20", "auc"]}
out["scott_knott"] = {k: v.to_dict() for k, v in sk.items()}

# ---- table: method summary
order = med.sort_values("popt", ascending=False).index
t = pd.DataFrame({"Method": order,
                  "Popt": med.loc[order, "popt"].values,
                  "IQR Popt": [f"{df[df.method == m].popt.quantile(.25):.3f}--{df[df.method == m].popt.quantile(.75):.3f}"
                               for m in order],
                  "SK$_{\\mathrm{Popt}}$": sk["popt"][order].values,
                  "Recall@20\\%": med.loc[order, "recall20"].values,
                  "PCI@20\\%": med.loc[order, "pci20"].values, "Prec@20\\%": med.loc[order, "prec20"].values,
                  "IFA": med.loc[order, "ifa"].values, "AUC": med.loc[order, "auc"].values})
tex.append(latex_table(t, None, f"Median over the {n_win} time-wise windows of each ranking method; IQR: interquartile range of the window-level Popt. SK: Scott-Knott ESD "
                       "rank on window-level Popt (1 = best). PCI@20\\%: share of the test commits that fall inside "
                       "the first 20\\% of churn; Prec@20\\%: share of those commits that are defect-inducing.",
                       "tab:p2-methods", fmt={"SK$_{\\mathrm{Popt}}$": lambda v: f"{int(v)}", "IFA": lambda v: f"{v:.1f}"}))

# table RQ2.1 (Popt, Recall@20, PCI, Prec)
def rq_tab(d, key, label, caption, keys):
    rows = []
    for k in keys:
        r = {key.title() if key != "method" else "Method": k}
        for met, lab in (("popt", "Popt"), ("recall20", "R@20"), ("pci20", "PCI@20"), ("prec20", "P@20")):
            s = d[(d[key] == k) & (d.metric == met)].iloc[0]
            r[f"$\\Delta${lab}"] = f"{s['diff']:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000")
            r[f"$p_H$ ({lab})"] = pfmt(s.p_holm)
            if met == "popt":
                r["$r_{rb}$"] = f"{s.rrb:+.2f}"
                r["proj."] = f"{s.proj_favour}/{s.n_proj}"
        rows.append(r)
    return latex_table(pd.DataFrame(rows), None, caption, label)
tex.append(rq_tab(rq1, "clf", "tab:p2-rq1",
                  "RQ2.1: density ranking minus probability ranking for each classifier (mean paired difference "
                  f"over {n_win} windows), Holm-adjusted Wilcoxon signed-rank $p$ over the eight classifiers per metric, the matched-pairs "
                  "rank-biserial $r_{rb}$ for Popt, and proj.: the number of projects (of six) whose median paired Popt difference "
                  "favours density ranking.",
                  ALL))
tex.append(rq_tab(rq2, "method", "tab:p2-rq2",
                  "RQ2.2: each method minus LR-density (mean paired difference over the windows), Holm-adjusted "
                  "Wilcoxon signed-rank $p$ over the eleven comparisons per metric, the matched-pairs rank-biserial $r_{rb}$ for Popt, "
                  "and proj.: the number of projects (of six) whose median paired Popt difference favours the method.", comp))

# ---- figure: Popt vs PCI@20 (the price of Popt)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, (x, y, xl, yl) in zip(axes, (("pci20", "popt", "PCI@20% (share of commits inspected)", "Popt"),
                                     ("pci20", "prec20", "PCI@20% (share of commits inspected)", "Precision@20%"))):
    for m in med.index:
        kind = "density" if m.endswith("density") else ("prob" if m.endswith("prob") else m)
        c = {"density": OI[0], "prob": OI[1], "EALR": OI[2], "ManualUp": OI[3], "ManualDown": OI[4], "CBS+": OI[5]}[kind]
        mk = {"density": "o", "prob": "s", "EALR": "^", "ManualUp": "D", "ManualDown": "v", "CBS+": "P"}[kind]
        ax.scatter(med.loc[m, x], med.loc[m, y], color=c, marker=mk, s=26, zorder=3)
        off = {"LR-density": (-42, 5), "ManualUp": (-12, -11), "LR-prob": (5, -4)}
        if kind in ("EALR", "ManualUp", "ManualDown", "CBS+") or m in ("LR-prob", "LR-density"):
            ax.annotate(m, (med.loc[m, x], med.loc[m, y]), fontsize=6.5, xytext=off.get(m, (3, 3)),
                        textcoords="offset points")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([], [], color=OI[i], marker=mk, ls="", label=l) for i, mk, l in
                        ((0, "o", "density p/(churn+1)"), (1, "s", "probability p"), (2, "^", "EALR"), (3, "D", "ManualUp"), (4, "v", "ManualDown"), (5, "P", "CBS+"))],
               fontsize=6.5, loc="lower right")
fig.tight_layout(); savefig(fig, 2, "popt_vs_pci")

# ---- per-project medians (descriptive)
pp = df[df.method.isin(["LR-density", "RF-density", "ManualUp", "LR-prob", "RF-prob", "EALR"])]
out["project_medians"] = pp.groupby(["project", "method"])[["popt", "recall20", "pci20", "prec20"]].median().reset_index()

# ---- exploratory (deviation 21): review-cost model churn + c0
hp = RAW / "p2jit_hybrid.csv"
if hp.exists():
    h = pd.read_csv(hp); h["unit"] = h.project + ":" + h.w.astype(str)
    h["c0s"] = h.c0.map(lambda v: "inf" if np.isinf(v) else str(int(v)))
    hm = h.groupby(["c0s", "method"])[["popt", "recall20", "pci20", "prec20"]].median().reset_index()
    out["hybrid_medians"] = hm
    best = {}
    for c0, g in h.groupby("c0s"):
        W = g.pivot(index="unit", columns="method", values="recall20")
        s = W.median().sort_values(ascending=False)
        rk = scott_knott_esd(W)
        best[c0] = {"top5": s.head(5).to_dict(), "sk_rank1": sorted(rk[rk == 1].index.tolist()),
                    "ManualUp": float(s["ManualUp"]), "LR-prob": float(s["LR-prob"]),
                    "LR-density": float(s["LR-density"]), "LR-costaware": float(s["LR-costaware"]),
                    "RF-costaware": float(s["RF-costaware"])}
        # cost-aware LR vs ManualUp and vs density under this cost model
        r1 = pair_test(W["LR-costaware"], W["ManualUp"]); r2 = pair_test(W["LR-costaware"], W["LR-density"])
        r3 = pair_test(W["LR-costaware"], W["LR-prob"])
        best[c0].update(costaware_vs_manualup=r1, costaware_vs_density=r2, costaware_vs_prob=r3)
    order_c = ["0", "10", "30", "100", "inf"]
    keys = [(c0, k) for c0 in order_c for k in ("costaware_vs_manualup", "costaware_vs_density", "costaware_vs_prob")]
    for (c0, k), p in zip(keys, holm([best[c0][k]["p"] for c0, k in keys])):  # one Holm family of 15 contrasts
        best[c0][k]["p_holm"] = p
    out["hybrid"] = best
    rows = []
    for c0 in order_c:
        g = hm[hm.c0s == c0].set_index("method")
        rows.append({"$c_0$ (LOC/commit)": "$\\infty$" if c0 == "inf" else c0,
                     "ManualUp": g.loc["ManualUp", "recall20"], "ManualDown": g.loc["ManualDown", "recall20"],
                     "LR-prob": g.loc["LR-prob", "recall20"],
                     "LR-density": g.loc["LR-density", "recall20"], "LR-cost-aware": g.loc["LR-costaware", "recall20"],
                     "XGB-cost-aware": g.loc["XGB-costaware", "recall20"],
                     "PCI (LR-c.a.)": g.loc["LR-costaware", "pci20"],
                     "$p_H$ c.a. vs ManualUp": pfmt(best[c0]["costaware_vs_manualup"]["p_holm"]),
                     "$r_{rb}$": f"{best[c0]['costaware_vs_manualup']['rrb']:+.2f}"})
    tex.append(latex_table(pd.DataFrame(rows), None,
                           "Exploratory review-cost model (deviation 21): each commit costs its churn plus $c_0$ "
                           "LOC-equivalents of context-switch overhead ($c_0=\\infty$: only the number of commits "
                           "counts). Median Recall@20\\% of the cost budget over the windows. Cost-aware ranking: "
                           "$p/(\\mathrm{churn}+c_0+1)$, the greedy rule for this cost model. Last two columns: LR "
                           "cost-aware vs ManualUp, Wilcoxon $p$ with Holm over all 15 cost-aware contrasts (three "
                           "per $c_0$), and the matched-pairs rank-biserial $r_{rb}$.", "tab:p2-hybrid"))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    xs = np.arange(len(order_c))
    for m, c, mk in (("ManualUp", OI[3], "D"), ("LR-prob", OI[1], "s"), ("LR-density", OI[0], "o"),
                     ("LR-costaware", OI[2], "^")):
        ax.plot(xs, [hm[(hm.c0s == c0) & (hm.method == m)].recall20.iloc[0] for c0 in order_c], marker=mk, color=c,
                label=m.replace("costaware", "cost-aware"))
    ax.set_xticks(xs, ["0", "10", "30", "100", "∞"]); ax.set_xlabel("per-commit overhead $c_0$ (LOC-equivalents)")
    ax.set_ylabel("median Recall@20% cost"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 2, "hybrid_cost")

# ---- sensitivity: 12-month training windows
p12 = RAW / "p2jit_perf12.csv"
if p12.exists():
    d12 = pd.read_csv(p12); d12["unit"] = d12.project + ":" + d12.w.astype(str)
    out["sens12_n_windows"] = d12.unit.nunique()
    out["sens12_medians"] = d12.groupby("method")[PRIMARY + ["pci20", "prec20"]].median().reset_index()
    rows = []
    W = wide(d12, "popt")
    for m in comp:
        r = pair_test(W[m], W[REF]); r.update(method=m); rows.append(r)
    out["sens12_rq22_popt"] = add_holm(pd.DataFrame(rows))
    rows = []
    for clf in ALL:
        r = pair_test(W[f"{clf}-density"], W[f"{clf}-prob"]); r.update(clf=clf); rows.append(r)
    out["sens12_rq21_popt"] = add_holm(pd.DataFrame(rows))

# ---- RQ2.3 cost (requires the sequential timing run)
cp = RAW / "p2jit_cost.csv"
if cp.exists():
    c = pd.read_csv(cp)
    c["unit"] = c.project + ":" + c.w.astype(str)
    if "n_est" in c:
        c["n_est"] = c.n_est.fillna("").astype(str).str.replace(".0", "", regex=False)
        cv = c[c.n_est != ""].groupby(["clf", "n_est", "unit"])[["fit_s", "latency_ms_per_1k"]].median().reset_index()
        out["cost_tree_variants"] = cv.groupby(["clf", "n_est"])[["fit_s", "latency_ms_per_1k"]].median().reset_index()
        c = c[c.n_est == ""]
    c = c.groupby(["clf", "unit", "project", "w"])[["fit_s", "latency_ms_per_1k", "size_kb"]].median().reset_index()
    complete = c.groupby("unit").clf.nunique().eq(len(ALL))
    c = c[c.unit.isin(complete[complete].index)]
    out["cost_n_windows"] = c.unit.nunique()
    cm = c.groupby("clf")[["fit_s", "latency_ms_per_1k", "size_kb"]].median().reindex(ALL)
    lr = c[c.clf == "LR"].set_index("unit")
    ratio = {}
    for clf in ALL:
        g = c[c.clf == clf].set_index("unit")
        fr = g.fit_s / lr.fit_s; lr_ = g.latency_ms_per_1k / lr.latency_ms_per_1k
        ratio[clf] = {"fit_ratio_median": float(fr.median()), "fit_ratio_q1": float(fr.quantile(.25)),
                      "fit_ratio_q3": float(fr.quantile(.75)), "lat_ratio_median": float(lr_.median()),
                      "lat_ratio_q1": float(lr_.quantile(.25)), "lat_ratio_q3": float(lr_.quantile(.75))}
    pm = med.loc[[f"{c_}-density" for c_ in ALL], "popt"].values
    out["cost_medians"] = cm.reset_index(); out["cost_ratio_vs_lr"] = ratio
    out["spearman_fit_popt"] = {"rho": float(spearmanr(cm.fit_s.values, pm).statistic),
                                "p": float(spearmanr(cm.fit_s.values, pm).pvalue)}
    out["spearman_lat_popt"] = {"rho": float(spearmanr(cm.latency_ms_per_1k.values, pm).statistic),
                                "p": float(spearmanr(cm.latency_ms_per_1k.values, pm).pvalue)}
    rq = rq2[rq2.metric == "popt"].set_index("method")
    rows = []
    for clf in ALL:
        m = f"{clf}-density"
        rows.append({"Classifier": clf, "fit (s)": f"{cm.loc[clf, 'fit_s']:.3f}",
                     "fit / LR [IQR]": f"{ratio[clf]['fit_ratio_median']:.1f} [{ratio[clf]['fit_ratio_q1']:.1f}, {ratio[clf]['fit_ratio_q3']:.1f}]",
                     "latency (ms/1k)": f"{cm.loc[clf, 'latency_ms_per_1k']:.2f}",
                     "lat. / LR [IQR]": f"{ratio[clf]['lat_ratio_median']:.1f} [{ratio[clf]['lat_ratio_q1']:.1f}, {ratio[clf]['lat_ratio_q3']:.1f}]",
                     "size (KB)": f"{cm.loc[clf, 'size_kb']:.1f}" if cm.loc[clf, "size_kb"] < 10 else f"{cm.loc[clf, 'size_kb']:.0f}",
                     "Popt": f"{med.loc[m, 'popt']:.3f}",
                     "$\\Delta$Popt vs LR": "--" if clf == "LR" else f"{rq.loc[m, 'diff']:+.3f}".replace("-0.000", "0.000").replace("-", "$-$"),
                     "$p_H$": "--" if clf == "LR" else pfmt(rq.loc[m, "p_holm"])})
    tex.append(latex_table(pd.DataFrame(rows), None,
                           f"RQ2.3: single-threaded cost (median over {out['cost_n_windows']} windows) against "
                           "density-ranked Popt. Ratios are medians [interquartile ranges] of per-window ratios to LR, each window value being the median of three fits; $\\Delta$Popt and "
                           "$p_H$ repeat Table~\\ref{tab:p2-rq2}.", "tab:p2-cost"))
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for i, clf in enumerate(ALL):
        ax.scatter(cm.loc[clf, "fit_s"], med.loc[f"{clf}-density", "popt"], color=OI[i % 8], s=28, zorder=3)
        ax.annotate(clf, (cm.loc[clf, "fit_s"], med.loc[f"{clf}-density", "popt"]), fontsize=7, xytext=(3, 2),
                    textcoords="offset points")
    ax.axhline(med.loc["ManualUp", "popt"], color=OI[3], ls="--", lw=1)
    ax.text(ax.get_xlim()[0] if False else cm.fit_s.min(), med.loc["ManualUp", "popt"] + 0.004, "ManualUp (no training)",
            fontsize=6.5, color=OI[3])
    ax.set_xscale("log"); ax.set_xlabel("median training time (s, log scale)"); ax.set_ylabel("median Popt (density)")
    ax.grid(alpha=.3); fig.tight_layout(); savefig(fig, 2, "cost_value")

# ==== revision round 1 (deviation 31; all exploratory unless stated) ====
def f3s(v):
    return f"{v:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000")


SESOI = 0.01  # smallest Popt / Recall@20% difference of interest, declared in deviation 31
rng = np.random.default_rng(20261018)


def cluster_ci(d, a, b, met, n=4000, alpha=0.10):
    """Two-stage cluster bootstrap of the mean paired difference a - b: resample projects, then windows within them."""
    w = d[d.method.isin([a, b])].pivot_table(index=["project", "unit"], columns="method", values=met).dropna()
    diff = (w[a] - w[b]); groups = [g.values for _, g in diff.groupby(level=0)]
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        stats.append(np.concatenate([groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in pick]).mean())
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    pm = diff.groupby(level=0).mean()
    return {"diff": float(diff.mean()), "lo": float(lo), "hi": float(hi), "equal_weight": float(pm.mean()),
            "proj_pos": int((pm > 0).sum()), "n_proj": int(len(pm)), "n": int(len(diff))}


def tost(d, a, b, met, sesoi=SESOI):
    """Equivalence: the 90% two-stage cluster-bootstrap CI of the mean paired difference lies inside +-sesoi."""
    r = cluster_ci(d, a, b, met)
    r["equivalent"] = bool(r["lo"] > -sesoi and r["hi"] < sesoi)
    r["ci95"] = list(cluster_ci(d, a, b, met, alpha=0.05).values())[1:3]
    from scipy.stats import t as tdist
    w = d[d.method.isin([a, b])].pivot_table(index=["project", "unit"], columns="method", values=met).dropna()
    pm = (w[a] - w[b]).groupby(level=0).mean(); h = tdist.ppf(0.95, len(pm) - 1) * pm.std(ddof=1) / np.sqrt(len(pm))
    r["t90_proj"] = [float(pm.mean() - h), float(pm.mean() + h)]
    return r


def mixed_slope(d, a, b, met):
    """Random intercept and random method slope per project, window as a nested variance component."""
    import statsmodels.formula.api as smf
    x = d[d.method.isin([a, b])].dropna(subset=[met]).copy(); x["is_a"] = (x.method == a).astype(int)
    x = x.rename(columns={met: "v"})
    try:
        r = smf.mixedlm("v ~ is_a", x, groups=x["project"], re_formula="~is_a",
                        vc_formula={"win": "0 + C(w)"}).fit(reml=True, method="lbfgs")
        lo, hi = r.conf_int().loc["is_a"]
        return {"est": float(r.params["is_a"]), "lo": float(lo), "hi": float(hi), "p": float(r.pvalues["is_a"])}
    except Exception as e:  # report, do not hide, a failed fit
        return {"error": str(e)[:120]}


eq = {}
for met in PRIMARY:
    for m in comp:
        r = tost(df, m, REF, met)
        if met == "popt" and m in ("ManualUp", "RF-density", "MLP-density", "XGB-density", "ManualDown", "CBS+"):
            r["mixed_slope"] = mixed_slope(df, m, REF, met)
            r["mixed_project_only"] = mixed_diff(df, met, "method", m, REF)  # the pre-registered wording before deviation 23
            from scipy.stats import binomtest
            r["sign_p"] = float(binomtest(r["proj_pos"], r["n_proj"]).pvalue)
        eq[f"{met}:{m}"] = r
out["equivalence"] = eq
rows = []
for m in comp:
    a, b = eq[f"popt:{m}"], eq[f"recall20:{m}"]
    rows.append({"Method": m, "$\\Delta$Popt": a["diff"], "90\\% CI (cluster)": f"[{a['lo']:+.3f}, {a['hi']:+.3f}]",
                 "Equal-weight": a["equal_weight"], "proj. $>0$": f"{a['proj_pos']}/{a['n_proj']}",
                 "Equiv. Popt": "yes" if a["equivalent"] else "no",
                 "$\pm0.005$": "yes" if (a["lo"] > -0.005 and a["hi"] < 0.005) else "no",
                 "$\\Delta$R@20": b["diff"], "90\\% CI (R@20)": f"[{b['lo']:+.3f}, {b['hi']:+.3f}]",
                 "Equiv. R@20": "yes" if b["equivalent"] else "no"})
tex.append(latex_table(pd.DataFrame(rows), None,
                       "Equivalence and dependence checks for RQ2.2 (exploratory, deviation 31): mean paired "
                       "difference to LR-density, 90\\% two-stage cluster-bootstrap percentile CI (4{,}000 replicates; "
                       "projects, then windows within projects), the mean of the six per-project mean differences (equal "
                       "project weight), proj.\\ $>0$: the number of projects (of six) whose mean paired difference is "
                       f"positive, and equivalence (the 90\\% CI inside the bounds) at the post-hoc margin $\\pm{SESOI}$ and, "
                       "as a post-hoc sensitivity check, at $\\pm0.005$ Popt. With six projects the cluster CI is likely too narrow. "
                       "Table~\\ref{tab:p2-rq2} counts projects by the median paired difference, so its counts can differ.",
                       "tab:p2-equiv",
                       fmt={"$\\Delta$Popt": f3s, "Equal-weight": f3s, "$\\Delta$R@20": f3s}))

# Prec@20% sample sizes and the window-level share below the base rate
out["prec20_n"] = df.groupby("method").prec20.apply(lambda s: int(s.notna().sum())).to_dict()
base = df.pos_test / df.n_test
out["prec_below_base_share"] = (df.assign(b=df.prec20 < base).dropna(subset=["prec20"])
                                .groupby("method").b.mean().to_dict())

# zero-churn sensitivity: the same rankings scored on commits with churn > 0
nz = df.dropna(subset=["popt_nz"])
out["nz_n_windows"] = int(nz.unit.nunique())
out["nz_medians"] = nz.groupby("method")[["popt_nz", "recall20_nz"]].median().reset_index()
rows = []
for m in comp:
    W = nz.pivot(index="unit", columns="method", values="popt_nz")
    r = ptest(W[m], W[REF]); r.update(method=m); rows.append(r)
out["nz_rq22_popt"] = add_holm(pd.DataFrame(rows))
rows = []
for clf in ALL:
    W = nz.pivot(index="unit", columns="method", values="popt_nz")
    r = ptest(W[f"{clf}-density"], W[f"{clf}-prob"]); r.update(clf=clf); rows.append(r)
out["nz_rq21_popt"] = add_holm(pd.DataFrame(rows))

# review-cost model: ManualDown and CBS+-free baselines, break-even
if hp.exists():
    be = {}
    for c0, g in h.groupby("c0s"):
        W = g.pivot(index="unit", columns="method", values="recall20")
        for ref in ("ManualUp", "ManualDown"):
            r = pair_test(W["LR-costaware"], W[ref]); be[f"{c0}:LR-costaware_vs_{ref}"] = r
        be[f"{c0}:ManualDown_vs_ManualUp"] = pair_test(W["ManualDown"], W["ManualUp"])
        be[f"{c0}:LR-costaware_vs_best"] = {"best": W.median().idxmax(), "best_median": float(W.median().max()),
                                            "lr_ca_median": float(W["LR-costaware"].median())}
    out["hybrid_extra"] = be
    out["hybrid_md_medians"] = hm[hm.method.isin(["ManualDown", "ManualUp", "LR-costaware", "XGB-costaware",
                                                    "RF-costaware"])]

# sensitivity runs (gap, label masking, ensemble size, seeds)
SENS = {}
for mode, key in (("gap", "gap"), ("mask", "mask"), ("trees", "n_est"), ("seeds", "seed_offset")):
    f = RAW / f"p2jit_{mode}.csv"
    if not f.exists():
        continue
    x = pd.read_csv(f); x["unit"] = x.project + ":" + x.w.astype(str)
    res = {}
    for v, g in x.groupby(key):
        r = {"n_windows": int(g.unit.nunique()), "medians": g.groupby("method")[["popt", "recall20"]].median().to_dict()}
        if mode in ("gap", "mask"):
            W = g.pivot(index="unit", columns="method", values="popt")
            r["ManualUp_vs_LR"] = pair_test(W["ManualUp"], W["LR-density"])
            r["RF_vs_LR"] = pair_test(W["RF-density"], W["LR-density"])
            r["LR_density_vs_prob"] = pair_test(W["LR-density"], W["LR-prob"])
        else:  # same windows as the main run: paired against the main-run value of the same learner
            for clf in g.clf.unique():
                a = g[g.method == f"{clf}-density"].set_index("unit").popt
                b = df[df.method == f"{clf}-density"].set_index("unit").popt.reindex(a.index)
                r[f"{clf}_vs_main"] = pair_test(a, b)
                r[f"{clf}_vs_LR"] = pair_test(a, df[df.method == "LR-density"].set_index("unit").popt.reindex(a.index))
        res[str(v)] = r
    SENS[mode] = res
out["sensitivity"] = SENS
if "trees" in SENS:
    _t = pd.read_csv(RAW / "p2jit_trees.csv"); _t["unit"] = _t.project + ":" + _t.w.astype(str)
    _a = _t[(_t.n_est == 50) & (_t.method == "XGB-density")].set_index("unit").popt
    out["xgb50_vs_manualup"] = pair_test(_a, df[df.method == "ManualUp"].set_index("unit").popt.reindex(_a.index))
    _d = pd.concat([_t[(_t.n_est == 50) & (_t.method == "XGB-density")].assign(method="XGB-50")[["project", "unit", "w", "method", "popt"]],
                    df[(df.method == "ManualUp") & df.unit.isin(_a.index)][["project", "unit", "w", "method", "popt"]]], ignore_index=True)
    _r = cluster_ci(_d, "XGB-50", "ManualUp", "popt")
    _r["mixed_slope"] = mixed_slope(_d, "XGB-50", "ManualUp", "popt")
    out["xgb50_vs_manualup_cluster"] = _r
if SENS:
    def cell(r):
        return f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]", pfmt(r["p"]), f"{r['rrb']:+.2f}"
    main_ma = pair_test(W0 := df[df.method == "ManualUp"].set_index("unit").popt,
                        df[df.method == "LR-density"].set_index("unit").popt.reindex(W0.index))
    main_rf = pair_test(df[df.method == "RF-density"].set_index("unit").popt.reindex(W0.index),
                        df[df.method == "LR-density"].set_index("unit").popt.reindex(W0.index))
    rows = [("main run (gap 2, mask 0, 300 trees)", out["n_windows"], main_rf, main_ma)]
    for mode, lab in (("gap", "gap {} months"), ("mask", "labels of last {} training month(s) hidden")):
        for v, r in SENS.get(mode, {}).items():
            rows.append((lab.format(v), r["n_windows"], r["RF_vs_LR"], r["ManualUp_vs_LR"]))
    for v, r in SENS.get("trees", {}).items():
        for clf in ("RF", "LGBM", "XGB"):
            rows.append((f"{clf} with {v} trees/rounds", r["n_windows"], r[f"{clf}_vs_LR"], None))
    for v, r in SENS.get("seeds", {}).items():
        for clf in ("RF", "MLP"):
            rows.append((f"{clf}, seed offset {v}", r["n_windows"], r[f"{clf}_vs_LR"], None))
    tr = []
    for lab, n, rf, ma in rows:
        a = cell(rf); b = cell(ma) if ma else ("--", "--", "--")
        tr.append({"Setting": lab, "$n$": n, "Learner$-$LR $\\Delta$Popt [95\\% CI]": a[0], "$p$": a[1],
                   "$r_{rb}$": a[2],
                   "ManualUp$-$LR $\\Delta$Popt [95\\% CI]": b[0], "$p$ ": b[1], "$r_{rb}$ ": b[2]})
    tex.append(latex_table(pd.DataFrame(tr), None,
                           "Sensitivity of the two key RQ2.2 contrasts on Popt (exploratory, deviation 31): mean "
                           "paired difference to LR-density over the windows of each setting, i.i.d.\\ bootstrap "
                           "95\\% CI, unadjusted Wilcoxon $p$ and $r_{rb}$. Label masking sets the labels of the "
                           "commits of the last training months to clean, as if their defects had not yet been "
                           "found. All CIs and $p$ in this table are window-level and not cluster-adjusted. Tree-size and seed variants refit only the named learner on the main windows. Learners: LR, RF and "
                           "XGB in the buffer and masking runs; RF, LGBM and XGB in the ensemble-size runs; RF, XGB, LGBM "
                           "and MLP in the seed runs (XGB and LGBM use no random subsampling here, so other seeds reproduce "
                           "the main run exactly and their rows are omitted). $\Delta$ and its CI describe the mean; $p$ is "
                           "from the signed-rank test, which concerns the location of the paired differences, so the two can "
                           "disagree (e.g., XGB with 100 rounds).",
                           "tab:p2-sens"))

import re
(ROOT / "paper2" / "tables.tex").write_text(re.sub(r"(?<![\w$])-(?=\d)", "$-$", "\n".join(tex)), encoding="utf-8")
dump(out, "p2jit.json")
print(med.sort_values("popt", ascending=False).round(3).to_string())
print(rq1[rq1.metric == "popt"][["clf", "diff", "ci_lo", "ci_hi", "p_holm", "delta", "proj_favour"]].round(3).to_string())
print(rq2[rq2.metric.isin(PRIMARY)][["metric", "method", "diff", "ci_lo", "ci_hi", "p_holm", "delta", "proj_favour",
                                     "mixed_est", "mixed_p"]].round(3).to_string())
print({k: v.round(0).to_dict() for k, v in sk.items()})
