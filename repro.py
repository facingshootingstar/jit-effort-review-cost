"""Reproduce every statistic, table and figure of the paper (effort-aware JIT defect prediction). See README.md.

  python repro.py check                     data manifest (SHA-256) + package versions
  python repro.py fast    [p2]              shipped raw results -> analysis -> compare with reference/
  python repro.py full    [p2] [-j N]       re-run the experiments -> compare raw -> analysis -> compare
  python repro.py compare [p2]              compare only (after fast/full)

FAST is exact: the analysis scripts are deterministic, so every JSON value, table and figure must match.
FULL re-trains every model; on another OS/CPU tree learners can differ in the last digits, and the
timing/latency measurements (p2jit_cost.csv) are hardware-dependent by nature.
"""
import os, sys, json, math, re, shutil, subprocess, time, pathlib, importlib.metadata as md

ROOT = pathlib.Path(__file__).resolve().parent
RAW, REF, SUM = ROOT / "results" / "raw", ROOT / "reference", ROOT / "results" / "summary"
OUTD = ROOT / "out"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["PYTHONIOENCODING"] = "utf-8"; os.environ.setdefault("MPLBACKEND", "Agg")

JIT = ["bugzilla", "columba", "jdt", "mozilla", "platform", "postgres"]
PAPERS = {
    "p2": dict(scripts=["p2jit.py"], json=["p2jit.json"], csv=[], tables=["paper2/tables.tex"],
               raw=[f"p2jit_{m}.csv" for m in ("perf", "perf12", "hybrid", "cost", "gap", "mask", "trees", "seeds")]
                   + ["p2jit_info.json"]),
}
# values that measure the machine, not the method (RQ2.3 training time / latency / memory)
HW = re.compile(r"(time|timing|latency|fit_s|_ms|ms_|peak|mem|speed|ratio_fit|cost_tab|seconds)", re.I)


def sh(args, log=None):
    print("$", " ".join(map(str, args)), flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-W", "ignore", *map(str, args)], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True); log.write_text(r.stdout, encoding="utf-8")
    if r.returncode:
        print(r.stdout[-4000:]); raise SystemExit(f"FAILED ({r.returncode}): {' '.join(map(str, args))}")
    print(f"  ok {time.time() - t0:.0f}s", flush=True)
    return r.stdout


def check():
    sys.path.insert(0, str(ROOT))
    from core.corpora import verify_manifest
    verify_manifest(); print("data manifest: all SHA-256 hashes match")
    want = dict(l.split("==") for l in (ROOT / "requirements.txt").read_text().split() if "==" in l)
    bad = []
    for p, v in want.items():
        try:
            got = md.version(p)
        except md.PackageNotFoundError:
            got = None
        if got != v:
            bad.append(f"{p}: want {v}, have {got}")
    print(f"python {sys.version.split()[0]} (original runs: 3.12.10)")
    print("packages: all pinned versions installed" if not bad else "PACKAGE MISMATCH:\n  " + "\n  ".join(bad))
    return not bad


def raw_files(base, p):
    return sorted(f for g in PAPERS[p]["raw"] for f in base.glob(g) if f.is_file())


def stage_raw(p):
    """FAST: copy the shipped raw experiment outputs of paper p into results/raw."""
    for f in raw_files(REF / "raw", p):
        d = RAW / f.relative_to(REF / "raw"); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, d)


def analyse(p):
    SUM.mkdir(parents=True, exist_ok=True)
    for s in PAPERS[p]["scripts"]:
        sh(["analysis/" + s], OUTD / "logs" / f"analysis_{s}.txt")


# ---------------------------------------------------------------- FULL: re-run the experiments
def par(cmds, jobs):
    """run independent runner invocations, at most `jobs` at a time (each process is single-threaded)."""
    run, todo = [], list(cmds)
    while todo or run:
        while todo and len(run) < jobs:
            c = todo.pop(0); print("$", " ".join(c), flush=True)
            run.append((c, subprocess.Popen([sys.executable, "-W", "ignore", *c], cwd=ROOT, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE, text=True)))
        for c, pr in list(run):
            if pr.poll() is not None:
                run.remove((c, pr))
                if pr.returncode:
                    raise SystemExit(f"FAILED: {' '.join(c)}\n{pr.stderr.read()[-3000:]}")
        time.sleep(1)


def experiments(p, jobs):
    for f in raw_files(RAW, p):  # runners skip outputs that already exist, so start from nothing
        f.unlink()
    E = "experiments/"
    if p == "p2":
        for m in ("perf", "hybrid", "perf12", "gap", "mask", "trees", "seeds"):
            par([[E + "run_p2_jit.py", m, pr] for pr in JIT], jobs)
            sh([E + "run_p2_jit.py", "merge", m])
        sh([E + "run_p2_jit.py", "info"])
        sh([E + "run_p2_jit.py", "timing"])  # alone: wall-clock measurements


# ---------------------------------------------------------------- comparison
def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def same(a, b, rtol, atol):
    if num(a) and num(b):
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
            return (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b))
        return abs(a - b) <= atol + rtol * abs(b)
    return a == b


def cmp_json(new, ref, rtol, atol):
    A, B = dict(leaves(json.loads(new.read_text(encoding="utf-8")))), dict(leaves(json.loads(ref.read_text(encoding="utf-8"))))
    diff = [k for k in B if k in A and not same(A[k], B[k], rtol, atol)]
    miss = [k for k in B if k not in A]
    hw = [k for k in diff if HW.search(k)]
    return dict(n=len(B), n_num=sum(num(v) for v in B.values()), equal=len(B) - len(diff) - len(miss),
                diff=[k for k in diff if k not in hw], hw=hw, missing=miss, A=A, B=B)


def cmp_csv(new, ref, rtol, atol):
    import pandas as pd, numpy as np
    a, b = pd.read_csv(new), pd.read_csv(ref)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    bad = 0
    for i in range(b.shape[1]):
        x, y = a.iloc[:, i], b.iloc[:, i]
        if pd.api.types.is_numeric_dtype(y) and pd.api.types.is_numeric_dtype(x):
            bad += int((~np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=rtol, atol=atol, equal_nan=True)).sum())
        else:
            bad += int((x.astype(str) != y.astype(str)).sum())
    return "" if not bad else f"{bad} cells differ"


def cmp_png(new, ref):
    import numpy as np, matplotlib.image as mi
    a, b = mi.imread(new), mi.imread(ref)
    if a.shape != b.shape:
        return None, f"size {a.shape[:2]} vs {b.shape[:2]}"
    d = np.abs(a.astype(float) - b.astype(float))
    return float(d.mean()), f"mean |diff| {d.mean():.2e}, pixels changed {(d.max(axis=-1) > 1e-6).mean():.3%}"


def compare(p, mode, rtol, atol):
    L, ok = [f"# {p.upper()} — {mode} reproduction check", ""], True
    for j in PAPERS[p]["json"]:
        new, ref = SUM / j, REF / "summary" / j
        if not new.exists():
            L.append(f"- **{j}: NOT PRODUCED**"); ok = False; continue
        r = cmp_json(new, ref, rtol, atol)
        good = not r["diff"] and not r["missing"]
        ok &= good
        L.append(f"- {j}: {r['equal']}/{r['n']} values identical ({r['n_num']} numeric) — "
                 + ("**OK**" if good else f"**{len(r['diff'])} differ, {len(r['missing'])} missing**")
                 + (f"; {len(r['hw'])} hardware-timing values differ (expected)" if r["hw"] else ""))
        for k in (r["diff"] + r["missing"])[:25]:
            L.append(f"    - `{k}`: new={r['A'].get(k)!r} ref={r['B'][k]!r}")
        for k in r["hw"][:8]:
            L.append(f"    - (timing) `{k}`: new={r['A'][k]!r} ref={r['B'][k]!r}")
    for c in PAPERS[p]["csv"]:
        m = cmp_csv(SUM / c, REF / "summary" / c, rtol, atol); ok &= not m
        L.append(f"- {c}: " + ("**OK**" if not m else f"**{m}**"))
    for t in PAPERS[p]["tables"]:
        new, ref = ROOT / t, REF / t
        if not new.exists():
            L.append(f"- **{t}: NOT PRODUCED**"); ok = False; continue
        same_t = new.read_text(encoding="utf-8") == ref.read_text(encoding="utf-8")
        ok &= same_t or mode == "full"
        L.append(f"- {t}: " + ("**identical**" if same_t else "**differs** (see diff with reference/)"))
    n = p[1]
    for ref in sorted((REF / f"paper{n}" / "figs").glob("*.png")):
        new = ROOT / f"paper{n}" / "figs" / ref.name
        if not new.exists():
            L.append(f"- **figure {ref.name}: NOT PRODUCED**"); ok = False; continue
        m, msg = cmp_png(new, ref)
        good = m is not None and m < 2e-3
        ok &= good or mode == "full"
        L.append(f"- figure {ref.name}: {msg} — " + ("**OK**" if good else "**differs**"))
    if mode == "full":
        for f in raw_files(REF / "raw", p):
            g = RAW / f.relative_to(REF / "raw")
            if f.suffix != ".csv":
                continue
            m = "missing" if not g.exists() else cmp_csv(g, f, 1e-6, 1e-9)
            tag = " (hardware timing, expected)" if f.name == "p2jit_cost.csv" else ""
            L.append(f"- raw {f.relative_to(REF / 'raw')}: " + ("identical within 1e-6" if not m else m + tag))
    L += ["", "**RESULT: " + ("ALL REPRODUCED**" if ok else "SEE DIFFERENCES ABOVE**")]
    OUTD.mkdir(exist_ok=True); (OUTD / f"report_{p}_{mode}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)
    return ok


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    cmd = a[0]
    jobs = int(a[a.index("-j") + 1]) if "-j" in a else (os.cpu_count() or 2)
    ps = [x for x in a[1:] if x in PAPERS] or list(PAPERS)
    if cmd == "check":
        check(); return
    check()
    res = {}
    for p in ps:
        print(f"\n===== {p.upper()} ({cmd}) =====", flush=True)
        if cmd == "fast":
            stage_raw(p); analyse(p); res[p] = compare(p, "fast", 1e-9, 1e-12)
        elif cmd == "full":
            experiments(p, jobs); analyse(p); res[p] = compare(p, "full", 1e-6, 1e-9)
        elif cmd == "compare":
            res[p] = compare(p, "fast", 1e-9, 1e-12)
    print("\nSUMMARY:", {p: ("OK" if v else "DIFF") for p, v in res.items()})
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
