# Effort-aware just-in-time defect prediction: value vs cost — reproduction package

Data and code that reproduce every number, table and figure of the paper (Paper 2): effort-aware
just-in-time (JIT) defect prediction on 168 time-wise windows of six projects of the Kamei et al. (2013)
corpus, comparing eight learners (LR, NB, DT, RF, LightGBM, XGBoost, kNN, MLP) under probability and
density ranking with the training-free baselines ManualUp, ManualDown, EALR and a CBS+-style ranking,
and relating the inspection value of each ranking to its training and scoring cost and to a review-cost
model.

## Contents

| Path | What it is |
|---|---|
| `data/raw/jit/` | The six Kamei et al. (2013) JIT change data sets (bugzilla, columba, jdt, mozilla, platform, postgres), as distributed with Fu & Menzies (2017), *Revisiting unsupervised learning for defect prediction* (GitHub: WeiFoo/RevisitUnsupervised). `data/raw/MANIFEST.sha256` holds their hashes. |
| `core/` | Loading and cleaning (`corpora.py`), time-wise windows (`folds.py`: 6 months training, 2-month gap, 3 months test, step 3), learners (`models.py`), metrics (Popt, Recall@20%, PCI@20%, ...), statistics. Seed 20261018. |
| `experiments/run_p2_jit.py` | Main run (`perf`) and the additional runs (`perf12`, `hybrid`, `timing`, `info`, `gap`, `mask`, `trees`, `seeds`); see the docstring. |
| `analysis/p2jit.py` | Statistics, LaTeX tables and figures from the raw results. |
| `sap/` | The frozen statistical analysis plan (`SAP_v2.md`, hash in `SAP_v2.sha256`; the sections *Shared* and *Paper 2* apply) and the deviations that concern this paper (`deviations_p2.md`). |
| `reference/` | The outputs of the original run: raw per-window results, `summary/p2jit.json`, `paper2/tables.tex`, figures. |
| `repro.py` | One entry point that runs everything and compares it with `reference/`. |

## Run

Python 3.12 (original runs: 3.12.10, Windows 11).

```bash
pip install -r requirements.txt
```

```bash
python repro.py check
```

```bash
python repro.py fast
```

`fast` takes the shipped raw results (`reference/raw/`), re-runs the full statistical analysis and checks
that every JSON value, the LaTeX tables and the figures match `reference/`. The report is written to
`out/` and ends with `RESULT: ALL REPRODUCED` or a list of differences.

```bash
python repro.py full -j 8
```

`full` re-trains every model on every window from `data/`, then analyses and compares. It takes hours.
Tree learners may differ in the last digits on another OS or CPU, and the training-time and latency
measurements (`p2jit_cost.csv`, RQ2.3) depend on the hardware; run the `timing` mode alone.

## Cleaning rules

`core/corpora.py` drops rows with missing or negative values. Identical commits are kept (a commit stream
legitimately contains them). Commits with identical metrics but different labels are removed from the
training block of each window only; test blocks are never altered. Effort is the absolute churn LA + LD of
a commit, converted back from the relative values stored in the Kamei files as in the original authors'
R scripts (where LT = 0, LA and LD are already absolute).
