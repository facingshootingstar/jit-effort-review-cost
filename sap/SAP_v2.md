# Statistical Analysis Plan v2: new corpora for Papers 2–4 (frozen 2026-10-22, before any v2 experiment was run)

Paper 1 keeps the NASA MDP design of `SAP.md`. Papers 2–4 move to three other public corpora, so that each paper has its own data, question and unit of analysis. The NASA results for Papers 2–4 under `SAP.md` are archived and not reported in v2.

## Shared
- **Seed and data.**
  - Seed 20261018. Data are local copies in `data/raw/`; the SHA-256 hashes are in `data/raw/MANIFEST.sha256`.
- **Cleaning.** Same rules as `core.data.load`:
  - drop rows with missing values;
  - drop rows with negative values in non-negative metrics. Net churn in AEEEM is signed and is exempt;
  - drop every member of each group of conflicting rows (identical metrics, different labels);
  - drop exact duplicates.
- **Learners and preprocessing.** `core.models.make`, fixed hyperparameters, log1p on non-negative columns, and scaling for LR, kNN and MLP.
- **Statistical tests.**
  - Two-sided exact or normal-approximation Wilcoxon signed-rank test, Holm within each pre-declared family, α = 0.05.
  - Cliff's δ with the Romano thresholds.
  - Percentile bootstrap 95% CI on the median paired difference (10,000 resamples).
- **Effort.** Effort-aware metrics use Popt (Kamei et al. 2013) and Recall@20% effort from `core.metrics.effort_metrics`.

## Paper 2: JIT effort-aware value vs cost (Kamei corpus: bugzilla, columba, jdt, mozilla, platform, postgres)
- **Effort definition.** Effort = absolute churn LA+LD. The file stores LA and LD relative to LT, and LT relative to NF, so churn is reconstructed exactly as in the corpus authors' scripts. LA and LD stay in the feature set in their relative form.
- **Units: time-wise windows.** Commits are grouped by calendar month.
  - Training uses 6 consecutive months, followed by a gap of 2 months (label latency), and testing uses the next 3 months. The window advances by 3 months.
  - A window is kept only if its training part has ≥ 20 defect-inducing commits and its test part has ≥ 30 commits and ≥ 5 defect-inducing commits.
- **Methods.** All 8 classifiers of `core.models.ALL`, each ranked in two ways:
  - by probability (`prob`);
  - by predicted density p/(churn+1) (`density`).
- **Baselines.**
  - ManualUp: ascending churn.
  - EALR: linear regression of y/(churn+1) on the same log-transformed features, then ranking by the prediction.
- **Dependent variables.**
  - Popt and Recall@20% effort (primary).
  - IFA: the number of clean commits ranked above the first defect-inducing one, under the density ordering.
  - AUC.
  - Cost:
    - training time;
    - prediction latency per 1,000 commits, as the median of 5 timed calls after a warm-up;
    - pickled model size.
  - Everything is measured single-threaded.
- **RQ2.1: does density ranking beat probability ranking for each classifier?**
  - Wilcoxon on the windows, Holm over the 8 classifiers, separately per metric.
- **RQ2.2: which method is best?** Each method (classifier×density, EALR, ManualUp) is compared with LR-density.
  - Wilcoxon on the windows, Holm over the 9 comparisons, per primary metric.
  - Scott-Knott ESD rank across all methods on the window-level Popt.
- **RQ2.3: does the extra cost pay off?** Each classifier's training-time and latency ratio vs LR is set against its Popt difference vs LR.
  - Spearman correlation, over the 8 classifiers, between the median training time and the median Popt.
- **Robustness of the window-as-unit tests.** The windows are not independent across a project, so each RQ2.1/RQ2.2 test is repeated with:
  - (a) a linear mixed model, metric ~ method + (1 | project/window), using the windows of the two methods compared;
  - (b) project-level medians (n = 6), reported descriptively as the number of projects agreeing in sign.
- **Sensitivity.** A 12-month training window (exploratory).

## Paper 3: label noise vs real release-to-release shift (Jureczko/PROMISE CK, 12 projects)
- **Units: consecutive release pairs (v_i → v_{i+1}).**
  - Projects: ant, camel, ivy, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces, prop.
  - A pair is excluded if either release has a defect ratio < 5% or > 95%, or fewer than 5 defective classes.
- **Learners.** LR, NB, DT, RF, LGBM, kNN.
- **Conditions per pair.**
  - (W) Within-release reference: 5×2 stratified CV on v_{i+1}.
  - (V) Cross-version: train on clean v_i, test on v_{i+1}.
  - (P) Cross-project: train on each of 3 releases drawn at random (seeded) from other projects, test on v_{i+1}, and take the mean.
  - (N) Noise injected into v_i's training labels before cross-version testing:
    - types: symmetric flips and missed defects (1→0 only);
    - rates: 10, 20 and 30%;
    - 5 noise seeds per type and rate, averaged per pair.
  - (F) Noise filter: a 5-fold out-of-fold RF classification filter (Brodley & Friedl) removes training rows whose out-of-fold probability of their own label is < 0.5. It is applied to clean v_i (V+F) and to 20%-noisy v_i (N20+F).
- **Dependent variables.** AUC (primary), MCC, recall at threshold 0.5, and Popt (effort = LOC).
- **RQ3.1: how large is real release-to-release shift?** The AUC gap W − V, tested per learner (Wilcoxon over the pairs, Holm over the 6 learners).
- **RQ3.2: how does injected noise compare with real shift?**
  - The AUC loss V − N for each type × rate × learner, Holm per type × rate over the learners.
  - The noise rate at which the median noise loss exceeds the median shift loss W − V, reported descriptively.
- **RQ3.3: is cross-project much worse than cross-version?** V − P per learner, Holm over the 6 learners.
- **RQ3.4: does a classification filter help?**
  - (V+F) − V and (N20+F) − N20 for each noise type, per learner, Holm over the 6 learners and per metric (AUC and recall).

## Paper 4: explanation stability and minimal metric sets (AEEEM change metrics: 5 systems; Zimmermann Eclipse 2.0/2.1/3.0 code metrics)
- **Data and resampling.**
  - 8 datasets.
  - 5×5 repeated stratified CV with stored folds.
  - Label: bugs > 0 (AEEEM) and post > 0 (Eclipse). Eclipse keeps `pre` (pre-release defects) as a feature.
- **Models.** LR, RF, LGBM.
- **Feature sets.**
  - "all".
  - "filtered": AutoSpearman-style unsupervised filter, |Spearman ρ| > 0.7, then VIF > 5. It is fitted once per dataset and never sees labels.
- **Explanations.**
  - SHAP: mean |SHAP| over ≤ 1,000 training rows.
  - Permutation importance: AUC, 5 repeats, on the test fold.
  - Built-in importance: |coef|, impurity importance or gain.
  - On "all", only SHAP is computed, because permutation importance over 199 Eclipse metrics is out of computational budget.
- **Stability measures.**
  - Kendall's W of the importance ranks across the 25 folds.
  - The mean pairwise top-3 Jaccard across folds.
  - The top-1 agreement rate.
- **RQ4.1: which method gives the most stable explanations?**
  - On "filtered", Friedman test over the 3 methods.
  - Unit: the dataset × model case (n = 24).
  - Pairwise Wilcoxon on those cases, Holm over 3 comparisons.
- **RQ4.2: does removing correlated metrics stabilise SHAP?**
  - SHAP W on "filtered" vs "all", Wilcoxon over the 24 cases.
- **RQ4.3: how few metrics are needed?**
  - Retrain on the SHAP top-k filtered metrics (ranked on the training fold), k ∈ {1, 2, 3, 5, 8}.
  - Measure the AUC and PR-AUC loss vs all filtered metrics.
  - Minimal k: the smallest k whose median per-case AUC loss is ≤ 0.01 with a bootstrap upper CI bound ≤ 0.02 (non-inferiority margin).
  - Wilcoxon per k over the cases, Holm over k.
- **RQ4.4: do explanations transfer across releases and systems?**
  - Eclipse: the top-5 SHAP Jaccard between models trained on different releases vs the within-release top-5 Jaccard.
  - AEEEM: how often each change metric is in the top 3 across the 5 systems.
  - Descriptive, with bootstrap CIs.
