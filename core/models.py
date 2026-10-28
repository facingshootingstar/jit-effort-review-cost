"""Classifier factory and the shared preprocessing (log1p + optional scaling)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 20261018
SCALED = {"LR", "kNN", "MLP"}
ALL = ["LR", "NB", "DT", "RF", "LGBM", "XGB", "kNN", "MLP"]


def make(name, seed=SEED, n_jobs=1, balanced=False):
    cw = "balanced" if balanced else None
    if name == "LR":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed)
    if name == "NB":
        return GaussianNB()
    if name == "DT":
        return DecisionTreeClassifier(min_samples_leaf=5, class_weight=cw, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, n_jobs=n_jobs, class_weight=cw, random_state=seed)
    if name == "LGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, n_jobs=n_jobs,
                              class_weight=cw, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, eval_metric="logloss",
                             n_jobs=n_jobs, random_state=seed)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=7, n_jobs=n_jobs)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=seed)
    raise ValueError(name)


class Prep:
    """log1p on non-negative columns, then StandardScaler for scale-sensitive models."""

    def __init__(self, model_name):
        self.scale = model_name in SCALED

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.logcols = (X >= 0).all(axis=0)
        Z = self._log(X)
        self.sc = StandardScaler().fit(Z) if self.scale else None
        return self

    def _log(self, X):
        Z = np.array(X, dtype=float, copy=True)
        Z[:, self.logcols] = np.log1p(Z[:, self.logcols])
        return Z

    def transform(self, X):
        Z = self._log(np.asarray(X, dtype=float))
        return self.sc.transform(Z) if self.sc is not None else Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)
