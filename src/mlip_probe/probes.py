"""The two probes of the paper, with its hyperparameters: XGBoost and a standardised linear model
(ridge for regression, logistic regression for classes). Scores are R^2 and macro-F1."""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROBE_SEED


def make_probe(task: str, method: str, n_jobs: int) -> Pipeline | xgb.XGBModel:
    if method == "xgboost":
        common = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs, random_state=PROBE_SEED)
        if task == "reg":
            return xgb.XGBRegressor(**common)
        return xgb.XGBClassifier(eval_metric="mlogloss", **common)
    if task == "reg":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 6)))
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def fit_predict(task: str, method: str, X_train: np.ndarray, y_train: np.ndarray,
                X_eval: np.ndarray, n_jobs: int) -> np.ndarray | None:
    """Predictions on X_eval, or None when a classification draw holds a single class."""
    if task == "clf":
        classes, y_enc = np.unique(y_train, return_inverse=True)
        if len(classes) < 2:
            return None
        return classes[make_probe(task, method, n_jobs).fit(X_train, y_enc).predict(X_eval)]
    return make_probe(task, method, n_jobs).fit(X_train, y_train).predict(X_eval)


def score(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if task == "clf":
        return float(f1_score(y_true, y_pred, average="macro"))
    return float(r2_score(y_true, y_pred))
