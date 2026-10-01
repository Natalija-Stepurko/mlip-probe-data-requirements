"""On a small dataset, is k-fold cross-validation a better estimate of a probe's score than one
80/20 split or a handful of repeated splits?

Its results feed the cross-validation section of docs/index.html; this experiment is not in the paper.

Per arm, target and dataset size n (XGBoost probe): cv_draws disjoint draws of n rows from the
training pool. The truth for a draw is the score of a probe fitted on all n rows and scored on
the frozen test set. Three estimators try to recover it from the n rows alone, each with a 95 %
interval:

    single    one 80/20 split; interval = bootstrap over the held-out rows
    repeated  cv_repeat_splits 80/20 splits; interval = t-interval of their mean
    cv        cv_folds-fold cross-validation scored on the pooled out-of-fold predictions;
              interval = bootstrap over the out-of-fold pairs

The summary gives, per (arm, target, n, estimator), the bias, sd and RMSE of the error, the
coverage of the interval and its mean width.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .config import CV_SEED, DEFAULT_DESIGN, Design, task_of
from .dataset import Inputs
from .probes import fit_predict, score
from .results_io import CV_RAW_COLS, CV_SUMMARY_COLS, append_rows, fmt, group, read_rows, write_rows
from .sampling import frozen_split, target_seed

log = logging.getLogger(__name__)

PERMUTATION_DRAW = 10**6   # draw index of the stream that permutes the pool; per-draw streams use 0..k-1
MIN_RESAMPLES = 10         # fewer usable bootstrap resamples give no percentile interval
Estimate = tuple[float, float, float]   # estimate, interval low, interval high


def _rng(target: str, n: int, draw: int) -> np.random.Generator:
    return np.random.default_rng([CV_SEED, target_seed(target), n, draw])


def _bootstrap_interval(task: str, y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator,
                        design: Design) -> tuple[float, float]:
    """Percentile interval; resamples holding a single class are skipped (macro-F1 is undefined)."""
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(design.cv_bootstrap):
        b = rng.choice(idx, len(idx), replace=True)
        if task == "clf" and len(np.unique(y_true[b])) < 2:
            continue
        vals.append(score(task, y_true[b], y_pred[b]))
    if len(vals) < MIN_RESAMPLES:
        return float("nan"), float("nan")
    return float(np.quantile(vals, design.cv_alpha / 2)), float(np.quantile(vals, 1 - design.cv_alpha / 2))


def _split(task: str, y: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    strat = y if task == "clf" and min(np.unique(y, return_counts=True)[1]) >= 2 else None
    return train_test_split(np.arange(len(y)), test_size=0.2, random_state=int(rng.integers(2**31)), stratify=strat)


def _single(task: str, X: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_jobs: int,
            design: Design) -> Estimate | None:
    tr, te = _split(task, y, rng)
    p = fit_predict(task, "xgboost", X[tr], y[tr], X[te], n_jobs)
    if p is None:
        return None
    return (score(task, y[te], p), *_bootstrap_interval(task, y[te], p, rng, design))


def _repeated(task: str, X: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_jobs: int,
              design: Design) -> Estimate | None:
    scores = []
    for _ in range(design.cv_repeat_splits):
        tr, te = _split(task, y, rng)
        p = fit_predict(task, "xgboost", X[tr], y[tr], X[te], n_jobs)
        if p is not None:
            scores.append(score(task, y[te], p))
    if len(scores) < 2:
        return None
    m, sd = float(np.mean(scores)), float(np.std(scores, ddof=1))
    half = t_dist.ppf(1 - design.cv_alpha / 2, len(scores) - 1) * sd / np.sqrt(len(scores))
    return m, m - half, m + half


def _kfold(task: str, X: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_jobs: int,
           design: Design) -> Estimate | None:
    folds = (StratifiedKFold if task == "clf" else KFold)(design.cv_folds, shuffle=True,
                                                           random_state=int(rng.integers(2**31)))
    try:
        splits = list(folds.split(X, y))
    except ValueError:   # StratifiedKFold refuses when every class has fewer members than folds
        return None
    oof = np.empty(len(y), dtype=object if task == "clf" else float)
    for tr, te in splits:
        p = fit_predict(task, "xgboost", X[tr], y[tr], X[te], n_jobs)
        if p is None:
            return None
        oof[te] = p
    return (score(task, y, oof), *_bootstrap_interval(task, y, oof, rng, design))


def one_draw(task: str, X: np.ndarray, y: np.ndarray, X_ref: np.ndarray, y_ref: np.ndarray,
             rng: np.random.Generator, n_jobs: int, design: Design) -> tuple[float, dict[str, Estimate]] | None:
    """Truth and the estimates for one draw. The estimators share one stream in a fixed order."""
    pred = fit_predict(task, "xgboost", X, y, X_ref, n_jobs)
    if pred is None:
        return None
    truth = score(task, y_ref, pred)
    single = _single(task, X, y, rng, n_jobs, design)
    if single is None:
        return None
    estimates = {"single": single}
    for name, estimator in (("repeated", _repeated), ("cv", _kfold)):
        e = estimator(task, X, y, rng, n_jobs, design)
        if e is not None:
            estimates[name] = e
    return truth, estimates


def run(inputs: Inputs, out: Path, targets: Sequence[str], draws: int | None = None, n_jobs: int = 1,
        design: Design = DEFAULT_DESIGN) -> None:
    """raw/cv_vs_holdout_{arm}.csv; resumable per (target, n)."""
    draws = design.cv_draws if draws is None else draws
    path = out / "raw" / f"cv_vs_holdout_{inputs.arm}.csv"
    done = {(r["target"], int(r["n"])) for r in read_rows(path)}
    for t in targets:
        task, y = task_of(t), inputs.targets[t]
        test, pool = frozen_split(t, task, y, design)
        X_ref, y_ref = inputs.X[test], y[test]
        for n in design.cv_sizes:
            k = min(draws, len(pool) // n)
            if (t, n) in done or k == 0:
                continue
            perm = pool[_rng(t, n, PERMUTATION_DRAW).permutation(len(pool))]
            rows, t0 = [], time.time()
            for d in range(k):
                rows_d = perm[d * n:(d + 1) * n]
                res = one_draw(task, inputs.X[rows_d], y[rows_d], X_ref, y_ref, _rng(t, n, d), n_jobs, design)
                if res is not None:
                    rows += _raw_rows(inputs.arm, t, n, d, *res)
            append_rows(path, CV_RAW_COLS, rows)
            log.info("[%s] %s n=%d: %d draws (%.0fs)", inputs.arm, t, n, k, time.time() - t0)


def _raw_rows(arm: str, t: str, n: int, d: int, truth: float, estimates: dict[str, Estimate]) -> list[dict]:
    return [dict(arm=arm, target=t, n=n, draw=d, estimator=name, estimate=f"{est:.6f}", lo=fmt(lo), hi=fmt(hi),
                 truth=f"{truth:.6f}", error=f"{est - truth:.6f}",
                 covered=int(lo <= truth <= hi) if not math.isnan(lo) else "")
            for name, (est, lo, hi) in estimates.items()]


def summarise(out: Path) -> None:
    """cv_vs_holdout.csv from every raw/cv_vs_holdout_{arm}.csv."""
    raw = sorted((out / "raw").glob("cv_vs_holdout_*.csv"))
    if not raw:
        raise FileNotFoundError(f"no cross-validation runs in {out}/raw; run `mlip-probe cv` first")
    rows = []
    for p in raw:
        for (arm, t, n, est), g in sorted(group(read_rows(p), ["arm", "target", "n", "estimator"]).items()):
            err = np.array([float(r["error"]) for r in g])
            cov = [int(r["covered"]) for r in g if r["covered"] != ""]
            width = [float(r["hi"]) - float(r["lo"]) for r in g if r["lo"] != ""]
            rows.append(dict(arm=arm, target=t, n=int(n), estimator=est, draws=len(g),
                             bias=f"{err.mean():.6f}", sd=f"{err.std(ddof=1):.6f}" if len(err) > 1 else "",
                             rmse=f"{np.sqrt((err ** 2).mean()):.6f}",
                             coverage=f"{np.mean(cov):.4f}" if cov else "",
                             mean_width=f"{np.mean(width):.6f}" if width else ""))
    write_rows(out / "cv_vs_holdout.csv", CV_SUMMARY_COLS, rows)
    log.info("  cv_vs_holdout.csv: %d rows", len(rows))
