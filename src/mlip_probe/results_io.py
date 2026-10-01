"""CSV reading and writing, and the columns of every result file.

Files are written with csv.DictWriter and floats as fixed-decimal strings (six places; the
cross-validation coverage four), so a re-run on the same inputs and machine reproduces every
file byte for byte.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TRAIN_COLS = ["arm", "target", "method", "axis", "n", "rep", "score", "n_test", "n_pool"]
TEST_COLS = ["arm", "target", "method", "axis", "n", "rep", "score", "n_train_fit"]
FROZEN_IDS_COLS = ["target", "mp_id"]
CEILING_COLS = ["arm", "target", "method", "ceiling"]
SUMMARY_COLS = ["arm", "target", "method", "axis", "n", "reps", "median", "p10", "p90", "spread",
                "spread_direct", "ceiling", "ceiling_source", "gap"]
THRESHOLD_COLS = ["arm", "target", "method", "n_stable", "n_saturated", "n_train_max", "test_n_stable",
                  "ceiling", "ceiling_source"]
GAIN_COLS = ["arm", "target", "n", "n_pairs", "median", "p10", "p90", "frac_positive"]
SPLIT_COLS = ["arm", "target", "N", "fraction", "n_train", "n_test", "score", "shortfall", "spread", "sd", "total"]
CV_RAW_COLS = ["arm", "target", "n", "draw", "estimator", "estimate", "lo", "hi", "truth", "error", "covered"]
CV_SUMMARY_COLS = ["arm", "target", "n", "estimator", "draws", "bias", "sd", "rmse", "coverage", "mean_width"]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def append_rows(path: Path, cols: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def write_rows(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(v: object) -> object:
    """Floats to six decimals, None and NaN to an empty cell, anything else unchanged."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.6f}" if isinstance(v, float) else v


def formatted(rows: list[dict]) -> list[dict]:
    return [{k: fmt(v) for k, v in r.items()} for r in rows]


def group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g
