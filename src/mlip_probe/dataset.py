"""Inputs of one arm from the paper's dataset release.

The release is a directory DIR with

    DIR/data/materials.parquet                             mp_id and the nine targets (canonical row order)
    DIR/embeddings/node/orb-node-backbone.parquet          mp_id, emb: ORB-v3 pooled node embedding
    DIR/results/magpie_features.parquet                    mp_id, features: 132 Magpie statistics
    DIR/results/probe_metrics.parquet                      full-corpus probe scores (ORB and UMA ceilings)
    DIR/results/controls/composition_baseline.parquet      full-corpus Magpie scores (Magpie ceilings)

and the UMA-S embeddings ship in a separate access-gated release UMA_DIR (pass --uma-dataset;
default DIR/uma):

    UMA_DIR/embeddings/node/uma-node-ang_norm-backbone.parquet
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARMS, DATASET_COLUMNS, TARGETS, task_of

log = logging.getLogger(__name__)

Ceilings = dict[tuple[str, str], float]
ESTIMATOR_METHOD = {"xgb": "xgboost", "linear": "linear"}   # probe_metrics spelling -> METHODS


@dataclass
class Inputs:
    """Canonical material ids, one arm's features in that order, the target arrays in that
    order, and the arm's full-corpus scores {(target, method): score}."""

    mp_ids: np.ndarray
    X: np.ndarray | None
    targets: dict[str, np.ndarray]
    ceilings: Ceilings
    arm: str

    def __post_init__(self) -> None:
        if self.X is not None and len(self.X) != len(self.mp_ids):
            raise ValueError(f"{self.arm}: {len(self.X)} feature rows for {len(self.mp_ids)} canonical ids")


def align(X: np.ndarray, ids: Sequence[str], canon: np.ndarray, arm: str) -> np.ndarray:
    """Rows of X reordered to the canonical ids. A missing id is an error: a subset of the
    materials would change every draw of every arm."""
    ids = np.asarray(ids).astype(str)
    if len(ids) == len(canon) and (ids == canon).all():
        return X
    pos = {m: i for i, m in enumerate(ids)}
    missing = [m for m in canon if m not in pos]
    if missing:
        raise ValueError(f"{arm}: {len(missing)} canonical materials absent (e.g. {missing[:3]})")
    log.info("  %s: rows reindexed to the canonical order", arm)
    return X[np.fromiter((pos[m] for m in canon), dtype=np.int64, count=len(canon))]


def clean_labels(raw: dict) -> dict[str, np.ndarray]:
    """Regression targets as float (NaN = unlabelled); class labels as str or None."""
    out = {}
    for t, y in raw.items():
        if task_of(t) == "reg":
            out[t] = np.asarray(y, dtype=float)
        else:
            out[t] = np.array([None if v is None or v == "" or (isinstance(v, float) and np.isnan(v))
                               else str(v) for v in y], dtype=object)
    return out


def uma_release_dir(dataset_dir: Path, uma_dir: Path | None = None) -> Path | None:
    if uma_dir:
        return Path(uma_dir)
    cand = Path(dataset_dir) / "uma"
    return cand if cand.exists() else None


def _embedding(path: Path) -> tuple[np.ndarray, pd.Series]:
    e = pd.read_parquet(path)
    e = e[e["level"] == "node"]
    return np.stack(e["emb"].to_numpy()).astype(np.float32), e["mp_id"]


def _features(arm: str, dataset_dir: Path, uma_dir: Path | None) -> tuple[np.ndarray, pd.Series]:
    if arm == "orb":
        return _embedding(dataset_dir / "embeddings" / "node" / "orb-node-backbone.parquet")
    if arm == "uma":
        uma = uma_release_dir(dataset_dir, uma_dir)
        if uma is None:
            raise FileNotFoundError("the UMA arm needs the gated UMA release: pass --uma-dataset DIR")
        return _embedding(uma / "embeddings" / "node" / "uma-node-ang_norm-backbone.parquet")
    if arm == "magpie":
        e = pd.read_parquet(dataset_dir / "results" / "magpie_features.parquet")
        return np.stack(e["features"].to_numpy()).astype(np.float32), e["mp_id"]
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def reference_ceilings(arm: str, dataset_dir: Path) -> Ceilings:
    """The paper's full-corpus score of each probe on this arm (R^2, or macro-F1 for classes)."""
    if arm in ("orb", "uma"):
        return _probe_ceilings(arm, dataset_dir / "results" / "probe_metrics.parquet")
    if arm == "magpie":
        return _composition_ceilings(dataset_dir / "results" / "controls" / "composition_baseline.parquet")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def _probe_ceilings(arm: str, path: Path) -> Ceilings:
    pm = pd.read_parquet(path)
    sel = (pm["model"].str.upper() == arm.upper()) & (pm["read_out"] == "node") & (pm["layer"] == "backbone")
    if arm == "uma":
        sel &= pm["family"] == "ang_norm"
    ceil = {}
    for _, r in pm[sel].iterrows():
        t = r["target"]
        method = ESTIMATOR_METHOD.get(r["estimator"])
        if t not in TARGETS or not method:
            continue
        v = pd.to_numeric(r["macro_f1"] if task_of(t) == "clf" else r["r2"], errors="coerce")
        if pd.notna(v):
            ceil[(t, method)] = float(v)
    return ceil


def _composition_ceilings(path: Path) -> Ceilings:
    cb = pd.read_parquet(path)
    ceil = {}
    for _, r in cb.iterrows():
        if r["target"] not in TARGETS:
            continue
        for method, col in (("xgboost", "comp_xgb"), ("linear", "comp_linear")):
            v = pd.to_numeric(r[col], errors="coerce")
            if pd.notna(v):
                ceil[(r["target"], method)] = float(v)
    return ceil


def load_labels(dataset_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Canonical material ids and the nine cleaned target arrays from materials.parquet."""
    mat = pd.read_parquet(Path(dataset_dir) / "data" / "materials.parquet")
    canon = mat["mp_id"].astype(str).to_numpy()
    return canon, clean_labels({t: mat[DATASET_COLUMNS.get(t, t)].to_numpy() for t in TARGETS})


def load(arm: str, dataset_dir: Path, uma_dir: Path | None = None, need_features: bool = True) -> Inputs:
    """Canonical ids and labels from materials.parquet, the arm's features in that order, and its
    reference ceilings."""
    dataset_dir = Path(dataset_dir)
    canon, targets = load_labels(dataset_dir)
    X = None
    if need_features:
        feats, ids = _features(arm, dataset_dir, uma_dir)
        X = align(feats, ids, canon, arm)
    return Inputs(canon, X, targets, reference_ceilings(arm, dataset_dir), arm)
