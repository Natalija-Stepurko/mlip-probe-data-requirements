"""The result tables behind the page, loaded once from results/: summaries, raw draws, paired
gain, split fraction, thresholds and the cross-validation summary. Curves are kept for XGBoost
only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARMS
from .targets import ordered

ARM_LABEL = {"orb": "ORB", "uma": "UMA", "magpie": "Magpie"}
MODEL_NAME = {"orb": "ORB-v3", "uma": "UMA-S"}


def _key(target: str) -> str:
    return target.replace("/", "_per_")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    if "target" in d.columns:
        d["target"] = d["target"].map(_key)
    return d


class Results:
    def __init__(self, indir: Path):
        self.indir = Path(indir)
        self.summary = {}      # arm -> XGBoost summary rows, both axes
        self.raw = {}          # (arm, axis) -> per-draw XGBoost scores
        for arm in ARMS:
            s = _read(self.indir / f"summary_{arm}.csv")
            if len(s):
                self.summary[arm] = s[s.method == "xgboost"].copy()
            for axis in ("train", "test"):
                r = _read(self.indir / "raw" / f"curves_{axis}_{arm}.csv")
                if len(r):
                    r = r[r.method == "xgboost"].copy()
                    r["score"] = pd.to_numeric(r["score"], errors="coerce")
                    self.raw[(arm, axis)] = r
        self.gain = _read(self.indir / "paired_gain.csv")
        self.split = _read(self.indir / "split_fraction.csv")
        self.cv = _read(self.indir / "cv_vs_holdout.csv")
        self.thresholds = _read(self.indir / "thresholds.csv")

    def curve(self, arm: str, axis: str, target: str | None = None) -> pd.DataFrame:
        s = self.summary.get(arm)
        if s is None:
            return pd.DataFrame()
        s = s[s.axis == axis]
        if target is not None:
            s = s[s.target == target]
        return s.sort_values(["target", "n"])

    def targets(self) -> list[str]:
        present = set()
        for arm in ARMS:
            present |= set(self.curve(arm, "train").target)
        return ordered(present)

    def ceiling(self, arm: str, target: str) -> float | None:
        c = self.curve(arm, "train", target)
        return float(c.ceiling.iloc[0]) if len(c) else None

    def extremes(self, arm: str, axis: str) -> pd.DataFrame:
        """worst / best / draw count per (target, n) from the raw draws."""
        r = self.raw.get((arm, axis))
        if r is None:
            return pd.DataFrame(columns=["target", "n", "worst", "best", "draws"])
        return (r.groupby(["target", "n"]).score.agg(worst="min", best="max", draws="count")
                 .reset_index())

    def gain_rows(self, arm: str) -> pd.DataFrame:
        """target, n, k, med, p10, p90, win — the embedding minus Magpie on paired draws."""
        if not len(self.gain):
            return pd.DataFrame(columns=["target", "n", "k", "med", "p10", "p90", "win"])
        g = self.gain[self.gain.arm == arm]
        return (g.rename(columns={"n_pairs": "k", "median": "med", "frac_positive": "win"})
                 [["target", "n", "k", "med", "p10", "p90", "win"]].sort_values(["target", "n"]))

    def cv_cells(self, arm: str) -> pd.DataFrame:
        """One row per (target, n): the three estimators side by side.
        red = 1 - RMSE(cv) / RMSE(repeated holdout)."""
        cols = ["target", "n", "draws", "holdout_rmse", "rh_rmse", "cv_rmse", "red",
                "ho_cov", "rh_cov", "cv_cov", "ho_sd", "rh_sd", "cv_sd"]
        if not len(self.cv):
            return pd.DataFrame(columns=cols)
        c = self.cv[self.cv.arm == arm]
        if not len(c):
            return pd.DataFrame(columns=cols)
        w = c.pivot_table(index=["target", "n"], columns="estimator",
                          values=["rmse", "coverage", "sd", "draws"], aggfunc="first")
        out = pd.DataFrame({
            "draws": w[("draws", "cv")] if ("draws", "cv") in w else np.nan,
            "holdout_rmse": w.get(("rmse", "single")), "rh_rmse": w.get(("rmse", "repeated")),
            "cv_rmse": w.get(("rmse", "cv")),
            "ho_cov": w.get(("coverage", "single")), "rh_cov": w.get(("coverage", "repeated")),
            "cv_cov": w.get(("coverage", "cv")),
            "ho_sd": w.get(("sd", "single")), "rh_sd": w.get(("sd", "repeated")), "cv_sd": w.get(("sd", "cv")),
        }).reset_index()
        out["red"] = 1.0 - out.cv_rmse / out.rh_rmse
        return out[cols].sort_values(["target", "n"])

    def at(self, arm: str, axis: str, target: str, col: str, n: float) -> float | None:
        """`col` of the curve at size n, log-linear between measured sizes."""
        c = self.curve(arm, axis, target)
        c = c[c[col].notna()]
        if not len(c):
            return None
        xs, ys = np.log(c.n.to_numpy(float)), c[col].to_numpy(float)
        return float(np.interp(np.log(n), xs, ys))
