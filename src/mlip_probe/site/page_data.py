"""The DATA object inlined in docs/index.html.

The page's prose numbers and its interactive figures both read this object, so the two cannot
drift. The page reports the XGBoost probe; the linear probe's rows are in results/.
"""
from __future__ import annotations

import math

from ..config import ARMS, CLASSIFICATION, EMBEDDINGS
from .data import Results
from .targets import label


def _round6(v: float | None) -> float | None:
    """The precision of results/; the page rounds once, at display."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 6)


def build(results: Results) -> dict:
    targets = results.targets()
    page = {"targets": targets,
            "labels": {t: label(t) for t in targets},
            "metric": {t: ("macro-F1" if t in CLASSIFICATION else "R²") for t in targets},
            "ceilings": {}, "train": {}, "test": {}, "gain": {}, "cv": {}}
    for arm in ARMS:
        tr, te = results.curve(arm, "train"), results.curve(arm, "test")
        if not len(tr):
            continue
        page["ceilings"][arm] = {t: _round6(g.ceiling.iloc[0]) for t, g in tr.groupby("target")}
        for key, frame in (("train", tr), ("test", te)):
            page[key][arm] = [{"target": row.target, "n": int(row.n), "reps": int(row.reps),
                               "median": _round6(row.median), "p10": _round6(row.p10),
                               "p90": _round6(row.p90), "spread": _round6(row.spread), "gap": _round6(row.gap)}
                              for row in frame.itertuples()]
    for arm in EMBEDDINGS:
        g = results.gain_rows(arm)
        if len(g):
            page["gain"][arm] = [{"target": x.target, "n": int(x.n), "k": int(x.k), "med": _round6(x.med),
                                  "p10": _round6(x.p10), "p90": _round6(x.p90), "win": _round6(x.win)}
                                 for x in g.itertuples()]
        c = results.cv_cells(arm)
        if len(c):
            page["cv"][arm] = [{"target": x.target, "n": int(x.n), "draws": int(x.draws),
                                "rh_rmse": _round6(x.rh_rmse), "cv_rmse": _round6(x.cv_rmse),
                                "holdout_rmse": _round6(x.holdout_rmse), "red": _round6(x.red),
                                "cv_cov": _round6(x.cv_cov), "rh_cov": _round6(x.rh_cov),
                                "ho_cov": _round6(x.ho_cov), "ho_sd": _round6(x.ho_sd),
                                "rh_sd": _round6(x.rh_sd), "cv_sd": _round6(x.cv_sd)}
                               for x in c.itertuples()]
    return page
