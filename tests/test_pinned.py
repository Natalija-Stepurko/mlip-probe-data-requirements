"""Pins what the committed raw draws depend on: seeds, hyperparameters, metrics and every random
stream. A change to any of them would change results/ without any other test noticing."""
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from mlip_probe import config, crossval, curves
from mlip_probe.analysis import read_ceilings
from mlip_probe.config import Design
from mlip_probe.probes import make_probe, score
from mlip_probe.results_io import CV_RAW_COLS, read_rows, write_rows
from mlip_probe.sampling import frozen_split, train_draws
from mlip_probe.site.build import render


def test_constants_are_the_published_ones():
    assert (config.DRAW_SEED, config.PROBE_SEED, config.CV_SEED) == (7, 42, 11)
    assert (config.P90_P10_OVER_SD, config.MAX_SPREAD, config.MAX_GAP, config.MIN_PAIRS) == (2.5631, 0.05, 0.05, 3)
    assert config.DATASET_COLUMNS == {"formation_energy/atom": "formation_energy_per_atom"}


def test_probe_hyperparameters():
    p = make_probe("reg", "xgboost", 1).get_params()
    assert {k: p[k] for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                              "random_state")} == dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                                                       subsample=0.8, colsample_bytree=0.8, random_state=42)


def test_scores_are_r2_and_macro_f1():
    assert score("clf", np.array(["a", "a", "a", "b"]), np.array(["a", "a", "b", "b"])) == pytest.approx(11 / 15)
    assert score("reg", np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])) == pytest.approx(0.5)


def test_frozen_split_and_train_draws_are_pinned():
    d = Design(frozen_test_n=100)
    y = np.arange(1000.0)
    y[::7] = np.nan
    test, pool = frozen_split("formation_energy/atom", "reg", y, d)
    draws = train_draws("formation_energy/atom", pool, 50, d)
    digest = hashlib.sha256(np.concatenate([test, pool, *draws]).astype(np.int64).tobytes()).hexdigest()[:16]
    assert (len(test), len(pool), len(draws), digest) == (100, 757, 15, "2acce9054ccce915")


def _fingerprint_probe(task, method, X_train, y_train, X_eval, n_jobs):
    # Depends on which rows were fitted and on the order of the evaluated rows, so the "score"
    # below fingerprints every index the design chose, with no estimator involved.
    return np.arange(len(X_eval), dtype=float) + float(np.nansum(np.asarray(y_train, dtype=float)))


def _fingerprint_score(task, y_true, y_pred):
    return float(np.dot(np.asarray(y_true, dtype=float), np.asarray(y_pred) % 97) % 1000)


RAW_DIGESTS = {"curves_test_orb.csv": "84460fc3f18a09bd", "curves_train_orb.csv": "9698181aebb8f0bc",
               "cv_vs_holdout_orb.csv": "2b659220895c9cfa"}


def test_every_random_stream_is_pinned(world, design, tmp_path, monkeypatch):
    for module in (curves, crossval):
        monkeypatch.setattr(module, "fit_predict", _fingerprint_probe)
        monkeypatch.setattr(module, "score", _fingerprint_score)
    inp = world["orb"]
    curves.run_train_axis(inp, tmp_path, ["band_gap"], 1, design)
    curves.run_test_axis(inp, tmp_path, ["band_gap"], 1, design)
    crossval.run(inp, tmp_path, ["band_gap"], n_jobs=1, design=design)
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((tmp_path / "raw").iterdir())}
    assert digests == RAW_DIGESTS


def test_page_template_markers_are_checked():
    with pytest.raises(ValueError, match="without a generator"):
        render("<!--{{a}}--><!--{{b}}-->", {"a": "x"})
    with pytest.raises(ValueError, match="without a template marker"):
        render("<!--{{a}}-->", {"a": "x", "b": "y"})


def test_ceilings_round_trip_exactly(tmp_path, world):
    curves.record_ceilings(tmp_path, replace(world["orb"], ceilings={("band_gap", "xgboost"): 0.1 + 0.2}))
    assert read_ceilings(tmp_path, "orb") == {("band_gap", "xgboost"): 0.1 + 0.2}


def test_cv_summary_follows_from_raw(tmp_path):
    rows = [dict(arm="orb", target="band_gap", n=100, draw=d, estimator="cv", estimate="0", lo=lo, hi=hi,
                 truth="0", error=e, covered=c)
            for d, (e, lo, hi, c) in enumerate([("0.1", "-0.2", "0.2", 1), ("-0.3", "0.1", "0.5", 0)])]
    write_rows(tmp_path / "raw" / "cv_vs_holdout_orb.csv", CV_RAW_COLS, rows)
    crossval.summarise(tmp_path)
    (r,) = read_rows(tmp_path / "cv_vs_holdout.csv")
    assert (r["bias"], r["rmse"], r["coverage"], r["mean_width"]) == ("-0.100000", "0.223607", "0.5000", "0.400000")
