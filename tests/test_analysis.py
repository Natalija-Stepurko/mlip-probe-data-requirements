import numpy as np
import pytest

from mlip_probe.analysis import first_sustained, paired_gain, split_fraction, summarise, summarise_arm
from mlip_probe.config import MIN_PAIRS, P90_P10_OVER_SD
from mlip_probe.results_io import SUMMARY_COLS, TRAIN_COLS, read_rows, write_rows

NAN = float("nan")


def _points(values):
    return [dict(n=n, spread=v) for n, v in values]


def test_first_sustained_needs_every_larger_size():
    assert first_sustained(_points([(50, 0.2), (100, 0.04), (200, 0.06), (500, 0.03), (1000, 0.01)]),
                           "spread", 0.05) == 500


def test_first_sustained_skips_undefined_points():
    assert first_sustained(_points([(50, 0.2), (100, 0.04), (200, NAN)]), "spread", 0.05) == 100


def test_first_sustained_never():
    assert first_sustained(_points([(50, 0.2), (100, 0.04), (200, 0.3)]), "spread", 0.05) is None


def _train_rows(arm, scores_by_n):
    return [dict(arm=arm, target="band_gap", method="xgboost", axis="train", n=n, rep=i, score=f"{s:.6f}",
                 n_test=100, n_pool=1000) for n, scores in scores_by_n.items() for i, s in enumerate(scores)]


def test_summary_spread_is_scaled_sd(tmp_path):
    scores = [0.50, 0.55, 0.61, 0.58]
    write_rows(tmp_path / "raw" / "curves_train_orb.csv", TRAIN_COLS, _train_rows("orb", {100: scores, 200: [0.7]}))
    rows = summarise_arm("orb", tmp_path, {("band_gap", "xgboost"): 0.8})
    assert rows[0]["spread"] == pytest.approx(P90_P10_OVER_SD * np.std(scores, ddof=1))
    assert rows[0]["gap"] == pytest.approx(0.8 - np.median(scores))
    assert np.isnan(rows[1]["spread"])                          # a single draw has no spread
    measured = summarise_arm("orb", tmp_path, {})
    assert measured[0]["ceiling"] == 0.7 and measured[0]["ceiling_source"] == "measured"


def test_paired_gain_needs_min_pairs(tmp_path):
    few, many = MIN_PAIRS - 1, MIN_PAIRS
    write_rows(tmp_path / "raw" / "curves_train_magpie.csv", TRAIN_COLS,
               _train_rows("magpie", {100: [0.3] * few, 200: [0.4] * many}))
    write_rows(tmp_path / "raw" / "curves_train_orb.csv", TRAIN_COLS,
               _train_rows("orb", {100: [0.5] * few, 200: [0.6] * many}))
    paired_gain(tmp_path)
    rows = read_rows(tmp_path / "paired_gain.csv")
    assert [(r["n"], r["n_pairs"], r["median"]) for r in rows] == [("200", str(many), "0.200000")]


def test_split_fraction_leaves_cells_outside_the_grid_blank(tmp_path, design):
    def row(axis, n, median, spread):
        return dict(arm="orb", target="band_gap", method="xgboost", axis=axis, n=n, reps=10, median=median,
                    spread=spread, ceiling="0.900000", ceiling_source="reference")
    rows = [row("train", n, m, "0.1") for n, m in ((50, "0.5"), (200, "0.7"))]
    rows += [row("test", n, "0.85", s) for n, s in ((20, "0.3"), (500, "0.05"))]
    write_rows(tmp_path / "summary_orb.csv", SUMMARY_COLS, rows)
    split_fraction(tmp_path, design)
    cells = {(int(r["N"]), float(r["fraction"])): r for r in read_rows(tmp_path / "split_fraction.csv")}
    inside, outside = cells[(100, 0.5)], cells[(1000, 0.2)]      # 50 training rows; 800 is beyond the grid
    assert float(inside["shortfall"]) == pytest.approx(0.4) and inside["total"] != ""
    assert outside["score"] == outside["shortfall"] == outside["total"] == ""


def test_summarise_refuses_curves_without_ceilings(tmp_path):
    write_rows(tmp_path / "raw" / "curves_train_orb.csv", TRAIN_COLS, _train_rows("orb", {100: [0.5, 0.6]}))
    with pytest.raises(ValueError, match="ceilings"):
        summarise(tmp_path)


@pytest.mark.parametrize("step", [summarise, paired_gain, split_fraction])
def test_steps_refuse_a_directory_without_their_inputs(tmp_path, step):
    with pytest.raises(FileNotFoundError, match="mlip-probe"):
        step(tmp_path)
    assert not any(tmp_path.iterdir())
