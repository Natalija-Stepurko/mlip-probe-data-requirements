"""Every pass on the synthetic world, three arms, a few targets."""
import hashlib
from dataclasses import replace

import pytest

from mlip_probe import analysis, crossval, curves
from mlip_probe.config import ARMS
from mlip_probe.results_io import read_rows

E2E_TARGETS = ["band_gap", "bulk_modulus", "magnetic_label"]


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def out(world, design, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    for arm in ARMS:
        curves.record_frozen_ids(out, world[arm], design)      # raises if the arms' test sets disagree
        curves.record_ceilings(out, world[arm])
        curves.run_train_axis(world[arm], out, E2E_TARGETS, n_jobs=1, design=design)
        curves.run_test_axis(world[arm], out, E2E_TARGETS, n_jobs=1, design=design)
    analysis.summarise(out)
    analysis.paired_gain(out)
    analysis.split_fraction(out, design)
    crossval.run(world["orb"], out, ["band_gap"], n_jobs=1, design=design)
    crossval.summarise(out)
    return out


def test_every_pass_writes_its_file(out):
    for name in ["frozen_test_ids.csv", "ceilings.csv", "summary_orb.csv", "summary_uma.csv", "summary_magpie.csv",
                 "thresholds.csv", "paired_gain.csv", "split_fraction.csv", "cv_vs_holdout.csv"]:
        assert len(read_rows(out / name)) > 0, name
    assert {r["estimator"] for r in read_rows(out / "cv_vs_holdout.csv")} == {"single", "repeated", "cv"}


def test_resume_adds_nothing(out, world, design):
    raw = [out / "raw" / name for name in ("curves_train_orb.csv", "curves_test_orb.csv", "cv_vs_holdout_orb.csv")]
    before = [_md5(p) for p in raw]
    curves.run_train_axis(world["orb"], out, E2E_TARGETS, n_jobs=1, design=design)
    curves.run_test_axis(world["orb"], out, E2E_TARGETS, n_jobs=1, design=design)
    crossval.run(world["orb"], out, ["band_gap"], n_jobs=1, design=design)
    assert [_md5(p) for p in raw] == before


def test_recording_other_ceilings_is_refused(out, world):
    changed = replace(world["orb"], ceilings={("band_gap", "xgboost"): 0.5})
    with pytest.raises(ValueError, match="ceilings differ"):
        curves.record_ceilings(out, changed)


def test_interrupted_test_axis_resumes_to_the_uninterrupted_file(out, world, design, tmp_path):
    full = out / "raw" / "curves_test_orb.csv"
    lines = full.read_text().splitlines(keepends=True)
    first_size = [line for line in lines[1:] if line.split(",")[4] == str(design.test_grid[0])]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "curves_test_orb.csv").write_text(lines[0] + "".join(first_size))   # stopped after one size
    curves.run_test_axis(world["orb"], tmp_path, E2E_TARGETS, n_jobs=1, design=design)
    assert sorted((tmp_path / "raw" / "curves_test_orb.csv").read_text().splitlines()) == \
        sorted(full.read_text().splitlines())


def test_embedding_beats_the_noisier_composition_arm(out):
    gains = [r for r in read_rows(out / "paired_gain.csv") if r["target"] == "band_gap"]
    assert gains and all(float(r["median"]) > 0 for r in gains)
