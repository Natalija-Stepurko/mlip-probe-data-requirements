"""The loader and the dataset-reading commands on a 40-material release built in tmp_path."""
import numpy as np
import pandas as pd
import pytest

from mlip_probe.cli import main
from mlip_probe.config import CLASSIFICATION, DATASET_COLUMNS, TARGETS
from mlip_probe.dataset import load
from mlip_probe.results_io import read_rows


def _probe_row(read_out, target, estimator, r2=None, macro_f1=None):
    return dict(model="ORB", read_out=read_out, family=None, layer="backbone", target=target, estimator=estimator,
                r2=r2, macro_f1=macro_f1)


@pytest.fixture
def release(tmp_path):
    root, n = tmp_path / "release", 40
    rng = np.random.default_rng(1)
    ids = [f"mp-{i}" for i in range(n)]
    mat = pd.DataFrame({"mp_id": ids})
    for t in TARGETS:
        mat[DATASET_COLUMNS.get(t, t)] = rng.choice(["x", "y"], n) if t in CLASSIFICATION else rng.normal(size=n)
    for d in ("data", "embeddings/node", "results/controls"):
        (root / d).mkdir(parents=True)
    mat.to_parquet(root / "data" / "materials.parquet")
    order = rng.permutation(n)                      # the embedding file is not in the canonical order
    pd.DataFrame({"mp_id": [ids[i] for i in order], "level": "node",
                  "emb": [[float(i), 0.0] for i in order]}).to_parquet(
        root / "embeddings" / "node" / "orb-node-backbone.parquet")
    pd.DataFrame({"mp_id": ids, "features": [[1.0, 2.0, 3.0]] * n}).to_parquet(
        root / "results" / "magpie_features.parquet")
    pd.DataFrame([_probe_row("node", "band_gap", "xgb", r2=0.8), _probe_row("node", "band_gap", "linear", r2=0.6),
                  _probe_row("node", "crystal_system", "xgb", macro_f1=0.7),
                  _probe_row("graph", "density", "xgb", r2=0.1)]).to_parquet(
        root / "results" / "probe_metrics.parquet")
    pd.DataFrame([dict(target="band_gap", comp_xgb=0.5, comp_linear=0.4)]).to_parquet(
        root / "results" / "controls" / "composition_baseline.parquet")
    return root, ids


def test_load_realigns_features_and_reads_the_ceilings(release):
    root, ids = release
    orb = load("orb", root)
    assert list(orb.mp_ids) == ids and np.array_equal(orb.X[:, 0], np.arange(len(ids)))
    assert orb.ceilings == {("band_gap", "xgboost"): 0.8, ("band_gap", "linear"): 0.6,
                            ("crystal_system", "xgboost"): 0.7}
    assert np.isfinite(orb.targets["formation_energy/atom"]).all()
    assert load("magpie", root).ceilings == {("band_gap", "xgboost"): 0.5, ("band_gap", "linear"): 0.4}


def test_load_refuses_an_unknown_arm(release):
    with pytest.raises(ValueError, match="unknown arm"):
        load("soap", release[0])


def test_cli_ceilings_and_design(release, tmp_path, capsys):
    root, _ = release
    main(["ceilings", "--dataset", str(root), "--out", str(tmp_path / "out")])
    assert [(r["arm"], r["method"]) for r in read_rows(tmp_path / "out" / "ceilings.csv")] == [
        ("orb", "xgboost"), ("orb", "linear"), ("orb", "xgboost"), ("magpie", "xgboost"), ("magpie", "linear")]
    main(["design", "--dataset", str(root)])
    assert "band_gap" in capsys.readouterr().out


def test_cli_rejects_an_unknown_target_before_writing(release, tmp_path):
    with pytest.raises(SystemExit, match="unknown targets"):
        main(["curves", "--arm", "orb", "--axis", "train", "--targets", "bandgap", "--dataset", str(release[0]),
              "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
