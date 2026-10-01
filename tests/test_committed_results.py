"""The committed results follow from the committed raw draws and the committed code: every
derived table and the page are rebuilt from results/ alone (no dataset needed) through the
command line, and compared byte for byte."""
import collections
import filecmp
import shutil
from pathlib import Path

import numpy as np
import pytest

from mlip_probe.cli import main
from mlip_probe.config import ARMS, DEFAULT_DESIGN
from mlip_probe.results_io import read_rows
from mlip_probe.sampling import train_draws

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DERIVED = ["summary_orb.csv", "summary_uma.csv", "summary_magpie.csv", "thresholds.csv", "paired_gain.csv",
           "split_fraction.csv", "cv_vs_holdout.csv"]


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    out = tmp_path_factory.mktemp("rebuilt")
    shutil.copytree(RESULTS / "raw", out / "raw")
    shutil.copy(RESULTS / "ceilings.csv", out)
    for command in ("summarise", "gain", "split-fraction", "cv-summarise"):
        main([command, "--out", str(out)])
    return out


@pytest.mark.parametrize("name", DERIVED)
def test_derived_table_matches(rebuilt, name):
    assert filecmp.cmp(rebuilt / name, RESULTS / name, shallow=False), name


def test_page_matches(tmp_path):
    main(["page", "--results", str(RESULTS), "--template", str(ROOT / "docs" / "template.html"),
          "--output", str(tmp_path / "index.html")])
    assert filecmp.cmp(tmp_path / "index.html", ROOT / "docs" / "index.html", shallow=False)


def test_committed_raw_draw_counts_follow_the_design():
    """Every (target, method, size) of the raw files holds as many draws as the design gives it,
    and the frozen test sets have the sizes the raw rows report."""
    n_test = {}
    for arm in ARMS:
        train = collections.Counter((r["target"], r["method"], int(r["n"]), int(r["n_pool"]), int(r["n_test"]))
                                    for r in read_rows(RESULTS / "raw" / f"curves_train_{arm}.csv"))
        for (t, method, n, pool, nt), k in train.items():
            assert k == len(train_draws(t, np.arange(pool), n, DEFAULT_DESIGN)), (arm, t, method, n)
            assert nt == min(DEFAULT_DESIGN.frozen_test_n, (nt + pool) // 2), (arm, t)
            n_test[t] = nt
        test = collections.Counter((r["target"], r["method"], int(r["n"]))
                                   for r in read_rows(RESULTS / "raw" / f"curves_test_{arm}.csv"))
        for (t, method, n), k in test.items():
            assert k == (1 if n == n_test[t] else DEFAULT_DESIGN.test_resamples), (arm, t, method, n)
    assert dict(collections.Counter(r["target"] for r in read_rows(RESULTS / "frozen_test_ids.csv"))) == n_test


def test_committed_cv_draw_counts_follow_the_design():
    """Every (target, n) of the cross-validation runs holds the design's draws, or as many disjoint
    n-row datasets as the training pool supplies."""
    pools = {r["target"]: int(r["n_pool"]) for r in read_rows(RESULTS / "raw" / "curves_train_orb.csv")}
    for arm in ("orb", "uma"):
        draws = collections.defaultdict(set)
        for r in read_rows(RESULTS / "raw" / f"cv_vs_holdout_{arm}.csv"):
            draws[(r["target"], int(r["n"]))].add(int(r["draw"]))
        assert {n for _, n in draws} == set(DEFAULT_DESIGN.cv_sizes), arm
        for (t, n), d in draws.items():
            assert len(d) == min(DEFAULT_DESIGN.cv_draws, pools[t] // n), (arm, t, n)
