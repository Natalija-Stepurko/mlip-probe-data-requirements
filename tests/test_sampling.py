from dataclasses import replace

import numpy as np
import pytest

from mlip_probe.config import DEFAULT_DESIGN, TARGETS
from mlip_probe.curves import record_frozen_ids
from mlip_probe.dataset import align
from mlip_probe.results_io import read_rows
from mlip_probe.sampling import frozen_split, labelled_rows, train_draws


def test_frozen_ids_are_in_row_order_and_a_changed_design_is_refused(world, design, tmp_path):
    record_frozen_ids(tmp_path, world["orb"], design)
    record_frozen_ids(tmp_path, world["magpie"], design)             # same labels: accepted
    rows = read_rows(tmp_path / "frozen_test_ids.csv")
    targets = [r["target"] for r in rows]
    assert targets == sorted(targets, key=TARGETS.index)
    for t in TARGETS:
        idx = [int(r["mp_id"].split("-")[1]) for r in rows if r["target"] == t]
        assert idx == sorted(idx)
    with pytest.raises(ValueError, match="inputs changed"):
        record_frozen_ids(tmp_path, world["orb"], replace(design, frozen_test_n=design.frozen_test_n - 1))


def test_frozen_split_is_deterministic_and_covers_the_labelled_rows(world, design):
    y = world["orb"].targets["bulk_modulus"]
    test, pool = frozen_split("bulk_modulus", "reg", y, design)
    again, _ = frozen_split("bulk_modulus", "reg", y, design)
    assert np.array_equal(test, again)
    assert len(test) == min(design.frozen_test_n, len(labelled_rows("reg", y)) // 2)
    assert sorted(np.concatenate([test, pool])) == list(labelled_rows("reg", y))


def test_train_draws_are_disjoint_and_come_from_the_pool(world, design):
    _, pool = frozen_split("band_gap", "reg", world["orb"].targets["band_gap"], design)
    draws = train_draws("band_gap", pool, 40, design)
    assert len(draws) == min(design.repeats(40), len(pool) // 40)
    rows = np.concatenate(draws)
    assert len(set(rows)) == len(rows) and set(rows) <= set(pool)


@pytest.mark.parametrize("n, k", [(50, 30), (2000, 30), (2001, 10), (5000, 10), (7000, 7), (20000, 7)])
def test_repeats_rule(n, k):
    assert DEFAULT_DESIGN.repeats(n) == k


def test_train_draws_are_limited_by_the_pool(design):
    assert len(train_draws("density", np.arange(90), 40, design)) == 2
    assert train_draws("density", np.arange(30), 40, design) == []


def test_align_reorders_a_shuffled_arm():
    canon = np.array(["a", "b", "c", "d"])
    X = np.arange(4)[:, None] * np.ones((1, 2))
    shuffled = [2, 0, 3, 1]
    assert np.array_equal(align(X[shuffled], canon[shuffled], canon, "orb"), X)


def test_align_refuses_a_missing_material():
    canon = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="absent"):
        align(np.zeros((2, 1)), ["a", "c"], canon, "orb")
