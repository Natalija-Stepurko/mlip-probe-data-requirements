"""A tiny synthetic world: latent factors Z drive every target; the "embeddings" see Z with little
noise and the "composition" arm sees two of its columns with a lot of noise."""
from __future__ import annotations

import numpy as np
import pytest

from mlip_probe.config import ARMS, CLASSIFICATION, METHODS, TARGETS, Design
from mlip_probe.dataset import Inputs

N_MATERIALS = 800


@pytest.fixture(scope="session")
def world() -> dict[str, Inputs]:
    rng = np.random.default_rng(0)
    mp_ids = np.array([f"mp-{i}" for i in range(N_MATERIALS)])
    Z = rng.normal(size=(N_MATERIALS, 6))
    targets = {}
    for t in TARGETS:
        if t in CLASSIFICATION:
            y = np.where(Z[:, 0] > 0.3, "A", np.where(Z[:, 1] > 0, "B", "C")).astype(object)
            if t == "magnetic_label":
                y[rng.random(N_MATERIALS) < 0.3] = None
        else:
            y = Z[:, 0] * 2 + Z[:, 1] + rng.normal(scale=0.5, size=N_MATERIALS)
            if t == "bulk_modulus":
                y[rng.random(N_MATERIALS) < 0.6] = np.nan
        targets[t] = y
    features = {"orb": Z + rng.normal(scale=0.1, size=Z.shape),
                "uma": Z + rng.normal(scale=0.2, size=Z.shape),
                "magpie": Z[:, :2] + rng.normal(scale=1.0, size=(N_MATERIALS, 2))}
    ceilings = {(t, m): 0.9 for t in TARGETS for m in METHODS}
    return {arm: Inputs(mp_ids, features[arm].astype(np.float32), targets, dict(ceilings), arm) for arm in ARMS}


@pytest.fixture(scope="session")
def design() -> Design:
    return Design(train_grid=[40, 100, 250], test_grid=[20, 50, 200], test_resamples=10, frozen_test_n=200,
                  test_axis_train_n=400, split_sizes=[100, 300, 1000], split_fractions=[0.2, 0.5],
                  cv_sizes=[100], cv_draws=2, cv_bootstrap=50)
