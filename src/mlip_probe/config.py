"""Every constant of the experiment: targets, feature sets, probes, seeds, thresholds and the
draw design. The defaults of `Design` are the values that produced results/."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

TARGETS = ("formation_energy/atom", "energy_above_hull", "band_gap", "density", "bulk_modulus",
           "mag_density", "crystal_system", "metal_insulator", "magnetic_label")
CLASSIFICATION = frozenset({"crystal_system", "metal_insulator", "magnetic_label"})
ARMS = ("orb", "uma", "magpie")          # two frozen MLIP embeddings and the composition baseline
EMBEDDINGS = ("orb", "uma")
METHODS = ("xgboost", "linear")

# Column names in the dataset release, for the targets whose spelling differs.
DATASET_COLUMNS = MappingProxyType({"formation_energy/atom": "formation_energy_per_atom"})

# Plot colour of each arm, shared by the matplotlib figure and the page's SVG fallbacks.
ARM_COLOURS = MappingProxyType({"orb": "#1F6F6B", "uma": "#C2681A", "magpie": "#4A4F55"})

DRAW_SEED = 7            # frozen test sets, training draws and test-axis subsamples
PROBE_SEED = 42          # estimator randomness, as in the paper
CV_SEED = 11             # the cross-validation experiment's draws and splits

P90_P10_OVER_SD = 2.5631  # p90 - p10 of a normal distribution, in units of sd
MAX_SPREAD = 0.05         # a curve point is stable when its p10-p90 spread is at most this
MAX_GAP = 0.05            # a curve point is saturated when it is within this of the ceiling
MIN_PAIRS = 3             # a paired difference needs a spread; sizes with fewer draws are not reported


def task_of(target: str) -> str:
    return "clf" if target in CLASSIFICATION else "reg"


@dataclass(frozen=True)
class Design:
    """Grids and repeat counts of both experiments."""

    train_grid: tuple[int, ...] = (50, 100, 200, 500, 1000, 2000, 5000, 7000, 10000, 20000)
    test_grid: tuple[int, ...] = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
    test_resamples: int = 200        # subsamples of the frozen test set per test-axis size
    frozen_test_n: int = 20_000      # capped at half the labelled rows for sparse targets
    test_axis_train_n: int = 20_000  # training rows of the single test-axis probe
    split_sizes: tuple[int, ...] = (200, 500, 1000, 2000, 5000, 10000, 20000)
    split_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    cv_sizes: tuple[int, ...] = (100, 200, 500, 1000, 2000)
    cv_draws: int = 20               # disjoint n-row datasets per (target, n)
    cv_repeat_splits: int = 5        # 80/20 splits of the repeated-holdout estimator
    cv_folds: int = 5
    cv_bootstrap: int = 200          # resamples per percentile interval
    cv_alpha: float = 0.05           # intervals cover 1 - cv_alpha

    def __post_init__(self) -> None:
        for name in ("train_grid", "test_grid", "split_sizes", "split_fractions", "cv_sizes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def repeats(self, n: int) -> int:
        """Training draws per size: small sizes are noisy and cheap, large ones the opposite."""
        if n <= 2000:
            return 30
        return 10 if n <= 5000 else 7


DEFAULT_DESIGN = Design()
