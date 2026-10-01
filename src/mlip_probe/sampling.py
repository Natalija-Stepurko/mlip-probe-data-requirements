"""The draw design: which materials each probe is trained and tested on.

Every random choice is seeded by (DRAW_SEED, target) and depends only on the labels, so for a
given target all three arms see the same frozen test set and the same training draws, and the
embedding-minus-Magpie difference is a paired quantity at every point of a curve.
"""
from __future__ import annotations

import numpy as np

from .config import DRAW_SEED, TARGETS, Design, task_of


def target_seed(target: str) -> int:
    # stable across processes; hash() of a str is salted per interpreter
    return int.from_bytes(target.encode(), "little") % 2**31


def labelled_rows(task: str, y: np.ndarray) -> np.ndarray:
    if task == "reg":
        return np.flatnonzero(np.isfinite(y))
    return np.flatnonzero(np.array([v is not None for v in y]))


def frozen_split(target: str, task: str, y: np.ndarray, design: Design) -> tuple[np.ndarray, np.ndarray]:
    """(test_rows, pool_rows): a seeded permutation of the labelled rows; the first
    frozen_test_n rows, or half of them for sparse targets, are the test set."""
    rows = labelled_rows(task, y)
    rng = np.random.default_rng([DRAW_SEED, target_seed(target)])
    perm = rows[rng.permutation(len(rows))]
    n_test = min(design.frozen_test_n, len(rows) // 2)
    return perm[:n_test], perm[n_test:]


def train_draws(target: str, pool: np.ndarray, n: int, design: Design) -> list[np.ndarray]:
    """Disjoint draws of n rows: consecutive chunks of a permutation seeded by (target, n),
    as many as design.repeats(n) allows and the pool can supply."""
    k = min(design.repeats(n), len(pool) // n)
    if k == 0:
        return []
    rng = np.random.default_rng([DRAW_SEED, target_seed(target), n])
    perm = pool[rng.permutation(len(pool))]
    return [perm[i * n:(i + 1) * n] for i in range(k)]


def design_table(targets: dict[str, np.ndarray], design: Design) -> list[dict]:
    """Per target: labelled rows, frozen test and pool sizes, and the draws at each training size."""
    rows = []
    for t in TARGETS:
        task = task_of(t)
        test, pool = frozen_split(t, task, targets[t], design)
        sizes = [(n, len(train_draws(t, pool, n, design))) for n in design.train_grid]
        rows.append(dict(target=t, task=task, labelled=len(test) + len(pool), n_test=len(test), n_pool=len(pool),
                         draws=" ".join(f"{n}x{k}" for n, k in sizes if k)))
    return rows
