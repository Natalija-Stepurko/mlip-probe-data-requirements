"""The two learning-curve axes of one arm, written as raw per-draw scores.

Training axis: at each size n, a probe per disjoint training draw, scored on the frozen test set.
Test axis: one probe fitted on a large training set, its predictions re-scored on random
subsamples of the frozen test set at each test size.

Both runners append per (target, method, size) and skip what the raw file already holds, so an
interrupted run resumes where it stopped and ends with the rows of an uninterrupted run.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import ARMS, DRAW_SEED, METHODS, TARGETS, Design, task_of
from .dataset import Inputs
from .probes import fit_predict, score
from .results_io import CEILING_COLS, FROZEN_IDS_COLS, TEST_COLS, TRAIN_COLS, append_rows, fmt, read_rows, write_rows
from .sampling import frozen_split, target_seed, train_draws

log = logging.getLogger(__name__)

TEST_AXIS_STREAM = 999   # third seed component of the test-axis subsampling stream


def record_frozen_ids(out: Path, inputs: Inputs, design: Design) -> None:
    """frozen_test_ids.csv: every target's test materials. Written once; a later run on other
    inputs is refused."""
    path = out / "frozen_test_ids.csv"
    rows = []
    for t in TARGETS:
        test, _ = frozen_split(t, task_of(t), inputs.targets[t], design)
        rows += [dict(target=t, mp_id=m) for m in inputs.mp_ids[np.sort(test)]]
    if path.exists():
        old = read_rows(path)
        if [(r["target"], r["mp_id"]) for r in old] != [(r["target"], r["mp_id"]) for r in rows]:
            raise ValueError(f"{path} differs from the current design; the inputs changed")
        return
    write_rows(path, FROZEN_IDS_COLS, rows)


def record_ceilings(out: Path, inputs: Inputs) -> None:
    """ceilings.csv: the arm's full-corpus reference scores, so the summaries can be recomputed
    from results/ alone. repr() keeps every float exact through the round trip."""
    path = out / "ceilings.csv"
    new = [dict(arm=inputs.arm, target=t, method=m, ceiling=repr(inputs.ceilings[(t, m)]))
           for t in TARGETS for m in METHODS if (t, m) in inputs.ceilings]
    by_arm = {a: [r for r in read_rows(path) if r["arm"] == a] for a in ARMS}
    if by_arm[inputs.arm] and by_arm[inputs.arm] != new:
        raise ValueError(f"{path}: the {inputs.arm} ceilings differ from the current inputs")
    by_arm[inputs.arm] = new
    write_rows(path, CEILING_COLS, [r for a in ARMS for r in by_arm[a]])


def run_train_axis(inputs: Inputs, out: Path, targets: Sequence[str], n_jobs: int, design: Design) -> None:
    path = out / "raw" / f"curves_train_{inputs.arm}.csv"
    done = {(r["target"], r["method"], int(r["n"])) for r in read_rows(path)}
    for t in targets:
        task, y = task_of(t), inputs.targets[t]
        test, pool = frozen_split(t, task, y, design)
        X_test, y_test = inputs.X[test], y[test]
        log.info("[%s train] %s: %s test, %s pool", inputs.arm, t, f"{len(test):,}", f"{len(pool):,}")
        for n in design.train_grid:
            draws = train_draws(t, pool, n, design)
            for method in METHODS:
                if not draws or (t, method, n) in done:
                    continue
                t0, rows = time.time(), []
                for rep, rows_tr in enumerate(draws):
                    pred = fit_predict(task, method, inputs.X[rows_tr], y[rows_tr], X_test, n_jobs)
                    s = score(task, y_test, pred) if pred is not None else float("nan")
                    rows.append(dict(arm=inputs.arm, target=t, method=method, axis="train", n=n, rep=rep,
                                     score=fmt(s), n_test=len(test), n_pool=len(pool)))
                append_rows(path, TRAIN_COLS, rows)
                med = np.nanmedian([float(r["score"]) for r in rows])
                log.info("    n=%6d %-8s %2d draws  median=%.3f  (%.0fs)", n, method, len(draws), med, time.time() - t0)


def run_test_axis(inputs: Inputs, out: Path, targets: Sequence[str], n_jobs: int, design: Design) -> None:
    path = out / "raw" / f"curves_test_{inputs.arm}.csv"
    done = {(r["target"], r["method"], int(r["n"])) for r in read_rows(path)}
    for t in targets:
        task, y = task_of(t), inputs.targets[t]
        test, pool = frozen_split(t, task, y, design)
        y_test = y[test]
        sizes = [m for m in design.test_grid if m <= len(test)]
        fit_rows = pool[:design.test_axis_train_n]   # frozen_split permuted the pool, so its head is a random sample
        for method in METHODS:
            if all((t, method, m) in done for m in sizes):
                continue
            pred = fit_predict(task, method, inputs.X[fit_rows], y[fit_rows], inputs.X[test], n_jobs)
            log.info("[%s test] %s %s: fitted on %s, full-test score %.3f", inputs.arm, t, method,
                     f"{len(fit_rows):,}", score(task, y_test, pred))
            rng = np.random.default_rng([DRAW_SEED, target_seed(t), TEST_AXIS_STREAM])
            for m in sizes:
                k = 1 if m == len(test) else design.test_resamples
                subs = [rng.choice(len(test), m, replace=False) if m < len(test) else np.arange(len(test))
                        for _ in range(k)]
                if (t, method, m) in done:
                    continue   # drawn anyway, so later sizes see the stream of an uninterrupted run
                append_rows(path, TEST_COLS, [dict(arm=inputs.arm, target=t, method=method, axis="test", n=m,
                                                   rep=rep, score=fmt(score(task, y_test[s], pred[s])),
                                                   n_train_fit=len(fit_rows))
                                              for rep, s in enumerate(subs)])
