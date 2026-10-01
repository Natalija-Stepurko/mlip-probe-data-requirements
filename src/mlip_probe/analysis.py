"""Everything computed from the raw draws: per-point summaries, stability and saturation
thresholds, the paired embedding-minus-Magpie gain, and the split-fraction table. Needs only
results/raw/ and results/ceilings.csv."""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import (ARM_COLOURS, ARMS, CLASSIFICATION, DEFAULT_DESIGN, EMBEDDINGS, MAX_GAP, MAX_SPREAD,
                     MIN_PAIRS, P90_P10_OVER_SD, TARGETS, Design)
from .results_io import (GAIN_COLS, SPLIT_COLS, SUMMARY_COLS, THRESHOLD_COLS, fmt, formatted, group, read_rows,
                         write_rows)

log = logging.getLogger(__name__)
NAN = float("nan")


def _scores(rows: list[dict]) -> np.ndarray:
    return np.array([float(r["score"]) for r in rows if r["score"] not in ("", "nan")])


def read_ceilings(out: Path, arm: str) -> dict[tuple[str, str], float]:
    return {(r["target"], r["method"]): float(r["ceiling"]) for r in read_rows(out / "ceilings.csv")
            if r["arm"] == arm}


def summarise_arm(arm: str, out: Path, ceilings: dict[tuple[str, str], float]) -> list[dict]:
    """One row per (axis, target, method, n). The ceiling is the reference score when there is
    one, else the median at the largest measured size ("measured"). spread is the p10-p90
    width estimated from the sd, which stays unbiased at a handful of draws."""
    rows = []
    for axis in ("train", "test"):
        raw = read_rows(out / "raw" / f"curves_{axis}_{arm}.csv")
        for (t, method), g in sorted(group(raw, ["target", "method"]).items()):
            by_n = group(g, ["n"])
            if (t, method) in ceilings:
                full, src = ceilings[(t, method)], "reference"
            else:
                n_max = max(int(n) for (n,) in by_n)
                full, src = float(np.nanmedian(_scores(by_n[(str(n_max),)]))), "measured"
            for (n,), gg in sorted(by_n.items(), key=lambda kv: int(kv[0][0])):
                s = _scores(gg)
                if len(s) == 0:
                    continue
                median = float(np.median(s))
                p10, p90 = float(np.quantile(s, 0.10)), float(np.quantile(s, 0.90))
                rows.append(dict(arm=arm, target=t, method=method, axis=axis, n=int(n), reps=len(s),
                                 median=median, p10=p10, p90=p90,
                                 spread=P90_P10_OVER_SD * float(np.std(s, ddof=1)) if len(s) > 1 else NAN,
                                 spread_direct=p90 - p10, ceiling=full, ceiling_source=src, gap=full - median))
    return rows


def first_sustained(points: list[dict], col: str, cut: float) -> int | None:
    """Smallest n from which `col` <= cut holds at every larger measured n; None if never.
    Undefined values (a single draw has no spread) are skipped."""
    pts = [r for r in sorted(points, key=lambda r: r["n"]) if not math.isnan(r[col])]
    ok = [r[col] <= cut for r in pts]
    for i in range(len(pts)):
        if all(ok[i:]):
            return pts[i]["n"]
    return None


def thresholds(summary: list[dict]) -> list[dict]:
    """n_stable (spread <= MAX_SPREAD) and n_saturated (gap <= MAX_GAP) per curve."""
    rows = []
    for (arm, t, method), g in sorted(group(summary, ["arm", "target", "method"]).items()):
        tr = [r for r in g if r["axis"] == "train"]
        te = [r for r in g if r["axis"] == "test"]
        rows.append(dict(arm=arm, target=t, method=method,
                         n_stable=first_sustained(tr, "spread", MAX_SPREAD) if tr else None,
                         n_saturated=first_sustained(tr, "gap", MAX_GAP) if tr else None,
                         n_train_max=max(r["n"] for r in tr) if tr else None,
                         test_n_stable=first_sustained(te, "spread", MAX_SPREAD) if te else None,
                         ceiling=tr[0]["ceiling"] if tr else None,
                         ceiling_source=tr[0]["ceiling_source"] if tr else None))
    return rows


def _require(found: bool, what: str, command: str) -> None:
    if not found:
        raise FileNotFoundError(f"no {what}; run `mlip-probe {command}` first")


def summarise(out: Path) -> None:
    """summary_{arm}.csv for every arm with raw training curves, and thresholds.csv. An arm with
    curves but no rows in ceilings.csv is refused: its ceilings would silently become "measured"."""
    _require(any((out / "raw").glob("curves_train_*.csv")), f"learning curves in {out}/raw", "curves")
    summary = []
    for arm in ARMS:
        if not (out / "raw" / f"curves_train_{arm}.csv").exists():
            log.info("  %s: no raw curves, skipped", arm)
            continue
        ceilings = read_ceilings(out, arm)
        if not ceilings:
            raise ValueError(f"{out}/ceilings.csv has no {arm} rows; run `mlip-probe ceilings` first")
        rows = summarise_arm(arm, out, ceilings)
        write_rows(out / f"summary_{arm}.csv", SUMMARY_COLS, formatted(rows))
        summary += rows
        log.info("  %s: %d summary points", arm, len(rows))
    write_rows(out / "thresholds.csv", THRESHOLD_COLS, formatted(thresholds(summary)))


def paired_gain(out: Path) -> None:
    """paired_gain.csv: embedding minus Magpie, XGBoost, over draws that share their materials."""
    def draws(arm: str) -> dict[tuple[str, int, int], float]:
        raw = read_rows(out / "raw" / f"curves_train_{arm}.csv")
        return {(r["target"], int(r["n"]), int(r["rep"])): float(r["score"])
                for r in raw if r["method"] == "xgboost" and r["score"] not in ("", "nan")}

    _require((out / "raw" / "curves_train_magpie.csv").exists(), f"Magpie learning curves in {out}/raw",
             "curves --arm magpie")
    base = draws("magpie")
    rows = []
    for arm in EMBEDDINGS:
        emb = draws(arm)
        for t, n in sorted({(k[0], k[1]) for k in emb}):
            d = np.array([emb[k] - base[k] for k in emb if k[0] == t and k[1] == n and k in base])
            if len(d) < MIN_PAIRS:
                continue
            rows.append(dict(arm=arm, target=t, n=n, n_pairs=len(d), median=fmt(float(np.median(d))),
                             p10=fmt(float(np.quantile(d, 0.1))), p90=fmt(float(np.quantile(d, 0.9))),
                             frac_positive=fmt(float((d > 0).mean()))))
    write_rows(out / "paired_gain.csv", GAIN_COLS, rows)
    log.info("  paired_gain.csv: %d rows", len(rows))


def log_interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """ys at x, linear in log(x); NaN outside the measured range."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if x < xs.min() or x > xs.max():
        return NAN
    return float(np.interp(np.log(x), np.log(xs), ys))


def split_fraction(out: Path, design: Design = DEFAULT_DESIGN) -> None:
    """split_fraction.csv: total error of one train/test split of N materials with a fraction f
    held out, from the XGBoost summaries. shortfall = ceiling minus the training curve at (1-f)N;
    the test-size noise is the test-axis spread at fN, rescaled by (1 - score) / (1 - ceiling)
    because a weaker probe scatters more, as an sd; total adds the two in quadrature. Cells
    outside the measured grids are blank."""
    _require(any(out.glob("summary_*.csv")), f"summaries in {out}", "summarise")
    rows = []
    for arm in ARMS:
        summ = [r for r in read_rows(out / f"summary_{arm}.csv") if r["method"] == "xgboost"]
        for t in TARGETS:
            tr = sorted([r for r in summ if r["target"] == t and r["axis"] == "train"], key=lambda r: int(r["n"]))
            te = sorted([r for r in summ if r["target"] == t and r["axis"] == "test" and r["spread"] != ""],
                        key=lambda r: int(r["n"]))
            if not tr or not te:
                continue
            ceiling = float(tr[0]["ceiling"])
            for N in design.split_sizes:
                for f in design.split_fractions:
                    rows.append(_split_cell(arm, t, N, f, ceiling, tr, te))
    write_rows(out / "split_fraction.csv", SPLIT_COLS, rows)
    log.info("  split_fraction.csv: %d rows", len(rows))


def _split_cell(arm: str, t: str, N: int, f: float, ceiling: float, tr: list[dict], te: list[dict]) -> dict:
    n_test = int(round(f * N))
    n_train = N - n_test
    score = log_interp([int(r["n"]) for r in tr], [float(r["median"]) for r in tr], n_train)
    spread = log_interp([int(r["n"]) for r in te], [float(r["spread"]) for r in te], n_test)
    short = NAN if np.isnan(score) else max(0.0, ceiling - score)
    rescaled = spread * (1.0 - score) / max(1e-9, 1.0 - ceiling)   # a ceiling of 1.0 would divide by zero
    sd = rescaled / P90_P10_OVER_SD
    return dict(arm=arm, target=t, N=N, fraction=f, n_train=n_train, n_test=n_test, score=fmt(score),
                shortfall=fmt(short), spread=fmt(rescaled), sd=fmt(sd), total=fmt(float(np.hypot(short, sd))))


def plot_train_curves(out: Path, path: Path) -> None:
    """XGBoost training curves of every arm, median with the p10-p90 band, one panel per target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _require(any(out.glob("summary_*.csv")), f"summaries in {out}", "summarise")
    summary = [r for arm in ARMS for r in read_rows(out / f"summary_{arm}.csv")
               if r["method"] == "xgboost" and r["axis"] == "train"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5))
    for ax, t in zip(axes.flat, TARGETS):
        for arm in ARMS:
            pts = sorted([r for r in summary if r["arm"] == arm and r["target"] == t], key=lambda r: int(r["n"]))
            if not pts:
                continue
            n = [int(r["n"]) for r in pts]
            ax.fill_between(n, [float(r["p10"]) for r in pts], [float(r["p90"]) for r in pts],
                            color=ARM_COLOURS[arm], alpha=0.15, lw=0)
            ax.plot(n, [float(r["median"]) for r in pts], color=ARM_COLOURS[arm], lw=2, marker="o", ms=3, label=arm)
        ax.set_xscale("log")
        ax.set_title(t, fontsize=9)
        ax.set_ylabel("macro-F1" if t in CLASSIFICATION else "R²")
        ax.set_xlabel("training materials")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
