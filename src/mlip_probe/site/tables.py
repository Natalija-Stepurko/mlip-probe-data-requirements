"""The generated table bodies of docs/index.html (the <tbody> rows only; the headers are
authored in the template next to the prose that describes the columns).

    settle(axis)        every size, three arms: median, worst and best draw, spread, ceiling, shortfall
    gain_summary()      per target: from which size each embedding is ahead of Magpie on ≥ 90 % of
                        draws, where composition is ahead, and the gain at the largest size
    gain_full()         every target, embedding and size of the paired difference
    split_median()      total error per dataset size and split fraction, median over cells
    split_wins()        which fraction wins each cell
    training_floor()    shortfall at every training size: median and range across targets
    test_floor()        spread at every test size: median and range across targets
    floor_by_target()   per target, the training and test size each property needs on its own
    shortfall_by_size_targets()   shortfall per target at 5,000 / 7,000 / 10,000 / 20,000
    cv_target(target)   the three estimators at every dataset size for one target
    cv_budget(target)   bias and noise per estimator, added in quadrature
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ARMS, EMBEDDINGS
from .data import ARM_LABEL, Results
from .targets import label, ordered

LARGE_SIZES = [5000, 7000, 10000, 20000]
TEST_FRACTION = 0.2
TRAIN_FLOOR, TEST_FLOOR = 5000, 500   # the page's recommended floors, highlighted in the floor tables


def _n(n) -> str:
    return f"{int(n):,}"


def _f(v, fmt="{:.3f}", dash="&mdash;") -> str:
    return dash if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def design_example(results: Results, axis: str, target: str = "band_gap", arm: str = "orb") -> str:
    """The worked example in the design section: one target and arm, median with the worst
    and best draw at every size."""
    g = (results.curve(arm, axis, target).merge(results.extremes(arm, axis), on=["target", "n"], how="left")
         .sort_values("n"))
    return "\n".join(f'<tr><td>{_n(r.n)}</td><td class="num">{_f(r.median)}</td>'
                     f'<td class="num">{_f(r.worst)}</td><td class="num">{_f(r.best)}</td></tr>' for r in g.itertuples())


def _stable_n(results: Results, arm: str, target: str, axis: str) -> int | None:
    """The XGBoost curve's first stable size, from thresholds.csv."""
    th = results.thresholds
    row = th[(th.arm == arm) & (th.target == target) & (th.method == "xgboost")]
    v = row["n_stable" if axis == "train" else "test_n_stable"].iloc[0] if len(row) else float("nan")
    return None if pd.isna(v) else int(v)


def settle(results: Results, axis: str) -> str:
    out = []
    for target in results.targets():
        for arm in ARMS:
            g = results.curve(arm, axis, target)
            if g.empty:
                continue
            g = g.merge(results.extremes(arm, axis), on=["target", "n"], how="left").sort_values("n")
            stable_n = _stable_n(results, arm, target, axis)
            first = True
            for r in g.itertuples():
                cls = f' class="settled hl-{arm}"' if r.n == stable_n else ""
                head = (f'<td rowspan="{len(g)}">{label(target)}</td><td rowspan="{len(g)}">{ARM_LABEL[arm]}</td>'
                        if first else "")
                first = False
                out.append(f'<tr{cls}>{head}<td class="num">{_n(r.n)}</td><td class="num">{_f(r.draws, "{:.0f}")}</td>'
                           f'<td class="num">{_f(r.median)}</td><td class="num">{_f(r.worst)}</td>'
                           f'<td class="num">{_f(r.best)}</td><td class="num">{_f(r.spread)}</td>'
                           f'<td class="num">{_f(r.ceiling)}</td><td class="num">{_f(r.gap)}</td></tr>')
    return "\n".join(out)


def _first_ahead(g: pd.DataFrame):
    g = g.sort_values("n")
    p10 = g.p10.to_numpy(float)
    for i in range(len(g)):
        if (p10[i:] > 0).all():
            return int(g.n.iloc[i])
    return None


def _comp_ahead(g: pd.DataFrame) -> str:
    ns = [int(r.n) for r in g.sort_values("n").itertuples() if r.p90 < 0]
    if not ns:
        return "&mdash;"
    return _n(ns[0]) if len(ns) == 1 else f"{_n(ns[0])}&ndash;{_n(ns[-1])}"


def gain_summary(results: Results) -> str:
    gains = {e: results.gain_rows(e) for e in EMBEDDINGS}
    rows = []
    for t in results.targets():
        cells, n_last = [], None
        for e in EMBEDDINGS:
            g = gains[e][gains[e].target == t]
            if g.empty:
                cells.append(("&mdash;", "&mdash;", "&mdash;"))
                continue
            last = g.sort_values("n").iloc[-1]
            first = _first_ahead(g)
            cells.append(("never" if first is None else _n(first), _comp_ahead(g), f"{last.med:+.3f}"))
            n_last = n_last or _n(last.n)
        if n_last is None:
            continue
        rows.append(f"<tr><td>{label(t)}</td>" + "".join(f'<td class="num">{a}</td><td class="num">{b}</td>'
                                                          f'<td class="num">{c}</td>' for a, b, c in cells)
                    + f'<td class="num">{n_last}</td></tr>')
    return "\n".join(rows)


def gain_full(results: Results) -> str:
    rows = []
    for e in EMBEDDINGS:
        g = results.gain_rows(e)
        for t in ordered(set(g.target)):
            grp = g[g.target == t].sort_values("n")
            first = True
            for r in grp.itertuples():
                head = f'<td rowspan="{len(grp)}">{label(t)}</td><td rowspan="{len(grp)}">{ARM_LABEL[e]}</td>' if first else ""
                first = False
                cls = f' class="settled hl-{e}"' if r.p10 > 0 else ""
                rows.append(f'<tr{cls}>{head}<td class="num">{_n(r.n)}</td><td class="num">{int(r.k)}</td>'
                            f'<td class="num">{r.med:+.3f}</td><td class="num">{r.p10:+.3f}</td>'
                            f'<td class="num">{r.p90:+.3f}</td><td class="num">{100 * r.win:.0f}&nbsp;%</td></tr>')
    return "\n".join(rows)


def _split_cells(results: Results) -> pd.DataFrame:
    s = results.split
    if not len(s):
        return s
    s = s[s.arm.isin(EMBEDDINGS)].dropna(subset=["total"]).copy()
    s["split"] = s.fraction.map(lambda f: f"{round((1 - f) * 100)}/{round(f * 100)}")
    return s


def split_median(results: Results) -> str:
    c = _split_cells(results)
    if not len(c):
        return ""
    out = []
    for N, g in c.groupby("N"):
        m = g.groupby(["fraction", "split", "n_train", "n_test"]).agg(
            score=("score", "median"), shortfall=("shortfall", "median"), spread=("spread", "median"),
            total=("total", "median")).reset_index().sort_values("fraction")
        best = m.total.idxmin()
        for i, r in enumerate(m.itertuples()):
            cls = ' class="settled"' if r.Index == best else ""
            first = f'<td rowspan="{len(m)}" class="num">{_n(N)}</td>' if i == 0 else ""
            out.append(f'<tr{cls}>{first}<td>{r.split}</td><td class="num">{_n(r.n_train)}</td>'
                       f'<td class="num">{_n(r.n_test)}</td><td class="num">{_f(r.score)}</td>'
                       f'<td class="num">{_f(r.shortfall)}</td><td class="num">{_f(r.spread)}</td>'
                       f'<td class="num">{_f(r.total)}</td></tr>')
    k = c.groupby("N").size().min() // c.fraction.nunique()
    return "\n".join(out) + f"\n<!-- median over {k} cells -->"


def split_wins(results: Results) -> str:
    c = _split_cells(results)
    if not len(c):
        return ""
    out = []
    for N, g in c.groupby("N"):
        w = g.pivot_table(index=["arm", "target"], columns="fraction", values="total")
        w = w.dropna()
        if w.empty:
            continue
        b = w.idxmin(axis=1)
        k = len(w)
        beats = (w[0.2] < w[0.5]).sum() if 0.2 in w and 0.5 in w else 0
        out.append(f'<tr><td class="num">{_n(N)}</td><td class="num">{beats} of {k}</td>'
                   f'<td class="num">{(b == 0.2).sum()} of {k}</td><td class="num">{(b == 0.3).sum()} of {k}</td>'
                   f'<td class="num">{(b == 0.1).sum()} of {k}</td></tr>')
    return "\n".join(out)


def _gaps(results: Results) -> pd.DataFrame:
    frames = [results.curve(e, "train") for e in EMBEDDINGS]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames)
    d = d[d.n.isin(LARGE_SIZES)]
    return d.groupby(["target", "n"]).gap.mean().unstack("n")



def shortfall_by_size_targets(results: Results) -> str:
    g = _gaps(results)
    if g.empty:
        return ""
    g = g.reindex(columns=LARGE_SIZES).loc[ordered(g.index)].sort_values(LARGE_SIZES[0])
    hardest = g[LARGE_SIZES[0]].idxmax()
    rows = []
    for t, r in g.iterrows():
        cells = "".join(f'<td class="num">{_f(r[n])}</td>' for n in LARGE_SIZES)
        name = label(t)
        if t == hardest:
            name = f"<strong>{name}</strong>"
            cells = cells.replace('<td class="num">', '<td class="num"><strong>').replace("</td>", "</strong></td>")
        rows.append(f"<tr><td>{name}</td>{cells}</tr>")
    full = g.dropna()
    if len(full):
        span = "".join(f'<td class="num">{full[n].max() / full[n].min():.1f}&times;</td>' for n in LARGE_SIZES)
        rows.append(f"<tr><td>span, hardest / easiest</td>{span}</tr>")
    return "\n".join(rows)


def _shortfall_at(results: Results, arm: str, target: str, n_train: float):
    c = results.ceiling(arm, target)
    m = results.at(arm, "train", target, "median", n_train)
    return None if c is None or m is None else max(0.0, c - m)


def cv_target(results: Results, target: str = "band_gap") -> str:
    rows = []
    for e in EMBEDDINGS:
        c = results.cv_cells(e)
        c = c[c.target == target].sort_values("n")
        for r in c.itertuples():
            rows.append(f'<tr><td>{ARM_LABEL[e]}</td><td class="num">{_n(r.n)}</td><td class="num">{_f(r.draws, "{:.0f}")}</td>'
                        f'<td class="num">{_f(r.rh_rmse, "{:.4f}")}</td><td class="num">{_f(r.cv_rmse, "{:.4f}")}</td>'
                        f'<td class="num">{_f(r.red * 100, "{:+.0f}&nbsp;%")}</td><td class="num">{_f(r.rh_cov, "{:.2f}")}</td>'
                        f'<td class="num">{_f(r.cv_cov, "{:.2f}")}</td>'
                        f'<td class="num">{_f(_shortfall_at(results, e, target, 0.8 * r.n))}</td></tr>')
    return "\n".join(rows)


def cv_budget(results: Results, target: str = "band_gap", arm: str = "orb") -> str:
    """Bias (shortfall at 0.8n) and each estimator's noise (sd of its error), in quadrature."""
    c = results.cv_cells(arm)
    c = c[c.target == target].sort_values("n")
    rows = []
    for r in c.itertuples():
        bias = _shortfall_at(results, arm, target, 0.8 * r.n)
        noise = [r.ho_sd, r.rh_sd, r.cv_sd]
        tot = [None if bias is None or pd.isna(x) else float(np.hypot(bias, x)) for x in noise]
        rows.append(f'<tr><td class="num">{_n(r.n)}</td><td class="num">{_f(bias)}</td>'
                    + "".join(f'<td class="num">{_f(x)}</td>' for x in noise)
                    + "".join(f'<td class="num">{_f(x)}</td>' for x in tot) + "</tr>")
    return "\n".join(rows)


def _per_target(results: Results, axis: str, col: str) -> pd.DataFrame:
    """`col` of the XGBoost curves, mean of the two embeddings, one row per target and one column per size."""
    frames = [f for f in (results.curve(e, axis) for e in EMBEDDINGS) if len(f)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).groupby(["target", "n"])[col].mean().unstack("n")


def _floor_rows(d: pd.DataFrame, floor: int, extra) -> str:
    rows = []
    for n in d.columns:
        v = d[n].dropna()
        if v.empty:
            continue
        cls = ' class="settled"' if n == floor else ""
        rows.append(f'<tr{cls}><td class="num">{_n(n)}</td><td class="num">{v.median():.3f}</td>'
                    f'<td class="num">{v.min():.3f}&ndash;{v.max():.3f}</td>{extra(n, v)}</tr>')
    return "\n".join(rows)


def training_floor(results: Results) -> str:
    """Shortfall below the full-corpus score at every training size, over the eight targets measured at
    every size (bulk modulus stops at 5,000)."""
    d = _per_target(results, "train", "gap")
    if d.empty:
        return ""
    d = d.drop(index="bulk_modulus", errors="ignore").dropna()
    return _floor_rows(d, TRAIN_FLOOR,
                       lambda n, v: f'<td class="num">{round(n / (1 - TEST_FRACTION)):,}</td>')


def test_floor(results: Results) -> str:
    """Spread of the score across re-drawn test sets at every test size, over every target measured there."""
    d = _per_target(results, "test", "spread")
    if d.empty:
        return ""
    return _floor_rows(d, TEST_FLOOR, lambda n, v: f'<td class="num">&plusmn;{v.median() / 2:.3f}</td>')


def floor_by_target(results: Results) -> str:
    """Per target: the training size from which the score stays within 0.05 of its ceiling, and the test
    size from which its spread stays at or below 0.05, ORB-v3 / UMA-S, from thresholds.csv."""
    th = results.thresholds
    if not len(th):
        return ""
    th = th[th.method == "xgboost"]

    def cell(t, col):
        vals = []
        for e in EMBEDDINGS:
            r = th[(th.arm == e) & (th.target == t)]
            if r.empty or pd.isna(r[col].iloc[0]):
                vals.append("not reached" if col == "n_saturated" else "&mdash;")
                continue
            n = int(r[col].iloc[0])
            last = col == "n_saturated" and n == int(r["n_train_max"].iloc[0])
            vals.append(f"{n:,}{'&dagger;' if last else ''}")
        return " / ".join(vals)
    rows = [f'<tr><td>{label(t)}</td><td class="num">{cell(t, "n_saturated")}</td>'
            f'<td class="num">{cell(t, "test_n_stable")}</td></tr>' for t in ordered(set(th.target))]
    return "\n".join(rows)


def floor_strip(results: Results) -> str:
    """The headline numbers of the two floors, for the stat strip above the floor figures."""
    tr = _per_target(results, "train", "gap").drop(index="bulk_modulus", errors="ignore")
    te = _per_target(results, "test", "spread")
    if tr.empty or te.empty:
        return ""
    stats = [("training floor", f"{TRAIN_FLOOR:,}"),
             ("median shortfall there", f"{tr[TRAIN_FLOOR].median():.3f}"),
             ("test floor", f"{TEST_FLOOR:,}"),
             ("median spread there", f"{te[TEST_FLOOR].median():.3f} (&plusmn;{te[TEST_FLOOR].median() / 2:.3f})"),
             ("materials at 80/20", f"{round(TRAIN_FLOOR / (1 - TEST_FRACTION)):,}")]
    return "".join(f'<div class="cvx-stat"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in stats)


ESTIMATORS = (("single", "one 80/20 split", 1), ("repeated", "five 80/20 splits, averaged", 5),
              ("cv", "5-fold cross-validation", 5))


def _cv_draws(results: Results) -> pd.DataFrame:
    """Every cross-validation draw of both embeddings, with the expected score of a probe trained on
    0.8n (the training curve's median, log-interpolated), the quantity every estimator fits for."""
    frames = []
    for arm in EMBEDDINGS:
        path = results.indir / "raw" / f"cv_vs_holdout_{arm}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d["target"] = d["target"].str.replace("/", "_per_")
        curve = {t: g.sort_values("n") for t, g in results.curve(arm, "train").groupby("target")}
        d["expected"] = [float(np.interp(np.log(0.8 * n), np.log(curve[t].n), curve[t]["median"]))
                         for t, n in zip(d.target, d.n)]
        frames.append(d.assign(arm=arm))
    return pd.concat(frames) if frames else pd.DataFrame()


def cv_overview(results: Results, sizes=(200, 1000)) -> str:
    """Per estimator: fits, typical error against the true score (median over cells of the RMSE) at
    two dataset sizes, and how often its 95 % interval contains the true score, averaged over cells."""
    d = _cv_draws(results)
    if d.empty:
        return ""
    d = d[d.lo.notna()].copy()
    d["in_true"] = (d.lo <= d.truth) & (d.truth <= d.hi)
    d["in_expected"] = (d.lo <= d.expected) & (d.expected <= d.hi)
    cell = d.groupby(["arm", "target", "n", "estimator"]).agg(
        rmse=("error", lambda e: float(np.sqrt((e ** 2).mean()))), in_true=("in_true", "mean"),
        in_expected=("in_expected", "mean")).reset_index()
    rows = []
    for key, name, fits in ESTIMATORS:
        c = cell[cell.estimator == key]
        err = "".join(f'<td class="num">{c[c.n == n].rmse.median():.3f}</td>' for n in sizes)
        rows.append(f'<tr><td>{name}</td><td class="num">{fits}</td>{err}'
                    f'<td class="num">{c.in_true.mean():.2f}</td><td class="num">{c.in_expected.mean():.2f}</td></tr>')
    return "\n".join(rows)
