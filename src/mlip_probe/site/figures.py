"""The SVG figures of docs/index.html, XGBoost, one panel per target.

Colour encodes the feature set: ORB against UMA against the Magpie composition baseline.
Each arm's full-corpus score is a dashed line in its own colour, so a curve and the
ceiling it approaches match. Colours are the page's CSS custom properties; most carry a
literal fallback for use outside the page.

    grid(kind)   train | test | shortfall | gain  — one panel per target
    gainmap()    one dot per embedding, target and size: does the sign of the paired
                 difference hold across the p10–p90 band, and which way
    percell(n)   per-target shortfall bars at one training size, three arms
    cv_gain(), cv_coverage()   the two cross-validation panels, one per embedding
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ..config import ARM_COLOURS, ARMS, EMBEDDINGS
from .data import ARM_LABEL, MODEL_NAME, Results
from .targets import label, ordered

ORB = f"var(--orb,{ARM_COLOURS['orb']})"
UMA = f"var(--uma,{ARM_COLOURS['uma']})"
MAG = f"var(--mag,{ARM_COLOURS['magpie']})"
COLOUR = {"orb": ORB, "uma": UMA, "magpie": MAG}
GRID = "var(--rule,#DDE1E4)"
AXIS = "var(--axis,#B9C0C6)"
MUTED = "var(--muted,#5B646E)"
INK = "var(--ink,#16191D)"
PANEL = "var(--panel,#FFFFFF)"

W, H = 330, 218
X0, X1 = 50.0, 304.0
Y0, Y1 = 16.0, 162.0
Y_AXIS, Y_TICK, Y_LABEL = 178, 189, 212
NOMINAL = 0.95


def _fmt_n(n: float) -> str:
    n = int(n)
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else f"{n:,}"


def _scales(ns: Sequence[float], lo: float, hi: float) -> tuple[Callable, Callable]:
    ln = np.log(np.asarray(ns, float))
    lo_l, hi_l = ln.min(), ln.max()

    def sx(n):
        f = 0.0 if hi_l == lo_l else (np.log(n) - lo_l) / (hi_l - lo_l)
        return X0 + f * (X1 - X0)

    def sy(v):
        f = 0.0 if hi == lo else (v - lo) / (hi - lo)
        return Y1 - f * (Y1 - Y0)
    return sx, sy


def panel(title: str, curves: list[dict], ns: Sequence[int], xlabel: str = "", ylabel: str = "",
          zero: bool = False) -> str:
    """One figure. `curves`: dicts with colour, n, median, p10, p90, ceiling (or None).
    With `zero`, a solid line marks y = 0."""
    vals = [v for c in curves for v in list(c["p10"]) + list(c["p90"])
            + ([c["ceiling"]] if c["ceiling"] is not None else [])]
    vals = [v for v in vals if not math.isnan(v)]
    if zero:
        vals.append(0.0)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.05
    lo, hi = lo - pad, hi + pad
    sx, sy = _scales(ns, lo, hi)
    out = [f'<figure class="panel"><figcaption>{title}</figcaption>'
           f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">']
    for frac in (0.0, 1 / 3, 2 / 3, 1.0):
        v = lo + frac * (hi - lo)
        y = sy(v)
        out.append(f'<line x1="{X0:.0f}" y1="{y:.1f}" x2="{X1:.0f}" y2="{y:.1f}" style="stroke:{GRID};stroke-width:1"/>'
                   f'<text x="{X0 - 6:.0f}" y="{y + 3.5:.1f}" text-anchor="end" font-size="9" fill="{MUTED}" '
                   f'font-family="monospace">{v:.2f}</text>')
    if zero:
        y = sy(0.0)
        out.append(f'<line x1="{X0:.0f}" y1="{y:.1f}" x2="{X1:.0f}" y2="{y:.1f}" style="stroke:{INK};stroke-width:1"/>')
    for c in curves:
        if c["ceiling"] is None:
            continue
        y = sy(c["ceiling"])
        out.append(f'<line x1="{X0:.0f}" y1="{y:.1f}" x2="{X1:.0f}" y2="{y:.1f}" '
                   f'style="stroke:{c["colour"]};stroke-width:1.2;stroke-dasharray:4 3;opacity:0.9"/>')
    for c in curves:
        xs = [sx(n) for n in c["n"]]
        band = ([f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, c["p90"])]
                + [f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs[::-1], c["p10"][::-1])])
        out.append(f'<polygon points="{" ".join(band)}" style="fill:{c["colour"]};fill-opacity:0.13;stroke:none"/>')
        line = " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, c["median"]))
        out.append(f'<polyline points="{line}" style="fill:none;stroke:{c["colour"]};stroke-width:1.8"/>')
        for x, v in zip(xs, c["median"]):
            out.append(f'<circle cx="{x:.1f}" cy="{sy(v):.1f}" r="2.2" style="fill:{c["colour"]}"/>')
    last = {0: -1e9, 1: -1e9}
    for n in dict.fromkeys(ns):
        x = sx(n)
        row = 0 if x - last[0] >= 20 else 1
        last[row] = x
        out.append(f'<line x1="{x:.1f}" y1="{Y_AXIS}" x2="{x:.1f}" y2="{Y_AXIS + 3}" style="stroke:{AXIS};stroke-width:1"/>')
        out.append(f'<text x="{x:.1f}" y="{Y_TICK + 9 * row}" text-anchor="middle" font-size="8.5" fill="{MUTED}" '
                   f'font-family="monospace">{_fmt_n(n)}</text>')
    out.append(f'<line x1="{X0:.0f}" y1="{Y_AXIS}" x2="{X1:.0f}" y2="{Y_AXIS}" style="stroke:{AXIS};stroke-width:1"/>')
    if xlabel:
        out.append(f'<text x="{(X0 + X1) / 2:.0f}" y="{Y_LABEL}" text-anchor="middle" font-size="9.5" fill="{MUTED}">{xlabel}</text>')
    if ylabel:
        out.append(f'<text transform="rotate(-90)" x="{-(Y0 + Y1) / 2:.0f}" y="11" text-anchor="middle" '
                   f'font-size="9.5" fill="{MUTED}">{ylabel}</text>')
    return "".join(out) + "</svg></figure>"


def grid(results: Results, kind: str) -> str:
    axis = "test" if kind == "test" else "train"
    xlabel = "number of test materials" if kind == "test" else "number of training materials"
    ylabel = {"shortfall": "shortfall from ceiling", "gain": "embedding − Magpie"}.get(kind, "score (R² or macro-F1)")
    panels = []
    for target in results.targets():
        curves, seen = [], set()
        for arm in (EMBEDDINGS if kind == "gain" else ARMS):
            if kind == "gain":
                g = results.gain_rows(arm)
                g = g[g.target == target].sort_values("n")
                if g.empty:
                    continue
                seen.update(int(v) for v in g.n)
                curves.append(dict(colour=COLOUR[arm], n=list(g.n), median=list(g.med), p10=list(g.p10),
                                   p90=list(g.p90), ceiling=None))
                continue
            g = results.curve(arm, axis, target)
            if g.empty:
                continue
            seen.update(int(v) for v in g.n)
            if kind == "shortfall":
                curves.append(dict(colour=COLOUR[arm], n=list(g.n), median=list(g.gap), p10=list(g.gap),
                                   p90=list(g.gap), ceiling=0.0))
            else:
                curves.append(dict(colour=COLOUR[arm], n=list(g.n), median=list(g["median"]), p10=list(g.p10),
                                   p90=list(g.p90), ceiling=float(g.ceiling.iloc[0])))
        if curves:
            panels.append(panel(label(target), curves, sorted(seen), xlabel=xlabel, ylabel=ylabel,
                                zero=(kind == "gain")))
    return '<div class="grid">\n' + "\n".join(panels) + "\n</div>"


def gainmap(results: Results) -> str:
    gains = {arm: results.gain_rows(arm) for arm in EMBEDDINGS}
    gains = {a: g for a, g in gains.items() if len(g)}
    if not gains:
        return ""
    targets = ordered(set().union(*(set(g.target) for g in gains.values())))
    ns = sorted(set().union(*(set(int(v) for v in g.n) for g in gains.values())))
    WM, XL, XR, ROW, SUB, TOP = 700, 190.0, 660.0, 34.0, 12.0, 16.0
    HM = int(TOP + ROW * len(targets) + 72)
    ln = np.log(ns)

    def sx(n: float) -> float:
        return XL + (np.log(n) - ln.min()) / max(ln.max() - ln.min(), 1e-9) * (XR - XL)

    out = [f'<svg viewBox="0 0 {WM} {HM}" role="img" aria-label="whether the embedding beats composition alone, '
           f'per target and training size">']
    for n in ns:
        out.append(f'<line x1="{sx(n):.1f}" y1="{TOP}" x2="{sx(n):.1f}" y2="{TOP + ROW * len(targets):.1f}" '
                   f'style="stroke:{GRID};stroke-width:1"/><text x="{sx(n):.1f}" y="{TOP + ROW * len(targets) + 14:.1f}" '
                   f'text-anchor="middle" font-size="9" fill="{MUTED}" font-family="monospace">{_fmt_n(n)}</text>')
    for i, t in enumerate(targets):
        yc = TOP + ROW * (i + 0.5)
        if i % 2 == 0:
            out.append(f'<rect x="{XL - 14:.0f}" y="{yc - ROW / 2:.1f}" width="{XR - XL + 28:.0f}" height="{ROW:.0f}" '
                       f'style="fill:var(--band,#EEF1F2);opacity:.45"/>')
        out.append(f'<text x="{XL - 44:.0f}" y="{yc:.1f}" text-anchor="end" font-size="10.5" fill="{INK}" '
                   f'dominant-baseline="middle">{label(t)}</text>')
        for j, arm in enumerate(a for a in EMBEDDINGS if a in gains):
            y = yc + (j - 0.5) * SUB
            out.append(f'<text x="{XL - 22:.0f}" y="{y:.1f}" text-anchor="end" font-size="7.5" fill="{MUTED}" '
                       f'font-family="monospace" dominant-baseline="middle">{ARM_LABEL[arm]}</text>')
            g = gains[arm][gains[arm].target == t].set_index("n")
            for n in ns:
                if n not in g.index:
                    continue
                row = g.loc[n]
                if row.p10 > 0:
                    style = f"fill:{COLOUR[arm]}"
                elif row.p90 < 0:
                    style = f"fill:{MAG}"
                else:
                    style = f"fill:{PANEL};stroke:{AXIS};stroke-width:1.2"
                out.append(f'<circle cx="{sx(n):.1f}" cy="{y:.1f}" r="4.6" style="{style}"><title>{label(t)}, '
                           f'{MODEL_NAME[arm]} minus Magpie at {n:,}: median {row.med:+.3f}, '
                           f'p10 {row.p10:+.3f}, p90 {row.p90:+.3f}, embedding ahead on {100 * row.win:.0f} % of '
                           f'{int(row.k)} draws</title></circle>')
    out.append(f'<text x="{(XL + XR) / 2:.0f}" y="{TOP + ROW * len(targets) + 30:.1f}" text-anchor="middle" '
               f'font-size="9.5" fill="{MUTED}">number of training materials</text>')
    legend = [(f"fill:{ORB}", "ORB-v3 ahead on ≥ 90 % of draws"), (f"fill:{UMA}", "UMA-S ahead on ≥ 90 % of draws"),
              (f"fill:{MAG}", "composition ahead on ≥ 90 % of draws"),
              (f"fill:{PANEL};stroke:{AXIS};stroke-width:1.2", "draws disagree on the sign")]
    for i, (style, text) in enumerate(legend):
        x = 24 + (i % 2) * 300
        yb = HM - 20 + (i // 2) * 14
        out.append(f'<circle cx="{x + 5:.1f}" cy="{yb - 3.5}" r="4.2" style="{style}"/>'
                   f'<text x="{x + 14:.1f}" y="{yb}" font-size="9" fill="{MUTED}">{text}</text>')
    return "".join(out) + "</svg>"


def percell(results: Results, n: int = 5000) -> str:
    """Per-target shortfall bars at one training size, each against its own arm's ceiling."""
    PW, PX0, PX1, ROW = 700, 150.0, 660.0, 34.0
    rows = []
    for t in results.targets():
        vals = {}
        for arm in ARMS:
            g = results.curve(arm, "train", t)
            g = g[g.n == n]
            if len(g):
                vals[arm] = float(g.gap.iloc[0])
        if set(ARMS) <= vals.keys():
            rows.append((t, vals["orb"], vals["uma"], vals["magpie"]))
    if not rows:
        return ""
    rows.sort(key=lambda x: (x[1] + x[2]) / 2)
    HP = int(ROW * len(rows) + 46)
    hi = max(max(o, u, m) for _, o, u, m in rows) * 1.08 or 0.05
    out = [f'<svg viewBox="0 0 {PW} {HP}" role="img" aria-label="shortfall below the full-corpus score at {n:,} '
           f'training materials, per target, ORB, UMA and Magpie">']
    for frac in (0, .25, .5, .75, 1):
        x, v = PX0 + frac * (PX1 - PX0), frac * hi
        out.append(f'<line x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{HP - 30}" style="stroke:var(--rule);stroke-width:1"/>'
                   f'<text x="{x:.1f}" y="{HP - 18}" text-anchor="middle" font-size="9" fill="var(--muted)" '
                   f'font-family="monospace">{v:.2f}</text>')
    for i, (t, o, u, m) in enumerate(rows):
        y = 18 + i * ROW
        out.append(f'<text x="{PX0 - 10:.0f}" y="{y + ROW / 2:.1f}" text-anchor="end" font-size="10.5" fill="var(--ink)" '
                   f'dominant-baseline="middle">{label(t)}</text>')
        for j, (v, colour) in enumerate(((o, "var(--orb)"), (u, "var(--uma)"), (m, "var(--mag)"))):
            bh = ROW * 0.30
            by = y + ROW / 2 - 1.5 * bh + j * bh
            w = max(0.0, v) / hi * (PX1 - PX0)
            out.append(f'<rect x="{PX0:.1f}" y="{by:.1f}" width="{w:.1f}" height="{bh - 1.5:.1f}" style="fill:{colour}"/>')
            out.append(f'<text x="{PX0 + w + 5:.1f}" y="{by + bh / 2 - 0.5:.1f}" font-size="8.5" fill="var(--muted)" '
                       f'font-family="monospace" dominant-baseline="middle">{v:.3f}</text>')
    out.append(f'<line x1="{PX0:.1f}" y1="14" x2="{PX0:.1f}" y2="{HP - 30}" style="stroke:var(--axis);stroke-width:1"/>')
    out.append(f'<text x="{(PX0 + PX1) / 2:.0f}" y="{HP - 4}" text-anchor="middle" font-size="9.5" fill="var(--muted)">'
               f'shortfall below the full-corpus score, at {n:,} training materials</text>')
    return "".join(out) + "</svg>"


# ---- cross-validation panels ---------------------------------------------------------------
CW, CH, CX0, CX1, CY0, CY1 = 700, 250, 54, 664, 24, 196


def _cv_means(results: Results) -> pd.DataFrame | None:
    out = []
    for arm in EMBEDDINGS:
        c = results.cv_cells(arm)
        if not len(c):
            continue
        g = c.groupby("n").agg(gain=("red", "mean"), cv=("cv_cov", "mean"), rh=("rh_cov", "mean")).reset_index()
        g["emb"] = arm
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else None


def _cv_panels(df: pd.DataFrame, lo: float, hi: float, fmt: Callable, draw: Callable, ylabel: str = "") -> str:
    ns = sorted(df.n.unique())
    pad, pw = 18, (CX1 - CX0 - 18) / 2
    out = [f'<svg viewBox="0 0 {CW} {CH}" role="img">']
    if ylabel:
        out.append(f'<text transform="rotate(-90)" x="{-(CY0 + CY1) / 2:.0f}" y="11" text-anchor="middle" '
                   f'font-size="9.5" fill="{MUTED}">{ylabel}</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = lo + frac * (hi - lo)
        y = CY1 - frac * (CY1 - CY0)
        out.append(f'<line x1="{CX0}" y1="{y:.1f}" x2="{CX1}" y2="{y:.1f}" style="stroke:{GRID};stroke-width:1"/>'
                   f'<text x="{CX0 - 6}" y="{y + 3.5:.1f}" text-anchor="end" font-size="9" fill="{MUTED}" '
                   f'font-family="monospace">{fmt(v)}</text>')
    for i, emb in enumerate(EMBEDDINGS):
        colour = COLOUR[emb]
        px = CX0 + i * (pw + pad)
        sub = df[df.emb == emb].sort_values("n")
        step = pw / len(ns)
        out.append(f'<text x="{px + pw / 2:.0f}" y="{CH - 8}" text-anchor="middle" font-size="10" fill="{INK}" '
                   f'font-family="monospace">{MODEL_NAME[emb]} &#183; number of materials per dataset</text>')
        for j, n in enumerate(ns):
            cx = px + (j + 0.5) * step
            out.append(f'<text x="{cx:.1f}" y="{CY1 + 14:.0f}" text-anchor="middle" font-size="8" fill="{MUTED}" '
                       f'font-family="monospace">{_fmt_n(n)}</text>')
            row = sub[sub.n == n]
            if not row.empty:
                out += draw(row.iloc[0], cx, step, colour, lo, hi)
        out.append(f'<line x1="{px:.1f}" y1="{CY1}" x2="{px + pw:.1f}" y2="{CY1}" style="stroke:{AXIS};stroke-width:1"/>')
    return "".join(out) + "</svg>"


def _cy(v: float, lo: float, hi: float) -> float:
    return CY1 - (v - lo) / (hi - lo) * (CY1 - CY0)


def cv_gain(results: Results) -> str:
    df = _cv_means(results)
    if df is None:
        return ""
    lo, hi = -0.2, 0.5

    def draw(r, cx, step, colour, lo, hi):
        y0, y1 = _cy(0, lo, hi), _cy(min(max(r.gain, lo), hi), lo, hi)
        top, ht = min(y0, y1), abs(y1 - y0)
        bw = step * 0.45
        return [f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{ht:.1f}" '
                f'style="fill:{colour};fill-opacity:{0.85 if r.gain > 0 else 0.35}"/>',
                f'<text x="{cx:.1f}" y="{top - 3:.1f}" text-anchor="middle" font-size="8" fill="{MUTED}" '
                f'font-family="monospace">{r.gain * 100:+.0f}%</text>']
    svg = _cv_panels(df, lo, hi, lambda v: f"{v * 100:+.0f}%", draw, ylabel="RMSE reduction vs repeated holdout")
    zero = _cy(0, lo, hi)
    return svg.replace("</svg>", f'<line x1="{CX0}" y1="{zero:.1f}" x2="{CX1}" y2="{zero:.1f}" '
                                 f'style="stroke:{INK};stroke-width:1.2"/></svg>')


def cv_coverage(results: Results) -> str:
    df = _cv_means(results)
    if df is None:
        return ""
    lo, hi = 0.70, 1.00

    def draw(r, cx, step, colour, lo, hi):
        o = step * 0.16
        return [f'<circle cx="{cx - o:.1f}" cy="{_cy(min(max(r.rh, lo), hi), lo, hi):.1f}" r="3.2" style="fill:{INK}"/>',
                f'<circle cx="{cx + o:.1f}" cy="{_cy(min(max(r.cv, lo), hi), lo, hi):.1f}" r="3.2" style="fill:{colour}"/>']
    svg = _cv_panels(df, lo, hi, lambda v: f"{v:.2f}", draw, ylabel="coverage of the 95% interval")
    y = _cy(NOMINAL, lo, hi)
    return svg.replace("</svg>", f'<line x1="{CX0}" y1="{y:.1f}" x2="{CX1}" y2="{y:.1f}" '
                                 f'style="stroke:{INK};stroke-width:1;stroke-dasharray:4 3"/></svg>')
