"""docs/index.html from docs/template.html and results/.

The template is the authored page with every generated block replaced by a marker
<!--{{name}}-->. Every marker must have a generator and every generator a marker, so a missing
result or a renamed block fails the build.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import figures, tables
from .data import Results
from .page_data import build as build_data

log = logging.getLogger(__name__)
MARK = re.compile(r"<!--\{\{(\w+)\}\}-->")


def blocks(results: Results) -> dict[str, str]:
    return {
        "data": json.dumps(build_data(results), separators=(",", ":"), ensure_ascii=False),
        "fig_train": figures.grid(results, "train"),
        "fig_shortfall": figures.grid(results, "shortfall"),
        "fig_test": figures.grid(results, "test"),
        "fig_gain": figures.grid(results, "gain"),
        "fig_gainmap": figures.gainmap(results),
        "fig_percell": figures.percell(results, 5000),
        "tbl_cv_accuracy": tables.cv_accuracy(results),
        "tbl_cv_intervals": tables.cv_intervals(results),
        "fig_cv_gain": figures.cv_gain(results),
        "fig_cv_coverage": figures.cv_coverage(results),
        "tbl_design_train": tables.design_example(results, "train"),
        "tbl_design_test": tables.design_example(results, "test"),
        "tbl_train": tables.settle(results, "train"),
        "tbl_test": tables.settle(results, "test"),
        "tbl_gain_summary": tables.gain_summary(results),
        "tbl_gain_full": tables.gain_full(results),
        "tbl_split_median": tables.split_median(results),
        "tbl_split_wins": tables.split_wins(results),
        "floor_strip": tables.floor_strip(results),
        "fig_floors": figures.floor_panels(results),
        "tbl_floor_train": tables.training_floor(results),
        "tbl_floor_test": tables.test_floor(results),
        "tbl_floor_by_target": tables.floor_by_target(results),
        "tbl_shortfall_targets": tables.shortfall_by_size_targets(results),
        "tbl_cv_target": tables.cv_target(results, "band_gap"),
        "tbl_cv_budget": tables.cv_budget(results, "band_gap", "orb"),
    }


def render(template: str, filled: dict[str, str]) -> str:
    missing = [m for m in MARK.findall(template) if m not in filled]
    if missing:
        raise ValueError(f"template markers without a generator: {missing}")
    unused = [k for k in filled if f"<!--{{{{{k}}}}}-->" not in template]
    if unused:
        raise ValueError(f"generators without a template marker: {unused}")
    return MARK.sub(lambda m: filled[m.group(1)], template)


def build_page(results: Path, template: Path, output: Path) -> str:
    if not any(Path(results).glob("summary_*.csv")):
        raise FileNotFoundError(f"no summary_*.csv in {results}; run `mlip-probe summarise` first")
    filled = blocks(Results(results))
    empty = [k for k, v in filled.items() if not v]
    if empty:
        log.info("  empty blocks (no data yet): %s", ", ".join(empty))
    html = render(template.read_text(), filled)
    output.write_text(html)
    log.info("wrote %s (%s bytes)", output, f"{len(html):,}")
    return html
