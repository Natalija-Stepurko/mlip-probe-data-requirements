"""Command line: `mlip-probe <command>` or `python -m mlip_probe <command>`.

    design          print the draw design of every target
    ceilings        write results/ceilings.csv from the dataset's reference scores
    curves          one arm, one axis of the learning curves (resumable)
    summarise       summary_{arm}.csv and thresholds.csv from the raw draws
    gain            paired_gain.csv
    split-fraction  split_fraction.csv
    cv              the cross-validation experiment for one arm (resumable)
    cv-summarise    cv_vs_holdout.csv
    plot            a PNG of the XGBoost training curves
    page            docs/index.html from results/ and docs/template.html
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import analysis, crossval, curves
from .config import ARMS, DEFAULT_DESIGN, TARGETS
from .dataset import Inputs, load, load_labels
from .sampling import design_table
from .site.build import build_page

log = logging.getLogger("mlip_probe")


def _targets(arg: str | None) -> list[str]:
    targets = arg.split(",") if arg else list(TARGETS)
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown targets {unknown}; valid: {list(TARGETS)}")
    return targets


def _load(a: argparse.Namespace, arm: str, need_features: bool = True) -> Inputs:
    return load(arm, a.dataset, a.uma_dataset, need_features)


def _record(out: Path, inputs: Inputs) -> None:
    out.mkdir(parents=True, exist_ok=True)
    curves.record_frozen_ids(out, inputs, DEFAULT_DESIGN)
    curves.record_ceilings(out, inputs)


def cmd_design(a: argparse.Namespace) -> None:
    _, targets = load_labels(a.dataset)
    for r in design_table(targets, DEFAULT_DESIGN):
        print(f"  {r['target']:<22} {r['task']} labelled={r['labelled']:>7,} test={r['n_test']:>6,} "
              f"pool={r['n_pool']:>7,}  draws: {r['draws']}")


def cmd_ceilings(a: argparse.Namespace) -> None:
    # Every arm's reference scores are in the public release, so the gated UMA release is not needed.
    a.out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        inputs = _load(a, arm, need_features=False)
        curves.record_ceilings(a.out, inputs)
        log.info("  %s: %d reference scores", arm, len(inputs.ceilings))


def cmd_curves(a: argparse.Namespace) -> None:
    targets = _targets(a.targets)
    inputs = _load(a, a.arm)
    _record(a.out, inputs)
    log.info("%s: %s materials, %d features; ceilings for %d (target, probe) pairs",
             a.arm, f"{len(inputs.mp_ids):,}", inputs.X.shape[1], len(inputs.ceilings))
    run = curves.run_train_axis if a.axis == "train" else curves.run_test_axis
    run(inputs, a.out, targets, a.n_jobs, DEFAULT_DESIGN)


def cmd_summarise(a: argparse.Namespace) -> None:
    analysis.summarise(a.out)


def cmd_gain(a: argparse.Namespace) -> None:
    analysis.paired_gain(a.out)


def cmd_split_fraction(a: argparse.Namespace) -> None:
    analysis.split_fraction(a.out)


def cmd_cv(a: argparse.Namespace) -> None:
    targets = _targets(a.targets)
    inputs = _load(a, a.arm)
    _record(a.out, inputs)
    crossval.run(inputs, a.out, targets, a.draws, a.n_jobs, DEFAULT_DESIGN)


def cmd_cv_summarise(a: argparse.Namespace) -> None:
    crossval.summarise(a.out)


def cmd_plot(a: argparse.Namespace) -> None:
    analysis.plot_train_curves(a.out, a.path)
    log.info("wrote %s", a.path)


def cmd_page(a: argparse.Namespace) -> None:
    build_page(a.results, a.template, a.output)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mlip-probe", description="Learning curves of probes on frozen MLIP "
                                 "embeddings. Run `mlip-probe <command> -h` for options.")
    sub = ap.add_subparsers(dest="command", required=True)

    def command(name: str, func, summary: str, dataset=False, uma=True, out=True, arm=False, targets=False,
                jobs=False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=summary, description=summary)
        p.set_defaults(func=func)
        if dataset:
            p.add_argument("--dataset", type=Path, required=True, help="the dataset release directory")
        if dataset and uma:
            p.add_argument("--uma-dataset", type=Path, help="the gated UMA release (default: DATASET/uma)")
        if out:
            p.add_argument("--out", type=Path, default=Path("results"), help="results directory (default: results)")
        if arm:
            p.add_argument("--arm", choices=ARMS, required=True,
                           help="feature set: orb or uma (MLIP embeddings), magpie (composition)")
        if targets:
            p.add_argument("--targets", help="comma-separated; default all nine")
        if jobs:
            p.add_argument("--n-jobs", type=int, default=8, help="XGBoost threads (default: 8)")
        return p

    command("design", cmd_design, "print the draw design of every target", dataset=True, uma=False, out=False)
    command("ceilings", cmd_ceilings, "write ceilings.csv from the dataset's reference scores", dataset=True)
    command("curves", cmd_curves, "learning curves of one arm and axis", dataset=True, arm=True, targets=True,
            jobs=True).add_argument("--axis", choices=["train", "test"], required=True,
                                    help="train: vary the training-set size; test: vary the test-set size")
    command("summarise", cmd_summarise, "summaries and thresholds from the raw draws")
    command("gain", cmd_gain, "paired embedding-minus-Magpie gain")
    command("split-fraction", cmd_split_fraction, "total error per split fraction")
    command("cv", cmd_cv, "cross-validation vs holdout for one arm", dataset=True, arm=True, targets=True,
            jobs=True).add_argument("--draws", type=int, default=DEFAULT_DESIGN.cv_draws,
                                    help=f"datasets per (target, n) (default: {DEFAULT_DESIGN.cv_draws})")
    command("cv-summarise", cmd_cv_summarise, "summarise the cross-validation runs")
    command("plot", cmd_plot, "PNG of the XGBoost training curves").add_argument(
        "--path", type=Path, default=Path("work/curves_train.png"), help="output PNG (default: work/curves_train.png)")
    page = command("page", cmd_page, "build docs/index.html from results/ and the template", out=False)
    page.add_argument("--results", type=Path, default=Path("results"), help="(default: results)")
    page.add_argument("--template", type=Path, default=Path("docs/template.html"), help="(default: docs/template.html)")
    page.add_argument("--output", type=Path, default=Path("docs/index.html"), help="(default: docs/index.html)")
    return ap


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = _parser().parse_args(argv)
    try:
        a.func(a)
    except (FileNotFoundError, ValueError) as e:   # bad or missing inputs: one line, no traceback
        raise SystemExit(f"error: {e}") from e
