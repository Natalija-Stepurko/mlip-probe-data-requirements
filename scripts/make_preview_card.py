"""Draw docs/card.png, the 1200 x 630 image shown when the page's link is shared.

    python scripts/make_preview_card.py [--results results] [--out docs/card.png]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlip_probe.config import ARM_COLOURS  # noqa: E402
from mlip_probe.site.data import MODEL_NAME, Results  # noqa: E402

TITLE = "When is a probe score\ntrustworthy under limited data?"
LEDE = ("How many labelled materials a probe on\n"
        "frozen ORB-v3 and UMA-S embeddings needs:\n"
        "learning curves on 154,875 Materials\n"
        "Project crystals, nine properties, and a\n"
        "composition-only baseline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("docs/card.png"))
    a = ap.parse_args()
    res = Results(a.results)
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor="#F7F8F9")
    fig.text(0.05, 0.86, TITLE, fontsize=27, fontweight="bold", color="#16191D", va="top", linespacing=1.15)
    fig.text(0.05, 0.56, LEDE, fontsize=15, color="#5B646E", va="top", linespacing=1.4)
    fig.text(0.05, 0.08, "natalija-stepurko.github.io/mlip-probe-data-requirements", fontsize=12,
             color="#5B646E", family="monospace")
    ax = fig.add_axes([0.55, 0.17, 0.40, 0.42])
    for arm in ("orb", "uma"):
        c = res.curve(arm, "train")
        c = c[c.target != "bulk_modulus"].groupby("n").gap.median()
        ax.plot(c.index, c.values, marker="o", ms=4, lw=2.4, color=ARM_COLOURS[arm], label=MODEL_NAME[arm])
    ax.axvline(5000, color="#16191D", lw=1, ls=":")
    ax.text(4400, 0.30, "training floor\n5,000", fontsize=10.5, color="#16191D", va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xticks([50, 500, 5000, 20000], ["50", "500", "5,000", "20,000"])
    ax.minorticks_off()
    ax.set_xlabel("training materials", fontsize=12, color="#5B646E")
    ax.set_ylabel("median shortfall from\nthe full-data score", fontsize=12, color="#5B646E")
    ax.tick_params(colors="#5B646E", labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor("#F7F8F9")
    ax.legend(frameon=False, fontsize=11)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100, facecolor=fig.get_facecolor())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
