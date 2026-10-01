# How much labelled data does a probe on a frozen MLIP embedding need?

Machine-learned interatomic potentials (MLIPs) are trained on energies and forces, yet their
internal embeddings also predict properties they never saw: band gaps, densities, crystal
systems. A small probe trained on a frozen embedding is the usual way to read those out. This
repository measures how many labelled materials that probe needs: how many before its score
stops changing from one sample to the next, and how many before it stops improving.

The measurement covers learning curves for XGBoost probes on two frozen MLIP embeddings,
ORB-v3 and UMA-S, and on the composition-only Magpie features. They run over 154,875 Materials
Project crystals and nine properties. Training-set size and test-set size are varied
separately, and every training draw is shared across the three feature sets. The difference
between an embedding and composition alone is therefore measured on identical materials.

**Results page:** https://natalija-stepurko.github.io/mlip-probe-data-requirements/ has
interactive figures, a calculator for a given dataset size, and every table. The page is built
from [`docs/template.html`](docs/template.html) and [`results/`](results).

Companion to Stepurko, N. & Laulainen, J. (2026), *Out-of-training properties in frozen
interatomic potential embeddings are decodable and ordered by physical scale*. The paper was
accepted at the AI4Sci and AI4Mat workshops, NeurIPS 2026. The paper, its code and the Hugging
Face datasets with the embeddings are released in November 2026.

## Findings

Each cell reads ORB-v3 / UMA-S, for the XGBoost probe. Sizes are numbers of materials.

| target | training size where the score is stable | training size where it is within 0.05 of the full-corpus score | test size where the score is stable | training size from which the embedding beats composition |
|---|---|---|---|---|
| formation energy / atom | 500 / 500 | 2,000 / 1,000 | 20 / 20 | 200 / 100 |
| energy above hull | 2,000 / 2,000 | 10,000 / 5,000 | 1,000 / 1,000 | 50 / 50 |
| band gap | 500 / 500 | 7,000 / 7,000 | 5,000 / 5,000 | 100 / 50 |
| density | 1,000 / 500 | 7,000 / 7,000 | 200 / 200 | 2,000 / 2,000 |
| magnetic density | 1,000 / 500 | 20,000 / 20,000 | 5,000 / 5,000 | not by 20,000 |
| bulk modulus | 500 / 500 | 2,000 / 2,000 | 500 / 500 | 1,000 / 500 |
| crystal system | 500 / 50 | 20,000 / 20,000 | 1,000 / 1,000 | 500 / 50 |
| metal / insulator | 100 / 100 | 1,000 / 1,000 | 500 / 500 | 100 / 50 |
| magnetic / not | 100 / 200 | 5,000 / 2,000 | 500 / 500 | 200 / 50 |

- **Stable** means the p10–p90 spread of the score across draws is at most 0.05 at that size
  and at every larger size measured. The full definitions are under *Design* below.
- **Stability comes early; saturation comes late.** Every target gives a reproducible score
  from between 100 and 2,000 training materials. Getting within 0.05 of the full-corpus score takes
  between 1,000 and 20,000, depending on the target. The largest training size measured is
  20,000, or 5,000 for bulk modulus, whose corpus is smaller.
- **Test size sets reproducibility separately.** Band gap and magnetic density need 5,000
  test materials before the spread across re-drawn test sets falls to 0.05; formation energy
  needs 20.
- **The embeddings beat composition early on most targets.** On six of the nine targets both
  embeddings beat Magpie on at least 90 % of paired draws from 500 training materials or
  fewer. Density needs 2,000. On magnetic density, composition alone is never
  beaten within the measured range.
- **Cross-validation helps a little on small datasets.** Over the 72 cells of 100 to 1,000
  materials (two embeddings, nine targets, four sizes), 5-fold cross-validation estimates the
  score with a median 17 % lower RMSE than five repeated 80/20 splits. Every estimator trains
  on 80 % of the dataset, so all three run low against a probe trained on all of it. Averaged
  over the 90 cells, their nominal 95 % intervals cover that score in 81 % (cross-validation),
  85 % (repeated splits) and 88 % (one split) of draws.

The numbers come from [`results/thresholds.csv`](results/thresholds.csv),
[`results/paired_gain.csv`](results/paired_gain.csv) and
[`results/cv_vs_holdout.csv`](results/cv_vs_holdout.csv). The page also covers the
per-target shortfall, the choice of split fraction and a minimum dataset size.

## Design

- **Frozen test set per target.** A seeded permutation of the labelled materials. The first
  20,000, or half of them for sparse targets, are the test set; the rest are the training
  pool.
- **Training axis.** At each size from 50 to 20,000, disjoint draws from the pool: 30 draws
  up to 2,000, 10 at 5,000 and 7 above. A probe is fitted on each draw and scored on the frozen
  test set.
- **Test axis.** One probe is fitted on 20,000 training materials. Its predictions are
  re-scored on 200 random subsamples of the test set at each size from 20 to 20,000.
- **Pairing.** Draws depend only on the target and the size. ORB-v3, UMA-S and Magpie are
  therefore fitted on the same materials, and the embedding-minus-composition difference is
  paired draw by draw.
- **Scores and ceilings.** R² for regression, macro-F1 for classification. The ceiling is
  the same probe's score on the full corpus, taken from the paper's reference results
  ([`results/ceilings.csv`](results/ceilings.csv)).
- **Summaries.**
  - *Spread* is the p10–p90 width, estimated as 2.5631 × the standard deviation across draws.
  - *Stable* is the smallest size from which the spread stays at or below 0.05.
  - *Saturated* is the smallest size from which the gap to the ceiling stays at or below 0.05.

Probe hyperparameters and seeds are in [`src/mlip_probe/config.py`](src/mlip_probe/config.py)
and [`src/mlip_probe/probes.py`](src/mlip_probe/probes.py). A linear probe is measured
alongside XGBoost and kept in the raw tables.

## Repository

```
src/mlip_probe/
  config.py      targets, feature sets, seeds, thresholds, and the Design grids
  dataset.py     load the dataset release and align every feature set to one row order
  sampling.py    frozen test sets and disjoint training draws
  probes.py      XGBoost and linear probes, scoring
  curves.py      the training and test axes (resumable), frozen test ids, ceilings
  analysis.py    summaries, thresholds, paired gain, split-fraction error budget
  crossval.py    k-fold cross-validation against single and repeated holdout
  cli.py         the `mlip-probe` command
  site/          builds docs/index.html from results/ and docs/template.html
results/         raw per-draw scores and every derived table (PROVENANCE.md)
docs/            the results page: template.html (authored) and index.html (built)
tests/           unit tests, an end-to-end run on synthetic data, and a rebuild of results/
scripts/         reproduce.sh (full re-run), check_page.js (runs the page in jsdom)
```

## Running it

Python 3.11 or newer. `requirements-lock.txt` pins the exact environment that produced
`results/`.

```bash
make install        # pip install -r requirements-lock.txt && pip install -e ".[test]"
make test           # about a minute; no dataset needed
make page           # rebuild docs/index.html from results/
```

`make test` includes a check anyone can run. It regenerates every summary, threshold,
paired-gain and split-fraction table from the committed raw draws. It rebuilds the page, and
compares all of them byte for byte with the committed files.

Re-measuring from scratch needs the paper's datasets, released in November 2026. There are
two: the Materials Project metadata, targets, ORB-v3 embeddings, Magpie features and
reference scores; and the access-gated UMA-S embeddings.

```bash
make reproduce DATASET=/path/to/orb-release UMA=/path/to/uma-release OUT=work/results
# or step by step:
mlip-probe design   --dataset DIR                       # the draw design of every target
mlip-probe curves   --arm orb --axis train --dataset DIR --out work/results
mlip-probe summarise --out work/results
```

Learning curves take about 40 minutes for Magpie, 1 h 20 for ORB-v3 and 2 h 20 for UMA-S on
8 CPU cores. Every step resumes where it stopped. In the locked environment every score is
deterministic for a fixed thread count. A different `--n-jobs` changes about one XGBoost score
in 500, in the fourth decimal. The number of BLAS threads moves linear-probe scores in the
sixth decimal, and newer scikit-learn versions move them by about 0.001.

The cross-validation experiment (`mlip-probe cv`) compares 5-fold cross-validation with single
and repeated holdout on datasets of 100 to 2,000 materials, 20 draws per cell. It took 7 to 13 hours
per embedding with 4 threads on a shared machine.

## Data and licences

Code: Apache-2.0 ([`LICENSE`](LICENSE)). Result tables: CC-BY-4.0
([`LICENSE-DATA`](LICENSE-DATA)). The inputs are Materials Project data (CC-BY-4.0), ORB-v3
embeddings (Apache-2.0) and UMA-S embeddings (FAIR Chemistry License, access-gated). The
ORB-v3 and Magpie arms run without the UMA-S release.

## Citation

If you use these results, cite the paper ([`CITATION.cff`](CITATION.cff)):

> Stepurko, N. & Laulainen, J. (2026). Out-of-training properties in frozen interatomic
> potential embeddings are decodable and ordered by physical scale. AI4Sci and AI4Mat
> workshops, NeurIPS 2026.
