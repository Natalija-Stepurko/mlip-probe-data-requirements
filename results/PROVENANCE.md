# Provenance

Every file here is written by this repository's code (`scripts/reproduce.sh`, or the
`mlip-probe` commands it runs) from the paper's dataset release.

| | |
|---|---|
| measured | learning curves 2026-09-30; cross-validation 2026-09-30 to 2026-10-01 |
| inputs | 154,875 Materials Project crystals: targets, ORB-v3 and UMA-S pooled node embeddings (backbone layer), Magpie composition features, and the paper's full-corpus probe scores |
| environment | Python 3.12, `requirements-lock.txt` (numpy 2.4.4, pandas 2.3.3, scikit-learn 1.8.0, xgboost 3.3.0) |
| machine | 8-core CPU, no GPU |

| file | written by | what it holds |
|---|---|---|
| `raw/curves_{train,test}_{arm}.csv` | `mlip-probe curves` | one score per target, probe, size and draw |
| `frozen_test_ids.csv` | `mlip-probe curves` | the materials of every target's frozen test set |
| `ceilings.csv` | `mlip-probe ceilings` | the full-corpus reference score of every arm, target and probe |
| `summary_{arm}.csv`, `thresholds.csv` | `mlip-probe summarise` | median, p10, p90, spread and gap per curve point; stable and saturated sizes |
| `paired_gain.csv` | `mlip-probe gain` | embedding minus Magpie over paired draws |
| `split_fraction.csv` | `mlip-probe split-fraction` | total error of one train/test split, per dataset size and test fraction |
| `raw/cv_vs_holdout_{orb,uma}.csv` | `mlip-probe cv` | per draw: the n-material truth and each estimator's score and 95 % interval |
| `cv_vs_holdout.csv` | `mlip-probe cv-summarise` | bias, sd, RMSE, interval coverage and width per estimator and cell |

## Checks

- **Derived tables.** Every table except the raw draws, `frozen_test_ids.csv` and
  `ceilings.csv` is regenerated from those three by the test suite. The page is rebuilt from
  them too, and every file is compared byte for byte (`tests/test_committed_results.py`).
- **Raw draws.** Every score is deterministic for a fixed thread count. Re-running
  `mlip-probe curves` in the locked environment reproduced the committed rows exactly, apart
  from about one XGBoost score in 500 that moves in the fourth decimal with `--n-jobs`. The
  number of BLAS threads moves linear-probe scores in the sixth decimal. Re-running
  `mlip-probe cv` reproduced the committed band-gap rows at 100 and 200 materials exactly.
