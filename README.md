# Total Microcystin Predict Workflow

This repository contains the reproducible Python workflow for the one-week
total microcystin prediction originally developed in
`ML_toxin_Collection_Kfold3_predict2.py`. It intentionally excludes unrelated
Chla-only experiments so the public repository stays focused on the toxin
analysis used for publication.

## What is included

```text
ML_toxin_Collection_Kfold3_predict2.py   Legacy-name compatibility entry point
run_toxin_predict.py                     Script entry point
src/ml_toxin_predict/                    Installable Python package
data/raw/                                 Required source workbook
data/processed/                           Optional generated intermediate data
data/outputs/                             Generated metrics, predictions, plots
docs/                                     Structure and WRR/AGU notes
tests/                                    Lightweight dataset smoke test
```

## Quick start

Clone the repository, create a virtual environment, install the package in
editable mode, and run the prediction workflow:

```bash
git clone https://github.com/YOUR-USER/YOUR-REPO.git
cd YOUR-REPO

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

toxin-predict
```

The command writes:

```text
data/outputs/toxin_predict/metrics.json
data/outputs/toxin_predict/predictions.csv
data/outputs/toxin_predict/cv_results.csv
```

The same workflow can also be run through either script:

```bash
python run_toxin_predict.py
python ML_toxin_Collection_Kfold3_predict2.py
```

## Reproduce the default analysis

By default, the workflow:

- reads `data/raw/western_lake_erie_water_quality_2012_2022.xlsx`;
- uses the sheet `2012-2022_surface_predict_toxi`;
- converts below-detection-limit values such as `<0.05` to half the reported
  detection limit;
- removes column outliers with Isolation Forest;
- imputes missing values with KNN imputation;
- linearly interpolates observations to daily values within each calendar year;
- predicts total microcystin seven days ahead;
- uses the model inputs from the legacy toxin script, including current total
  microcystin, Chla, phycocyanin, nutrients, and physical variables;
- trains a KNN regressor using an 80/20 random train/test split;
- tunes `n_neighbors`, `weights`, and `p` with 5-fold `GridSearchCV`;
- writes metrics, predictions, and cross-validation results to
  `data/outputs/toxin_predict/`.

Single-process grid search is the default for portability. Use `--n-jobs -1` on
systems that support parallel joblib execution.

## Optional processed data and interpretation outputs

```bash
toxin-predict \
  --write-processed \
  --make-explanations \
  --shap-samples 200 \
  --dependence-feature DO \
  --interaction-feature Cond
```

This writes:

- `data/processed/toxin_daily_interpolated.csv`;
- `data/processed/toxin_predict_dataset.csv`;
- permutation feature importance;
- SHAP values;
- SHAP summary and SHAP dependence plots.

The SHAP dependence plots are not partial dependence plots. They show
sample-level SHAP contributions across the observed feature distribution.

## Development checks

Install the optional test dependency and run the lightweight test suite:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

GitHub Actions runs the same smoke checks on push and pull request.

## Data and software availability

For Water Resources Research / AGU submission, archive a release of this
repository in Zenodo or a similar trusted repository and replace the placeholder
metadata in `CITATION.cff`. The source data should also be deposited or cited in
an appropriate repository, preferably with a DOI. See
`docs/WRR_OPEN_RESEARCH.md`.
