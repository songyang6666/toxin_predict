# Total Microcystin Prediction Workflow

This repository contains the reproducible Python workflow for the one-week
total microcystin prediction originally developed in
`ML_toxin_Collection_Kfold3_predict2.py`. It intentionally excludes unrelated
Chla-only experiments so the public repository stays focused on the toxin
analysis used for publication.

## Repository Layout

```text
ML_toxin_Collection_Kfold3_predict2.py   Legacy-name compatibility entry point
run_toxin_predict.py                     Script entry point
compare_toxin_predict_horizons.py        One-to-seven-day comparison entry point
src/ml_toxin_predict/                    Installable Python package
data/raw/                                 Required source workbook
data/processed/                           Optional generated intermediate data
data/outputs/                             Generated metrics, predictions, plots
docs/                                     WRR/AGU open research notes
tests/                                    Lightweight dataset smoke test
```

## Quick start

Clone the repository, create a virtual environment, install the package in
editable mode, and run the prediction workflow:

```bash
git clone https://github.com/songyang6666/toxin_predict.git
cd toxin_predict

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
- linearly interpolates observations to daily values within each monitoring
  site and calendar year;
- predicts total microcystin seven days ahead;
- prevents shifted targets from crossing site or calendar-year boundaries;
- uses the model inputs from the legacy toxin script, including current total
  microcystin, Chla, phycocyanin, nutrients, and physical variables;
- trains a KNN regressor using an 80/20 random train/test split;
- tunes `n_neighbors`, `weights`, and `p` with 5-fold `GridSearchCV`;
- fits standardization independently within every cross-validation fold using a
  scikit-learn `Pipeline`;
- writes metrics, predictions, and cross-validation results to
  `data/outputs/toxin_predict/`.

Single-process grid search is the default for portability. Use `--n-jobs -1` on
systems that support parallel joblib execution.

## Compare prediction horizons

Run the reproducible 1-7 day comparison with:

```bash
toxin-predict-horizons --horizons 1 2 3 4 5 6 7
```

The equivalent repository script is:

```bash
python compare_toxin_predict_horizons.py
```

The comparison reuses one cleaned and daily-interpolated dataset for all seven
horizons. It writes a summary table, a JSON record, an R2/RMSE comparison plot,
and complete metrics, predictions, and cross-validation results for each
horizon under `data/outputs/horizon_1_7_comparison/`.

Prediction CSV files include the source row index, date, and monitoring site so
held-out samples can be traced back to the processed dataset.

The default random split reproduces the legacy workflow. Because daily
interpolation creates temporally related samples, these scores should be
reported with that design choice clearly stated and interpreted alongside an
appropriate temporal sensitivity analysis.

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

## Author

This repository was developed by Yang Song, Independent Researcher.

- ORCID: https://orcid.org/0000-0002-7120-4583
- Contact: songyangscu@hotmail.com

## Citation

If you use or adapt this workflow, please cite the archived software release DOI
and the associated manuscript when they become available. Citation metadata are
provided in `CITATION.cff`.

## License

The analysis code is released under the MIT License. The underlying NOAA data
are governed by their original public data terms and should be cited separately.

## Data and software availability

For Water Resources Research / AGU submission, archive a release of this
repository in Zenodo or a similar trusted repository, then add the final
software DOI to the citation metadata. The source data should also be deposited
or cited in an appropriate repository, preferably with a DOI. See
`docs/WRR_OPEN_RESEARCH.md`.
