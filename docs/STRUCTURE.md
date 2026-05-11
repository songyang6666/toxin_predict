# Code Structure

## Data flow

```text
data/raw/western_lake_erie_water_quality_2012_2022.xlsx
  -> src/ml_toxin_predict/datasets.py
  -> src/ml_toxin_predict/preprocessing.py
  -> src/ml_toxin_predict/predicting.py
  -> src/ml_toxin_predict/modeling.py
  -> src/ml_toxin_predict/explainability.py
  -> data/outputs/toxin_predict/
```

## Modules

- `config.py`: project paths.
- `datasets.py`: workbook loading and source-column mapping.
- `preprocessing.py`: detection-limit conversion, missing-value handling,
  Isolation Forest outlier removal, and KNN imputation.
- `predicting.py`: within-year daily interpolation and shifted prediction target
  construction.
- `modeling.py`: KNN regressor, feature standardization, GridSearchCV, metrics,
  and JSON output.
- `explainability.py`: permutation importance and SHAP summary/dependence
  outputs.

## Entry points

- `run_toxin_predict.py`: primary reproducible workflow.
- `ML_toxin_Collection_Kfold3_predict2.py`: compatibility wrapper preserving
  the legacy script name used in earlier analysis notes.
- `toxin-predict`: console command installed by `python -m pip install -e .`.
