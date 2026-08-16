# Non-interpolated weekly blocked validation

This analysis is a sensitivity test designed to address temporal dependence,
interpolation, and baseline-comparison concerns. It is separate from the legacy
daily-interpolated random-split analysis.

## Weekly observed-pair dataset

Each input row is paired only with the immediately following observed sample
from the same monitoring site. Pairs are retained when the actual interval is
5-9 days, representing a nominal 7-day prediction with a +/-2-day observation
window. No response or predictor is temporally interpolated. The current and
future total microcystin values must both be observed; other missing predictors
are imputed within each training fold. All observed toxin targets are retained;
the sensitivity analysis does not apply an outcome-based outlier filter.

## Blocked validation

Two outer validation designs are reported:

- `year`: leave one complete target year out;
- `station_year`: leave one complete monitoring-site target-year block out.

Candidate model hyperparameters are selected inside each outer training set
with grouped inner cross-validation. Median imputation and, where appropriate,
standardization are included in the scikit-learn Pipeline and are therefore
fitted using training-fold data only. Predictions are clipped at zero because
negative concentrations are not physically meaningful.

## Candidate models

- K-nearest neighbors (KNN)
- Linear Regression
- second-order Polynomial Regression with Ridge regularization
- Random Forest
- Extreme Gradient Boosting (XGBoost)
- artificial neural network implemented with `MLPRegressor`
- support vector machine implemented with RBF-kernel `SVR`
- Extra Trees, included as an additional small-tabular-data ensemble

The search grids are deliberately compact because the observed-pair dataset is
small. Linear Regression has no tuned hyperparameters. All stochastic models
use a fixed random seed.

## Baselines

- `persistence`: the current observed total microcystin concentration.
- `seasonal_climatology`: the mean target concentration for the same site and
  calendar month calculated from outer-fold training data only. If unavailable,
  the training month mean and then the global training mean are used.

## Event metrics

An event is total microcystin greater than or equal to 1 microgram/L.

- POD = hits / (hits + misses)
- FAR = false alarms / (hits + false alarms)
- CSI = hits / (hits + misses + false alarms)

Undefined fold-level ratios are stored as missing values. Summary values are
the unweighted mean and sample standard deviation across valid held-out blocks;
the CSV also records the number of valid folds for every metric. Supplementary
pooled metrics are written separately and do not replace the requested
block-level `mean +/- SD` results.

## Reproduction

```bash
toxin-predict-weekly \
  --horizon-days 7 \
  --tolerance-days 2 \
  --split-types year station_year \
  --models knn linear polynomial random_forest xgboost ann svm extra_trees \
  --threshold 1.0
```
