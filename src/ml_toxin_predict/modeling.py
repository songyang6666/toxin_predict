from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return common regression metrics as JSON-serializable floats."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def knn_param_grid() -> dict[str, list]:
    """Hyperparameter grid used in the legacy toxin predict script."""
    return {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }


def fit_knn_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    param_grid: dict,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
    n_jobs: int = 1,
    verbose: int = 1,
):
    """Fit the KNN prediction model with train/test split and GridSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grid_search = GridSearchCV(
        estimator=KNeighborsRegressor(),
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=verbose,
    )
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    metadata = {
        "best_params": grid_search.best_params_,
        "best_cv_mse": float(-grid_search.best_score_),
        "cv": cv,
        "scoring": "neg_mean_squared_error",
        "train_metrics": regression_metrics(y_train, y_train_pred),
        "test_metrics": regression_metrics(y_test, y_test_pred),
    }
    predictions = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_test_pred,
        }
    )
    cv_results = pd.DataFrame(grid_search.cv_results_)
    return best_model, scaler, metadata["test_metrics"], predictions, metadata, cv_results


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
