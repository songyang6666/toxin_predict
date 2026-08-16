from __future__ import annotations

import os
from pathlib import Path

from .config import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def write_permutation_importance(
    model,
    X_scaled,
    y,
    feature_names: list[str],
    output_path: Path,
    *,
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Write model-agnostic permutation feature importance on the test split."""
    result = permutation_importance(
        model,
        X_scaled,
        y,
        scoring="neg_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    total = frame["importance_mean"].sum()
    frame["importance_normalized"] = frame["importance_mean"] / total if total else 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def write_shap_analysis(
    model,
    X_model_df: pd.DataFrame,
    X_display_df: pd.DataFrame,
    output_dir: Path,
    *,
    max_samples: int = 300,
    random_state: int = 42,
    dependence_feature: str = "DO",
    interaction_feature: str = "Cond",
) -> None:
    """Write SHAP summary values and dependence plots for observed samples.

    SHAP dependence plots are not partial dependence plots. They show the
    sample-level SHAP contribution for a feature across the observed feature
    distribution, optionally colored by another feature to reveal interactions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(X_model_df) > max_samples:
        X_sample = X_model_df.sample(max_samples, random_state=random_state)
    else:
        X_sample = X_model_df.copy()
    X_display_sample = X_display_df.loc[X_sample.index]

    def predict(values):
        frame = pd.DataFrame(values, columns=X_sample.columns)
        return model.predict(frame)

    explainer = shap.KernelExplainer(predict, X_sample)
    shap_values = explainer(X_sample)
    shap_frame = pd.DataFrame(shap_values.values, columns=X_sample.columns, index=X_sample.index)
    shap_frame.to_csv(output_dir / "shap_values.csv", index_label="sample_index")

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=300)
    plt.close()

    feature = dependence_feature if dependence_feature in X_sample.columns else X_sample.columns[0]
    color_feature = (
        interaction_feature if interaction_feature in X_sample.columns else X_sample.columns[min(1, len(X_sample.columns) - 1)]
    )
    y = shap_frame[feature]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_display_sample[feature], y, s=20, alpha=0.75)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    fig.tight_layout()
    fig.savefig(output_dir / f"shap_dependence_{feature}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter = ax.scatter(
        X_display_sample[feature],
        y,
        c=X_display_sample[color_feature],
        s=24,
        alpha=0.8,
        cmap="plasma",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_feature)
    fig.tight_layout()
    fig.savefig(output_dir / f"shap_dependence_{feature}_colored_by_{color_feature}.png", dpi=300)
    plt.close(fig)
