"""
evaluate.py
───────────
Evaluation utilities shared between training and inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    label: str = "Model",
    price_scale: float = 100_000,
) -> dict:
    """
    Compute MAE, RMSE, R², and MAPE.

    Args:
        y_true:       Ground-truth target values (in $100k units).
        y_pred:       Model predictions (in $100k units).
        label:        Name shown in the printed report.
        price_scale:  Multiplier to convert units back to dollars.

    Returns:
        dict with keys: mae, rmse, r2, mape
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — skip rows where actual is zero to avoid division error
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"\n{'─' * 45}")
    print(f"  {label} — Evaluation Report")
    print(f"{'─' * 45}")
    print(f"  MAE   : ${mae  * price_scale:>12,.0f}   (avg error per house)")
    print(f"  RMSE  : ${rmse * price_scale:>12,.0f}   (penalises large errors)")
    print(f"  R²    :  {r2:>12.4f}   (1.0 = perfect)")
    print(f"  MAPE  :  {mape:>11.2f}%   (mean abs % error)")
    print(f"{'─' * 45}\n")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def feature_importance_report(
    model,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Print and return feature importances for tree-based models,
    or coefficients for linear models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        kind = "importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        kind = "abs_coefficient"
    else:
        print("Model does not expose feature importances or coefficients.")
        return pd.DataFrame()

    report = (
        pd.DataFrame({"feature": feature_names, kind: importances})
        .sort_values(kind, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} features by {kind}:")
    print(report.to_string(index=False))
    return report
