"""Cost-based threshold optimization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def find_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> tuple[float, float, pd.DataFrame, float]:
    """
    Find the cost-minimizing threshold and the F1-maximizing threshold.

    Thresholds are scanned from 0.05 to 0.95 in 0.01 increments.
    """
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "total_cost": float((fn * fn_cost) + (fp * fp_cost)),
                "f1_default": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    results = pd.DataFrame(rows)
    cost_row = results.loc[results["total_cost"].idxmin()]
    f1_row = results.loc[results["f1_default"].idxmax()]
    return (
        float(cost_row["threshold"]),
        float(cost_row["total_cost"]),
        results,
        float(f1_row["threshold"]),
    )
