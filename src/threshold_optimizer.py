"""Cost-based and F1 threshold optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    """Named threshold optimization output to avoid ambiguous tuple positions."""

    cost_minimizing_threshold: float
    min_cost: float
    f1_optimal_threshold: float
    f1_at_optimal: float
    threshold_table: pd.DataFrame
    fn_cost: float
    fp_cost: float

    def to_summary_dict(self) -> dict[str, float]:
        """Return JSON-serializable scalar threshold policy fields."""
        return {
            "cost_minimizing_threshold": self.cost_minimizing_threshold,
            "min_cost": self.min_cost,
            "f1_optimal_threshold": self.f1_optimal_threshold,
            "f1_at_optimal": self.f1_at_optimal,
            "fn_cost": self.fn_cost,
            "fp_cost": self.fp_cost,
        }


def find_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> ThresholdOptimizationResult:
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
    return ThresholdOptimizationResult(
        cost_minimizing_threshold=float(cost_row["threshold"]),
        min_cost=float(cost_row["total_cost"]),
        f1_optimal_threshold=float(f1_row["threshold"]),
        f1_at_optimal=float(f1_row["f1_default"]),
        threshold_table=results,
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
