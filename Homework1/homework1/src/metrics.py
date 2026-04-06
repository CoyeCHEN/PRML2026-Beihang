from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(residual ** 2))
