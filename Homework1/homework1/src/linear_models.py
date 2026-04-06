from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.metrics import mean_squared_error


@dataclass
class LinearRegressionModel:
    weights: np.ndarray
    history: list[float] = field(default_factory=list)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        design = _design_matrix(x)
        return design @ self.weights


def fit_least_squares_linear(x: np.ndarray, y: np.ndarray) -> LinearRegressionModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = _design_matrix(x)
    weights, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return LinearRegressionModel(weights=weights)


def fit_gradient_descent_linear(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 1e-2,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> LinearRegressionModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = _design_matrix(x)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    history: list[float] = []

    for _ in range(max_iter):
        pred = design @ weights
        error = pred - y
        loss = mean_squared_error(y, pred)
        history.append(loss)
        gradient = (2.0 / len(x)) * (design.T @ error)
        next_weights = weights - learning_rate * gradient
        if np.linalg.norm(next_weights - weights) < tolerance:
            weights = next_weights
            history.append(mean_squared_error(y, design @ weights))
            break
        weights = next_weights

    return LinearRegressionModel(weights=weights, history=history)


def fit_newton_linear(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 20,
    tolerance: float = 1e-10,
) -> LinearRegressionModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = _design_matrix(x)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    history: list[float] = []
    hessian = (2.0 / len(x)) * (design.T @ design)

    for _ in range(max_iter):
        pred = design @ weights
        error = pred - y
        history.append(mean_squared_error(y, pred))
        gradient = (2.0 / len(x)) * (design.T @ error)
        step = np.linalg.solve(hessian, gradient)
        next_weights = weights - step
        weights = next_weights
        if np.linalg.norm(step) < tolerance:
            break

    return LinearRegressionModel(weights=weights, history=history)


def _design_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.column_stack([np.ones_like(x), x])
