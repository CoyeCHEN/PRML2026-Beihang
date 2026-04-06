from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.metrics import mean_squared_error


@dataclass
class BasisRegressionModel:
    weights: np.ndarray
    feature_fn: Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, float | int | str]

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = self.feature_fn(np.asarray(x, dtype=np.float64))
        return features @ self.weights


@dataclass
class GaussianKernelRegressionModel:
    x_train: np.ndarray
    y_train: np.ndarray
    bandwidth: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        distances = (x[:, None] - self.x_train[None, :]) / self.bandwidth
        weights = np.exp(-0.5 * distances ** 2)
        denom = np.sum(weights, axis=1)
        return (weights @ self.y_train) / denom


def fit_polynomial_regression(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
) -> BasisRegressionModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def feature_fn(values: np.ndarray) -> np.ndarray:
        return np.vander(values, N=degree + 1, increasing=True)

    weights, _, _, _ = np.linalg.lstsq(feature_fn(x), y, rcond=None)
    return BasisRegressionModel(
        weights=weights,
        feature_fn=feature_fn,
        metadata={"degree": degree},
    )


def fit_gaussian_kernel_regression(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
) -> GaussianKernelRegressionModel:
    return GaussianKernelRegressionModel(
        x_train=np.asarray(x, dtype=np.float64),
        y_train=np.asarray(y, dtype=np.float64),
        bandwidth=float(bandwidth),
    )


def fit_trigonometric_regression(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    order: int,
) -> BasisRegressionModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def feature_fn(values: np.ndarray) -> np.ndarray:
        columns = [np.ones_like(values), values]
        for k in range(1, order + 1):
            columns.append(np.sin(k * alpha * values))
            columns.append(np.cos(k * alpha * values))
        return np.column_stack(columns)

    weights, _, _, _ = np.linalg.lstsq(feature_fn(x), y, rcond=None)
    return BasisRegressionModel(
        weights=weights,
        feature_fn=feature_fn,
        metadata={"alpha": alpha, "order": order},
    )


def search_polynomial_degree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    degrees: list[int],
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    best_trial: dict[str, object] | None = None

    for degree in degrees:
        model = fit_polynomial_regression(x_train, y_train, degree=degree)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        trial = {
            "degree": degree,
            "model": model,
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "train_pred": train_pred,
            "test_pred": test_pred,
        }
        trials.append(trial)
        if best_trial is None or trial["test_mse"] < best_trial["test_mse"]:
            best_trial = trial

    return {"trials": trials, "best": best_trial}


def search_kernel_bandwidth(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    bandwidths: list[float],
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    best_trial: dict[str, object] | None = None

    for bandwidth in bandwidths:
        model = fit_gaussian_kernel_regression(x_train, y_train, bandwidth=bandwidth)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        trial = {
            "bandwidth": bandwidth,
            "model": model,
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "train_pred": train_pred,
            "test_pred": test_pred,
        }
        trials.append(trial)
        if best_trial is None or trial["test_mse"] < best_trial["test_mse"]:
            best_trial = trial

    return {"trials": trials, "best": best_trial}


def search_trigonometric_configuration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
    orders: list[int],
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    best_trial: dict[str, object] | None = None

    for alpha in alphas:
        for order in orders:
            model = fit_trigonometric_regression(
                x_train,
                y_train,
                alpha=alpha,
                order=order,
            )
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            trial = {
                "alpha": alpha,
                "order": order,
                "model": model,
                "train_mse": mean_squared_error(y_train, train_pred),
                "test_mse": mean_squared_error(y_test, test_pred),
                "train_pred": train_pred,
                "test_pred": test_pred,
            }
            trials.append(trial)
            if best_trial is None or trial["test_mse"] < best_trial["test_mse"]:
                best_trial = trial

    return {"trials": trials, "best": best_trial}
