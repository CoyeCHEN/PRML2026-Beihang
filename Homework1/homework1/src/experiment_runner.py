from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_loader import load_regression_data
from src.linear_models import (
    fit_gradient_descent_linear,
    fit_least_squares_linear,
    fit_newton_linear,
)
from src.metrics import mean_squared_error
from src.nonlinear_models import (
    search_kernel_bandwidth,
    search_polynomial_degree,
    search_trigonometric_configuration,
)
from src.plotting import (
    save_data_scatter,
    save_kernel_bandwidth_curve,
    save_linear_fit,
    save_loss_curve,
    save_polynomial_mse_curve,
    save_single_fit,
)
from src.report_generator import write_markdown_report, write_pdf_report


def run_all_experiments(
    data_path: Path,
    output_dir: Path | None = None,
    generate_report: bool = False,
    generate_pdf: bool = False,
) -> dict[str, object]:
    data = load_regression_data(data_path)
    output_dir = output_dir or Path("Homework1/output")

    least_squares = fit_least_squares_linear(data.x_train, data.y_train)
    gradient_descent = fit_gradient_descent_linear(
        data.x_train,
        data.y_train,
        learning_rate=0.01,
        max_iter=10000,
        tolerance=1e-10,
    )
    newton = fit_newton_linear(
        data.x_train,
        data.y_train,
        max_iter=10,
        tolerance=1e-12,
    )

    polynomial = search_polynomial_degree(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        degrees=list(range(2, 21)),
    )
    kernel = search_kernel_bandwidth(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        bandwidths=[float(x) for x in np.linspace(0.03, 1.5, 40)],
    )
    trigonometric = search_trigonometric_configuration(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        alphas=[round(float(x), 2) for x in np.linspace(0.3, 1.5, 25)],
        orders=list(range(1, 16)),
    )

    results = {
        "data": data,
        "linear": {
            "least_squares": _linear_result(data, least_squares),
            "gradient_descent": _linear_result(data, gradient_descent),
            "newton": _linear_result(data, newton),
        },
        "polynomial": polynomial,
        "kernel": kernel,
        "trigonometric": trigonometric,
    }
    _write_figures(results, output_dir)
    if generate_report:
        report_path = write_markdown_report(results, output_dir)
        results["report_path"] = report_path
    if generate_pdf:
        pdf_path = write_pdf_report(results, output_dir)
        results["pdf_path"] = pdf_path
    return results


def _linear_result(data, model) -> dict[str, object]:
    train_pred = model.predict(data.x_train)
    test_pred = model.predict(data.x_test)
    return {
        "model": model,
        "weights": model.weights,
        "history": model.history,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_mse": mean_squared_error(data.y_train, train_pred),
        "test_mse": mean_squared_error(data.y_test, test_pred),
    }


def _write_figures(results: dict[str, object], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = results["data"]
    linear = results["linear"]
    polynomial_best = results["polynomial"]["best"]
    kernel_best = results["kernel"]["best"]
    trigonometric_best = results["trigonometric"]["best"]

    save_data_scatter(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        figures_dir / "data_scatter.png",
    )
    save_linear_fit(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        linear,
        figures_dir / "linear_fit.png",
    )
    save_loss_curve(
        linear["gradient_descent"]["history"],
        figures_dir / "gd_loss_curve.png",
    )
    save_kernel_bandwidth_curve(
        results["kernel"]["trials"],
        figures_dir / "kernel_bandwidth_curve.png",
    )
    save_single_fit(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        kernel_best["model"],
        "Gaussian Kernel Regression Fit",
        figures_dir / "kernel_fit.png",
    )
    save_polynomial_mse_curve(
        results["polynomial"]["trials"],
        figures_dir / "polynomial_mse.png",
    )
    save_single_fit(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        polynomial_best["model"],
        "Polynomial Regression Fit",
        figures_dir / "polynomial_fit.png",
    )
    save_single_fit(
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        trigonometric_best["model"],
        "Trigonometric Regression Fit",
        figures_dir / "trigonometric_fit.png",
    )
