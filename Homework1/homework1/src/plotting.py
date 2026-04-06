from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_data_scatter(x_train, y_train, x_test, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, s=24, alpha=0.8, label="Train")
    ax.scatter(x_test, y_test, s=24, alpha=0.8, label="Test")
    ax.set_title("Training/Test Data Scatter")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_linear_fit(x_train, y_train, x_test, y_test, linear_results, path: Path) -> None:
    xs = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, s=20, alpha=0.65, label="Train")
    ax.scatter(x_test, y_test, s=20, alpha=0.65, label="Test")
    for label, result in linear_results.items():
        ax.plot(xs, result["model"].predict(xs), linewidth=2, label=label)
    ax.set_title("Linear Regression Fits")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_loss_curve(history, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(history) + 1), history, linewidth=2)
    ax.set_title("Gradient Descent Loss Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_kernel_bandwidth_curve(trials, path: Path) -> None:
    bandwidths = [trial["bandwidth"] for trial in trials]
    train_mse = [trial["train_mse"] for trial in trials]
    test_mse = [trial["test_mse"] for trial in trials]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bandwidths, train_mse, label="Train MSE", linewidth=2)
    ax.plot(bandwidths, test_mse, label="Test MSE", linewidth=2)
    ax.set_title("Kernel Bandwidth Search")
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_polynomial_mse_curve(trials, path: Path) -> None:
    degrees = [trial["degree"] for trial in trials]
    train_mse = [trial["train_mse"] for trial in trials]
    test_mse = [trial["test_mse"] for trial in trials]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, train_mse, marker="o", label="Train MSE")
    ax.plot(degrees, test_mse, marker="o", label="Test MSE")
    ax.set_title("Polynomial Degree Search")
    ax.set_xlabel("Degree")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_single_fit(x_train, y_train, x_test, y_test, model, title: str, path: Path) -> None:
    xs = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, s=20, alpha=0.65, label="Train")
    ax.scatter(x_test, y_test, s=20, alpha=0.65, label="Test")
    ax.plot(xs, model.predict(xs), linewidth=2.2, label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
