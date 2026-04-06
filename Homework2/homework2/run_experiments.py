from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

@dataclass
class SearchSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]
    needs_scaling: bool = False
    kernel_name: str | None = None

def make_moons_3d(n_per_class: int = 500, noise: float = 0.2, seed: int = 2026):
                                                                   
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_per_class, endpoint=False)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    positive = np.column_stack([x, y, z])
    negative = np.column_stack([-x, y - 1, -z])
    X = np.vstack([positive, negative])
    labels = np.hstack(
        [
            np.zeros(n_per_class, dtype=int),
            np.ones(n_per_class, dtype=int),
        ]
    )
    X += rng.normal(scale=noise, size=X.shape)
    return X, labels

def build_specs() -> list[SearchSpec]:
    return [
        SearchSpec(
            name="Decision Tree",
            estimator=DecisionTreeClassifier(random_state=2026),
            param_grid={
                "max_depth": [3, 4, 5, 6, 8, 10, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy"],
            },
        ),
        SearchSpec(
            name="AdaBoost + Decision Tree",
            estimator=AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=2026),
                algorithm="SAMME",
                random_state=2026,
            ),
            param_grid={
                "estimator__max_depth": [1, 2, 3, 4],
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.3, 0.5, 1.0],
            },
        ),
        SearchSpec(
            name="SVM (Linear)",
            estimator=SVC(kernel="linear", random_state=2026),
            param_grid={"svc__C": [0.1, 0.5, 1, 2, 5, 10]},
            needs_scaling=True,
            kernel_name="linear",
        ),
        SearchSpec(
            name="SVM (Polynomial)",
            estimator=SVC(kernel="poly", random_state=2026),
            param_grid={
                "svc__C": [0.1, 1, 5, 10],
                "svc__degree": [2, 3, 4],
                "svc__gamma": ["scale", 0.1, 0.5, 1.0],
                "svc__coef0": [0.0, 1.0],
            },
            needs_scaling=True,
            kernel_name="poly",
        ),
        SearchSpec(
            name="SVM (RBF)",
            estimator=SVC(kernel="rbf", random_state=2026),
            param_grid={
                "svc__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
                "svc__gamma": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            },
            needs_scaling=True,
            kernel_name="rbf",
        ),
        SearchSpec(
            name="SVM (Sigmoid)",
            estimator=SVC(kernel="sigmoid", random_state=2026),
            param_grid={
                "svc__C": [0.1, 1, 5, 10],
                "svc__gamma": ["scale", 0.1, 0.5, 1.0],
                "svc__coef0": [-1.0, 0.0, 1.0],
            },
            needs_scaling=True,
            kernel_name="sigmoid",
        ),
    ]

def make_estimator(spec: SearchSpec):
    if spec.needs_scaling:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", clone(spec.estimator)),
            ]
        )
    return clone(spec.estimator)

def extract_model(estimator):
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["svc"]
    return estimator

def sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.floating)):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean

def evaluate_model(name: str, estimator, X_train, y_train, X_test, y_test):
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)

    result = {
        "model": name,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_test_pred, digits=4, output_dict=True
        ),
    }

    model = extract_model(estimator)
    if hasattr(model, "n_support_"):
        result["support_vectors"] = int(np.sum(model.n_support_))
    if isinstance(model, DecisionTreeClassifier):
        result["tree_depth"] = int(model.get_depth())
        result["n_leaves"] = int(model.get_n_leaves())
    return result

def search_and_train(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    results = []
    best_estimators = {}
    rbf_grid_scores = None

    for spec in build_specs():
        estimator = make_estimator(spec)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=spec.param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The SAMME.R algorithm .* is deprecated",
                category=FutureWarning,
            )
            search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        metrics = evaluate_model(
            spec.name, search.best_estimator_, X_train, y_train, X_test, y_test
        )
        metrics["cv_best_accuracy"] = float(search.best_score_)
        metrics["search_time_sec"] = elapsed
        metrics["best_params"] = sanitize_params(search.best_params_)
        results.append(metrics)
        best_estimators[spec.name] = search.best_estimator_

        if spec.name == "SVM (RBF)":
            cv_df = pd.DataFrame(search.cv_results_)
            rbf_grid_scores = cv_df[
                ["param_svc__C", "param_svc__gamma", "mean_test_score"]
            ].copy()

    return results, best_estimators, rbf_grid_scores

def plot_dataset(X_train, y_train, X_test, y_test):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(y_train == 0, "#2b6cb0", "#d94841")
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        X_train[:, 2],
        c=colors,
        s=14,
        alpha=0.55,
        label="训练集",
    )
    colors_test = np.where(y_test == 0, "#7fb3d5", "#f5b7b1")
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        X_test[:, 2],
        c=colors_test,
        s=24,
        alpha=0.35,
        marker="^",
        label="测试集",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Make-Moons 训练集与测试集分布")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_distribution.png", dpi=220)
    plt.close(fig)

def plot_comparison(results_df: pd.DataFrame):
    plot_df = (
        results_df.set_index("model")[
            ["cv_best_accuracy", "test_accuracy", "f1"]
        ].sort_values("test_accuracy", ascending=False)
    )
    x = np.arange(len(plot_df.index))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, plot_df["cv_best_accuracy"], width=width, label="CV 最优准确率")
    ax.bar(x, plot_df["test_accuracy"], width=width, label="测试准确率")
    ax.bar(x + width, plot_df["f1"], width=width, label="测试 F1")

    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("分数")
    ax.set_title("不同模型的交叉验证与测试表现对比")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=220)
    plt.close(fig)

def plot_confusion_matrices(best_estimators, X_test, y_test):
    names = list(best_estimators.keys())
    ncols = 3
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        estimator = best_estimators[name]
        ConfusionMatrixDisplay.from_estimator(
            estimator,
            X_test,
            y_test,
            cmap="Blues",
            colorbar=False,
            ax=ax,
            values_format="d",
        )
        ax.set_title(name)

    for ax in axes[len(names) :]:
        ax.axis("off")

    fig.suptitle("测试集混淆矩阵", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=220)
    plt.close(fig)

def plot_decision_slices(best_estimators, X_test, y_test):
    names = list(best_estimators.keys())
    ncols = 3
    nrows = math.ceil(len(names) / ncols)

    x_min, x_max = X_test[:, 0].min() - 0.4, X_test[:, 0].max() + 0.4
    y_min, y_max = X_test[:, 1].min() - 0.4, X_test[:, 1].max() + 0.4
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 240),
        np.linspace(y_min, y_max, 240),
    )
    zz = np.zeros_like(xx)
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    band = np.abs(X_test[:, 2]) < 0.18

    bg_cmap = ListedColormap(["#d9e8fb", "#f8d9d6"])
    pt_colors = np.where(y_test[band] == 0, "#2b6cb0", "#d94841")

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        estimator = best_estimators[name]
        pred = estimator.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], cmap=bg_cmap, alpha=0.8)
        ax.scatter(
            X_test[band, 0],
            X_test[band, 1],
            c=pt_colors,
            s=18,
            edgecolor="k",
            linewidth=0.2,
        )
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for ax in axes[len(names) :]:
        ax.axis("off")

    fig.suptitle("固定 z = 0 切片时的决策边界", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decision_slices.png", dpi=220)
    plt.close(fig)

def plot_rbf_heatmap(rbf_grid_scores: pd.DataFrame):
    heatmap_df = rbf_grid_scores.copy()
    heatmap_df["param_svc__C"] = heatmap_df["param_svc__C"].astype(float)
    heatmap_df["param_svc__gamma"] = heatmap_df["param_svc__gamma"].astype(float)
    pivot = heatmap_df.pivot(
        index="param_svc__gamma",
        columns="param_svc__C",
        values="mean_test_score",
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("C")
    ax.set_ylabel("gamma")
    ax.set_title("RBF-SVM 参数网格的 5 折交叉验证准确率")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j,
                i,
                f"{pivot.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rbf_heatmap.png", dpi=220)
    plt.close(fig)

def run_robustness(best_estimators, seeds=range(10)):
    records = []
    template_estimators = {name: clone(est) for name, est in best_estimators.items()}
    for seed in seeds:
        X_train, y_train = make_moons_3d(n_per_class=500, noise=0.2, seed=seed)
        X_test, y_test = make_moons_3d(n_per_class=250, noise=0.2, seed=seed + 10_000)
        for name, estimator in template_estimators.items():
            model = clone(estimator)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The SAMME.R algorithm .* is deprecated",
                    category=FutureWarning,
                )
                model.fit(X_train, y_train)
            pred = model.predict(X_test)
            records.append(
                {
                    "seed": seed,
                    "model": name,
                    "test_accuracy": accuracy_score(y_test, pred),
                    "f1": f1_score(y_test, pred),
                }
            )
    return pd.DataFrame(records)

def plot_robustness(robust_df: pd.DataFrame):
    models = list(dict.fromkeys(robust_df["model"]))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    series = [robust_df.loc[robust_df["model"] == model, "test_accuracy"].values for model in models]
    bp = ax.boxplot(series, tick_labels=models, patch_artist=True, showmeans=True)
    palette = ["#8fbddc", "#f8c291", "#b8e0a5", "#c3bef0", "#ffd08a", "#f0b3d4"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("测试准确率")
    ax.set_title("固定最优超参数后，10 组随机数据上的稳健性比较")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "robustness_boxplot.png", dpi=220)
    plt.close(fig)

def main():
                                   
    X_train, y_train = make_moons_3d(n_per_class=500, noise=0.2, seed=2026)
    X_test, y_test = make_moons_3d(n_per_class=250, noise=0.2, seed=2027)

    plot_dataset(X_train, y_train, X_test, y_test)

    results, best_estimators, rbf_grid_scores = search_and_train(
        X_train, y_train, X_test, y_test
    )
    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    results_df.to_csv(OUT_DIR / "model_results.csv", index=False)
    with open(OUT_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    best_params = {
        row["model"]: row["best_params"]
        for row in results
    }
    with open(OUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    plot_comparison(results_df)
    plot_confusion_matrices(best_estimators, X_test, y_test)
    plot_decision_slices(best_estimators, X_test, y_test)

    if rbf_grid_scores is not None:
        rbf_grid_scores.to_csv(OUT_DIR / "rbf_grid_scores.csv", index=False)
        plot_rbf_heatmap(rbf_grid_scores)

    robust_df = run_robustness(best_estimators, seeds=range(10))
    robust_df.to_csv(OUT_DIR / "robustness.csv", index=False)
    plot_robustness(robust_df)

    summary = {
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_class_balance": np.bincount(y_train).tolist(),
        "test_class_balance": np.bincount(y_test).tolist(),
    }
    with open(OUT_DIR / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
