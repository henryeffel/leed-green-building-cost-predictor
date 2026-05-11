"""Generate PNG visualizations from ML experiment report CSV files."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "visualizations"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd


MODEL_EXPERIMENTS_PATH = REPORTS_DIR / "model_experiments.csv"
FEATURE_IMPORTANCE_SCORE_PATH = REPORTS_DIR / "feature_importance_score.csv"
FEATURE_IMPORTANCE_COST_PATH = REPORTS_DIR / "feature_importance_cost.csv"
RATING_CONFUSION_MATRIX_PATH = REPORTS_DIR / "rating_confusion_matrix.csv"


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_model_experiment_comparison() -> Path:
    experiments = pd.read_csv(MODEL_EXPERIMENTS_PATH)
    tasks = [
        ("score_regression", ["mae", "rmse", "r2"], "Score Regression"),
        ("cost_regression", ["mae", "rmse", "r2"], "Cost Regression"),
        (
            "rating_classification",
            ["accuracy", "macro_f1", "weighted_f1"],
            "Rating Classification",
        ),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), constrained_layout=True)
    colors = ["#3b82f6", "#10b981", "#f59e0b"]

    for ax, (task, metrics, title) in zip(axes, tasks):
        task_df = experiments.loc[experiments["task"] == task, ["model", *metrics]].copy()
        task_df = task_df.set_index("model")

        task_df.plot(kind="bar", ax=ax, color=colors, width=0.78)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(title="", loc="best")
        _style_axes(ax)

    fig.suptitle("ML Model Experiment Comparison", fontsize=18, fontweight="bold")
    output_path = OUTPUT_DIR / "model_experiment_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(csv_path: Path, output_name: str, title: str) -> Path:
    importance_df = pd.read_csv(csv_path).nlargest(10, "importance")
    importance_df = importance_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color="#2563eb",
    )

    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_xlim(0, max(importance_df["importance"]) * 1.18)
    _style_axes(ax)

    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rating_confusion_matrix() -> Path:
    matrix_df = pd.read_csv(RATING_CONFUSION_MATRIX_PATH, index_col=0)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(matrix_df.values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Rating Classification Confusion Matrix", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Rating")
    ax.set_ylabel("Actual Rating")
    ax.set_xticks(range(len(matrix_df.columns)), labels=matrix_df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix_df.index)), labels=matrix_df.index)

    threshold = matrix_df.values.max() / 2
    for row_idx, row_label in enumerate(matrix_df.index):
        for col_idx, col_label in enumerate(matrix_df.columns):
            value = int(matrix_df.loc[row_label, col_label])
            color = "white" if value > threshold else "#111827"
            ax.text(col_idx, row_idx, value, ha="center", va="center", color=color, fontsize=10)

    output_path = OUTPUT_DIR / "rating_confusion_matrix.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_paths = [
        plot_model_experiment_comparison(),
        plot_feature_importance(
            FEATURE_IMPORTANCE_SCORE_PATH,
            "feature_importance_score.png",
            "Top 10 Features for LEED Score Prediction",
        ),
        plot_feature_importance(
            FEATURE_IMPORTANCE_COST_PATH,
            "feature_importance_cost.png",
            "Top 10 Features for Construction Cost Prediction",
        ),
        plot_rating_confusion_matrix(),
    ]

    print("Generated visualizations:")
    for path in generated_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
