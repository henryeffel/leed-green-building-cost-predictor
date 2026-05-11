"""Train ML models for synthetic LEED portfolio demonstration data."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import build_feature_matrix
from src.preprocessing import load_training_data


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _empty_metric_row(task: str, model_name: str, notes: str) -> dict[str, object]:
    return {
        "task": task,
        "model": model_name,
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "notes": notes,
    }


def _save_feature_importance(model, feature_columns: list[str], path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
        total = values.sum()
        if total > 0:
            values = values / total
    else:
        return
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(path, index=False)


def train_and_save_models(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    """Run ML experiments, select final models, and persist reports."""
    df = load_training_data()
    x = build_feature_matrix(df)
    y_score = df["leed_total_score"]
    y_rating = df["leed_rating"]
    y_cost = df["additional_construction_cost_usd"]

    x_train, x_test, y_score_train, y_score_test, y_rating_train, y_rating_test, y_cost_train, y_cost_test = (
        train_test_split(x, y_score, y_rating, y_cost, test_size=0.25, random_state=42, stratify=y_rating)
    )

    regression_candidates = {
        "DummyRegressor_mean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=220, random_state=42, min_samples_leaf=2),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }
    classification_candidates = {
        "DummyClassifier_most_frequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression_balanced": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "RandomForestClassifier_balanced": RandomForestClassifier(
            n_estimators=220, random_state=42, class_weight="balanced"
        ),
    }

    experiment_rows: list[dict[str, object]] = []
    score_models = {}
    cost_models = {}
    rating_models = {}

    for model_name, model in regression_candidates.items():
        fitted = clone(model).fit(x_train, y_score_train)
        score_models[model_name] = fitted
        metrics = _regression_metrics(y_score_test, fitted.predict(x_test))
        row = _empty_metric_row("score_regression", model_name, "LEED total score prediction")
        row.update(metrics)
        experiment_rows.append(row)

    for model_name, model in regression_candidates.items():
        fitted = clone(model).fit(x_train, y_cost_train)
        cost_models[model_name] = fitted
        metrics = _regression_metrics(y_cost_test, fitted.predict(x_test))
        row = _empty_metric_row("cost_regression", model_name, "Additional construction cost prediction")
        row.update(metrics)
        experiment_rows.append(row)

    for model_name, model in classification_candidates.items():
        fitted = clone(model).fit(x_train, y_rating_train)
        rating_models[model_name] = fitted
        metrics = _classification_metrics(y_rating_test, fitted.predict(x_test))
        row = _empty_metric_row("rating_classification", model_name, "LEED rating class prediction")
        row.update(metrics)
        experiment_rows.append(row)

    experiments = pd.DataFrame(
        experiment_rows,
        columns=["task", "model", "mae", "rmse", "r2", "accuracy", "macro_f1", "weighted_f1", "notes"],
    )

    score_experiments = experiments[experiments["task"] == "score_regression"].sort_values(
        ["r2", "mae"], ascending=[False, True]
    )
    cost_experiments = experiments[experiments["task"] == "cost_regression"].sort_values(
        ["r2", "mae"], ascending=[False, True]
    )
    rating_experiments = experiments[experiments["task"] == "rating_classification"].sort_values(
        ["macro_f1", "accuracy"], ascending=[False, False]
    )

    selected_score_model_name = str(score_experiments.iloc[0]["model"])
    selected_cost_model_name = str(cost_experiments.iloc[0]["model"])
    selected_rating_model_name = str(rating_experiments.iloc[0]["model"])

    score_model = score_models[selected_score_model_name]
    cost_model = cost_models[selected_cost_model_name]
    rating_model = rating_models[selected_rating_model_name]

    rating_pred = rating_model.predict(x_test)
    labels = sorted(y_rating.unique())

    evaluation = {
        "score_regression": _regression_metrics(y_score_test, score_model.predict(x_test)),
        "cost_regression": _regression_metrics(y_cost_test, cost_model.predict(x_test)),
        "rating_classification": {
            **_classification_metrics(y_rating_test, rating_pred),
            "classification_report": classification_report(y_rating_test, rating_pred, zero_division=0),
        },
    }
    bundle = {
        "score_model": score_model,
        "rating_model": rating_model,
        "cost_model": cost_model,
        "feature_columns": list(x.columns),
        "evaluation": evaluation,
        "selected_models": {
            "score_regression": selected_score_model_name,
            "cost_regression": selected_cost_model_name,
            "rating_classification": selected_rating_model_name,
        },
        "experiment_results": experiments.to_dict(orient="records"),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    experiments.to_csv(REPORT_DIR / "model_experiments.csv", index=False)
    _save_feature_importance(score_model, list(x.columns), REPORT_DIR / "feature_importance_score.csv")
    _save_feature_importance(cost_model, list(x.columns), REPORT_DIR / "feature_importance_cost.csv")
    (REPORT_DIR / "rating_classification_report.txt").write_text(
        evaluation["rating_classification"]["classification_report"], encoding="utf-8"
    )
    pd.DataFrame(confusion_matrix(y_rating_test, rating_pred, labels=labels), index=labels, columns=labels).to_csv(
        REPORT_DIR / "rating_confusion_matrix.csv"
    )
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    return bundle


if __name__ == "__main__":
    trained = train_and_save_models()
    print(trained["evaluation"])
