"""Train ML models for synthetic LEED portfolio demonstration data."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import build_feature_matrix
from src.preprocessing import load_training_data


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_save_models(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    """Train and persist score, rating, and construction cost models."""
    df = load_training_data()
    x = build_feature_matrix(df)
    y_score = df["leed_total_score"]
    y_rating = df["leed_rating"]
    y_cost = df["additional_construction_cost_usd"]

    x_train, x_test, y_score_train, y_score_test, y_rating_train, y_rating_test, y_cost_train, y_cost_test = (
        train_test_split(x, y_score, y_rating, y_cost, test_size=0.25, random_state=42, stratify=y_rating)
    )

    score_model = RandomForestRegressor(n_estimators=220, random_state=42, min_samples_leaf=2)
    cost_model = GradientBoostingRegressor(random_state=42)
    rating_model = RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")

    score_model.fit(x_train, y_score_train)
    cost_model.fit(x_train, y_cost_train)
    rating_model.fit(x_train, y_rating_train)

    score_pred = score_model.predict(x_test)
    cost_pred = cost_model.predict(x_test)
    rating_pred = rating_model.predict(x_test)

    evaluation = {
        "score_regression": _regression_metrics(y_score_test, score_pred),
        "cost_regression": _regression_metrics(y_cost_test, cost_pred),
        "rating_classification": {
            "Accuracy": float(accuracy_score(y_rating_test, rating_pred)),
            "classification_report": classification_report(y_rating_test, rating_pred, zero_division=0),
        },
    }
    bundle = {
        "score_model": score_model,
        "rating_model": rating_model,
        "cost_model": cost_model,
        "feature_columns": list(x.columns),
        "evaluation": evaluation,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    return bundle


if __name__ == "__main__":
    trained = train_and_save_models()
    print(trained["evaluation"])
