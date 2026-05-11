"""Prediction helpers for trained scikit-learn models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import build_feature_matrix
from src.rating import get_leed_rating


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_model_bundle(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    """Load trained model bundle from disk."""
    bundle_path = model_dir / "model_bundle.joblib"
    if not bundle_path.exists():
        from src.train_models import train_and_save_models

        train_and_save_models()
    return joblib.load(bundle_path)


def align_features(features: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    """Align inference features to training-time columns."""
    return features.reindex(columns=training_columns, fill_value=0)


def predict_project(df: pd.DataFrame, model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """Predict LEED score, rating, and additional construction cost."""
    bundle = load_model_bundle(model_dir)
    features = align_features(build_feature_matrix(df), bundle["feature_columns"])
    output = df.copy()
    output["predicted_leed_score"] = bundle["score_model"].predict(features).clip(0, 110)
    output["predicted_rating_class"] = bundle["rating_model"].predict(features)
    output["predicted_rating_from_score"] = output["predicted_leed_score"].apply(get_leed_rating)
    output["predicted_additional_cost_usd"] = bundle["cost_model"].predict(features).clip(0)
    return output
