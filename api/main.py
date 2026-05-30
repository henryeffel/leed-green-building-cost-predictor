"""FastAPI extension for the Streamlit LEED portfolio MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI

from api.llm_report import generate_recommendation_report
from api.schemas import BuildingInput, PredictionResponse, RecommendationResponse
from src.cost_optimizer import recommend_minimum_cost_credits
from src.energyplus_parser import analyze_energyplus_outputs
from src.predict import predict_project
from src.rating import score_gap_to_target


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LIMITATION_NOTE = (
    "Prototype only: not an official LEED tool, does not replace USGBC review, "
    "uses synthetic data, and keeps OpenAI reporting optional."
)

app = FastAPI(
    title="LEED Green Building Cost Predictor API",
    description="Lightweight FastAPI extension for the Streamlit MVP.",
    version="0.1.0",
)


def _to_project_frame(payload: BuildingInput) -> pd.DataFrame:
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    data["project_id"] = "API-INPUT"
    return pd.DataFrame([data])


def _prediction_dict(payload: BuildingInput) -> dict[str, Any]:
    analyzed = analyze_energyplus_outputs(_to_project_frame(payload))
    prediction = predict_project(analyzed)
    row = prediction.iloc[0]
    gap = score_gap_to_target(float(row["predicted_leed_score"]), payload.target_rating)

    return {
        "predicted_leed_score": float(row["predicted_leed_score"]),
        "predicted_rating_from_score": str(row["predicted_rating_from_score"]),
        "predicted_rating_class": str(row["predicted_rating_class"]),
        "predicted_additional_cost_usd": float(row["predicted_additional_cost_usd"]),
        "energy_saving_rate": float(row["energy_saving_rate"]),
        "annual_energy_cost_saving": float(row["annual_energy_cost_saving"]),
        "estimated_co2_reduction": float(row["estimated_co2_reduction"]),
        "target_rating": payload.target_rating,
        "score_gap": float(gap.gap),
    }


def _load_credit_candidates() -> pd.DataFrame:
    credit_path = DATA_DIR / "leed_credit_reference.csv"
    if credit_path.exists():
        return pd.read_csv(credit_path)
    return pd.read_csv(DATA_DIR / "cost_assumptions.csv")


def _recommendation_dict(prediction: dict[str, Any]) -> dict[str, Any]:
    result = recommend_minimum_cost_credits(
        current_score=prediction["predicted_leed_score"],
        target_rating=prediction["target_rating"],
        credit_candidates=_load_credit_candidates(),
    )
    recommendations = result["recommendations"]
    return {
        "current_score": float(result["current_score"]),
        "target_rating": str(result["target_rating"]),
        "target_score": float(result["target_score"]),
        "score_gap": float(result["score_gap"]),
        "recommended_points": float(result["recommended_points"]),
        "estimated_additional_cost": float(result["estimated_additional_cost"]),
        "is_target_reachable": bool(result["is_target_reachable"]),
        "credits": recommendations.to_dict(orient="records"),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: BuildingInput) -> PredictionResponse:
    return PredictionResponse(**_prediction_dict(payload))


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: BuildingInput) -> RecommendationResponse:
    prediction = _prediction_dict(payload)
    recommendations = _recommendation_dict(prediction)
    llm_report = generate_recommendation_report(prediction, recommendations)
    return RecommendationResponse(
        prediction=PredictionResponse(**prediction),
        recommendations=recommendations,
        llm_report=llm_report,
        limitation_note=LIMITATION_NOTE,
    )
