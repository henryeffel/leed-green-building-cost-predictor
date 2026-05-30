"""Pydantic schemas for the optional FastAPI extension."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BuildingInput(BaseModel):
    project_name: str
    building_type: str
    climate_zone: str
    gross_floor_area_m2: float = Field(gt=0)
    num_floors: int = Field(gt=0)
    baseline_energy_kwh: float = Field(ge=0)
    proposed_energy_kwh: float = Field(ge=0)
    energy_cost_per_kwh: float = Field(ge=0)
    emissions_factor_kg_co2_per_kwh: float = Field(ge=0)
    water_saving_rate: float = Field(ge=0)
    recycled_material_ratio: float = Field(ge=0)
    ieq_score: float = Field(ge=0)
    renewable_energy_rate: float = Field(ge=0)
    site_density_score: float = Field(ge=0)
    target_rating: str


class PredictionResponse(BaseModel):
    predicted_leed_score: float
    predicted_rating_from_score: str
    predicted_rating_class: str
    predicted_additional_cost_usd: float
    energy_saving_rate: float
    annual_energy_cost_saving: float
    estimated_co2_reduction: float
    target_rating: str
    score_gap: float


class RecommendationResponse(BaseModel):
    prediction: PredictionResponse
    recommendations: dict[str, Any]
    llm_report: str
    limitation_note: str
