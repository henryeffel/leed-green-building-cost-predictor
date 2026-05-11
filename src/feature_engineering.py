"""Feature engineering for LEED prediction models."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "gross_floor_area_m2",
    "num_floors",
    "baseline_energy_kwh",
    "proposed_energy_kwh",
    "energy_saving_rate",
    "annual_energy_cost_saving",
    "estimated_co2_reduction",
    "water_saving_rate",
    "recycled_material_ratio",
    "ieq_score",
    "renewable_energy_rate",
    "site_density_score",
]

CATEGORICAL_COLUMNS = ["building_type", "climate_zone"]


def build_model_frame(projects: pd.DataFrame, energy: pd.DataFrame) -> pd.DataFrame:
    """Merge project metadata with analyzed EnergyPlus-style metrics."""
    merged = projects.merge(energy, on="project_id", how="left", suffixes=("", "_energy"))
    return merged


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return one-hot encoded ML features for model training and inference."""
    missing = [column for column in FEATURE_COLUMNS + CATEGORICAL_COLUMNS if column not in df]
    if missing:
        raise ValueError(f"Feature input missing columns: {missing}")

    numeric = df[FEATURE_COLUMNS].copy()
    encoded = pd.get_dummies(df[CATEGORICAL_COLUMNS], columns=CATEGORICAL_COLUMNS, dtype=float)
    return pd.concat([numeric, encoded], axis=1).fillna(0)
