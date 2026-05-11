"""ESG summary metric preparation."""

from __future__ import annotations

import pandas as pd


ESG_COLUMNS = [
    "energy_saving_rate",
    "annual_energy_cost_saving",
    "estimated_co2_reduction",
    "water_saving_rate",
    "recycled_material_ratio",
    "ieq_score",
]


def build_esg_summary(row: pd.Series | dict[str, float]) -> dict[str, float]:
    """Build dashboard-ready ESG summary metrics."""
    data = dict(row)
    return {
        "energy_saving_rate": float(data.get("energy_saving_rate", 0)),
        "annual_energy_cost_saving": float(data.get("annual_energy_cost_saving", 0)),
        "estimated_co2_reduction_tonnes": float(data.get("estimated_co2_reduction", 0)),
        "water_saving_rate": float(data.get("water_saving_rate", 0)),
        "recycled_material_ratio": float(data.get("recycled_material_ratio", 0)),
        "indoor_environmental_quality_score": float(data.get("ieq_score", 0)),
    }


def esg_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return ESG columns plus project identifiers."""
    columns = ["project_id", "project_name"] + [c for c in ESG_COLUMNS if c in df.columns]
    return df[columns].copy()
