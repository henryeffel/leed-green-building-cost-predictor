"""EnergyPlus-style CSV output analysis.

This project does not execute EnergyPlus. It analyzes CSV rows shaped like
post-processed EnergyPlus outputs for portfolio demonstration purposes.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "project_id",
    "baseline_energy_kwh",
    "proposed_energy_kwh",
    "energy_cost_per_kwh",
    "emissions_factor_kg_co2_per_kwh",
}


def load_energyplus_outputs(path: str) -> pd.DataFrame:
    """Load and validate EnergyPlus-style output CSV data."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"EnergyPlus-style CSV missing columns: {sorted(missing)}")
    return analyze_energyplus_outputs(df)


def analyze_energyplus_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate energy savings, annual cost savings, and CO2 reduction."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"EnergyPlus-style data missing columns: {sorted(missing)}")

    analyzed = df.copy()
    baseline = analyzed["baseline_energy_kwh"].replace(0, pd.NA)
    savings_kwh = analyzed["baseline_energy_kwh"] - analyzed["proposed_energy_kwh"]
    analyzed["energy_saving_rate"] = (savings_kwh / baseline).fillna(0).clip(lower=0)
    analyzed["annual_energy_cost_saving"] = (
        savings_kwh.clip(lower=0) * analyzed["energy_cost_per_kwh"]
    )
    analyzed["estimated_co2_reduction"] = (
        savings_kwh.clip(lower=0) * analyzed["emissions_factor_kg_co2_per_kwh"] / 1000
    )
    return analyzed
