import pandas as pd
import pytest

from src.energyplus_parser import analyze_energyplus_outputs


def test_analyze_energyplus_outputs_calculates_savings():
    df = pd.DataFrame(
        [
            {
                "project_id": "A",
                "baseline_energy_kwh": 1000,
                "proposed_energy_kwh": 700,
                "energy_cost_per_kwh": 0.1,
                "emissions_factor_kg_co2_per_kwh": 0.5,
            }
        ]
    )

    result = analyze_energyplus_outputs(df).iloc[0]

    assert result["energy_saving_rate"] == pytest.approx(0.3)
    assert result["annual_energy_cost_saving"] == pytest.approx(30)
    assert result["estimated_co2_reduction"] == pytest.approx(0.15)


def test_analyze_energyplus_outputs_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        analyze_energyplus_outputs(pd.DataFrame([{"project_id": "A"}]))
