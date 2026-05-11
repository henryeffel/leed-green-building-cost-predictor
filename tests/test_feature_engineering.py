import pandas as pd

from src.feature_engineering import build_feature_matrix


def test_build_feature_matrix_one_hot_encodes_categoricals():
    df = pd.DataFrame(
        [
            {
                "gross_floor_area_m2": 10000,
                "num_floors": 5,
                "baseline_energy_kwh": 1000000,
                "proposed_energy_kwh": 750000,
                "energy_saving_rate": 0.25,
                "annual_energy_cost_saving": 25000,
                "estimated_co2_reduction": 100,
                "water_saving_rate": 0.3,
                "recycled_material_ratio": 0.2,
                "ieq_score": 12,
                "renewable_energy_rate": 0.1,
                "site_density_score": 6,
                "building_type": "Office",
                "climate_zone": "3A",
            }
        ]
    )

    features = build_feature_matrix(df)

    assert "building_type_Office" in features.columns
    assert "climate_zone_3A" in features.columns
    assert features.loc[0, "energy_saving_rate"] == 0.25
