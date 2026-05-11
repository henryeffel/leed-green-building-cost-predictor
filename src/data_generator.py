"""Generate deterministic synthetic portfolio data for local demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.rating import get_leed_rating
except ModuleNotFoundError:
    from rating import get_leed_rating


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


BUILDING_TYPES = ["Office", "Healthcare", "Education", "Retail", "Multifamily"]
CLIMATE_ZONES = ["1A", "2A", "3A", "4A", "5A", "6A"]


def generate_sample_data(data_dir: Path = DATA_DIR, n_projects: int = 180, seed: int = 7) -> None:
    """Generate deterministic synthetic CSV datasets."""
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)

    project_ids = [f"PRJ-{i:04d}" for i in range(1, n_projects + 1)]
    gross_floor_area = rng.integers(2500, 85000, size=n_projects)
    num_floors = np.maximum(1, (gross_floor_area / rng.integers(1200, 4200, size=n_projects)).astype(int))
    baseline_energy = gross_floor_area * rng.uniform(145, 310, size=n_projects)
    energy_saving_rate = rng.uniform(0.04, 0.48, size=n_projects)
    proposed_energy = baseline_energy * (1 - energy_saving_rate)
    water_saving_rate = rng.uniform(0.05, 0.55, size=n_projects)
    recycled_material_ratio = rng.uniform(0.03, 0.45, size=n_projects)
    ieq_score = rng.uniform(4.0, 15.5, size=n_projects)
    renewable_energy_rate = rng.uniform(0.0, 0.32, size=n_projects)
    site_density_score = rng.uniform(1.0, 10.0, size=n_projects)

    latent_score = (
        24
        + energy_saving_rate * 48
        + water_saving_rate * 14
        + recycled_material_ratio * 13
        + ieq_score * 1.25
        + renewable_energy_rate * 18
        + site_density_score * 0.85
        + rng.normal(0, 4.5, size=n_projects)
    )
    leed_total_score = np.clip(latent_score, 24, 96).round(1)
    additional_cost = (
        gross_floor_area
        * (32 + energy_saving_rate * 150 + water_saving_rate * 35 + recycled_material_ratio * 70)
        * rng.uniform(0.86, 1.18, size=n_projects)
    ).round(0)

    projects = pd.DataFrame(
        {
            "project_id": project_ids,
            "project_name": [f"Synthetic Green Building {i}" for i in range(1, n_projects + 1)],
            "building_type": rng.choice(BUILDING_TYPES, size=n_projects),
            "climate_zone": rng.choice(CLIMATE_ZONES, size=n_projects),
            "gross_floor_area_m2": gross_floor_area,
            "num_floors": num_floors,
            "water_saving_rate": water_saving_rate.round(3),
            "recycled_material_ratio": recycled_material_ratio.round(3),
            "ieq_score": ieq_score.round(2),
            "renewable_energy_rate": renewable_energy_rate.round(3),
            "site_density_score": site_density_score.round(2),
            "leed_total_score": leed_total_score,
            "leed_rating": [get_leed_rating(score) for score in leed_total_score],
            "additional_construction_cost_usd": additional_cost,
        }
    )
    energy = pd.DataFrame(
        {
            "project_id": project_ids,
            "baseline_energy_kwh": baseline_energy.round(0),
            "proposed_energy_kwh": proposed_energy.round(0),
            "energy_cost_per_kwh": rng.uniform(0.09, 0.18, size=n_projects).round(3),
            "emissions_factor_kg_co2_per_kwh": rng.uniform(0.32, 0.55, size=n_projects).round(3),
        }
    )

    credits = pd.DataFrame(
        [
            ("EA-Optimize", "Optimize Energy Performance", "Energy and Atmosphere", 18, 42000),
            ("EA-Renewable", "Renewable Energy", "Energy and Atmosphere", 5, 65000),
            ("WE-Indoor", "Indoor Water Use Reduction", "Water Efficiency", 6, 18000),
            ("WE-Metering", "Water Metering", "Water Efficiency", 1, 9000),
            ("MR-EPD", "Environmental Product Declarations", "Materials and Resources", 2, 11000),
            ("MR-Sourcing", "Sourcing of Raw Materials", "Materials and Resources", 2, 14000),
            ("MR-Waste", "Construction Waste Management", "Materials and Resources", 2, 9500),
            ("EQ-Vent", "Enhanced Indoor Air Quality Strategies", "Indoor Environmental Quality", 2, 16000),
            ("EQ-Light", "Daylight", "Indoor Environmental Quality", 3, 22000),
            ("LT-Transit", "Access to Quality Transit", "Location and Transportation", 5, 8000),
            ("SS-Heat", "Heat Island Reduction", "Sustainable Sites", 2, 12000),
            ("IN-Innovation", "Innovation", "Innovation", 5, 15000),
            ("RP-Priority", "Regional Priority", "Regional Priority", 4, 7000),
        ],
        columns=["credit_id", "credit_name", "category", "available_points", "estimated_cost_per_point"],
    )
    prerequisites = pd.DataFrame(
        [
            ("IP-Prereq", "Integrative Project Planning and Design", "Integrative Process"),
            ("EA-Prereq-1", "Fundamental Commissioning and Verification", "Energy and Atmosphere"),
            ("EA-Prereq-2", "Minimum Energy Performance", "Energy and Atmosphere"),
            ("WE-Prereq", "Indoor Water Use Reduction", "Water Efficiency"),
            ("MR-Prereq", "Storage and Collection of Recyclables", "Materials and Resources"),
            ("EQ-Prereq", "Minimum Indoor Air Quality Performance", "Indoor Environmental Quality"),
        ],
        columns=["prerequisite_id", "prerequisite_name", "category"],
    )
    cost_assumptions = credits.copy()
    cost_assumptions["assumption_note"] = "Synthetic portfolio assumption, not a vendor quote."

    projects.to_csv(data_dir / "sample_projects.csv", index=False)
    energy.to_csv(data_dir / "energyplus_sample_outputs.csv", index=False)
    credits.to_csv(data_dir / "leed_credit_reference.csv", index=False)
    prerequisites.to_csv(data_dir / "prerequisite_reference.csv", index=False)
    cost_assumptions.to_csv(data_dir / "cost_assumptions.csv", index=False)


if __name__ == "__main__":
    generate_sample_data()
