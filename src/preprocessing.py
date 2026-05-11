"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.energyplus_parser import load_energyplus_outputs
from src.feature_engineering import build_model_frame


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_reference_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load static LEED reference and assumption data."""
    return {
        "credits": pd.read_csv(data_dir / "leed_credit_reference.csv"),
        "prerequisites": pd.read_csv(data_dir / "prerequisite_reference.csv"),
        "costs": pd.read_csv(data_dir / "cost_assumptions.csv"),
    }


def load_training_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load synthetic project and EnergyPlus-style data for model training."""
    projects = pd.read_csv(data_dir / "sample_projects.csv")
    energy = load_energyplus_outputs(str(data_dir / "energyplus_sample_outputs.csv"))
    return build_model_frame(projects, energy)
