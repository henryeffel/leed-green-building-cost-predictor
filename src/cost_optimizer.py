"""Minimum-cost LEED credit recommendation."""

from __future__ import annotations

import pandas as pd

from src.rating import score_gap_to_target


def recommend_minimum_cost_credits(
    current_score: float,
    target_rating: str,
    credit_candidates: pd.DataFrame,
) -> dict[str, object]:
    """Recommend credits by lowest estimated cost per point until target gap is covered."""
    gap_info = score_gap_to_target(current_score, target_rating)
    candidates = credit_candidates.copy()
    required = {"credit_id", "credit_name", "category", "available_points", "estimated_cost_per_point"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Credit candidates missing columns: {sorted(missing)}")

    candidates["cost_per_point"] = candidates["estimated_cost_per_point"]
    candidates = candidates[candidates["available_points"] > 0].sort_values(
        ["cost_per_point", "available_points"], ascending=[True, False]
    )

    selected_rows = []
    points_needed = gap_info.gap
    total_points = 0.0
    total_cost = 0.0
    for _, row in candidates.iterrows():
        if points_needed <= 0:
            break
        points = min(float(row["available_points"]), points_needed)
        cost = points * float(row["estimated_cost_per_point"])
        selected = row.to_dict()
        selected["recommended_points"] = points
        selected["estimated_additional_cost"] = cost
        selected_rows.append(selected)
        total_points += points
        total_cost += cost
        points_needed -= points

    return {
        "current_score": current_score,
        "target_rating": target_rating,
        "target_score": gap_info.target_score,
        "score_gap": gap_info.gap,
        "recommended_points": total_points,
        "estimated_additional_cost": total_cost,
        "is_target_reachable": points_needed <= 0,
        "recommendations": pd.DataFrame(selected_rows),
    }
