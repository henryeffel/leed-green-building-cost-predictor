"""Reference LEED rating thresholds and scorecard helpers."""

from __future__ import annotations

from dataclasses import dataclass


RATING_THRESHOLDS = {
    "Certified": 40,
    "Silver": 50,
    "Gold": 60,
    "Platinum": 80,
}

CATEGORY_MAX_POINTS = {
    "Location and Transportation": 16,
    "Sustainable Sites": 10,
    "Water Efficiency": 11,
    "Energy and Atmosphere": 33,
    "Materials and Resources": 13,
    "Indoor Environmental Quality": 16,
    "Innovation": 6,
    "Regional Priority": 4,
}


@dataclass(frozen=True)
class RatingGap:
    current_score: float
    target_rating: str
    target_score: int
    gap: float


def get_leed_rating(score: float) -> str:
    """Return the LEED rating implied by a total score."""
    if score >= RATING_THRESHOLDS["Platinum"]:
        return "Platinum"
    if score >= RATING_THRESHOLDS["Gold"]:
        return "Gold"
    if score >= RATING_THRESHOLDS["Silver"]:
        return "Silver"
    if score >= RATING_THRESHOLDS["Certified"]:
        return "Certified"
    return "Not Certified"


def score_gap_to_target(current_score: float, target_rating: str) -> RatingGap:
    """Calculate the score gap to a target LEED rating threshold."""
    if target_rating not in RATING_THRESHOLDS:
        raise ValueError(f"Unknown LEED target rating: {target_rating}")
    target_score = RATING_THRESHOLDS[target_rating]
    return RatingGap(
        current_score=current_score,
        target_rating=target_rating,
        target_score=target_score,
        gap=max(0.0, target_score - current_score),
    )


def category_score_share(category_scores: dict[str, float]) -> dict[str, float]:
    """Normalize category scores against the LEED scorecard reference structure."""
    shares: dict[str, float] = {}
    for category, max_points in CATEGORY_MAX_POINTS.items():
        score = float(category_scores.get(category, 0.0))
        shares[category] = min(max(score / max_points, 0.0), 1.0)
    return shares
