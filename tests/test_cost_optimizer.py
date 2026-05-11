import pandas as pd

from src.cost_optimizer import recommend_minimum_cost_credits


def test_recommend_minimum_cost_credits_selects_lowest_cost_points():
    credits = pd.DataFrame(
        [
            {
                "credit_id": "A",
                "credit_name": "Expensive",
                "category": "Energy and Atmosphere",
                "available_points": 5,
                "estimated_cost_per_point": 100,
            },
            {
                "credit_id": "B",
                "credit_name": "Cheap",
                "category": "Regional Priority",
                "available_points": 4,
                "estimated_cost_per_point": 10,
            },
        ]
    )

    result = recommend_minimum_cost_credits(57, "Gold", credits)

    assert result["score_gap"] == 3
    assert result["recommended_points"] == 3
    assert result["estimated_additional_cost"] == 30
    assert result["recommendations"].iloc[0]["credit_id"] == "B"


def test_recommend_minimum_cost_credits_no_gap_returns_empty():
    credits = pd.DataFrame(
        [
            {
                "credit_id": "A",
                "credit_name": "Credit",
                "category": "Innovation",
                "available_points": 1,
                "estimated_cost_per_point": 100,
            }
        ]
    )

    result = recommend_minimum_cost_credits(82, "Gold", credits)

    assert result["score_gap"] == 0
    assert result["recommendations"].empty
