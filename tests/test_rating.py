import pytest

from src.rating import category_score_share, get_leed_rating, score_gap_to_target


@pytest.mark.parametrize(
    ("score", "rating"),
    [(39.9, "Not Certified"), (40, "Certified"), (50, "Silver"), (60, "Gold"), (80, "Platinum")],
)
def test_get_leed_rating(score, rating):
    assert get_leed_rating(score) == rating


def test_score_gap_to_target():
    gap = score_gap_to_target(57, "Gold")
    assert gap.target_score == 60
    assert gap.gap == 3


def test_category_score_share_clips_to_reference_max():
    shares = category_score_share({"Energy and Atmosphere": 40})
    assert shares["Energy and Atmosphere"] == 1.0
