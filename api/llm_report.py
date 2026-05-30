"""Optional OpenAI reporting layer for API recommendation outputs."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv


FALLBACK_SUFFIX = (
    "This report is generated from backend ML and optimization outputs, "
    "not from official LEED review."
)


def generate_fallback_report(
    prediction: dict[str, Any],
    recommendations: dict[str, Any],
) -> str:
    """Return a deterministic report when OpenAI is unavailable."""
    rating = prediction.get("predicted_rating_from_score") or prediction.get("predicted_rating_class", "the predicted rating")
    target = prediction.get("target_rating", "the selected target rating")
    score_gap = prediction.get("score_gap", "unavailable")
    reachable = recommendations.get("is_target_reachable", "unavailable")

    return (
        f"Based on the predicted LEED score and cost-per-point recommendation, "
        f"the project is estimated to reach {rating}. The target rating is {target}, "
        f"with a score gap of {score_gap}. The recommended credits prioritize lower "
        f"cost-per-point options, and target reachability is {reachable}. {FALLBACK_SUFFIX}"
    )


def generate_recommendation_report(
    prediction: dict[str, Any],
    recommendations: dict[str, Any],
) -> str:
    """Generate an explanatory report with OpenAI, falling back deterministically."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return generate_fallback_report(prediction, recommendations)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        payload = {
            "prediction": prediction,
            "recommendations": recommendations,
        }
        prompt = (
            "You are writing a concise portfolio prototype report for a LEED green building "
            "cost predictor. Explain the already-computed backend outputs in plain language.\n\n"
            "Rules:\n"
            "- Do not invent numbers.\n"
            "- Use only the provided JSON.\n"
            "- If a field is missing, say it is unavailable.\n"
            "- Do not claim this is an official LEED certification result.\n"
            "- Mention this is a portfolio prototype using synthetic data.\n"
            "- Do not perform numerical calculation; the ML model and optimizer already did that.\n\n"
            f"JSON:\n{json.dumps(payload, indent=2, sort_keys=True)}"
        )

        response = client.responses.create(
            model=os.getenv("OPENAI_REPORT_MODEL", "gpt-4.1-mini"),
            input=prompt,
            temperature=0.2,
            max_output_tokens=450,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
    except Exception:
        return generate_fallback_report(prediction, recommendations)

    return generate_fallback_report(prediction, recommendations)
