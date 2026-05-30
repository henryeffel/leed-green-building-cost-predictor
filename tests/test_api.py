from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def sample_payload():
    return {
        "project_name": "API Test Project",
        "building_type": "Office",
        "climate_zone": "3A",
        "gross_floor_area_m2": 12000,
        "num_floors": 8,
        "baseline_energy_kwh": 1500000,
        "proposed_energy_kwh": 1050000,
        "energy_cost_per_kwh": 0.12,
        "emissions_factor_kg_co2_per_kwh": 0.42,
        "water_saving_rate": 0.32,
        "recycled_material_ratio": 0.25,
        "ieq_score": 10.5,
        "renewable_energy_rate": 0.18,
        "site_density_score": 7.0,
        "target_rating": "Gold",
    }


def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_fields():
    response = client.post("/predict", json=sample_payload())

    assert response.status_code == 200
    body = response.json()
    for field in [
        "predicted_leed_score",
        "predicted_rating_from_score",
        "predicted_rating_class",
        "predicted_additional_cost_usd",
        "energy_saving_rate",
        "annual_energy_cost_saving",
        "estimated_co2_reduction",
        "target_rating",
        "score_gap",
    ]:
        assert field in body


def test_recommend_works_without_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    response = client.post("/recommend", json=sample_payload())

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "recommendations" in body
    assert "llm_report" in body
    assert "official LEED review" in body["llm_report"]
