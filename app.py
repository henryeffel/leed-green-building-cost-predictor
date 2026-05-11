from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.cost_optimizer import recommend_minimum_cost_credits
from src.data_generator import generate_sample_data
from src.energyplus_parser import analyze_energyplus_outputs
from src.esg_report import build_esg_summary, esg_dataframe
from src.excel_exporter import build_excel_report
from src.predict import load_model_bundle, predict_project
from src.preprocessing import load_reference_data, load_training_data
from src.rating import CATEGORY_MAX_POINTS, RATING_THRESHOLDS, get_leed_rating
from src.train_models import train_and_save_models


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"


st.set_page_config(
    page_title="LEED Green Building Cost Predictor",
    page_icon="",
    layout="wide",
)


@st.cache_data
def ensure_data() -> pd.DataFrame:
    if not (DATA_DIR / "sample_projects.csv").exists():
        generate_sample_data(DATA_DIR)
    return load_training_data()


@st.cache_data
def get_references() -> dict[str, pd.DataFrame]:
    if not (DATA_DIR / "leed_credit_reference.csv").exists():
        generate_sample_data(DATA_DIR)
    return load_reference_data(DATA_DIR)


@st.cache_resource
def get_models() -> dict[str, object]:
    if not (ROOT / "models" / "model_bundle.joblib").exists():
        return train_and_save_models(ROOT / "models")
    return load_model_bundle(ROOT / "models")


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


@st.cache_data
def load_report_csv(filename: str) -> pd.DataFrame:
    path = REPORT_DIR / filename
    if not path.exists():
        train_and_save_models(ROOT / "models")
    return pd.read_csv(path)


@st.cache_data
def load_report_text(filename: str) -> str:
    path = REPORT_DIR / filename
    if not path.exists():
        train_and_save_models(ROOT / "models")
    return path.read_text(encoding="utf-8")


def project_input(default: pd.Series) -> pd.DataFrame:
    st.subheader("Project Input")
    with st.form("project_input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            project_name = st.text_input("Project name", value="Portfolio Demo Project")
            building_type = st.selectbox(
                "Building type",
                ["Office", "Healthcare", "Education", "Retail", "Multifamily"],
                index=["Office", "Healthcare", "Education", "Retail", "Multifamily"].index(default["building_type"]),
            )
            climate_zone = st.selectbox("ASHRAE climate zone", ["1A", "2A", "3A", "4A", "5A", "6A"], index=2)
            gross_floor_area_m2 = st.number_input("Gross floor area (m2)", 1000, 150000, int(default["gross_floor_area_m2"]))
        with c2:
            num_floors = st.number_input("Number of floors", 1, 120, int(default["num_floors"]))
            baseline_energy_kwh = st.number_input(
                "Baseline energy (kWh/year)", 10000, 30000000, int(default["baseline_energy_kwh"])
            )
            proposed_energy_kwh = st.number_input(
                "Proposed energy (kWh/year)", 10000, 30000000, int(default["proposed_energy_kwh"])
            )
            energy_cost_per_kwh = st.number_input("Energy cost ($/kWh)", 0.01, 1.0, float(default["energy_cost_per_kwh"]))
        with c3:
            emissions_factor = st.number_input(
                "Emissions factor (kg CO2/kWh)", 0.01, 1.5, float(default["emissions_factor_kg_co2_per_kwh"])
            )
            water_saving_rate = st.slider("Water saving rate", 0.0, 0.8, float(default["water_saving_rate"]))
            recycled_material_ratio = st.slider(
                "Recycled material ratio", 0.0, 0.8, float(default["recycled_material_ratio"])
            )
            ieq_score = st.slider("Indoor environmental quality score", 0.0, 16.0, float(default["ieq_score"]))
            renewable_energy_rate = st.slider("Renewable energy rate", 0.0, 0.6, float(default["renewable_energy_rate"]))
            site_density_score = st.slider("Site density score", 0.0, 10.0, float(default["site_density_score"]))
        submitted = st.form_submit_button("Run analysis")

    input_df = pd.DataFrame(
        [
            {
                "project_id": "USER-INPUT",
                "project_name": project_name,
                "building_type": building_type,
                "climate_zone": climate_zone,
                "gross_floor_area_m2": gross_floor_area_m2,
                "num_floors": num_floors,
                "baseline_energy_kwh": baseline_energy_kwh,
                "proposed_energy_kwh": proposed_energy_kwh,
                "energy_cost_per_kwh": energy_cost_per_kwh,
                "emissions_factor_kg_co2_per_kwh": emissions_factor,
                "water_saving_rate": water_saving_rate,
                "recycled_material_ratio": recycled_material_ratio,
                "ieq_score": ieq_score,
                "renewable_energy_rate": renewable_energy_rate,
                "site_density_score": site_density_score,
            }
        ]
    )
    analyzed = analyze_energyplus_outputs(input_df)
    st.session_state["active_project"] = analyzed
    if submitted:
        st.success("Analysis input updated.")
    return analyzed


def show_reference_layer(references: dict[str, pd.DataFrame]) -> None:
    st.subheader("Rule-Based LEED Reference Layer")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Rating thresholds")
        st.dataframe(pd.DataFrame(RATING_THRESHOLDS.items(), columns=["Rating", "Minimum points"]), use_container_width=True)
    with c2:
        st.write("Category-level score structure")
        st.dataframe(
            pd.DataFrame(CATEGORY_MAX_POINTS.items(), columns=["Category", "Maximum points"]),
            use_container_width=True,
        )
    st.write("Prerequisites")
    st.dataframe(references["prerequisites"], use_container_width=True)
    st.caption("This reference layer mirrors the public LEED scorecard structure for context. It is not the main prediction method.")


def show_energyplus_page(active_project: pd.DataFrame, training_df: pd.DataFrame) -> None:
    st.subheader("EnergyPlus-Style Output CSV Analysis")
    row = active_project.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Energy saving rate", format_percent(row["energy_saving_rate"]))
    c2.metric("Annual energy cost saving", f"${row['annual_energy_cost_saving']:,.0f}")
    c3.metric("Estimated CO2 reduction", f"{row['estimated_co2_reduction']:,.1f} tCO2/year")

    st.write("Synthetic portfolio comparison")
    st.dataframe(
        training_df[
            [
                "project_id",
                "project_name",
                "baseline_energy_kwh",
                "proposed_energy_kwh",
                "energy_saving_rate",
                "annual_energy_cost_saving",
                "estimated_co2_reduction",
            ]
        ].head(25),
        use_container_width=True,
    )


def show_prediction_page(active_project: pd.DataFrame) -> pd.DataFrame:
    st.subheader("ML-Based LEED Prediction")
    models = get_models()
    prediction = predict_project(active_project)
    row = prediction.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted LEED score", f"{row['predicted_leed_score']:.1f}")
    c2.metric("Rating classifier", row["predicted_rating_class"])
    c3.metric("Rating from score", row["predicted_rating_from_score"])

    selected_models = models.get("selected_models", {})
    st.write("Selected final models")
    st.dataframe(
        pd.DataFrame(
            [
                {"Task": "Score regression", "Selected model": selected_models.get("score_regression", "N/A")},
                {"Task": "Cost regression", "Selected model": selected_models.get("cost_regression", "N/A")},
                {"Task": "Rating classification", "Selected model": selected_models.get("rating_classification", "N/A")},
            ]
        ),
        use_container_width=True,
    )

    st.write("Model experiment comparison on held-out synthetic data")
    experiments = load_report_csv("model_experiments.csv")
    st.dataframe(experiments, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        st.write("Score prediction feature importance")
        score_importance = load_report_csv("feature_importance_score.csv")
        st.bar_chart(score_importance.head(12), x="feature", y="importance")
    with c5:
        st.write("Cost prediction feature importance")
        cost_importance = load_report_csv("feature_importance_cost.csv")
        st.bar_chart(cost_importance.head(12), x="feature", y="importance")

    st.text("Rating classification report")
    st.code(load_report_text("rating_classification_report.txt"))
    st.write("Rating confusion matrix")
    st.dataframe(load_report_csv("rating_confusion_matrix.csv"), use_container_width=True)
    st.info(
        "Class imbalance limitation: the synthetic portfolio contains fewer samples for some LEED rating classes. "
        "The pipeline uses stratified splitting and class_weight='balanced' where applicable, but these metrics should "
        "not be interpreted as production-level accuracy."
    )
    return prediction


def show_cost_page(prediction: pd.DataFrame, references: dict[str, pd.DataFrame]) -> dict[str, object]:
    st.subheader("Minimum-Cost Credit Recommendation")
    current_score = float(prediction.iloc[0]["predicted_leed_score"])
    target_rating = st.selectbox("Target LEED rating", ["Certified", "Silver", "Gold", "Platinum"], index=2)
    result = recommend_minimum_cost_credits(current_score, target_rating, references["credits"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Score gap", f"{result['score_gap']:.1f}")
    c2.metric("Recommended points", f"{result['recommended_points']:.1f}")
    c3.metric("Estimated additional cost", f"${result['estimated_additional_cost']:,.0f}")
    if result["recommendations"].empty:
        st.info("The selected target rating is already met by the predicted score.")
    else:
        st.dataframe(result["recommendations"], use_container_width=True)
    return result


def show_esg_page(active_project: pd.DataFrame) -> None:
    st.subheader("ESG Summary Dashboard")
    summary = build_esg_summary(active_project.iloc[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Energy saving rate", format_percent(summary["energy_saving_rate"]))
    c2.metric("Annual cost saving", f"${summary['annual_energy_cost_saving']:,.0f}")
    c3.metric("CO2 reduction", f"{summary['estimated_co2_reduction_tonnes']:,.1f} tCO2/year")
    c4, c5, c6 = st.columns(3)
    c4.metric("Water saving rate", format_percent(summary["water_saving_rate"]))
    c5.metric("Recycled material ratio", format_percent(summary["recycled_material_ratio"]))
    c6.metric("IEQ score", f"{summary['indoor_environmental_quality_score']:.1f} / 16")


def show_export_page(
    active_project: pd.DataFrame,
    prediction: pd.DataFrame,
    cost_result: dict[str, object],
    training_df: pd.DataFrame,
) -> None:
    st.subheader("Excel Export")
    sheets = {
        "project_input": active_project,
        "leed_prediction": prediction,
        "credit_recommendations": cost_result["recommendations"],
        "esg_summary": pd.DataFrame([build_esg_summary(active_project.iloc[0])]),
        "synthetic_portfolio_sample": training_df.head(50),
    }
    report = build_excel_report(sheets)
    st.download_button(
        "Download Excel report",
        data=report,
        file_name="leed_green_building_cost_predictor_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    training_df = ensure_data()
    references = get_references()
    default = training_df.iloc[0]

    st.title("LEED Green Building Cost Predictor")
    st.caption(
        "Synthetic AI portfolio project for LEED credit prediction, EnergyPlus-style data analysis, cost prediction, and ESG reporting."
    )

    active_project = project_input(default)
    tabs = st.tabs(
        [
            "Reference",
            "EnergyPlus Analysis",
            "LEED Prediction",
            "Cost Recommendation",
            "ESG Summary",
            "Excel Export",
        ]
    )
    with tabs[0]:
        show_reference_layer(references)
    with tabs[1]:
        show_energyplus_page(active_project, training_df)
    with tabs[2]:
        prediction = show_prediction_page(active_project)
    with tabs[3]:
        prediction = predict_project(active_project)
        cost_result = show_cost_page(prediction, references)
    with tabs[4]:
        show_esg_page(active_project)
        st.dataframe(esg_dataframe(training_df).head(30), use_container_width=True)
    with tabs[5]:
        prediction = predict_project(active_project)
        cost_result = recommend_minimum_cost_credits(
            float(prediction.iloc[0]["predicted_leed_score"]), "Gold", references["credits"]
        )
        show_export_page(active_project, prediction, cost_result, training_df)

    st.warning(
        "Limitations: this is not an official LEED certification tool, does not replace USGBC review, "
        "uses synthetic data, and analyzes EnergyPlus-style CSV output instead of executing EnergyPlus."
    )


if __name__ == "__main__":
    main()
