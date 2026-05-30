# Research-backed Case Study: LEED Green Building Cost Predictor

## 1. Background

LEED 인증 초기 검토에서는 건물 기본정보, 에너지 성능, 예상 점수, 추가공사비, ESG 지표를 함께 판단해야 합니다. 그러나 이러한 정보는 도면, 에너지 시뮬레이션 결과, 인증 항목표, 비용 산정표 등으로 분산되어 있어 초기 의사결정이 느려질 수 있습니다.

본 프로젝트는 이 문제를 포트폴리오 규모의 MVP로 축소하여, 친환경 건축 데이터를 전처리·예측·추천·리포팅까지 연결하는 ML 기반 decision-support workflow를 구현하는 것을 목표로 했습니다.

현재 구현은 실제 LEED 인증 도구가 아니며, USGBC 심사를 대체하지 않습니다. 실제 EnergyPlus를 실행하지 않고 EnergyPlus-style output CSV를 분석하며, synthetic dataset 기반의 포트폴리오 프로토타입입니다.

## 2. Research Reference

본 프로젝트는 LEED credit targeting을 머신러닝 문제로 다룬 연구를 참고했습니다.

참고 논문:

**Machine Learning for Leadership in Energy and Environmental Design Credit Targeting: Project Attributes and Climate Analysis Toward Sustainability**

논문에서는 USGBC LEED-certified project data와 US NCEI climate data를 결합하고, project attributes와 climate variables를 feature로 사용했습니다. 예측 target은 LEED points achieved였으며, Decision Tree, SVR, XGBoost 모델을 비교했습니다. 논문 결과에서는 XGBoost가 MAE, RMSE, R² 기준으로 가장 좋은 성능을 보였고, feature importance 분석에서는 LEED version, project type, climate variables가 중요한 변수로 나타났습니다.

본 프로젝트는 해당 논문을 재현한 것이 아닙니다. 논문에서 제시한 문제 구조, feature 구성 관점, 모델 벤치마킹 방식, 해석 가능성 접근을 참고하여 synthetic dataset 기반 portfolio MVP에 맞게 축소 구현했습니다.

## 3. What I Took from the Paper

1. **LEED credit targeting can be framed as supervised ML.**  
   논문은 LEED points achieved를 예측 target으로 두고, 프로젝트 속성과 기후 변수를 feature로 구성했습니다. 이 접근을 참고해 현재 MVP도 LEED score regression, rating classification, additional construction cost regression을 supervised ML task로 정의했습니다.

2. **Project attributes matter.**  
   LEED version, owner type, project type, gross floor area 같은 프로젝트 속성은 인증 결과를 설명하는 중요한 신호로 사용될 수 있습니다. 현재 MVP에서는 실제 USGBC 프로젝트 데이터 대신 synthetic dataset의 building type, climate zone, gross area, floors 등을 프로젝트 속성 feature로 사용했습니다.

3. **Climate and energy-related variables matter.**  
   논문에서는 CDD, PRCP, temperature, snowfall 등 실제 NCEI 기후 변수를 사용해 지역 환경 조건이 LEED credit targeting에 영향을 줄 수 있음을 분석했습니다. 현재 MVP에서는 실제 NCEI 기후 데이터 대신 EnergyPlus-style output에서 파생한 energy saving rate, CO₂ reduction, annual cost saving 등을 energy-related proxy로 사용했습니다.

4. **Model benchmarking is necessary.**  
   논문은 Decision Tree, SVR, XGBoost를 비교하여 단일 모델 선택보다 벤치마킹 기반 평가가 필요하다는 점을 보여줍니다. 현재 MVP도 Dummy, Linear, RandomForest, GradientBoosting을 비교하고, synthetic dataset 기준으로 task별 최종 모델을 선택했습니다.

5. **Feature importance improves interpretability.**  
   feature importance는 어떤 변수들이 예측에 영향을 주는지 설명하는 데 도움이 됩니다. 현재 MVP에서는 score prediction과 cost prediction에 대해 feature importance report를 생성해, 예측 결과를 단순 수치가 아니라 해석 가능한 decision-support output으로 제시하도록 구성했습니다.

## 4. Research-to-Implementation Mapping

| Research Paper | My Portfolio MVP |
|---|---|
| USGBC LEED-certified project dataset | Synthetic green building project dataset |
| NCEI climate data | EnergyPlus-style output CSV |
| LEED points achieved prediction | LEED score regression |
| Project attributes: LEED version, owner type, project type, gross floor area | Building type, climate zone, gross area, floors, energy/water/material/IEQ variables |
| Climate variables: CDD, PRCP, temperature, snowfall | Energy saving rate, CO₂ reduction, water saving rate, renewable energy rate |
| DT / SVR / XGBoost comparison | Dummy / Linear / RandomForest / GradientBoosting comparison |
| XGBoost feature importance | Feature importance for score and cost prediction |
| Research model analysis | Streamlit dashboard + FastAPI API + optional OpenAI reporting layer |

## 5. Current MVP Status

| Area | Status | Notes |
|---|---|---|
| Streamlit Dashboard | Implemented | Main portfolio MVP interface |
| EnergyPlus-style CSV Preprocessing | Implemented | Parses sample output CSV and derives energy-related metrics |
| LEED Score Prediction | Implemented | Synthetic dataset 기반 score regression |
| Rating Classification | Implemented | LEED rating class prediction with class-imbalance-aware selection |
| Additional Cost Prediction | Implemented | Synthetic cost regression task |
| Model Benchmarking | Implemented | Dummy / Linear / RandomForest / GradientBoosting comparison |
| Feature Importance / Confusion Matrix | Implemented | Reports and visualization assets generated under `reports/` |
| Minimum-cost Credit Recommendation | Implemented | Cost-per-point based recommendation logic |
| ESG Summary Dashboard | Implemented | Energy, cost, CO₂, water, material, IEQ indicators |
| Excel Export | Implemented | Exports inputs, predictions, recommendations, ESG summary, and samples |
| FastAPI `/predict`, `/recommend` API | Implemented | Optional backend extension |
| Optional OpenAI Reporting Layer | Implemented | Explains backend JSON only; deterministic fallback available |
| Real EnergyPlus Execution | Future Work | Current MVP analyzes EnergyPlus-style output CSV only |
| Real USGBC/NCEI Dataset | Future Work | Current MVP does not use real USGBC or NCEI data |
| React Frontend | Future Work | Current UI is Streamlit-based |
| XGBoost + Optuna tuning | Future Work | Not the current final model |
| Category-level LEED credit prediction | Future Work | Current MVP predicts total score and rating class |
| Budget-constrained optimization with real cost data | Future Work | Current cost assumptions are synthetic portfolio assumptions |

## 6. ML Experiment Pipeline

현재 MVP의 ML experiment pipeline은 `reports/` 산출물을 기준으로 정리할 수 있습니다.

3개의 prediction task를 구성했습니다.

- LEED score regression
- Rating classification
- Additional construction cost regression

모델 비교는 다음 후보군을 대상으로 수행했습니다.

- Dummy baseline
- Linear model
- RandomForest
- GradientBoosting

평가 지표는 task 유형에 따라 다음 지표를 사용했습니다.

- MAE
- RMSE
- R²
- Accuracy
- macro F1
- weighted F1

시각화 산출물은 다음과 같습니다.

- model experiment comparison
- score feature importance
- cost feature importance
- rating confusion matrix

Synthetic dataset 기준 실험 결과는 다음과 같습니다.

| Result | Value |
|---|---:|
| LEED score prediction | Dummy baseline 대비 MAE 38.2% 개선 |
| Additional cost prediction | Dummy baseline 대비 MAE 79.0% 개선 |
| Cost regression final R² | 0.95 |
| Rating classification | Macro F1 +20.6pp 개선 |
| Pytest | 15 tests passed |

위 수치는 synthetic dataset 기준이며, production-level accuracy를 주장하지 않습니다. 현재 실험 결과는 실제 인증 성능이나 실제 비용 예측 정확도가 아니라, 포트폴리오 MVP 내부에서 baseline 대비 모델링 workflow가 어떻게 개선되는지를 보여주는 상대 비교입니다.

## 7. LLM Integration Strategy

초기에는 LLM을 사용해 사용자의 누락 입력값을 보완하거나, LEED 비용·점수 계산을 자연어로 처리하는 방식을 고려했습니다. 그러나 LEED 인증과 비용 산정은 수치 정확성이 중요한 도메인이므로, LLM이 임의로 값을 생성하면 전체 예측 결과를 오염시킬 위험이 있습니다.

따라서 현재 프로젝트에서는 LLM을 계산 주체가 아니라 설명 레이어로 제한했습니다.

| Responsibility | Component |
|---|---|
| 점수 예측 | ML model |
| 비용 예측 | ML model |
| credit recommendation | cost optimizer |
| 자연어 리포트 | OpenAI API optional reporting layer |

OpenAI API는 `/recommend` 결과로 반환된 JSON만 설명합니다. API key가 없거나 호출 실패 시 deterministic fallback report가 반환됩니다. 이 구조는 LLM hallucination 위험을 줄이기 위해 수치 계산과 자연어 생성을 분리한 설계입니다.

## 8. Future Architecture

현재 프로젝트는 Streamlit 기반 MVP입니다. 향후 실제 SaaS 구조로 확장한다면 아래 구조가 적합합니다.

```text
User / Architect
 -> React or Next.js dashboard
 -> Architectural summary upload or form input
 -> JSON-schema-based parameter extraction
 -> FastAPI backend
 -> ML surrogate model trained on historical EnergyPlus outputs
 -> Cost optimizer
 -> Optional LLM Function Calling report layer
 -> Dashboard visualization / Excel report export
```

이 확장 방향은 현재 구현 완료 범위가 아니라 future architecture입니다. 현재 MVP는 실제 EnergyPlus 실행 병목을 해결한 것이 아니라, EnergyPlus-style output 기반으로 surrogate model 확장 가능성을 검토한 포트폴리오 프로토타입입니다.
