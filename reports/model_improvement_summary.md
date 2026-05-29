# Model Improvement Summary

This summary is calculated from `reports/model_experiments.csv`. The dataset is synthetic and generated for portfolio demonstration, so these numbers should not be presented as production-level accuracy or real-world certification performance.

## Selected Final Models

| Task | Baseline | Final selected model | Selection basis |
| --- | --- | --- | --- |
| LEED score regression | `DummyRegressor_mean` | `LinearRegression` | Highest R2, then lowest MAE |
| Additional cost regression | `DummyRegressor_mean` | `LinearRegression` | Highest R2, then lowest MAE |
| Rating classification | `DummyClassifier_most_frequent` | `LogisticRegression_balanced` | Highest macro F1, then accuracy |

## Improvement vs Baseline

| Task | MAE improvement | RMSE improvement | R2 improvement |
| --- | ---: | ---: | ---: |
| LEED score regression | 38.2% | 36.4% | +0.60 |
| Additional cost regression | 79.0% | 77.9% | +0.95 |

| Task | Accuracy change | Macro F1 change | Weighted F1 change |
| --- | ---: | ---: | ---: |
| Rating classification | -8.9pp | +20.6pp | +9.2pp |

## Conservative Resume Bullet Candidates

- Dummy baseline 대비 LEED 점수 예측 MAE를 38.2% 개선하고, R2를 -0.01에서 0.59로 향상시킨 최종 회귀 모델을 선택
- 추가공사비 예측에서 Dummy baseline 대비 MAE를 79.0% 개선하고, R2 0.95 수준의 선형 회귀 모델과 feature importance 기반 비용 영향 변수를 분석
- 등급 분류 모델은 stratified split과 `class_weight='balanced'`를 적용해 macro F1-score를 baseline 대비 20.6pp 개선했으며, accuracy는 낮아진 점을 고려해 macro F1 기준으로 보수적으로 최종 모델 선택

## Notes

- Rating classification improved macro F1 and weighted F1, but accuracy decreased by 8.9pp compared with the most-frequent dummy baseline.
- The classification result is better framed as class-imbalance-aware model selection, not as a broad accuracy improvement.
- Because the data is synthetic, these metrics are suitable for explaining the experiment process and relative model comparison only.
