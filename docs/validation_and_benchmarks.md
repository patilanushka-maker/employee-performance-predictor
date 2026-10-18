# Validation Strategy & Benchmark Framework

## 1. Validation Protocol & Temporal Partitioning
To ensure generalizability, prevent data leakage, and maintain realistic operational alignment:
- **Primary Holdout Scheme**: Strict Temporal Partitioning.
  - **Training Set (Periods 1–3)**: 3,000 observations (1,000 employees × 3 quarters).
  - **Holdout Test Set (Period 4)**: 1,000 observations (1,000 employees × 1 future evaluation period).
- **Intended Prediction Moment**: At the start of quarter $t$, predicting appraisal ratings realized at the conclusion of quarter $t$.
- **Leakage Controls**:
  - Preprocessing transformers (`StandardScaler`, `OneHotEncoder`) are encapsulated inside `sklearn.pipeline.Pipeline` objects and fitted strictly on training data (Periods 1–3).
  - Identifiers (`employee_id`) and temporal indices (`period`) are strictly excluded from predictive feature matrices.
  - No intra-period or post-outcome variables are accessible to predictors.
- **Cohort Boundary vs. Unseen-Employee Generalization**:
  - The temporal holdout evaluates forecasting accuracy on a future period of the **same employee cohort**.
  - For evaluating generalization to **entirely unseen employees**, Phase 6 introduces a distinct cross-sectional `GroupKFold` (grouped by `employee_id`) executed strictly on training data, reported separately rather than conflated with temporal forecasting.

---

## 2. Evaluation Metrics

### Primary Regression Metrics
- **MAE (Mean Absolute Error)**: Average absolute deviation in appraisal score units $[1.00, 5.00]$.
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals, heavily penalizing large estimation mistakes.
- **$R^2$ (Coefficient of Determination)**: Proportion of variance in future appraisal scores explained by model features.
- **Boundary Conformity**: Count of holdout predictions falling outside the allowable $[1.00, 5.00]$ evaluation range.

---

## 3. Verified Empirical Baseline Benchmarks (Holdout Period 4)

*Evaluated on identical 1,000 test observations from Period 4 (Dataset SHA-256: `2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79`):*

| Model Architecture | MAE ↓ | RMSE ↓ | $R^2$ ↑ | Prediction Range | Out-of-Bounds | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DummyRegressor (Mean)** | 0.4745 | 0.5946 | -0.0158 | [3.22, 3.22] | 0 | **Verified Baseline** |
| **Previous-Performance-Only** | 0.2299 | 0.2848 | 0.7669 | [1.65, 4.73] | 0 | **Verified Baseline** |
| **Full-Feature Ridge Model** | 0.2192 | 0.2713 | 0.7885 | [1.89, 4.89] | 0 | **Verified Baseline** |

### Key Benchmark Insights
1. **Autoregressive Anchor**: The single predictor `previous_performance_score` captures 76.69% of future score variance ($R^2 = 0.7669$, MAE = 0.2299).
2. **Multivariable Lift**: Incorporating full operational predictors (quality, productivity, attendance, projects, training hours, tenure, department, role) with L2 regularization yields an additional +2.16% variance explanation ($R^2 = 0.7885$, MAE = 0.2192).
3. **Natural Bounds**: All baseline predictions natively fall within $[1.00, 5.00]$ without clipping.

