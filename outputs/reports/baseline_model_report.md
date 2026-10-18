# Baseline Performance Predictor Experiment Report

**Project**: Employee Performance Predictor using Data Analytics  
**Execution Date**: 2026-10-16  
**Evaluation Design**: Temporal Holdout (Periods 1–3 Train, Period 4 Test)  
**Dataset SHA-256**: `2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79`  

---

## 1. Experimental Objective & Scope

The objective of this phase is to establish rigorous, reproducible empirical baselines for predicting `future_performance_score` on an out-of-time evaluation period. Three models of increasing complexity are benchmarked:
1. **DummyRegressor (Mean Baseline)**: Predicts constant training set mean (benchmark of zero explanatory skill).
2. **Previous-Performance-Only Model**: Single-predictor linear model using `previous_performance_score`.
3. **Full-Feature Ridge Model**: Multivariable pipeline incorporating all permitted historical predictors.

> [!IMPORTANT]
> **Temporal Integrity & Generalization Boundary**:
> - Models are trained strictly on **Periods 1, 2, and 3** (3,000 observations) and evaluated on **Period 4** (1,000 observations).
> - Preprocessing operations (`StandardScaler`, `OneHotEncoder`) are fitted strictly on training data with zero leakage from Period 4.
> - `employee_id` is excluded from all feature matrices (identifier key only).
> - `period` is treated as a temporal index and excluded from predictor sets.
> - This design estimates out-of-time performance for the existing employee population; evaluating generalization to entirely > unseen employees would require a separate group-aware partition.

---

## 2. Experimental Data Partitioning

- **Training Set (Periods 1, 2, 3)**: **3,000 records** (1,000 employees × 3 sequential observation periods)
- **Test Set (Period 4 Holdout)**: **1,000 records** (1,000 employees × 1 future evaluation period)
- **Target Variable ($y$)**: `future_performance_score` on continuous appraisal scale $[1.00, 5.00]$
- **Feature Sets**:
  - *Previous-Performance-Only*: `['previous_performance_score']`
  - *Full-Feature Predictors (10)*: Numeric (8): `['previous_performance_score', 'quality_score_previous_period', 'productivity_score_previous_period', 'project_completion_rate_previous_period', 'attendance_rate_previous_period', 'training_hours_previous_period', 'years_experience', 'tenure_months']`, Categorical (2): `['department', 'job_role']`

---

## 3. Quantitative Evaluation Results (Holdout Period 4)

Every model was evaluated on the identical 1,000 Period 4 observations:

| Model Architecture | MAE ↓ | RMSE ↓ | $R^2$ ↑ | Prediction Range [Min, Max] | Outside [1.0, 5.0] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DummyRegressor (Mean)** | **0.4745** | **0.5946** | **-0.0158** | [3.22, 3.22] | 0 |
| **Previous-Performance-Only** | **0.2299** | **0.2848** | **0.7669** | [1.65, 4.73] | 0 |
| **Full-Feature Ridge Model** | **0.2192** | **0.2713** | **0.7885** | [1.89, 4.89] | 0 |

### Interpretation of Evaluation Metrics in Plain Language:
- **MAE (Mean Absolute Error)**: Measures the average magnitude of absolute prediction errors in actual appraisal rating units. An MAE of 0.22 means predictions deviate by approximately 0.22 points on the 1–5 performance scale.
- **RMSE (Root Mean Squared Error)**: Similar to MAE but squares deviations before averaging, penalizing larger prediction mistakes more severely.
- **$R^2$ (Coefficient of Determination)**: Proportion of variance in Period 4 performance explained by the model relative to a naive horizontal line. An $R^2$ of 0.7885 indicates that 78.85% of variance in future ratings is captured by historical indicators.

---

## 4. Key Comparative Findings & Analysis

1. **Dominance of Prior Rating (Autoregressive Baseline)**:
   - The single-predictor `Previous-Performance-Only` model achieves $R^2 = 0.7669$ and $\text{MAE} = 0.2299$.
   - Prior performance rating alone accounts for over 76.7% of the variance in subsequent quarter ratings, highlighting strong autoregressive persistence.
2. **Incremental Value of Full Workforce Metrics**:
   - Adding quality, productivity, attendance, project completion, training hours, experience, tenure, and department/role boosts explanatory power to $R^2 = 0.7885$, reducing MAE to **0.2192** and RMSE to **0.2713**.
   - While the full model is superior across all three metrics, the incremental gain (+2.16% R²) confirms that in this synthetic data generating process, recent performance appraisal is the predominant anchor, with other metrics providing modest refinement.
3. **Boundary Integrity**:
   - For all models, 100% of holdout predictions naturally fell within the valid $[1.00, 5.00]$ range with zero out-of-bounds predictions. No artificial clipping was required.

---

## 5. Limitations & Ethical Considerations

- **Synthetic Generating Assumptions**: Performance persistence and feature correlations are properties of the simulation algorithm, not empirical observations from real corporate HR systems.
- **Selection Bias Caution**: Training hours was retained in the full feature set; as revealed in Phase 4 EDA, its marginal association is negative due to remedial training assignments for struggling performers.
- **No Automated Employment Decisions**: These predictive baselines are portfolio research demonstrations and should never be used to automate hiring, promotion, compensation, or termination decisions.

---

## 6. Reproduction Command

```powershell
python scripts/train_baselines.py
```
