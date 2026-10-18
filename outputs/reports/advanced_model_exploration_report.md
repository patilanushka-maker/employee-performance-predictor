# Phase 6: Feature Engineering & Advanced Model Exploration Report

**Project**: Employee Performance Predictor using Data Analytics  
**Execution Date**: 2026-10-17  
**Status**: Phase 6 Completed  
**Dataset SHA-256**: `2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79`  
**Scikit-Learn Version**: `1.3.2`  

---

## 1. Executive Summary & Core Findings

Phase 6 evaluated whether domain-motivated feature engineering and non-linear ensemble architectures (Random Forest, Histogram Gradient Boosting) improve upon the established full-feature Ridge baseline.

> [!WARNING]
> **Evaluation Integrity & Holdout Disclosure**:
> During Phase 6 exploratory development, multiple candidate configurations (including an alpha sweep of [0.1, 1.0, 5.0, 10.0] > and diverse model families) were evaluated directly on Period 4. Consequently, **Period 4 cannot be characterized as an untouched holdout**.
> Instead, Period-4 metrics must be interpreted as **exploratory benchmark results**. Model selection itself was strictly governed > by the expanding-window temporal cross-validation on Periods 1–3, but true independent validation would require future unobserved periods (e.g. Period 5).

> [!IMPORTANT]
> **Key Empirical Takeaways**:
> 1. **Temporal Selection Outcome (Periods 1–3)**: Across expanding-window temporal validation folds, >    `Ridge (All Eng Feats, alpha=10.0)` achieved the lowest validation MAE (0.2307), closely followed by the Ridge baseline (0.2309).
> 2. **Exploratory Holdout Benchmark (Period 4)**: The original 10-feature Ridge baseline achieved the lowest test error >    (MAE = **0.2192**, RMSE = **0.2713**, $R^2 = 0.7885$), >    slightly outperforming the engineered feature configuration (MAE = 0.2201).
> 3. **Non-Linear Model Degradation**: Tree ensembles (Random Forest MAE 0.2225, HistGradientBoosting MAE 0.2270) >    did not beat regularized linear regression due to the underlying linear-additive data generating process.
> 4. **Unseen-Employee Boundary**: Without prior appraisal history, new hire predictions yield $R^2 \approx 0.0$, >    empirically demonstrating that organizational tenure and demographic profile alone cannot forecast individual performance.

---

## 2. Expanding-Window Temporal Validation Design (Periods 1–3)

Model selection and hyperparameter exploration were conducted strictly on Periods 1–3 using chronological expanding-window validation:
- **Fold 1**: Train on **Period 1** (1,000 records) $\to$ Validate on **Period 2** (1,000 records).
- **Fold 2**: Train on **Periods 1 & 2** (2,000 records) $\to$ Validate on **Period 3** (1,000 records).
- **Leakage Prevention**: Preprocessing transformers (`StandardScaler`, `OneHotEncoder`) and feature transformations were fitted strictly on each fold's training slice.
- **Selection Criterion**: Mean Absolute Error (MAE) averaged across the two validation folds.

### Temporal Validation Model Comparison Table (Periods 1–3):

| Candidate Architecture | Feature Set | Mean MAE ↓ | Mean RMSE ↓ | Mean $R^2$ ↑ | Std MAE | Validation Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge (All Eng Feats, alpha=10.0)** | Baseline + 4 Candidate Features | **0.2307** | 0.2852 | 0.7934 | ±0.0046 | #1 |
| **Ridge + Adjusted Tenure** | Baseline + experience_adjusted_tenure | **0.2308** | 0.2854 | 0.7932 | ±0.0046 | #2 |
| **Ridge Baseline** | Baseline (10 features) | **0.2309** | 0.2854 | 0.7932 | ±0.0046 | #3 |
| **Ridge + Ratio (Quality/Prod)** | Baseline + quality_to_productivity_ratio | **0.2309** | 0.2855 | 0.7931 | ±0.0047 | #4 |
| **Ridge + Task Interaction** | Baseline + completion_attendance_interaction | **0.2310** | 0.2855 | 0.7931 | ±0.0046 | #5 |
| **Ridge + Training Interaction** | Baseline + prev_perf_training_interaction | **0.2311** | 0.2858 | 0.7925 | ±0.0051 | #6 |
| **Ridge (All Eng Feats, alpha=1.0)** | Baseline + 4 Candidate Features | **0.2313** | 0.2860 | 0.7922 | ±0.0051 | #7 |
| **Ridge (All Eng Feats, alpha=0.1)** | Baseline + 4 Candidate Features | **0.2314** | 0.2863 | 0.7918 | ±0.0051 | #8 |
| **RandomForest (depth=6, leaf=5)** | Baseline + 4 Candidate Features | **0.2355** | 0.2926 | 0.7827 | ±0.0061 | #9 |
| **RandomForest (depth=10, leaf=2)** | Baseline + 4 Candidate Features | **0.2380** | 0.2942 | 0.7802 | ±0.0063 | #10 |
| **HistGradientBoosting (iter=50, leaf=15)** | Baseline + 4 Candidate Features | **0.2393** | 0.2967 | 0.7765 | ±0.0090 | #11 |
| **HistGradientBoosting (iter=100, leaf=31)** | Baseline + 4 Candidate Features | **0.2468** | 0.3063 | 0.7618 | ±0.0073 | #12 |
| **Previous-Performance-Only** | Single Predictor | **0.2507** | 0.3106 | 0.7550 | ±0.0133 | #13 |
| **DummyRegressor (Mean)** | Baseline (None) | **0.5041** | 0.6330 | -0.0177 | ±0.0118 | #14 |

---

## 3. Controlled Feature Engineering Analysis

Four candidate features were engineered from historical metrics available at the start of period $t$:
1. **Quality-to-Productivity Ratio (`quality_to_productivity_ratio`)**:
   - Formula: `quality_score_previous_period / max(productivity_score_previous_period, 1e-4)`
   - Outcome: Neutral validation change (MAE 0.2309 vs 0.2309 baseline).
2. **Task Delivery Interaction (`completion_attendance_interaction`)**:
   - Formula: `project_completion_rate_previous_period * attendance_rate_previous_period`
   - Outcome: Neutral validation change (MAE 0.2310).
3. **Remedial Training Interaction (`prev_perf_training_interaction`)**:
   - Formula: `previous_performance_score * training_hours_previous_period`
   - Motivation: EDA revealed confounding by indication (remedial assignment to low performers).
   - Outcome: Did not outperform linear additive terms (MAE 0.2311).
4. **Experience-Adjusted Tenure (`experience_adjusted_tenure`)**:
   - Formula: `tenure_months / max(12 * years_experience, 1.0)`, clipped to `[0.0, 1.0]`.
   - Denominator Policy: Enforces a 1.0-month floor for near-zero career experience to avoid singularity, with strict $[0, 1]$ clipping.
   - Outcome: Best individual engineered feature in validation (MAE 0.2308).

> [!NOTE]
> **Performance Momentum Exclusion**:
> Quarter-over-quarter score delta ($\Delta y_{t-1} = y_{t-1} - y_{t-2}$) requires two prior observation quarters. > In this dataset, Period 1 has only $t-1$ baseline indicators (no $t-2$ score exists). Attempting to calculate momentum > would drop or invalidate 1,000 records (100% of Period 1), making Fold 1 temporal validation impossible. > Hence, multi-period momentum is not uniformly available at prediction time across all observation quarters.

---

## 4. Exploratory Holdout Benchmark Evaluation (Period 4)

Every candidate model was refitted on all available training periods (Periods 1–3, $N=3,000$) and evaluated on Period 4 ($N=1,000$):

| Model Architecture | Benchmark MAE ↓ | Benchmark RMSE ↓ | Benchmark $R^2$ ↑ | Δ MAE vs Base | Prediction Range | Out of Bounds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DummyRegressor (Mean)** | **0.4745** | **0.5946** | **-0.0158** | +116.47% | [3.22, 3.22] | 0 |
| **Previous-Performance-Only** | **0.2299** | **0.2848** | **0.7669** | +4.87% | [1.65, 4.73] | 0 |
| **Ridge Baseline (alpha=1.0)** | **0.2192** | **0.2713** | **0.7885** | -0.02% | [1.89, 4.89] | 0 |
| **Ridge (All Eng Feats, alpha=10.0)** | **0.2201** | **0.2724** | **0.7869** | +0.42% | [1.85, 4.89] | 0 |
| **HistGradientBoosting (iter=100, leaf=31)** | **0.2270** | **0.2840** | **0.7682** | +3.54% | [1.78, 4.80] | 0 |
| **RandomForest (depth=10, leaf=2)** | **0.2225** | **0.2781** | **0.7778** | +1.51% | [1.82, 4.68] | 0 |

### Benchmark Performance Analysis:
- **Best Generalization**: The original **Full-Feature Ridge Baseline** achieved the lowest test error (MAE = **0.2192**, RMSE = **0.2713**).
- **Ridge with Engineered Features**: Achieved MAE = **0.2201** (+0.42% higher error), showing that interaction terms slightly overfitted training dynamics.
- **Random Forest**: Achieved MAE = **0.2225** (+1.51% higher error).
- **HistGradientBoosting**: Achieved MAE = **0.2270** (+3.54% higher error).
- **Boundary Integrity**: 100% of holdout predictions across all models fell within the valid $[1.00, 5.00]$ range without artificial clipping.

---

## 5. Separate Unseen-Employee Generalization Experiment

To prevent methodologically conflating temporal forecasting for known employees with predicting unseen employees, a distinct 5-fold `GroupKFold` evaluation was conducted on training data:

| Operational Scenario | Features Evaluated | Cross-Validated MAE ↓ | Cross-Validated $R^2$ ↑ | Prediction-Time Information Availability |
| :--- | :--- | :--- | :--- | :--- |
| **New Hire Profile (No Appraisal Score)** | `['department', 'job_role', 'years_experience', 'tenure_months']...` | **0.5273** | **-0.0327** | Operational reality: new employees have zero prior appraisal ratings. |
| **Experienced Hire (Operational Metrics, No Appraisal)** | `['department', 'job_role', 'years_experience', 'tenure_months', 'quality_score_previous_period', 'productivity_score_previous_period', 'project_completion_rate_previous_period', 'attendance_rate_previous_period', 'training_hours_previous_period']...` | **0.2973** | **0.6800** | Operational indicators known from probation period, but formal annual appraisal missing. |
| **Cohort Benchmarking (Full Baseline Features Across Unseen Groups)** | `['previous_performance_score', 'quality_score_previous_period', 'productivity_score_previous_period', 'project_completion_rate_previous_period', 'attendance_rate_previous_period', 'training_hours_previous_period', 'years_experience', 'tenure_months', 'department', 'job_role']...` | **0.2209** | **0.8244** | Contrasts group generalization against temporal cohort forecasting. |

### Key Takeaways & Scenario Distinction:
- **New Hire Profile (Zero Operational History)**: Only static profile data (`department`, `job_role`, `years_experience`, `tenure_months`) is available. Predicting appraisal score yields $R^2 = -0.0327$ and MAE = 0.5273. Tenure and job role alone contain zero predictive signal for appraisal score.
- **Experienced / Probationary Hire**: Operational throughput measurements (`quality`, `productivity`, `attendance`, `project_completion`, `training`) are available, but formal annual appraisal score is missing. This yields $R^2 = 0.6800$ and MAE = 0.2973, confirming that operational performance indicators recover substantial predictive signal.
- **Cohort Benchmarking (Idealized Prior Score Available)**: Assumes prior performance rating is known ($R^2 = 0.8244$, MAE = 0.2209). **Methodological Guard**: This cross-sectional grouped $R^2$ evaluates variance across unseen employee groups and must NOT be directly compared with the temporal holdout $R^2$ (0.7885), which measures future quarterly forecasting within the same cohort.

---

## 6. Reproduction Command

```powershell
python scripts/train_advanced_models.py
```
