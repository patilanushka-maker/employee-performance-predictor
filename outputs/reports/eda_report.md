# Exploratory Data Analysis (EDA) Report

**Project**: Employee Performance Predictor using Data Analytics  
**Execution Date**: 2026-10-16  
**Status**: Phase 4 Completed  
**Dataset Analyzed**: `data/synthetic/employee_performance.csv` (Seed 42)  

---

## 1. Objective & Scope

This exploratory data analysis establishes the empirical foundation for subsequent predictive modeling. The objective is to rigorously inspect data distributions, evaluate linear and rank correlations, investigate departmental and occupational variance, track quarterly panel trends, and uncover structural data-generating dynamics without performing premature feature selection or training predictive algorithms.

> [!IMPORTANT]
> **Synthetic Nature & Causal Boundary**:
> All records in this dataset are synthetically generated for a fictional cohort of 1,000 employees. > All relationships documented below are mathematical artifacts of the underlying generator and must > **not** be interpreted as empirical truths or causal facts regarding real-world human workforce behavior.

---

## 2. Dataset Overview & Data-Quality Audit Summary

- **Total Rows**: 4,000
- **Total Columns**: 13
- **Unique Fictional Employees**: 1,000
- **Records per Period**: Quarter 1: 1,000, Quarter 2: 1,000, Quarter 3: 1,000, Quarter 4: 1,000
- **Missing Values**: 0 across all cells (100% completeness)
- **Exact Duplicate Rows**: 0
- **Duplicate Composite Keys `(employee_id, period)`**: 0
- **Departments Represented**: 7 divisions
- **Job Roles Represented**: 19 discrete titles

### Feature Dictionary & Measurement Units

| Feature Name | Scale / Domain | Unit | Modeling Role |
| :--- | :--- | :--- | :--- |
| `employee_id` | Identifier | String | Cluster / Panel Key |
| `period` | 1 to 4 | Integer Quarter | Temporal Index |
| `department` | 7 Categories | Categorical | Business Unit |
| `job_role` | 19 Categories | Categorical | Functional Title |
| `years_experience` | 0.5 to 30.0 | Years | Continuous Predictor |
| `tenure_months` | 1 to 300 | Months | Discrete Predictor |
| `training_hours_previous_period` | 0 to 60 | Hours | Continuous Predictor |
| `project_completion_rate_previous_period` | 0.0 to 1.0 | Fraction (0-1) | Ratio Predictor |
| `quality_score_previous_period` | 0 to 100 | Audit Index (0-100) | Continuous Predictor |
| `productivity_score_previous_period` | 0 to 100 | Output Index (0-100) | Continuous Predictor |
| `attendance_rate_previous_period` | 0.0 to 1.0 | Fraction (0-1) | Ratio Predictor |
| `previous_performance_score` | 1.0 to 5.0 | Appraisal Rating | Continuous Predictor |
| `future_performance_score` | 1.0 to 5.0 | Appraisal Rating | Prediction Target (y) |

### Parametric & Non-Parametric Numerical Statistics

| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `years_experience` | 7.60 | 3.84 | 0.92 | 4.79 | 6.91 | 9.51 | 28.46 | +1.25 | +2.42 |
| `tenure_months` | 47.64 | 35.54 | 1.00 | 21.00 | 39.00 | 65.00 | 232.00 | +1.47 | +2.78 |
| `training_hours_previous_period` | 12.35 | 3.98 | 2.00 | 9.17 | 12.00 | 15.00 | 30.00 | +0.41 | +0.19 |
| `project_completion_rate_previous_period` | 0.81 | 0.08 | 0.47 | 0.76 | 0.81 | 0.86 | 1.00 | -0.20 | +0.29 |
| `quality_score_previous_period` | 76.21 | 6.63 | 48.00 | 72.00 | 76.20 | 80.30 | 100.00 | -0.05 | +0.63 |
| `productivity_score_previous_period` | 72.71 | 7.00 | 40.00 | 68.20 | 72.90 | 77.20 | 99.50 | -0.21 | +0.73 |
| `attendance_rate_previous_period` | 0.94 | 0.03 | 0.81 | 0.92 | 0.94 | 0.96 | 1.00 | -0.27 | +0.01 |
| `previous_performance_score` | 3.16 | 0.64 | 1.00 | 2.73 | 3.15 | 3.58 | 5.00 | +0.01 | -0.02 |
| `future_performance_score` | 3.24 | 0.64 | 1.00 | 2.81 | 3.25 | 3.66 | 5.00 | -0.07 | -0.04 |

---

## 3. Univariate Distribution Findings

- **Target Variable (`future_performance_score`)**:
  - Demonstrates a symmetric, unimodal bell-shaped distribution with mean **3.24** and median **3.25**.
  - Standard deviation is **0.64**, with minimal skewness (-0.07) and negative excess kurtosis (-0.04), indicating healthy dispersion across the [1.0, 5.0] appraisal scale.
- **Ratio Features (`attendance_rate`, `project_completion_rate`)**:
  - Attendance rate exhibits high concentration near 1.0 (mean 0.941, median 0.942), reflecting typical corporate attendance floors.
  - Project completion rate is normally distributed around 0.810 with bounds spanning [0.473, 1.000].
- **Index Features (`quality_score`, `productivity_score`)**:
  - Both quality (76.2 ± 6.6) and productivity (72.7 ± 7.0) show wide, well-scaled variation across their 0–100 domains.

**Rendered Figures**:
- `outputs/figures/01_target_distribution.png`
- `outputs/figures/02_numeric_features_distributions.png`

---

## 4. Correlation Analysis & Feature Associations

Both Pearson ($r$) and Spearman rank ($\rho$) correlations were computed to detect linear and monotonic associations:

| Explanatory Predictor | Pearson Correlation ($r$) | Spearman Rank ($\rho$) | Interpretation |
| :--- | :--- | :--- | :--- |
| `years_experience` | **+0.0115** | **+0.0149** | Near-zero linear association |
| `tenure_months` | **-0.0234** | **-0.0245** | Near-zero linear association |
| `training_hours_previous_period` | **-0.2791** | **-0.2758** | Negative marginal correlation (Remedial assignment) |
| `project_completion_rate_previous_period` | **+0.5310** | **+0.5028** | Moderate-to-strong positive driver |
| `quality_score_previous_period` | **+0.7113** | **+0.6956** | Strong positive autoregressive persistence |
| `productivity_score_previous_period` | **+0.6294** | **+0.6011** | Moderate-to-strong positive driver |
| `attendance_rate_previous_period` | **+0.4200** | **+0.4013** | Moderate-to-strong positive driver |
| `previous_performance_score` | **+0.8662** | **+0.8566** | Strong positive autoregressive persistence |

> [!NOTE]
> **Absence of Multicollinearity Screening (VIF)**:
> Consistent with project scope restrictions, Variance Inflation Factor (VIF) screening is intentionally deferred > to Phase 5 (Feature Engineering) after candidate features, interaction terms, and categorical encodings are formalized.

**Rendered Figures**:
- `outputs/figures/03_correlation_heatmap.png`
- `outputs/figures/04_key_predictors_vs_target.png`

---

## 5. Investigation of the Training Hours Paradox

A prominent empirical finding in this dataset is the **negative marginal correlation** ($r = -0.2791$) between `training_hours_previous_period` and `future_performance_score`.

### Root Cause: Confounding by Indication (Selection Bias)
Review of the generator mechanics confirms that training hours are assigned conditionally based on previous performance. Struggling employees receive higher volumes of mandated remedial training:

| Prior Performance Tier | Observation Count | Average Training Prescribed | Mean Future Performance | Within-Tier Correlation ($r$) |
| :--- | :--- | :--- | :--- | :--- |
| **Low Tier (<=2.5)** | 617 | **14.9 hrs** | 2.37 | **-0.0781** |
| **Mid Tier (2.5-3.5)** | 2,238 | **12.5 hrs** | 3.14 | **-0.0432** |
| **High Tier (>3.5)** | 1,145 | **10.6 hrs** | 3.89 | **+0.0795** |

- **Key Insight**: Lower-tier performers receive an average of **14.9 hours** of training, whereas high performers average only **10.6 hours**.
- Within each stratum, the local correlation is near zero or weakly positive. In the synthetic generator formula, the structural linear weight parameter on training hours is +0.008 * (training_hours - 12.0). However, this mathematical parameter is an artifact of the simulation and must not be interpreted as an empirical causal effect. In real-world workforce analytics, observing a negative raw association would require formal observational causal inference methods (e.g. instrumental variables or panel fixed effects) to separate selection into training from training effectiveness.

**Rendered Figure**:
- `outputs/figures/08_training_hours_remedial_analysis.png`

---

## 6. Departmental & Job-Role Analysis

Performance distributions across organizational units:

| Department | Observation Count | Mean Target ($y$) | Std Dev | Median | IQR | Range [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Finance** | 356 | 3.302 | 0.712 | 3.325 | 0.970 | [1.48, 5.00] |
| **Marketing** | 544 | 3.274 | 0.640 | 3.280 | 0.862 | [1.49, 4.95] |
| **Product** | 312 | 3.264 | 0.647 | 3.265 | 0.834 | [1.52, 5.00] |
| **Engineering** | 1,140 | 3.263 | 0.626 | 3.310 | 0.820 | [1.00, 5.00] |
| **Customer Support** | 520 | 3.190 | 0.638 | 3.180 | 0.842 | [1.57, 5.00] |
| **Sales** | 796 | 3.190 | 0.628 | 3.160 | 0.860 | [1.58, 5.00] |
| **Human Resources** | 332 | 3.166 | 0.615 | 3.185 | 0.752 | [1.15, 4.80] |

- **Departmental Descriptive Summary**: Mean performance ratings across all 7 departments fall within a narrow descriptive band (3.166 to 3.302). These values describe the synthetic dataset and do not imply formal statistical equivalence across departments. Naive row-level ANOVA is deliberately omitted because it would violate independence assumptions.
- **Job Role Granularity**: Across all 19 job roles, median scores remain stable between 3.10 and 3.40.

**Rendered Figures**:
- `outputs/figures/05_performance_by_department.png`
- `outputs/figures/06_performance_by_job_role.png`

---

## 7. Quarterly Trends & Repeated-Measures Awareness

### A. Macroeconomic Period-Level Summary

| Quarter | Record Count | Mean Score | Std Dev | Median | [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Quarter 1** | 1,000 | 3.148 | 0.699 | 3.150 | [1.00, 5.00] |
| **Quarter 2** | 1,000 | 3.232 | 0.643 | 3.240 | [1.08, 5.00] |
| **Quarter 3** | 1,000 | 3.273 | 0.613 | 3.285 | [1.31, 4.86] |
| **Quarter 4** | 1,000 | 3.292 | 0.590 | 3.320 | [1.53, 5.00] |

### B. Within-Employee Transition Dynamics (n=3,000 Transitions)

- **Total Within-Employee Transitions**: 3,000
- **Mean Quarter-over-Quarter Change ($\Delta y$)**: +0.0480
- **Median Quarter-over-Quarter Change**: +0.0350
- **Standard Deviation of Change ($\sigma_\Delta$)**: 0.3045
- **Employees Showing Positive Change ($\Delta > +0.05$)*:** 47.6%
- **Employees Showing Negative Change ($\Delta < -0.05$)*:** 38.4%
- **Employees Stable ($-0.05 \le \Delta \le +0.05$)*:** 14.0%
  *(Note: The $\pm 0.05$ interval is an exploratory reporting convention to segment score fluctuations, not a validated business-significance threshold.)*

### C. Methodological Caution on Repeated Measures

> [!WARNING]
> **Clustered Panel Data Notice**:
> The 4,000 observations in this dataset represent **1,000 unique employees observed across 4 consecutive quarters**.
> They do **not** represent 4,000 independent and identically distributed (i.i.d.) observations.
> 
> Conventional inferential tests (such as row-level ordinary least squares standard errors or one-way ANOVA) that treat > all 4,000 rows as independent entities will substantially underestimate standard errors and artificially inflate test statistics.
> 
> In Phase 5 and Phase 6, cross-validation must utilize **GroupKFold (grouped by `employee_id`)** or **Temporal Out-of-Time Splitting** > (e.g. training on Quarters 1–3 and evaluating on Quarter 4) to ensure generalizability and eliminate data leakage.

**Rendered Figure**:
- `outputs/figures/07_quarterly_performance_trends.png`

---

## 8. Summary of Generated Figures

| Figure File | Description |
| :--- | :--- |
| `outputs/figures/01_target_distribution.png` | Boxplot and histogram of `future_performance_score` with parametric normal overlay |
| `outputs/figures/02_numeric_features_distributions.png` | 2x4 grid displaying parametric histograms for all numeric explanatory predictors |
| `outputs/figures/03_correlation_heatmap.png` | Annotated correlation matrix across all numeric features and target |
| `outputs/figures/04_key_predictors_vs_target.png` | Scatterplots with linear trend fits for key drivers vs target |
| `outputs/figures/05_performance_by_department.png` | Boxplot of performance scores across 7 corporate departments |
| `outputs/figures/06_performance_by_job_role.png` | Horizontal boxplots of ratings ranked across 19 job titles |
| `outputs/figures/07_quarterly_performance_trends.png` | Macro period trajectory alongside within-employee quarter-over-quarter delta distribution |
| `outputs/figures/08_training_hours_remedial_analysis.png` | Remedial selection breakdown explaining negative marginal training hours correlation |

---

## 9. Reproduction Instructions

To reproduce this analysis and regenerate all tables and figures from scratch:
```powershell
# Execute reproducible EDA script
python scripts/run_eda.py

# Or specify custom paths
python scripts/run_eda.py --data data/synthetic/employee_performance.csv --figures-dir outputs/figures --report-path outputs/reports/eda_report.md
```
