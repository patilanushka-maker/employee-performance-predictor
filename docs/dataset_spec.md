# Dataset Specification & Data Dictionary

## 1. Dataset Overview
- **Data Asset**: `data/synthetic/employee_performance.csv`
- **Source**: Synthetically generated via [`src/employee_performance/data_generator.py`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/src/employee_performance/data_generator.py)
- **Record Count**: 4,000 observations
- **Employee Cohort**: 1,000 unique fictional employees observed across 4 sequential quarters
- **Feature Count**: 12 predictor / metadata attributes + 1 continuous target variable
- **Primary Prediction Target**: `future_performance_score` (continuous rating on $[1.00, 5.00]$ scale)
- **Problem Formulation**: Supervised Regression / Longitudinal Panel Forecasting (with optional thresholding into multi-class performance tiers)

---

## 2. Feature Schema & Data Dictionary

| Field Name | Storage Type | Domain Range / Categories | Missing % | Description | Modeling Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `employee_id` | `object` (string) | `EMP-0001` - `EMP-1000` | 0.0% | Unique fictional identifier per employee | Index / Cluster Key |
| `period` | `int64` | `[1, 2, 3, 4]` | 0.0% | Sequential evaluation quarter | Temporal Grouping Key |
| `department` | `object` (string) | 7 Allowed Categories | 0.0% | Organizational business division | Categorical Feature |
| `job_role` | `object` (string) | 19 Allowed Categories | 0.0% | Specialized professional title | Categorical Feature |
| `years_experience`| `float64` | `[0.92, 28.46]` | 0.0% | Cumulative professional career tenure | Continuous Predictor |
| `tenure_months` | `int64` | `[1, 232]` | 0.0% | Cumulative months with this employer | Discrete Predictor |
| `training_hours_previous_period` | `float64` | `[2.0, 30.0]` | 0.0% | Professional training hours completed in $t-1$ | Continuous Predictor |
| `project_completion_rate_previous_period` | `float64` | `[0.473, 1.000]` | 0.0% | Ratio of assigned deliverables completed in $t-1$ | Ratio Predictor |
| `quality_score_previous_period` | `float64` | `[48.0, 100.0]` | 0.0% | Work delivery precision and audit rating in $t-1$ | Continuous Predictor |
| `productivity_score_previous_period` | `float64` | `[40.0, 99.5]` | 0.0% | Work throughput efficiency index in $t-1$ | Continuous Predictor |
| `attendance_rate_previous_period` | `float64` | `[0.815, 1.000]` | 0.0% | Proportion of working days present in $t-1$ | Ratio Predictor |
| `previous_performance_score` | `float64` | `[1.00, 5.00]` | 0.0% | Performance appraisal score awarded at $t-1$ | Continuous Predictor |
| `future_performance_score` | `float64` | `[1.00, 5.00]` | 0.0% | Performance appraisal score realized during period $t$ | **Target Variable ($y$)** |

---

## 3. Data Integrity & Quality Audit Results

| Audit Check | Target Criteria | Actual Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Completeness** | 0 nulls across all cells | 0 nulls found across 52,000 data cells | **PASSED** |
| **Duplicate Rows** | 0 exact duplicate rows | 0 duplicate rows detected | **PASSED** |
| **Key Uniqueness** | `(employee_id, period)` unique | 4,000 unique keys / 4,000 records | **PASSED** |
| **Period Integrity** | Monotonic quarters $[1, 2, 3, 4]$ | Coverage strictly spans quarters 1 to 4 | **PASSED** |
| **Temporal Stability** | Dept invariant; tenure non-decreasing | 0 department shifts; 100% monotonic tenure | **PASSED** |
| **Boundary Conformance**| All scores within theoretical ranges | 100% records conform to domain bounds | **PASSED** |
| **Target Distribution** | Non-degenerate, healthy variance | Mean: 3.24, Median: 3.25, Std: 0.640 | **PASSED** |

---

## 4. Generator Assumptions & Distinctions from Empirical Data
- **Synthetic Origin**: Generated relationships use a latent individual ability model combined with autoregressive persistence, domain interaction weights, and Gaussian noise.
- **Controlled Invariants**: No age or salary columns are generated or stored.
- **Interpretive Boundary**: Reviewers should note that high correlations between training hours or project completion and performance scores are mathematical assumptions of the generator function and must not be interpreted as empirical HR truths.

---

## 5. Prediction Moment & Leakage Prevention Boundary
- **Intended Prediction Moment**: Strictly at the beginning of observation period $t$ (Quarter $t$), before period $t$'s operational outcomes or appraisal ratings are realized.
- **Available Predictors at $t$**:
  - Organizational profile: `department`, `job_role`.
  - Cumulative tenure metrics: `years_experience`, `tenure_months` (as of the start of quarter $t$).
  - Lagged performance indicators realized during preceding quarter $t-1$: `previous_performance_score` (which identically equals quarter $t-1$'s `future_performance_score` for periods $t \ge 2$), `quality_score_previous_period`, `productivity_score_previous_period`, `project_completion_rate_previous_period`, `attendance_rate_previous_period`, `training_hours_previous_period`.
- **Target Variable ($y$)**: `future_performance_score` evaluated and awarded at the conclusion of quarter $t$.
- **Strict Leakage Prevention**:
  - Predictors never observe intra-period or post-outcome information for quarter $t$.
  - Identifier keys (`employee_id`, `period`) are excluded from model training feature matrices.
  - Preprocessing transformers (scalers, encoders) are fitted strictly on historical training periods ($t \in \{1, 2, 3\}$).
- **Cohort Generalization Boundary**:
  - The temporal holdout (Period 4) tests out-of-time forecasting for the *same employee cohort* across time.
  - It does **not** evaluate generalization to entirely unseen employees, which requires a distinct group-aware evaluation design (e.g., `GroupKFold` grouped by `employee_id`).

