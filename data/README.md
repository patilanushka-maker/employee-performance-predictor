# Data Directory & Synthetic Dataset Documentation

This directory organizes all data assets utilized and generated across the lifecycle of the **Employee Performance Predictor**.

---

## 📁 Directory Structure & Conventions

```
data/
├── raw/         # Immutable source data (reserved for external data imports)
├── processed/   # Cleaned, transformed, feature-engineered datasets
└── synthetic/   # Generated synthetic employee cohorts with controlled distributions
    └── employee_performance.csv  # Canonical longitudinal synthetic dataset
```

---

## 🧪 Synthetic Dataset Provenance

- **File Location**: `data/synthetic/employee_performance.csv`
- **Generator Module**: [`src/employee_performance/data_generator.py`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/src/employee_performance/data_generator.py)
- **Workflow Command**: `python scripts/generate_data.py`
- **Validation Command**: `python scripts/validate_data.py --data data/synthetic/employee_performance.csv`
- **Default Random Seed**: `42` (ensuring 100% byte-level reproducibility via NumPy `default_rng`)
- **Dataset Dimensions**: 4,000 rows $\times$ 13 columns (1,000 unique fictional employees observed over 4 sequential quarters)

> [!IMPORTANT]
> **Synthetic Artifact Notice**: All records and inter-variable relationships are generated via stochastic simulations with parameterized domain assumptions. They represent simulated organizational behavior for data science methodology and machine learning demonstration, **not** empirical findings or actual workforce data from a real enterprise.

---

## ⏱️ Temporal Modeling Architecture

To prevent look-ahead bias and eliminate data leakage, each row in this dataset strictly observes temporal order:

```
[ Period t-1 Historical Performance ] ──────────> [ Beginning of Period t ] ──────────> [ Period t Realized Evaluation ]
 • training_hours_previous_period                     • years_experience                 • future_performance_score (TARGET)
 • project_completion_rate_previous_period            • tenure_months
 • quality_score_previous_period                      • department, job_role
 • productivity_score_previous_period
 • attendance_rate_previous_period
 • previous_performance_score
```

- **Observation Timestamp**: The start of sequential observation `period` $t \in \{1, 2, 3, 4\}$.
- **Historical Features**: All predictor variables named `*_previous_period` and `previous_performance_score` represent metrics collected during period $t-1$ (prior to the prediction epoch).
- **Target Variable**: `future_performance_score` represents the performance rating evaluated at the end of period $t$.
- **Longitudinal Chaining**: For any employee observed across periods, period $t+1$'s `previous_performance_score` strictly equals period $t$'s realized `future_performance_score`. No future-period metrics are ever used to construct historical features.

---

## 📊 Data Dictionary & Schema Specification

| Field Name | Data Type | Range / Domain | Meaning | Modeling Role |
| :--- | :--- | :--- | :--- | :--- |
| `employee_id` | String | `EMP-0001` to `EMP-1000` | Fictional persistent employee identifier | Identifier (Key) |
| `period` | Integer | `[1, 2, 3, 4]` | Ordered sequential observation quarter | Temporal Index (Key) |
| `department` | Categorical | 7 Business Units | Operational unit (Engineering, Sales, etc.) | Categorical Feature |
| `job_role` | Categorical | 19 Specific Roles | Functional role mapped to department | Categorical Feature |
| `years_experience` | Float | `[0.92, 28.46]` | Total professional career experience | Continuous Predictor |
| `tenure_months` | Integer | `[1, 232]` | Months elapsed at the organization | Discrete Predictor |
| `training_hours_previous_period` | Float | `[2.0, 30.0]` | Formal professional development hours in $t-1$ | Continuous Predictor |
| `project_completion_rate_previous_period` | Float | `[0.473, 1.000]` | Percentage of assigned projects delivered in $t-1$ | Ratio Predictor |
| `quality_score_previous_period` | Float | `[48.0, 100.0]` | Work delivery audit and quality index in $t-1$ | Continuous Predictor |
| `productivity_score_previous_period` | Float | `[40.0, 99.5]` | Volume and velocity efficiency score in $t-1$ | Continuous Predictor |
| `attendance_rate_previous_period` | Float | `[0.815, 1.000]` | Working day presence / scheduled hours ratio in $t-1$ | Ratio Predictor |
| `previous_performance_score` | Float | `[1.00, 5.00]` | Official performance score awarded at $t-1$ | Continuous Predictor |
| `future_performance_score` | Float | `[1.00, 5.00]` | Performance score realized during period $t$ | **Prediction Target ($y$)** |

---

## 📈 Verified Dataset Descriptive Statistics

Computed directly from `data/synthetic/employee_performance.csv` (Seed 42):

| Metric | `years_experience` | `tenure_months` | `training_hours` | `project_completion` | `quality_score` | `productivity_score` | `attendance_rate` | `previous_perf` | `future_perf` (Target) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Count** | 4,000 | 4,000 | 4,000 | 4,000 | 4,000 | 4,000 | 4,000 | 4,000 | 4,000 |
| **Mean** | 7.60 | 47.64 | 12.35 | 0.810 | 76.21 | 72.71 | 0.941 | 3.16 | 3.24 |
| **Std** | 3.84 | 35.54 | 3.98 | 0.077 | 6.63 | 7.00 | 0.030 | 0.65 | 0.64 |
| **Min** | 0.92 | 1.00 | 2.00 | 0.473 | 48.00 | 40.00 | 0.815 | 1.00 | 1.00 |
| **25%** | 4.79 | 21.00 | 9.18 | 0.760 | 72.00 | 68.20 | 0.922 | 2.73 | 2.81 |
| **Median** | 6.91 | 39.00 | 12.00 | 0.812 | 76.20 | 72.90 | 0.942 | 3.15 | 3.25 |
| **75%** | 9.51 | 65.00 | 15.00 | 0.862 | 80.30 | 77.20 | 0.962 | 3.58 | 3.66 |
| **Max** | 28.46 | 232.00 | 30.00 | 1.000 | 100.00 | 99.50 | 1.000 | 5.00 | 5.00 |

### Department Breakdown
- **Engineering**: 1,140 rows (28.5%)
- **Sales**: 796 rows (19.9%)
- **Marketing**: 544 rows (13.6%)
- **Customer Support**: 520 rows (13.0%)
- **Finance**: 356 rows (8.9%)
- **Human Resources**: 332 rows (8.3%)
- **Product**: 312 rows (7.8%)

---

## 🛡️ Validation Rules Enforced

The validation suite in [`src/employee_performance/validation.py`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/src/employee_performance/validation.py) strictly enforces:
1. **Schema Integrity**: Presence of all 13 canonical columns; rejects extra or missing fields.
2. **Completeness**: 0 missing values or NaNs allowed across all attributes.
3. **Uniqueness**: 0 duplicate rows; exact uniqueness on composite key `(employee_id, period)`.
4. **Domain Boundaries**: Checks numerical bounds (e.g., $1.0 \le \text{score} \le 5.0$, $0.0 \le \text{rate} \le 1.0$).
5. **Categorical Conformity**: Strict membership within recognized departments and departmental role registries.
6. **Temporal Consistency**: Employee department cannot mutate; tenure cannot decrease across periods; target has healthy variance ($\sigma > 0$).
