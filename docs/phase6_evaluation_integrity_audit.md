# Phase 6 Evaluation Integrity Audit Report

**Project**: Employee Performance Predictor using Data Analytics  
**Date**: 2026-10-17  
**Role**: Senior Implementation Engineer  
**Audit Purpose**: Methodological verification of Phase 6 evaluation protocol, holdout contamination disclosure, model-selection provenance audit, and EDA correlation reconciliation.

---

## 1. Executive Summary & Audit Decision

This audit was conducted to rigorously verify the methodological soundness of Phase 6: Feature Engineering & Advanced Model Exploration. 

### Key Findings:
1. **Dataset Invariance**: The source dataset (`data/synthetic/employee_performance.csv`) remains untouched and verified with SHA-256 `2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79`.
2. **Holdout Status Disclosure**: Period 4 was directly evaluated multiple times during interactive exploratory development across candidate hyperparameter values (`alpha` $\in [0.1, 1.0, 5.0, 10.0]$) and multiple model families (Random Forest, HistGradientBoosting, Ridge, Dummy). Therefore, **Period 4 cannot be represented as an untouched, pristine holdout**. It must be characterized as an **exploratory test benchmark**.
3. **Model Selection Provenance**: Model selection was strictly governed by the chronological expanding-window validation on **Periods 1–3** (Fold 1: train P1, val P2; Fold 2: train P1–2, val P3). Across these folds, `Ridge (All Eng Feats, alpha=10.0)` achieved the lowest mean validation MAE (0.2307). The saved artifact [`models/advanced_pipeline.joblib`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/models/advanced_pipeline.joblib) faithfully matches this validation-selected configuration.
4. **Resolution of EDA Correlation Discrepancies**: The discrepancy between previously cited correlations (`project_completion_rate: +0.5310` vs `+0.5369` / `+0.5610` and `attendance_rate: +0.4200` vs `+0.4468`) was traced to a subtle unit test side-effect. In `scripts/run_eda.py`, `run_eda()` had unconditionally written report copies to `docs/eda_report.md` when executed on the 45-row unit test fixture `small_eda_df`. Decoupling documentation writes from unit tests restored full consistency across `outputs/reports/eda_report.md` and `docs/eda_report.md`.
5. **Audit Decision**: **PASS (METHODOLOGICALLY AUDITED & DISCLOSED)**. The baseline remains the top-performing production model, Phase 6 advanced models are preserved as exploratory benchmarks, and holdout limitations are fully disclosed.

---

## 2. Dataset Immutability Verification

The SHA-256 hash of `data/synthetic/employee_performance.csv` was verified at the start and conclusion of the audit:
- **Expected Hash**: `2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79`
- **Measured Hash**: `2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79`
- **Result**: Byte-level integrity confirmed. No records were modified, regenerated, imputed, or re-indexed.

---

## 3. Period-4 Evaluation History & Holdout Integrity Audit

### A. Development History of Direct Period-4 Evaluations
A line-by-line inspection of the project trajectory, code, and execution history reveals that Period 4 was evaluated repeatedly:
1. **Interactive Sweep**: An exploratory command evaluated Ridge pipelines with all engineered features across `alpha` values `[0.1, 1.0, 5.0, 10.0]` directly on Period 4:
   - `alpha=0.1`: Holdout MAE = 0.2198
   - `alpha=1.0`: Holdout MAE = 0.2198
   - `alpha=5.0`: Holdout MAE = 0.2200
   - `alpha=10.0`: Holdout MAE = 0.2201
2. **Batch Workflow Evaluation**: In [`scripts/train_advanced_models.py`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/scripts/train_advanced_models.py), multiple candidate pipelines were systematically fitted on Periods 1–3 and evaluated on Period 4 (`DummyRegressor`, `Previous-Performance-Only`, `Ridge Baseline`, `Ridge with Eng Feats`, `HistGradientBoosting`, and `RandomForest`).

### B. Influence on Model Selection Decisions
- **Selection Source**: In the expanding-window temporal validation on Periods 1–3, the candidates ranked as follows:
  - 1st: `Ridge (All Eng Feats, alpha=10.0)` — Val MAE: **0.2307**
  - 2nd: `Ridge + Adjusted Tenure (alpha=1.0)` — Val MAE: **0.2308**
  - 3rd: `Ridge Baseline (alpha=1.0)` — Val MAE: **0.2309**
  - 4th: `Ridge (All Eng Feats, alpha=1.0)` — Val MAE: **0.2313**
  - 5th: `Ridge (All Eng Feats, alpha=0.1)` — Val MAE: **0.2314**
- **Decision Trace**: The model selected for artifact serialization was `Ridge (All Eng Feats, alpha=10.0)`, exactly matching the top rank from Periods 1–3 temporal validation.
- **Holdout Divergence**: On Period 4, `alpha=10.0` achieved MAE 0.2201, whereas `alpha=1.0` and baseline `alpha=1.0` achieved MAE 0.2198 and 0.2192 respectively. Had Period 4 been used to choose the final model, the engineer would have selected `alpha=1.0` (or the original baseline). The fact that `alpha=10.0` was saved demonstrates that model selection strictly adhered to Periods 1–3 validation metrics.
- **Reporting Inaccuracy Corrected**: Describing Period 4 as an "untouched final holdout" was factually inaccurate. Reports have been updated to explicitly label Period-4 metrics as **exploratory benchmark results**.

---

## 4. Chronological Expanding-Window Validation Audit

The temporal cross-validation protocol implemented in [`src/employee_performance/modeling.py`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/src/employee_performance/modeling.py) was audited against strict methodological standards:

### Verification Items:
1. **Fold Structure**:
   - **Fold 1**: Train on Period 1 ($N=1,000$) $\to$ Validate on Period 2 ($N=1,000$).
   - **Fold 2**: Train on Periods 1 & 2 ($N=2,000$) $\to$ Validate on Period 3 ($N=1,000$).
   - Verified that Period 4 is completely excluded from all validation training and evaluation slices.
2. **Validation Row Consistency**:
   - Every candidate model (Dummy, Previous-Performance-Only, Ridge variants, Random Forest, HistGradientBoosting) predicts against the **exact same validation rows** in Fold 1 and Fold 2. Verified via unit test `test_temporal_validation_consistent_validation_rows()`.
3. **Preprocessing Isolation**:
   - `ColumnTransformer` (`StandardScaler` and `OneHotEncoder`) is encapsulated within `Pipeline` instances and fitted strictly on `X_train` within each fold. Verified via `test_preprocessing_isolation_per_fold()`.
4. **Prediction-Time Feature Engineering**:
   - All candidate features are computed from indicators realized in period $t-1$. Identifiers (`employee_id`), temporal keys (`period`), and targets (`future_performance_score`) are strictly barred from feature vectors.
5. **Performance Momentum Decision**:
   - Audited the rationale for excluding quarter-over-quarter score delta ($\Delta y_{t-1} = y_{t-1} - y_{t-2}$). Because Period 1 has only $t-1$ baseline indicators, two historical periods are unavailable, which would render 100% of Period 1 missing. The exclusion is methodologically sound and prevents structural data loss.

---

## 5. Unseen-Employee Generalization Audit

The group-aware evaluation (`evaluate_unseen_employee_generalization`) was audited to verify that it does not conflate same-cohort forecasting with new-hire prediction:

### A. Group Isolation
- Uses `GroupKFold(n_splits=5)` grouped strictly by `employee_id` across Periods 1–3.
- Verified that **zero employee overlap** exists between training and validation splits in any fold (`test_unseen_employee_zero_id_overlap_across_all_folds()`).

### B. Prediction-Time Information Availability Scenarios

| Scenario | Features Supplied at Prediction Time | Operational Reality | Cross-Validated MAE ↓ | Cross-Validated $R^2$ ↑ |
| :--- | :--- | :--- | :---: | :---: |
| **1. Truly New Hire** | `department`, `job_role`, `years_experience`, `tenure_months` | Zero historical appraisal or operational performance records exist upon joining. | **0.5273** | **-0.0327** |
| **2. Experienced / Probationary Hire** | Profile + `quality`, `productivity`, `project_completion`, `attendance`, `training_hours` | Operational task measurements exist (e.g. from probationary period), but formal annual appraisal score is absent. | **0.2973** | **0.6800** |
| **3. Cohort Benchmarking** | Full 10 baseline features (including prior appraisal score) | Assumes an external or prior appraisal rating is available across employee groups. | **0.2209** | **0.8244** |

### C. Methodological Separation
- **Key Insight**: Demographic and organizational tenure attributes alone have zero predictive power ($R^2 < 0$) for individual performance ratings. Operational throughput metrics restore substantial predictive signal ($R^2 = 0.6800$).
- **Metric Incomparability**: The cross-sectional grouped $R^2$ (0.8244) measures variance across distinct employee clusters, whereas the temporal holdout $R^2$ (0.7885) measures future-period forecasting for the same cohort. They originate from different evaluation designs and must not be compared directly.

---

## 6. Reconciliation of EDA Correlation Discrepancies

### A. Investigation & Root Cause Analysis
During forensic review, the following discrepancies were investigated:
- `project_completion_rate_previous_period`: reported as `+0.5310` vs `+0.5369` / `+0.5610`.
- `attendance_rate_previous_period`: reported as `+0.4200` vs `+0.4468`.

**Root Cause Found**:
In [`scripts/run_eda.py`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/scripts/run_eda.py), `run_eda()` originally had an unconditional hardcoded line:
`docs_copy = PROJECT_ROOT / "docs" / "eda_report.md"`
When `pytest` executed `test_run_eda_script_in_temporary_directory(small_eda_df)`, a temporary 45-row synthetic dataset (`num_employees=15, num_periods=3`) was passed to `run_eda()`. While figure and primary report outputs were directed to a temporary folder, line 721 silently overwrote `docs/eda_report.md` with the 45-row fixture statistics!
On that 45-row sample:
- `attendance_rate_previous_period` Pearson $r = 0.4468$
- `project_completion_rate_previous_period` Pearson $r = 0.5610$
Meanwhile, [`outputs/reports/eda_report.md`](file:///d:/Anushka/IIT/DS/Employee%20Performance%20Predictor%20using%20Data%20Analytics/outputs/reports/eda_report.md) had been generated from the full 4,000-row dataset, where:
- `attendance_rate_previous_period` Pearson $r = 0.41995 \to \mathbf{+0.4200}$
- `project_completion_rate_previous_period` Pearson $r = 0.53102 \to \mathbf{+0.5310}$

### B. Repair & Resolution
1. Modified `scripts/run_eda.py` so `docs_copy_path` is an explicit optional parameter, passed only when `main()` runs on the full dataset, eliminating test-suite side effects.
2. Regenerated both reports on the unchanged dataset. Both files now agree on the exact Pearson values:
   - `project_completion_rate_previous_period`: **+0.5310**
   - `attendance_rate_previous_period`: **+0.4200**
   - `quality_score_previous_period`: **+0.7113**
   - `productivity_score_previous_period`: **+0.6294**
   - `previous_performance_score`: **+0.8662**
   - `training_hours_previous_period`: **-0.2791**

---

## 7. Artifact & Metadata Consistency Audit

The saved artifacts in `models/` were audited against source code and metadata:
- **Target Variable Definition & Verification**:
  - The true ground-truth prediction target across the synthetic dataset specification, training module, pipeline artifact, and metadata is `future_performance_score`.
  - Any colloquial shorthand reference to `performance_score` in prior discussion refers strictly to `future_performance_score` (the continuous appraisal rating on $[1.00, 5.00]$ scale evaluated and awarded at the end of period $t$).
  - No column renaming or data transformations were applied to the target variable.
- **Feature Count & Preprocessing Breakdown**:
  - **Base Features (10 total)**:
    - *Categorical (2)*: `department`, `job_role`
    - *Numeric (8)*: `previous_performance_score`, `quality_score_previous_period`, `productivity_score_previous_period`, `project_completion_rate_previous_period`, `attendance_rate_previous_period`, `training_hours_previous_period`, `years_experience`, `tenure_months`
    - *Note on Prior Summary*: The previous audit summary accidentally referred to "8 base features" while explicitly listing all 10 features; this confusion stemmed from referencing the 8 numeric features. The pipeline actually ingests and processes all 10 base features.
  - **Engineered Candidate Features (4 total)**:
    - `quality_to_productivity_ratio` ($Q_{t-1} / \max(P_{t-1}, 0.01)$)
    - `completion_attendance_interaction` ($C_{t-1} \times A_{t-1}$)
    - `prev_perf_training_interaction` ($Score_{t-1} \times T_{t-1}$)
    - `experience_adjusted_tenure` ($Tenure / \max(12 \times Exp, 1.0)$ clipped to $[0.0, 1.0]$)
  - **Total Pre-Encoding Features**: 14 features (12 numeric + 2 categorical).
  - **Preprocessing Transformations**:
    - `ColumnTransformer` applies `StandardScaler()` to all 12 numeric features (8 base + 4 engineered).
    - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to the 2 categorical features.
    - Preprocessing is fitted strictly on training periods 1–3 ($N=3,000$).
- **Estimator & Hyperparameters**:
  - Estimator: `sklearn.linear_model.Ridge` with `alpha = 10.0`, `random_state = 42`.
  - Serialized pipeline matches `Pipeline[FeatureEngineer, ColumnTransformer[StandardScaler, OneHotEncoder], Ridge(alpha=10.0)]`.
- **Validation MAE Difference vs. Baseline**:
  - Across expanding-window temporal validation folds (Periods 1–3), `Ridge (All Eng Feats, alpha=10.0)` achieved validation MAE = **0.2307** vs. baseline Ridge = **0.2309** (a difference of just **0.0002** MAE; or **0.0016** MAE against fold-level baseline without candidate features).
  - This miniscule difference represents **practical parity** rather than a meaningful or substantial predictive breakthrough.
- **Model-Selection Provenance & Limitation**:
  - The automated selection function `run_candidate_model_exploration()` ranks candidates strictly by mean validation MAE on Periods 1–3 without accessing Period 4.
  - However, we explicitly acknowledge that source code mechanics cannot prove that historical human implementation decisions were completely uninfluenced by Period 4, given that Period 4 was examined repeatedly during development.
  - We preserve the documented distinction: chronological validation on Periods 1–3 serves as the quantitative model-selection evidence, while Period 4 results serve strictly as an exploratory benchmark.
- **`models/advanced_pipeline.joblib`**:
  - Successfully reloaded and verified.
  - Generates 1,000 finite predictions in $[1.00, 5.00]$ on Period 4 (range: $[1.8526, 4.8874]$, zero out-of-bounds).
- **`models/advanced_pipeline_metadata.json`**:
  - Accurately records estimator (`Ridge`), alpha (`10.0`), target column (`future_performance_score`), target definition, training periods (`[1, 2, 3]`), base features (10), engineered features (4), total pre-encoding features (14), preprocessing details, and the exploratory benchmark disclosure for Period 4.

---

## 8. Summary: What Remains Valid vs. What Is Compromised

| Component | Status | Analysis & Remediation |
| :--- | :---: | :--- |
| **Synthetic Dataset** | **Fully Valid** | Unchanged, verified by SHA-256 (`2df006a2...`). |
| **EDA Engine & Statistics** | **Fully Valid** | All correlation discrepancies resolved; unit test side-effects eliminated. |
| **Chronological Validation (Periods 1–3)** | **Fully Valid** | Expanding-window folds strictly respect temporal order; model selection based on mean MAE is reproducible and uncompromised. |
| **Unseen-Employee Evaluation** | **Fully Valid** | GroupKFold strictly isolates employees; distinct operational scenarios documented. |
| **Period-4 Untouched Holdout Status** | **Compromised** | Inspected repeatedly during exploratory development. Re-classified as an **exploratory benchmark**. |
| **Baseline Model Selection** | **Fully Valid** | Full-Feature Ridge model remains the verified primary production pipeline (MAE: 0.2192). |

### Protocol for Future Independent Holdout Validation:
To conduct a truly independent out-of-time evaluation in future milestones:
1. Simulate an uninspected future observation period (e.g. **Period 5**).
2. Sequester Period 5 behind an automated cryptographic evaluation gate that prevents interactive script access until all feature and model architectures are locked.

---

## 9. Final Test Suite Verification

```powershell
python -m pytest -v
```
- **Passed**: **84 tests** in 9.12s.
- **Failures**: 0.
- **Warnings**: 0.
- **Dataset Hash (Post-Audit)**: `2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79` (Verified identical).
