# Employee Performance Predictor using Data Analytics

An enterprise data science and machine learning system designed to forecast workforce appraisal scores on longitudinal panel data. The project incorporates chronological expanding-window cross-validation, confounding-by-indication analysis, serialized scikit-learn pipelines, and an interactive recruiter-ready Streamlit dashboard.

---

## Executive Summary

Workforce performance evaluations guide talent retention, resource planning, and internal mobility. Traditional appraisal reviews often rely on retroactive opinions or uncalibrated annual impressions. Machine learning applied to employee data offers operational visibility, but naive models frequently suffer from look-ahead leakage, selection bias, and confounding by indication.

This project delivers an audit-grade predictive modeling system that forecasts an employee's next quarterly performance appraisal rating (scale of 1.00 to 5.00) using operational metrics recorded strictly prior to the appraisal period.

### Core Business Objectives:
- Provide objective, data-informed forecasts of future performance ratings to assist workforce planning.
- Isolate the operational behaviors (task quality, throughput, project delivery) that correlate with appraisal ratings.
- Identify confounding administrative factors, such as mandatory remedial training assignments for struggling staff, preventing misleading policy decisions.
- Deliver an interactive, role-based decision support tool for managers and HR analysts without automated decision-making.

---

## System Overview and Prediction Mandate

- Prediction Target: future_performance_score on a bounded continuous rating scale from 1.00 to 5.00 (population mean: 3.24, standard deviation: 0.64).
- Longitudinal Cohort: 1,000 synthetic employee profiles tracked across 4 quarterly periods (4,000 total records).
- Predictive Horizon: Operational metrics realized in period t - 1 are used to predict appraisal performance realized at period t.
- Technology Stack: Python 3.10, scikit-learn 1.3.2, pandas 2.2, NumPy 1.26, Matplotlib 3.7, pytest 9.1, and Streamlit 1.65.
- Data Provenance Note: All employee records, organizational roles, operational outputs, and appraisal scores were generated via a parametric longitudinal simulation. No personal or proprietary records are used.

---

## Machine Learning Methodology

### 1. Leakage-Free Chronological Validation
Standard k-fold cross-validation randomly shuffles records across time periods, allowing future data to leak into past predictions. To prevent temporal look-ahead leakage, model training and hyperparameter selection strictly utilized expanding-window temporal folds:
- Fold 1: Train on Period 1 (1,000 records), validate on Period 2 (1,000 records).
- Fold 2: Train on Periods 1 and 2 (2,000 records), validate on Period 3 (1,000 records).
- Preprocessing transformers (StandardScaler, OneHotEncoder with handle_unknown='ignore') and feature transformations are fitted strictly within scikit-learn pipelines on training folds.

### 2. Candidate Model Benchmark Results

Candidate models were benchmarked on chronological validation (Periods 1 to 3) and compared against Period 4 exploratory benchmarks:

| Model Architecture | Feature Set | Chronological Val MAE | Chronological Val RMSE | Chronological Val R2 | Period 4 Benchmark MAE | Period 4 Benchmark R2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| DummyRegressor (Mean Baseline) | None (Predicts historical mean) | 0.5041 | 0.6330 | -0.0177 | 0.4745 | -0.0158 |
| Previous-Performance-Only | Single predictor (previous_performance_score) | 0.2507 | 0.3106 | 0.7550 | 0.2299 | 0.7669 |
| Ridge Regression Baseline (alpha = 1.0) | 10 Base Features | 0.2309 | 0.2854 | 0.7932 | 0.2192 | 0.7885 |
| Ridge + Feature Engineering (alpha = 10.0) [Selected] | 10 Base + 4 Interaction Features | 0.2307 | 0.2852 | 0.7934 | 0.2201 | 0.7869 |
| Random Forest Regressor (depth = 10, leaf = 2) | 10 Base + 4 Interaction Features | 0.2380 | 0.2942 | 0.7802 | 0.2225 | 0.7778 |
| HistGradientBoosting Regressor (iter = 50) | 10 Base + 4 Interaction Features | 0.2393 | 0.2967 | 0.7765 | 0.2270 | 0.7682 |

### 3. Key Analytical Insights
1. Practical Performance Parity: In temporal validation, Ridge with engineered features achieved a validation MAE of 0.2307 compared to 0.2309 for baseline Ridge. The 0.0002 MAE difference represents practical parity; the 10 base features already linearly explain the primary variance.
2. Status of Period 4: Period 4 was examined during exploratory iterations (including parameter sweeps). It is designated as an exploratory benchmark rather than a blind holdout. Model selection was governed strictly by Periods 1 to 3 validation.
3. Tree-Based Degradation: Decision tree ensembles exhibited higher error than regularized linear models due to step-function approximations on continuous linear-additive relationships.
4. Generalization Boundary (GroupKFold): When evaluated across new, unseen individuals without prior appraisal records, demographic and tenure attributes alone yield an R2 of -0.0327 (zero predictive power). Operational task metrics restore R2 to 0.6800, proving that operational delivery drives ratings, not static biographical tenure.

---

## Exploratory Data Analysis and Behavioral Dynamics

Pearson correlations evaluated on the complete longitudinal panel:

| Feature | Pearson r with Target | Domain Interpretation |
| :--- | :---: | :--- |
| previous_performance_score | +0.8662 | Autoregressive Anchor: Prior appraisal score explains approximately 76% of future variance. |
| quality_score_previous_period | +0.7113 | Task Fidelity: Strongest operational signal; defect-free work directly correlates with appraisal score. |
| productivity_score_previous_period | +0.6294 | Throughput Delivery: Consistent volume output correlates strongly with quarterly rating. |
| project_completion_rate_previous_period | +0.5310 | Milestone Adherence: Punctual delivery of assigned quarterly milestones. |
| attendance_rate_previous_period | +0.4200 | Engagement Signal: Reliable attendance supports consistent team contribution. |
| training_hours_previous_period | -0.2791 | Confounding by Indication: Administrative artifact where lower-rated employees receive remedial courses. |
| years_experience / tenure_months | ~0.0000 | Tenure Neutrality: Experience alone has negligible linear relationship with quarterly appraisal. |

### Confounding by Indication Explained
In bivariate correlation, training hours shows a negative correlation (-0.2791) with future appraisal rating. This occurs because underperforming employees are assigned mandatory remedial training. When controlling for prior appraisal rating in multivariate regression, the partial coefficient of training hours is positive (+0.0031). Training itself is beneficial; the bivariate negative sign is a classic confounding-by-indication phenomenon.

---

## Streamlit Dashboard and Visual Interface

An interactive Streamlit application provides visual access to model artifacts, analytics, and scenario simulations.

### Application Modules:
1. Overview and Scope: Executive summary, population demographics, dataset schema, and operational boundaries.
2. Exploratory Data Analysis: Interactive distribution plots, department benchmarks, correlation structures, and remedial training analysis.
3. Model Benchmarks and Validation: Expanding-window validation comparisons, error distributions, and holdout disclosures.
4. Interactive Performance Prediction: Real-time form evaluating the 10 operational features against the production Ridge pipeline.
5. What-If Scenario Analysis: Comparative simulation demonstrating how operational changes impact projected scores, with explicit non-causality notices.

### Dashboard Screenshots:

The images below illustrate key screens from the web application. Screenshots placed in the images/ directory are automatically referenced here:

#### 1. Overview and Project Scope
![Overview and Project Scope](images/01_overview.png)

#### 2. Exploratory Data Analysis and Correlation Structure
![Exploratory Data Analysis](images/02_eda_correlations.png)

#### 3. Model Benchmark Comparisons
![Model Benchmarks](images/03_model_benchmarks.png)

#### 4. Real-Time Interactive Prediction Interface
![Interactive Prediction](images/04_prediction_interface.png)

#### 5. What-If Scenario Simulation
![What-If Scenario Simulation](images/05_whatif_analysis.png)

Note: If screenshot images are not yet added to the repository images/ folder, the links above will render the designated captures once image files are placed in images/.

---

## Repository Structure

```
Employee Performance Predictor using Data Analytics/
├── .env.example                          # Environment variable configuration template
├── .gitignore                            # Excluded cache, log, and local runtime files
├── LICENSE                               # MIT Open Source License
├── pyproject.toml                        # Project packaging and metadata specification
├── README.md                             # Comprehensive project documentation
├── requirements.txt                      # Production and development dependencies
├── main.py                               # CLI entrypoint for running pipeline workflows
│
├── config/                               # Centralized system configurations
│   ├── __init__.py
│   └── config.py                         # Paths, schemas, hyperparameters, and seeds
│
├── dashboard/                            # Interactive Streamlit application
│   ├── app.py                            # Main dashboard application entrypoint
│   ├── utils.py                          # Metric cards, formatting, and UI helpers
│   └── pages/                            # Multi-page modular dashboard views
│
├── data/                                 # Data storage directories
│   ├── raw/                              # Immutable raw datasets (.gitkeep)
│   ├── processed/                        # Preprocessed and engineered data (.gitkeep)
│   └── synthetic/                        # Longitudinal synthetic panel (4,000 records)
│       └── employee_performance.csv
│
├── docs/                                 # Detailed architectural and interview guides
│   ├── architecture.md                   # System component architecture
│   ├── data_flow.md                      # Pipeline execution and data flow
│   ├── dataset_spec.md                   # Feature definitions and distributions
│   ├── interview_preparation.md          # Technical questions and explanations
│   ├── model_methodology.md              # Mathematical formulas and pipeline design
│   └── phase6_evaluation_integrity_audit.md # Evaluation protocol disclosure
│
├── images/                               # User interface screenshots and captures
│   ├── .gitkeep                          # Preserves images directory in version control
│   ├── 01_overview.png                   # Overview screen capture (placeholder link)
│   ├── 02_eda_correlations.png           # EDA correlation screen capture (placeholder link)
│   ├── 03_model_benchmarks.png           # Benchmarks comparison capture (placeholder link)
│   ├── 04_prediction_interface.png       # Prediction form capture (placeholder link)
│   └── 05_whatif_analysis.png            # Scenario simulator capture (placeholder link)
│
├── models/                               # Serialized machine learning artifacts
│   ├── advanced_pipeline.joblib          # Serialized Ridge pipeline with interactions
│   ├── advanced_pipeline_metadata.json   # Feature schema and validation metrics
│   ├── baseline_pipeline.joblib          # Serialized primary Ridge baseline pipeline
│   └── baseline_pipeline_metadata.json   # Baseline schema and evaluation metrics
│
├── notebooks/                            # Exploratory analysis notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_advanced_models.ipynb
│
├── outputs/                              # Exported artifacts and publication assets
│   ├── figures/                          # 8 publication-grade visualization figures
│   ├── metrics/                          # Benchmark metric summary CSVs and JSONs
│   └── reports/                          # Audit reports and data quality logs
│
├── scripts/                              # Standalone automation scripts
│   ├── generate_synthetic_data.py        # Data generator script
│   ├── train_baselines.py                # Baseline training pipeline
│   └── train_advanced_models.py          # Advanced training and serialization
│
├── sql/                                  # Database schema and analytical queries
│   ├── schema.sql                        # Relational schema definitions
│   └── queries.sql                       # Analytical SQL aggregation queries
│
├── src/                                  # Core modular library source code
│   ├── data/                             # Ingestion, validation, and splitting modules
│   ├── features/                         # Transformer and feature engineering classes
│   ├── models/                           # Model wrappers, pipelines, and evaluation
│   └── utils/                            # Logging, serialization, and helper utilities
│
└── tests/                                # Automated pytest test suite
    ├── conftest.py                       # Shared test fixtures and synthetic data
    ├── test_data.py                      # Data validation tests
    ├── test_features.py                  # Transformer correctness tests
    ├── test_models.py                    # Pipeline fit, predict, and benchmark tests
    └── test_dashboard.py                 # Dashboard utility and inference tests
```

---

## Installation and Setup

### 1. Prerequisites
- Python 3.10 or higher
- PowerShell, Bash, or Windows Command Prompt
- Git command line tools

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/patilanushka-maker/employee-performance-predictor.git

# Navigate into the project directory
cd "employee-performance-predictor"

# Create a Python virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy the sample environment configuration file:
```bash
# Windows (PowerShell):
Copy-Item .env.example .env
# Linux / macOS:
cp .env.example .env
```
The `.env` file is excluded in `.gitignore` to ensure credentials remain secure and private.

---

## Running the Application and Pipeline

### 1. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Access the application by opening `http://localhost:8501` in your web browser.

### 2. Execute Model Training Pipelines
```bash
# Train baseline models
python scripts/train_baselines.py

# Train advanced models with interaction features
python scripts/train_advanced_models.py
```

### 3. Run Automated Unit and Integration Tests
```bash
python -m pytest -v
```
All 110 unit and integration tests validate dataset integrity, leakage-free cross-validation, feature transformations, model serialization, and dashboard utility functions.

---

## Governance, Ethical Considerations, and Boundaries

1. Decision-Support Only: Predictions generated by this system are designed strictly as decision-support indicators. They must not be used as an automated mechanism for hiring, termination, compensation, or disciplinary actions.
2. Observational Associations: Model estimates represent statistical correlations in synthetic longitudinal data. They do not constitute causal treatment effects.
3. Privacy by Design: The dataset contains no personal identifying information (PII), names, or demographic proxies, preventing surveillance or disparate impact.
4. Continuous Monitoring: Quarterly appraisal distributions should be tracked over time to detect distribution drift and calibration shifts before updating production models.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
