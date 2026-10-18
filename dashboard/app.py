"""Streamlit Web Dashboard for Employee Performance Predictor using Data Analytics.

Provides an interactive, recruiter-ready interface covering:
- Executive Overview & Scope Boundaries
- Exploratory Data Analysis & Empirical Insights
- Expanding-Window Validation & Exploratory Holdout Benchmarks
- Interactive Real-Time Prediction using Validated Saved Pipeline
- What-If Scenario Exploration with Causality Disclaimers
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard.utils import (
    FIGURES_DIR,
    REQUIRED_BASE_FEATURES,
    STANDARD_DEPARTMENTS,
    STANDARD_JOB_ROLES,
    load_advanced_model,
    load_baseline_model,
    load_dataset,
    load_evaluation_metrics,
    predict_performance,
    run_what_if_analysis,
    validate_prediction_input,
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor | HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling System (Vanilla CSS injection for premium aesthetics)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    .metric-card {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    .metric-val {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }

    .banner-synthetic {
        background-color: #f8fafc;
        border-left: 4px solid #6366f1;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }

    .banner-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        border-top: 1px solid #fef3c7;
        border-right: 1px solid #fef3c7;
        border-bottom: 1px solid #fef3c7;
    }

    .score-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .badge-excellent { background-color: #dcfce7; color: #15803d; }
    .badge-good { background-color: #e0e7ff; color: #4338ca; }
    .badge-fair { background-color: #fef9c3; color: #a16207; }
    .badge-low { background-color: #fee2e2; color: #b91c1c; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> str:
    """Render the application sidebar with navigation and technical provenance."""
    with st.sidebar:
        st.markdown("### 🏢 HR Analytics Suite")
        st.caption("Employee Performance Predictor")
        st.markdown("---")

        nav_choice = st.radio(
            "Navigate Application",
            [
                "📌 Overview & Scope",
                "📈 Exploratory Data Analysis",
                "🔬 Model Benchmarks & Validation",
                "🎯 Interactive Prediction",
                "⚡ What-If Scenario Analysis",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("#### 🔒 Provenance & Governance")
        st.markdown(
            r"""
            - **Dataset**: Immutable Synthetic Longitudinal Panel
            - **SHA-256**: `2df006a2...8e79`
            - **Cohort**: 1,000 workers × 4 Quarters ($N=4,000$)
            - **Target**: `future_performance_score` $\in [1.00, 5.00]$
            - **Active Pipeline**: `Ridge (All Eng Feats, α=10.0)`
            """
        )

        st.markdown("---")
        st.caption(
            "⚠️ **Educational / Portfolio Project**  \n"
            "This application is built on synthetic data for data science evaluation. "
            "It must not be utilized for employment actions, appraisals, promotions, or compensation decisions."
        )

        return nav_choice


def render_overview():
    """Render the Overview & Scope section."""
    st.markdown(
        """
        <div class="main-header">
            <h1>Employee Performance Predictor</h1>
            <p>Longitudinal Data Analytics & Regularized Linear Regression Modeling</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="banner-synthetic">
            <strong>📢 Synthetic Data Notice:</strong> All employee profiles, department allocations, task completion metrics,
            and appraisal ratings displayed in this system were generated via a parametric synthetic simulation.
            No real individual or corporate entity is represented.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key statistics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Dataset Size</div>
                <div class="metric-val">4,000</div>
                <div class="metric-sub">Longitudinal records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Fictional Cohort</div>
                <div class="metric-val">1,000</div>
                <div class="metric-sub">Tracked employees</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Observation Window</div>
                <div class="metric-val">4</div>
                <div class="metric-sub">Quarterly periods</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Target Metric</div>
                <div class="metric-val">1.00 – 5.00</div>
                <div class="metric-sub">Continuous appraisal rating</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🎯 Prediction Task & Objectives")
    st.markdown(
        """
        The objective of this project is to predict an employee's future performance appraisal score 
        (`future_performance_score`) at period $t$ using strictly information available **before** period $t$ concludes.
        
        To prevent temporal look-ahead leakage:
        - **Historical Operational Signals**: Task quality, productivity, project completion rate, attendance rate, 
          training hours, and prior appraisal score are strictly lagged ($t-1$).
        - **Static Employee Attributes**: Department, job role, total career experience, and company tenure.
        - **Model Architecture**: Regularized Ridge regression equipped with 4 domain-motivated feature engineering transforms 
          and scikit-learn standard scaling / one-hot encoding.
        """
    )

    st.markdown("### ⚖️ Intended Use Boundaries & Governance")
    st.markdown(
        """
        <div class="banner-warning">
            <strong>Ethical & Operational Safeguards:</strong>
            <ul>
                <li><strong>No Real Employment Actions</strong>: Predictions must never serve as automated recommendations 
                    for hiring, firing, promotion, PIP assignment, compensation, or disciplinary action.</li>
                <li><strong>Correlation is Not Causation</strong>: Relationships identified in this dataset reflect 
                    observational synthetic correlations, not verified causal treatment effects.</li>
                <li><strong>Autoregressive Anchor</strong>: As established in EDA, prior appraisal performance captures ~76% 
                    of score variance, demonstrating strong persistence. The model cannot anticipate unobserved exogenous shocks.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eda():
    """Render the Exploratory Data Analysis section."""
    st.markdown("## 📈 Exploratory Data Analysis (EDA)")
    st.markdown(
        "Empirical analysis of the 4,000-record longitudinal dataset. All figures are rendered directly "
        "from the verified synthetic data repository."
    )

    # Load dataset for live summary table
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    eda_tabs = st.tabs(
        [
            "🎯 Target Distribution",
            "🏢 Departments & Roles",
            "📊 Correlation Structure",
            "🛠️ Remedial Training Analysis",
            "⏱️ Quarterly Trends",
        ]
    )

    with eda_tabs[0]:
        st.markdown("#### Distribution of Prediction Target (`future_performance_score`)")
        st.markdown(
            "The target follows a continuous, roughly Gaussian distribution bounded strictly within $[1.00, 5.00]$ "
            "with a population mean of **3.24** and standard deviation of **0.64**."
        )
        target_fig = FIGURES_DIR / "01_target_distribution.png"
        if target_fig.exists():
            st.image(str(target_fig), caption="Figure 1: Target Variable Distribution & Summary Statistics", use_container_width=True)
        else:
            st.warning(f"Figure file not found at {target_fig}")

    with eda_tabs[1]:
        st.markdown("#### Performance Variation by Department & Job Role")
        st.markdown(
            "Performance is evenly distributed across organizational departments (Engineering, Sales, Marketing, etc.) "
            "with minimal between-department variance, confirming that individual operational output drives scores rather than organizational silo."
        )
        c1, c2 = st.columns(2)
        with c1:
            dept_fig = FIGURES_DIR / "05_performance_by_department.png"
            if dept_fig.exists():
                st.image(str(dept_fig), caption="Figure 5: Performance by Department", use_container_width=True)
        with c2:
            role_fig = FIGURES_DIR / "06_performance_by_job_role.png"
            if role_fig.exists():
                st.image(str(role_fig), caption="Figure 6: Performance by Job Role", use_container_width=True)

    with eda_tabs[2]:
        st.markdown("#### Key Predictor Correlations with Target")
        st.markdown(
            r"""
            Ground-truth Pearson correlations computed across all 4,000 observations:
            - **`previous_performance_score`**: **+0.8662** (Dominant autoregressive predictor)
            - **`quality_score_previous_period`**: **+0.7113** (Strong task fidelity indicator)
            - **`productivity_score_previous_period`**: **+0.6294** (Operational throughput)
            - **`project_completion_rate_previous_period`**: **+0.5310** (Milestone adherence)
            - **`attendance_rate_previous_period`**: **+0.4200** (Engagement indicator)
            - **`training_hours_previous_period`**: **-0.2791** (Remedial assignment mechanism)
            - **`years_experience` / `tenure_months`**: $\approx 0.00$ (Minimal linear relationship)
            """
        )
        corr_fig = FIGURES_DIR / "03_correlation_heatmap.png"
        if corr_fig.exists():
            st.image(str(corr_fig), caption="Figure 3: Pearson Correlation Matrix", use_container_width=True)

    with eda_tabs[3]:
        st.markdown("#### Confounding by Indication: Training Hours Interpretation")
        st.markdown(
            """
            <div class="banner-warning">
                <strong>Crucial Analytical Finding:</strong><br>
                In this dataset, <code>training_hours_previous_period</code> displays a <strong>negative marginal correlation</strong> 
                (<em>r = -0.2791</em>) with the future appraisal score.
                <br><br>
                <strong>Why?</strong> This reflects <em>confounding by indication</em> (remedial assignment): in the synthetic data generating process, 
                employees who received lower ratings in period <em>t-1</em> were systematically assigned <strong>more remedial training</strong>. 
                <br><br>
                It does <strong>NOT</strong> demonstrate that training reduces employee performance! Once conditioning on <code>previous_performance_score</code>, 
                the partial coefficient of training is positive.
            </div>
            """,
            unsafe_allow_html=True,
        )
        train_fig = FIGURES_DIR / "08_training_hours_remedial_analysis.png"
        if train_fig.exists():
            st.image(str(train_fig), caption="Figure 8: Remedial Training Hours vs Appraisal Score", use_container_width=True)

    with eda_tabs[4]:
        st.markdown("#### Quarterly Performance Dynamics Across Periods 1–4")
        st.markdown(
            "Evaluation of macro stability across the four observation periods demonstrates stationarity in score distributions."
        )
        trend_fig = FIGURES_DIR / "07_quarterly_performance_trends.png"
        if trend_fig.exists():
            st.image(str(trend_fig), caption="Figure 7: Quarterly Performance Trends", use_container_width=True)


def render_model_performance():
    """Render the Model Performance & Validation section."""
    st.markdown("## 🔬 Model Performance & Validation Benchmarks")
    st.markdown(
        "Comprehensive audit of model architectures benchmarked across expanding-window temporal validation "
        "and exploratory Period 4 test benchmarks."
    )

    try:
        adv_metrics, base_metrics = load_evaluation_metrics()
        adv_model, adv_meta = load_advanced_model()
        base_model, base_meta = load_baseline_model()
    except Exception as e:
        st.error(f"Error loading model artifacts or metrics: {e}")
        return

    st.markdown(
        """
        <div class="banner-warning">
            <strong>Evaluation Integrity Disclosure:</strong><br>
            During Phase 6 exploratory development, Period 4 was examined repeatedly across an alpha hyperparameter sweep 
            (α ∈ [0.1, 1.0, 5.0, 10.0]) and tree ensemble evaluations. Therefore, <strong>Period 4 is an exploratory benchmark</strong>, 
            not an untouched blind holdout. Model selection was strictly governed by chronological cross-validation on Periods 1–3.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 1. Expanding-Window Cross-Validation (Periods 1–3)")
        st.caption("Folds: P1 → P2 (Fold 1) and P1–2 → P3 (Fold 2). Selection Metric: Mean Validation MAE.")

        val_data = {
            "Model Architecture": [
                "Ridge (All Eng Feats, α=10.0) [Selected Winner]",
                "Ridge Baseline (α=1.0)",
                "Ridge (All Eng Feats, α=1.0)",
                "RandomForest (depth=10, leaf=2)",
                "HistGradientBoosting (iter=50)",
                "Previous-Performance-Only",
                "DummyRegressor (Mean Baseline)",
            ],
            "Mean Val MAE ↓": [0.2307, 0.2309, 0.2313, 0.2380, 0.2393, 0.2507, 0.5041],
            "Mean Val RMSE ↓": [0.2852, 0.2854, 0.2860, 0.2942, 0.2967, 0.3106, 0.6330],
            "Mean Val R² ↑": [0.7934, 0.7932, 0.7922, 0.7802, 0.7765, 0.7550, -0.0177],
            "Rank": ["#1", "#2", "#3", "#10", "#11", "#13", "#14"],
        }
        st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

        st.markdown(
            """
            > **Performance Parity Takeaway**:  
            > The validation MAE difference between the winning engineered Ridge candidate (`0.2307`) and the baseline Ridge (`0.2309`) 
            > is just **0.0002 MAE** (0.09% relative). This represents **practical performance parity**, not a transformative breakthrough.
            """
        )

    with col_r:
        st.markdown("### 2. Period 4 Exploratory Test Benchmarks")
        st.caption("Fitted on Periods 1–3 (N=3,000), evaluated on Period 4 (N=1,000).")

        if not adv_metrics.empty:
            display_adv = adv_metrics[
                ["model_name", "mae", "rmse", "r2", "mae_pct_change_vs_baseline"]
            ].rename(
                columns={
                    "model_name": "Model Architecture",
                    "mae": "Holdout MAE ↓",
                    "rmse": "Holdout RMSE ↓",
                    "r2": "Holdout R² ↑",
                    "mae_pct_change_vs_baseline": "MAE vs Baseline (%)",
                }
            )
            st.dataframe(display_adv, use_container_width=True, hide_index=True)
        else:
            st.info("Advanced metrics CSV not found.")

        st.markdown(
            """
            > **Benchmark Takeaway**:  
            > On the Period 4 exploratory benchmark, the original 10-feature Ridge baseline achieved the lowest error (`MAE = 0.2192`, `R² = 0.7885`), 
            > slightly beating the engineered pipeline (`MAE = 0.2201`). Non-linear tree ensembles exhibited minor degradation due to the 
            > underlying linear-additive data generating process.
            """
        )

    st.markdown("---")
    st.markdown("### 👥 Unseen-Employee Generalization Scenarios (GroupKFold)")
    st.markdown(
        "Evaluated via `GroupKFold(n_splits=5)` grouped strictly by `employee_id` with **zero employee overlap** "
        "between training and validation partitions to test cross-sectional extrapolation:"
    )

    unseen_scenarios = {
        "Scenario": [
            "1. Truly New Hire (Profile Only)",
            "2. Experienced / Probationary Hire (Operational Metrics)",
            "3. Cohort Benchmarking (Full Baseline Features)",
        ],
        "Input Information Supplied": [
            "Department, job role, years experience, tenure months only",
            "Demographics + Quality, Productivity, Project Completion, Attendance, Training",
            "All operational metrics + Genuine Prior Appraisal Score",
        ],
        "Cross-Validated MAE ↓": [0.5273, 0.2973, 0.2209],
        "Cross-Validated R² ↑": [-0.0327, 0.6800, 0.8244],
        "Interpretation": [
            "Zero predictive capability (R² ≤ 0). Biographical profiles alone cannot forecast individual performance.",
            "Substantial predictive signal restored (68% variance explained) without formal appraisal history.",
            "Group generalization benchmark across individuals. (Not directly comparable to temporal holdout R²).",
        ],
    }
    st.dataframe(pd.DataFrame(unseen_scenarios), use_container_width=True, hide_index=True)


def render_prediction_form():
    """Render the Interactive Performance Prediction section."""
    st.markdown("## 🎯 Interactive Performance Prediction")
    st.markdown(
        "Generate a projected appraisal rating for period $t$ based on operational signals realized in period $t-1$. "
        "The active pipeline ingests the **10 base features** and internally constructs the 4 engineered feature interactions."
    )

    try:
        adv_model, adv_meta = load_advanced_model()
        base_model, base_meta = load_baseline_model()
    except Exception as e:
        st.error(f"Cannot load prediction pipelines: {e}")
        return

    st.markdown(
        """
        <div class="banner-synthetic">
            <strong>Simulation Note:</strong> Input features reflect historical operational measurements from quarter <em>t-1</em>.
            No employee identification numbers or personal identifying data are collected or processed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("#### 1. Organizational & Demographic Profile")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            department = st.selectbox(
                "Department",
                options=STANDARD_DEPARTMENTS,
                index=1,  # Engineering
                help="Functional department within the organization.",
            )
            years_experience = st.number_input(
                "Total Career Experience (Years)",
                min_value=0.0,
                max_value=45.0,
                value=7.5,
                step=0.5,
                help="Total professional experience across all prior and current employment.",
            )
        with col_c2:
            job_role = st.selectbox(
                "Job Role",
                options=STANDARD_JOB_ROLES,
                index=16,  # Software Engineer
                help="Standard job title classification.",
            )
            tenure_months = st.number_input(
                "Company Tenure (Months)",
                min_value=0.0,
                max_value=480.0,
                value=36.0,
                step=1.0,
                help="Duration of employment at current firm in months.",
            )

        st.markdown("#### 2. Prior Performance & Operational Signals (Realized at $t-1$)")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            previous_performance_score = st.slider(
                "Previous Performance Score (t-1)",
                min_value=1.00,
                max_value=5.00,
                value=3.25,
                step=0.05,
                help="Appraisal score awarded at previous quarterly review [1.00 - 5.00].",
            )
            quality_score = st.slider(
                "Quality Score (t-1)",
                min_value=40.0,
                max_value=100.0,
                value=76.0,
                step=0.5,
                help="Measure of task precision and low defect rate [0 - 100].",
            )
        with col_p2:
            productivity_score = st.slider(
                "Productivity Score (t-1)",
                min_value=40.0,
                max_value=100.0,
                value=73.0,
                step=0.5,
                help="Operational volume and output delivery score [0 - 100].",
            )
            project_completion_rate = st.slider(
                "Project Completion Rate (t-1)",
                min_value=0.40,
                max_value=1.00,
                value=0.82,
                step=0.01,
                help="Proportion of assigned deliverables completed on schedule [0.0 - 1.0].",
            )
        with col_p3:
            attendance_rate = st.slider(
                "Attendance Rate (t-1)",
                min_value=0.70,
                max_value=1.00,
                value=0.94,
                step=0.01,
                help="Ratio of active scheduled workdays attended [0.0 - 1.0].",
            )
            training_hours = st.slider(
                "Training Hours (t-1)",
                min_value=0.0,
                max_value=40.0,
                value=12.0,
                step=0.5,
                help="Hours of internal professional / remedial training attended in period t-1.",
            )

        submit_btn = st.form_submit_button("Generate Prediction", type="primary", use_container_width=True)

    if submit_btn:
        raw_input = {
            "department": department,
            "job_role": job_role,
            "years_experience": years_experience,
            "tenure_months": tenure_months,
            "previous_performance_score": previous_performance_score,
            "quality_score_previous_period": quality_score,
            "productivity_score_previous_period": productivity_score,
            "project_completion_rate_previous_period": project_completion_rate,
            "attendance_rate_previous_period": attendance_rate,
            "training_hours_previous_period": training_hours,
        }

        is_valid, errors, df_clean = validate_prediction_input(raw_input)

        if not is_valid:
            for err in errors:
                st.error(f"❌ Input Validation Error: {err}")
            return

        # Execute predictions
        adv_pred = predict_performance(adv_model, df_clean)
        base_pred = predict_performance(base_model, df_clean)

        # Determine illustrative heuristic rating band (for visualization only; not official HR standards)
        if adv_pred >= 4.0:
            badge_html = '<span class="score-badge badge-excellent">Illustrative Tier: Upper Quintile (≥ 4.00)</span>'
        elif adv_pred >= 3.0:
            badge_html = '<span class="score-badge badge-good">Illustrative Tier: Core Cohort (3.00 – 3.99)</span>'
        elif adv_pred >= 2.0:
            badge_html = '<span class="score-badge badge-fair">Illustrative Tier: Developing (2.00 – 2.99)</span>'
        else:
            badge_html = '<span class="score-badge badge-low">Illustrative Tier: Lower Boundary (&lt; 2.00)</span>'

        st.markdown("### 📊 Prediction Output")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.markdown(
                f"""
                <div class="metric-card" style="border: 2px solid #6366f1;">
                    <div class="metric-label">Selected Advanced Pipeline (Ridge α=10.0)</div>
                    <div class="metric-val" style="color: #4338ca;">{adv_pred:.2f} <span style="font-size: 1rem; color: #64748b;">/ 5.00</span></div>
                    <div style="margin-top: 0.5rem;">{badge_html}</div>
                    <div class="metric-sub" style="margin-top: 0.5rem;">Features: 10 Base + 4 Engineered Interactions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with p_col2:
            st.markdown(
                f"""
                <div class="metric-card" style="border: 1px solid #cbd5e1;">
                    <div class="metric-label">Baseline Production Pipeline (Ridge α=1.0)</div>
                    <div class="metric-val" style="color: #334155;">{base_pred:.2f} <span style="font-size: 1rem; color: #64748b;">/ 5.00</span></div>
                    <div class="metric-sub" style="margin-top: 0.75rem;">Model Agreement Delta: <strong>{abs(adv_pred - base_pred):.4f}</strong></div>
                    <div class="metric-sub">Features: 10 Base Features (No Feature Engineering)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "📌 **Prediction Notice**: This numerical projection is a regularized linear model estimate derived from synthetic data. "
            "It does not represent a deterministic future reality. No calibrated confidence intervals are claimed. "
            "Tier classifications are illustrative heuristic bands for visualization only; no official HR appraisal standards are implied."
        )


def render_what_if_analysis():
    """Render the What-If Scenario Analysis section."""
    st.markdown("## ⚡ What-If Scenario Analysis")
    st.markdown(
        "Evaluate hypothetical changes to employee operational indicators to examine model sensitivity. "
        "The same validated pipeline is executed across baseline and modified inputs."
    )

    try:
        adv_model, _ = load_advanced_model()
    except Exception as e:
        st.error(f"Cannot load prediction pipeline: {e}")
        return

    st.markdown(
        """
        <div class="banner-warning">
            <strong>⚠️ Causality Warning:</strong><br>
            The difference between the baseline prediction and scenario prediction (Δ) is a <strong>mathematical projection 
            of historical observational correlations</strong>. It is <strong>NOT a causal treatment effect</strong>. 
            Changing an employee's attendance or training hours in real life will not automatically produce the projected score change.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pre-populate sensible baseline
    base_col1, base_col2 = st.columns(2)

    with base_col1:
        st.markdown("### 📋 Original Baseline Scenario")
        dept = st.selectbox("Department (Baseline)", STANDARD_DEPARTMENTS, index=1, key="base_dept")
        role = st.selectbox("Job Role (Baseline)", STANDARD_JOB_ROLES, index=16, key="base_role")
        prev_perf = st.slider("Previous Score (Baseline)", 1.0, 5.0, 3.10, 0.05, key="base_prev_perf")
        quality = st.slider("Quality Score (Baseline)", 40.0, 100.0, 72.0, 0.5, key="base_qual")
        productivity = st.slider("Productivity Score (Baseline)", 40.0, 100.0, 70.0, 0.5, key="base_prod")
        completion = st.slider("Project Completion (Baseline)", 0.40, 1.00, 0.75, 0.01, key="base_comp")
        attendance = st.slider("Attendance Rate (Baseline)", 0.70, 1.00, 0.90, 0.01, key="base_att")
        training = st.slider("Training Hours (Baseline)", 0.0, 40.0, 15.0, 0.5, key="base_train")
        exp = st.number_input("Experience (Years)", 0.0, 45.0, 6.0, 0.5, key="base_exp")
        tenure = st.number_input("Tenure (Months)", 0.0, 480.0, 30.0, 1.0, key="base_tenure")

    with base_col2:
        st.markdown("### 🔄 Modified Hypothesis Scenario")
        st.info("Modify one or more operational levers below to observe the sensitivity of the model projection:")
        scen_dept = st.selectbox("Department (Scenario)", STANDARD_DEPARTMENTS, index=1, key="scen_dept")
        scen_role = st.selectbox("Job Role (Scenario)", STANDARD_JOB_ROLES, index=16, key="scen_role")
        scen_prev_perf = st.slider("Previous Score (Scenario)", 1.0, 5.0, prev_perf, 0.05, key="scen_prev_perf")
        scen_quality = st.slider("Quality Score (Scenario)", 40.0, 100.0, 85.0, 0.5, key="scen_qual")
        scen_productivity = st.slider("Productivity Score (Scenario)", 40.0, 100.0, 82.0, 0.5, key="scen_prod")
        scen_completion = st.slider("Project Completion (Scenario)", 0.40, 1.00, 0.90, 0.01, key="scen_comp")
        scen_attendance = st.slider("Attendance Rate (Scenario)", 0.70, 1.00, 0.98, 0.01, key="scen_att")
        scen_training = st.slider("Training Hours (Scenario)", 0.0, 40.0, 15.0, 0.5, key="scen_train")
        scen_exp = exp
        scen_tenure = tenure

    if st.button("Compute Scenario Comparison", type="primary", use_container_width=True):
        base_dict = {
            "department": dept,
            "job_role": role,
            "previous_performance_score": prev_perf,
            "quality_score_previous_period": quality,
            "productivity_score_previous_period": productivity,
            "project_completion_rate_previous_period": completion,
            "attendance_rate_previous_period": attendance,
            "training_hours_previous_period": training,
            "years_experience": exp,
            "tenure_months": tenure,
        }

        scen_dict = {
            "department": scen_dept,
            "job_role": scen_role,
            "previous_performance_score": scen_prev_perf,
            "quality_score_previous_period": scen_quality,
            "productivity_score_previous_period": scen_productivity,
            "project_completion_rate_previous_period": scen_completion,
            "attendance_rate_previous_period": scen_attendance,
            "training_hours_previous_period": scen_training,
            "years_experience": scen_exp,
            "tenure_months": scen_tenure,
        }

        result = run_what_if_analysis(adv_model, base_dict, scen_dict)

        if not result["success"]:
            for err in result["errors"]:
                st.error(err)
            return

        b_score = result["base_prediction"]
        s_score = result["scenario_prediction"]
        delta = result["delta"]

        st.markdown("### 📊 Scenario Comparison Results")
        res_c1, res_c2, res_c3 = st.columns(3)

        with res_c1:
            st.metric("Baseline Projection", f"{b_score:.2f} / 5.00")
        with res_c2:
            st.metric("Scenario Projection", f"{s_score:.2f} / 5.00")
        with res_c3:
            st.metric("Projected Delta (Δ)", f"{delta:+.4f}", delta=f"{delta:+.2f}")

        if result["changed_fields"]:
            st.markdown("#### Modified Attributes:")
            change_rows = []
            for field, vals in result["changed_fields"].items():
                change_rows.append(
                    {"Field": field, "Baseline Value": vals["original"], "Scenario Value": vals["scenario"]}
                )
            st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)

        st.caption(f"ℹ️ {result['disclaimer']}")


def main():
    """Main dashboard entry point."""
    nav_selection = render_sidebar()

    if nav_selection == "📌 Overview & Scope":
        render_overview()
    elif nav_selection == "📈 Exploratory Data Analysis":
        render_eda()
    elif nav_selection == "🔬 Model Benchmarks & Validation":
        render_model_performance()
    elif nav_selection == "🎯 Interactive Prediction":
        render_prediction_form()
    elif nav_selection == "⚡ What-If Scenario Analysis":
        render_what_if_analysis()


if __name__ == "__main__":
    main()
