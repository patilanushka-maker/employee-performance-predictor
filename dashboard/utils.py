"""Utility functions and backend logic for the Streamlit dashboard.

Provides cached artifact loading, robust input validation, prediction execution,
and scenario comparison utilities for the Employee Performance Predictor project.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import pandas as pd
import streamlit as st

# Path resolution relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATASET_PATH = PROJECT_ROOT / "data" / "synthetic" / "employee_performance.csv"
ADVANCED_MODEL_PATH = PROJECT_ROOT / "models" / "advanced_pipeline.joblib"
ADVANCED_METADATA_PATH = PROJECT_ROOT / "models" / "advanced_pipeline_metadata.json"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_pipeline.joblib"
BASELINE_METADATA_PATH = PROJECT_ROOT / "models" / "baseline_pipeline_metadata.json"
ADVANCED_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "advanced_model_metrics.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "baseline_model_metrics.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Required input features in canonical order
REQUIRED_BASE_FEATURES: List[str] = [
    "previous_performance_score",
    "quality_score_previous_period",
    "productivity_score_previous_period",
    "project_completion_rate_previous_period",
    "attendance_rate_previous_period",
    "training_hours_previous_period",
    "years_experience",
    "tenure_months",
    "department",
    "job_role",
]

NUMERIC_FEATURES: List[str] = [
    "previous_performance_score",
    "quality_score_previous_period",
    "productivity_score_previous_period",
    "project_completion_rate_previous_period",
    "attendance_rate_previous_period",
    "training_hours_previous_period",
    "years_experience",
    "tenure_months",
]

CATEGORICAL_FEATURES: List[str] = [
    "department",
    "job_role",
]

# Strict validation boundaries based on data generator & dataset specification
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "previous_performance_score": (1.00, 5.00),
    "quality_score_previous_period": (0.0, 100.0),
    "productivity_score_previous_period": (0.0, 100.0),
    "project_completion_rate_previous_period": (0.0, 1.00),
    "attendance_rate_previous_period": (0.0, 1.00),
    "training_hours_previous_period": (0.0, 100.0),
    "years_experience": (0.0, 50.0),
    "tenure_months": (0.0, 600.0),
}

STANDARD_DEPARTMENTS: List[str] = [
    "Customer Support",
    "Engineering",
    "Finance",
    "Human Resources",
    "Marketing",
    "Product",
    "Sales",
]

STANDARD_JOB_ROLES: List[str] = [
    "Account Manager",
    "Business Development Representative",
    "Content Strategist",
    "Customer Success Manager",
    "DevOps Engineer",
    "Financial Analyst",
    "HR Business Partner",
    "HR Specialist",
    "Marketing Specialist",
    "Product Manager",
    "QA Engineer",
    "Recruiter",
    "SEO Analyst",
    "Sales Executive",
    "Senior Accountant",
    "Senior Software Engineer",
    "Software Engineer",
    "Support Specialist",
    "UX Designer",
]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the immutable synthetic employee dataset.

    Returns:
        DataFrame containing 4,000 synthetic employee records across 4 periods.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at expected location: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_resource(show_spinner=False)
def load_advanced_model() -> Tuple[Any, Dict[str, Any]]:
    """Load the selected advanced pipeline and its sidecar metadata.

    Returns:
        Tuple containing (Pipeline, metadata_dict).

    Raises:
        FileNotFoundError: If artifact or metadata is missing.
        RuntimeError: If artifact or metadata is corrupt or unreadable.
    """
    if not ADVANCED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Advanced model artifact missing: {ADVANCED_MODEL_PATH}")
    if not ADVANCED_METADATA_PATH.exists():
        raise FileNotFoundError(f"Advanced model metadata missing: {ADVANCED_METADATA_PATH}")

    try:
        model = joblib.load(ADVANCED_MODEL_PATH)
        with open(ADVANCED_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse advanced model artifact/metadata: {e}") from e

    return model, metadata


@st.cache_resource(show_spinner=False)
def load_baseline_model() -> Tuple[Any, Dict[str, Any]]:
    """Load the primary baseline pipeline and its sidecar metadata.

    Returns:
        Tuple containing (Pipeline, metadata_dict).

    Raises:
        FileNotFoundError: If artifact or metadata is missing.
        RuntimeError: If artifact or metadata is corrupt or unreadable.
    """
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model artifact missing: {BASELINE_MODEL_PATH}")
    if not BASELINE_METADATA_PATH.exists():
        raise FileNotFoundError(f"Baseline model metadata missing: {BASELINE_METADATA_PATH}")

    try:
        model = joblib.load(BASELINE_MODEL_PATH)
        with open(BASELINE_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse baseline model artifact/metadata: {e}") from e

    return model, metadata


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical evaluation metrics CSVs for advanced and baseline models.

    Returns:
        Tuple of (advanced_metrics_df, baseline_metrics_df).
    """
    adv_df = pd.read_csv(ADVANCED_METRICS_PATH) if ADVANCED_METRICS_PATH.exists() else pd.DataFrame()
    base_df = pd.read_csv(BASELINE_METRICS_PATH) if BASELINE_METRICS_PATH.exists() else pd.DataFrame()
    return adv_df, base_df


def validate_prediction_input(
    raw_input: Dict[str, Any],
) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
    """Validate user input fields against the model schema and defensible bounds.

    Args:
        raw_input: Dictionary of feature names to values.

    Returns:
        Tuple of (is_valid, error_messages, cleaned_dataframe_or_None).
    """
    errors: List[str] = []

    # 1. Check for forbidden columns
    forbidden_cols = {"employee_id", "period", "future_performance_score", "performance_score"}
    found_forbidden = forbidden_cols.intersection(raw_input.keys())
    if found_forbidden:
        errors.append(f"Forbidden column(s) detected in input: {sorted(list(found_forbidden))}")

    # 2. Check for missing required features
    missing_features = [f for f in REQUIRED_BASE_FEATURES if f not in raw_input]
    if missing_features:
        errors.append(f"Missing required feature(s): {missing_features}")
        return False, errors, None

    cleaned_dict: Dict[str, Any] = {}

    # 3. Validate numeric features and boundaries
    for num_col in NUMERIC_FEATURES:
        val = raw_input.get(num_col)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Field '{num_col}' cannot be empty.")
            continue

        try:
            val_float = float(val)
        except (ValueError, TypeError):
            errors.append(f"Field '{num_col}' must be a numerical value (received '{val}').")
            continue

        # Reject non-finite numbers (NaN, Inf, -Inf)
        if math.isnan(val_float) or math.isinf(val_float):
            errors.append(f"Field '{num_col}' must be a finite numerical value (received '{val}').")
            continue

        # Check domain boundaries
        min_b, max_b = FEATURE_BOUNDS[num_col]
        if val_float < min_b or val_float > max_b:
            errors.append(
                f"Field '{num_col}' value {val_float:.2f} is outside the valid range [{min_b:.2f}, {max_b:.2f}]."
            )
        else:
            cleaned_dict[num_col] = val_float

    # 4. Validate categorical features
    for cat_col in CATEGORICAL_FEATURES:
        cat_val = raw_input.get(cat_col)
        if not cat_val or not isinstance(cat_val, str) or not cat_val.strip():
            errors.append(f"Categorical field '{cat_col}' cannot be blank.")
        else:
            cleaned_dict[cat_col] = str(cat_val).strip()

    # 5. Validate logical cross-feature relationships
    if "years_experience" in cleaned_dict and "tenure_months" in cleaned_dict:
        exp_years = cleaned_dict["years_experience"]
        tenure_years = cleaned_dict["tenure_months"] / 12.0
        # Company tenure cannot exceed total career experience (allowing a 0.5-year tolerance for rounding)
        if tenure_years > (exp_years + 0.5):
            errors.append(
                f"Company tenure ({tenure_years:.1f} years) cannot exceed total career experience ({exp_years:.1f} years)."
            )

    if errors:
        return False, errors, None

    # Construct single-row DataFrame ordered by REQUIRED_BASE_FEATURES
    df_clean = pd.DataFrame([cleaned_dict])[REQUIRED_BASE_FEATURES]
    return True, [], df_clean


def predict_performance(pipeline: Any, input_df: pd.DataFrame) -> float:
    """Execute prediction using the loaded pipeline artifact.

    Args:
        pipeline: Serialized scikit-learn Pipeline.
        input_df: Single-row DataFrame containing validated base features.

    Returns:
        Predicted appraisal score bounded to [1.00, 5.00] and rounded to 4 decimals.
    """
    preds = pipeline.predict(input_df[REQUIRED_BASE_FEATURES])
    score = float(preds[0])
    # Clip to valid theoretical appraisal scale [1.00, 5.00]
    score_bounded = max(1.00, min(5.00, score))
    return round(score_bounded, 4)


def run_what_if_analysis(
    pipeline: Any,
    base_input: Dict[str, Any],
    scenario_input: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute comparative scenario prediction between baseline and modified inputs.

    Args:
        pipeline: Serialized scikit-learn Pipeline.
        base_input: Original baseline feature dictionary.
        scenario_input: Modified scenario feature dictionary.

    Returns:
        Dictionary with status, validation messages, predictions, and deltas.
    """
    is_valid_base, errs_base, df_base = validate_prediction_input(base_input)
    if not is_valid_base:
        return {"success": False, "errors": [f"Baseline input error: {e}" for e in errs_base]}

    is_valid_scen, errs_scen, df_scen = validate_prediction_input(scenario_input)
    if not is_valid_scen:
        return {"success": False, "errors": [f"Scenario input error: {e}" for e in errs_scen]}

    score_base = predict_performance(pipeline, df_base)
    score_scen = predict_performance(pipeline, df_scen)
    delta = round(score_scen - score_base, 4)

    # Detect which fields were modified
    changed_fields = {}
    for feat in REQUIRED_BASE_FEATURES:
        if base_input.get(feat) != scenario_input.get(feat):
            changed_fields[feat] = {
                "original": base_input.get(feat),
                "scenario": scenario_input.get(feat),
            }

    return {
        "success": True,
        "base_prediction": score_base,
        "scenario_prediction": score_scen,
        "delta": delta,
        "changed_fields": changed_fields,
        "disclaimer": (
            "Hypothetical Scenario Notice: The predicted difference is an associative model projection "
            "based on synthetic correlations. It is not a causal treatment effect and does not establish that "
            "changing an employee's metrics will causally improve their appraisal rating."
        ),
    }
