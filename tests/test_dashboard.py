"""Automated unit and integration tests for dashboard utilities and prediction logic.

Verifies:
1. Saved model and artifact loading.
2. Correct input feature names and ordering.
3. Valid prediction input.
4. Out-of-range numerical values.
5. Missing or invalid required fields.
6. Unknown categorical values.
7. Scenario input validation.
8. Prediction logic uses the saved pipeline.
9. Immutability of synthetic CSV during dashboard operations.
10. Permanent documentation and reports are not modified by dashboard operations.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from dashboard.utils import (
    ADVANCED_METADATA_PATH,
    ADVANCED_MODEL_PATH,
    BASELINE_METADATA_PATH,
    BASELINE_MODEL_PATH,
    DATASET_PATH,
    PROJECT_ROOT,
    REQUIRED_BASE_FEATURES,
    load_advanced_model,
    load_baseline_model,
    load_dataset,
    load_evaluation_metrics,
    predict_performance,
    run_what_if_analysis,
    validate_prediction_input,
)
from employee_performance.train import ALL_PERMITTED_FEATURES


@pytest.fixture
def valid_input_dict() -> Dict[str, Any]:
    """Provide a canonical valid input dictionary for testing."""
    return {
        "department": "Engineering",
        "job_role": "Software Engineer",
        "years_experience": 6.5,
        "tenure_months": 28.0,
        "training_hours_previous_period": 10.0,
        "project_completion_rate_previous_period": 0.88,
        "quality_score_previous_period": 80.0,
        "productivity_score_previous_period": 75.0,
        "attendance_rate_previous_period": 0.96,
        "previous_performance_score": 3.40,
    }


def compute_file_sha256(filepath: Path) -> str:
    """Helper to compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# 1. Saved model and artifact loading
def test_saved_model_and_artifact_loading():
    """Verify that both saved models, metadata sidecars, dataset, and metrics load cleanly."""
    adv_model, adv_meta = load_advanced_model()
    base_model, base_meta = load_baseline_model()
    adv_metrics, base_metrics = load_evaluation_metrics()
    dataset = load_dataset()

    assert isinstance(adv_model, Pipeline)
    assert isinstance(base_model, Pipeline)
    assert adv_meta["target_column"] == "future_performance_score"
    assert base_meta["target_column"] == "future_performance_score"
    assert len(dataset) == 4000
    assert not adv_metrics.empty
    assert not base_metrics.empty


# 2. Correct input feature names and ordering
def test_correct_input_feature_names_and_ordering():
    """Verify REQUIRED_BASE_FEATURES matches ALL_PERMITTED_FEATURES in length and contents."""
    assert len(REQUIRED_BASE_FEATURES) == 10
    assert set(REQUIRED_BASE_FEATURES) == set(ALL_PERMITTED_FEATURES)


# 3. Valid prediction input
def test_valid_prediction_input(valid_input_dict: Dict[str, Any]):
    """Verify valid input passes validation and generates a bounded prediction."""
    is_valid, errors, df_clean = validate_prediction_input(valid_input_dict)
    assert is_valid is True
    assert errors == []
    assert df_clean is not None
    assert list(df_clean.columns) == REQUIRED_BASE_FEATURES

    adv_model, _ = load_advanced_model()
    score = predict_performance(adv_model, df_clean)
    assert isinstance(score, float)
    assert 1.00 <= score <= 5.00


# 4. Out-of-range numerical values
@pytest.mark.parametrize(
    "field,bad_val,expected_substr",
    [
        ("previous_performance_score", 0.50, "outside the valid range [1.00, 5.00]"),
        ("previous_performance_score", 5.50, "outside the valid range [1.00, 5.00]"),
        ("quality_score_previous_period", -5.0, "outside the valid range [0.00, 100.00]"),
        ("quality_score_previous_period", 105.0, "outside the valid range [0.00, 100.00]"),
        ("attendance_rate_previous_period", 1.20, "outside the valid range [0.00, 1.00]"),
        ("years_experience", -1.0, "outside the valid range [0.00, 50.00]"),
    ],
)
def test_out_of_range_numerical_values(valid_input_dict: Dict[str, Any], field: str, bad_val: float, expected_substr: str):
    """Verify that values violating domain bounds fail validation with specific error messages."""
    invalid_input = valid_input_dict.copy()
    invalid_input[field] = bad_val

    is_valid, errors, df_clean = validate_prediction_input(invalid_input)
    assert is_valid is False
    assert df_clean is None
    assert any(expected_substr in err for err in errors)


# 5. Missing or invalid required fields
def test_missing_required_fields(valid_input_dict: Dict[str, Any]):
    """Verify omitting required features is flagged."""
    incomplete_input = valid_input_dict.copy()
    del incomplete_input["department"]
    del incomplete_input["previous_performance_score"]

    is_valid, errors, df_clean = validate_prediction_input(incomplete_input)
    assert is_valid is False
    assert df_clean is None
    assert any("Missing required feature" in err for err in errors)


def test_invalid_type_fields(valid_input_dict: Dict[str, Any]):
    """Verify non-numeric string values in numeric fields are caught."""
    bad_type_input = valid_input_dict.copy()
    bad_type_input["quality_score_previous_period"] = "not_a_number"

    is_valid, errors, df_clean = validate_prediction_input(bad_type_input)
    assert is_valid is False
    assert df_clean is None
    assert any("must be a numerical value" in err for err in errors)


def test_forbidden_columns_flagged(valid_input_dict: Dict[str, Any]):
    """Verify passing target or identifier columns is rejected."""
    leaked_input = valid_input_dict.copy()
    leaked_input["future_performance_score"] = 4.0
    leaked_input["employee_id"] = 123

    is_valid, errors, df_clean = validate_prediction_input(leaked_input)
    assert is_valid is False
    assert df_clean is None
    assert any("Forbidden column" in err for err in errors)


# 6. Unknown categorical values
def test_unknown_categorical_values(valid_input_dict: Dict[str, Any]):
    """Verify that unknown categories are handled gracefully by OneHotEncoder."""
    unknown_cat_input = valid_input_dict.copy()
    unknown_cat_input["department"] = "QuantumRobotics"
    unknown_cat_input["job_role"] = "CosmicNavigator"

    is_valid, errors, df_clean = validate_prediction_input(unknown_cat_input)
    assert is_valid is True
    assert df_clean is not None

    adv_model, _ = load_advanced_model()
    score = predict_performance(adv_model, df_clean)
    assert 1.00 <= score <= 5.00


# 7. Scenario input validation
def test_scenario_input_validation(valid_input_dict: Dict[str, Any]):
    """Verify what-if scenario comparison calculates expected deltas and catches scenario errors."""
    adv_model, _ = load_advanced_model()

    scenario_input = valid_input_dict.copy()
    scenario_input["quality_score_previous_period"] = 95.0
    scenario_input["productivity_score_previous_period"] = 90.0

    res = run_what_if_analysis(adv_model, valid_input_dict, scenario_input)
    assert res["success"] is True
    assert "delta" in res
    assert "base_prediction" in res
    assert "scenario_prediction" in res
    assert res["scenario_prediction"] > res["base_prediction"]
    assert "quality_score_previous_period" in res["changed_fields"]

    # Negative test: invalid scenario input
    bad_scenario = valid_input_dict.copy()
    bad_scenario["previous_performance_score"] = 99.0  # Out of range

    bad_res = run_what_if_analysis(adv_model, valid_input_dict, bad_scenario)
    assert bad_res["success"] is False
    assert len(bad_res["errors"]) > 0


# 8. Ensuring that prediction logic uses the saved pipeline
def test_prediction_logic_uses_saved_pipeline(valid_input_dict: Dict[str, Any]):
    """Verify predict_performance directly invokes the pipeline's predict method."""
    adv_model, _ = load_advanced_model()
    _, _, df_clean = validate_prediction_input(valid_input_dict)
    assert df_clean is not None

    expected_raw = float(adv_model.predict(df_clean[REQUIRED_BASE_FEATURES])[0])
    expected_score = round(max(1.00, min(5.00, expected_raw)), 4)

    actual_score = predict_performance(adv_model, df_clean)
    assert actual_score == expected_score


# 9. Ensuring dashboard utilities do not mutate the original synthetic CSV
def test_dashboard_utilities_do_not_mutate_original_synthetic_csv(valid_input_dict: Dict[str, Any]):
    """Verify that dataset hash remains strictly invariant before and after dashboard operations."""
    expected_hash = "2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79"
    hash_before = compute_file_sha256(DATASET_PATH)
    assert hash_before == expected_hash

    # Execute all dashboard utility actions
    _ = load_dataset()
    adv_model, _ = load_advanced_model()
    _, _, df_clean = validate_prediction_input(valid_input_dict)
    _ = predict_performance(adv_model, df_clean)
    _ = run_what_if_analysis(adv_model, valid_input_dict, valid_input_dict)

    hash_after = compute_file_sha256(DATASET_PATH)
    assert hash_after == expected_hash


# 10. Ensuring dashboard execution does not overwrite permanent EDA or evaluation reports
def test_dashboard_execution_does_not_overwrite_permanent_reports():
    """Verify permanent markdown reports in docs/ and outputs/reports/ are not modified."""
    reports_to_check = [
        PROJECT_ROOT / "docs" / "eda_report.md",
        PROJECT_ROOT / "docs" / "advanced_model_exploration_report.md",
        PROJECT_ROOT / "outputs" / "reports" / "eda_report.md",
        PROJECT_ROOT / "outputs" / "reports" / "advanced_model_exploration_report.md",
    ]

    initial_states = {}
    for r in reports_to_check:
        if r.exists():
            initial_states[r] = (r.read_text(encoding="utf-8"), r.stat().st_mtime)

    # Invoke dashboard loaders and analysis
    _ = load_dataset()
    _ = load_evaluation_metrics()
    adv_model, _ = load_advanced_model()

    for r, (content_before, mtime_before) in initial_states.items():
        assert r.read_text(encoding="utf-8") == content_before
        assert r.stat().st_mtime == mtime_before


# 11. Edge case: NaN and Infinite numeric inputs
@pytest.mark.parametrize(
    "bad_numeric",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nan_and_inf_inputs_rejected(valid_input_dict: Dict[str, Any], bad_numeric: float):
    """Verify that NaN and Infinite numeric inputs are safely caught and rejected."""
    nan_input = valid_input_dict.copy()
    nan_input["previous_performance_score"] = bad_numeric

    is_valid, errors, df_clean = validate_prediction_input(nan_input)
    assert is_valid is False
    assert df_clean is None
    assert any("must be a finite numerical value" in err for err in errors)


# 12. Edge case: Tenure exceeding experience at zero experience
def test_tenure_exceeding_experience_at_zero_experience(valid_input_dict: Dict[str, Any]):
    """Verify that company tenure cannot exceed experience even when experience is 0."""
    impossible_input = valid_input_dict.copy()
    impossible_input["years_experience"] = 0.0
    impossible_input["tenure_months"] = 24.0  # 2 years company tenure with 0 total experience

    is_valid, errors, df_clean = validate_prediction_input(impossible_input)
    assert is_valid is False
    assert df_clean is None
    assert any("Company tenure (2.0 years) cannot exceed total career experience" in err for err in errors)


# 13. Edge case: Corrupt artifact handling
def test_corrupt_artifact_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that corrupted model or metadata sidecars raise a descriptive RuntimeError."""
    corrupt_joblib = tmp_path / "corrupt.joblib"
    corrupt_joblib.write_text("corrupted non-pickle binary content", encoding="utf-8")
    corrupt_json = tmp_path / "corrupt.json"
    corrupt_json.write_text("{invalid json", encoding="utf-8")

    monkeypatch.setattr("dashboard.utils.ADVANCED_MODEL_PATH", corrupt_joblib)
    monkeypatch.setattr("dashboard.utils.ADVANCED_METADATA_PATH", corrupt_json)

    # Clear cached resource so it hits the patched filesystem paths
    load_advanced_model.clear()
    try:
        with pytest.raises(RuntimeError, match="Failed to load or parse advanced model"):
            load_advanced_model()
    finally:
        load_advanced_model.clear()


# 14. Native Streamlit AppTest: Full navigation across all 5 sections
def test_streamlit_apptest_navigation_all_sections():
    """Exercise Streamlit rendered interface navigation across all 5 modules via AppTest."""
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    sections = [
        "📌 Overview & Scope",
        "📈 Exploratory Data Analysis",
        "🔬 Model Benchmarks & Validation",
        "🎯 Interactive Prediction",
        "⚡ What-If Scenario Analysis",
    ]

    for section in sections:
        at.sidebar.radio[0].set_value(section).run()
        assert len(at.exception) == 0, f"Exception occurred while navigating to {section}: {at.exception}"


# 15. Native Streamlit AppTest: Prediction form submission flow
def test_streamlit_apptest_prediction_flow():
    """Exercise Streamlit prediction form interaction and output rendering."""
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("🎯 Interactive Prediction").run()
    assert len(at.exception) == 0

    # Submit form with defaults
    at.button[0].click().run()
    assert len(at.exception) == 0

    # Verify prediction output is displayed in rendered markdown
    markdown_texts = [m.value for m in at.markdown]
    assert any("Prediction Output" in t for t in markdown_texts)
    assert any("Selected Advanced Pipeline" in t for t in markdown_texts)


# 16. Native Streamlit AppTest: What-If scenario execution flow
def test_streamlit_apptest_what_if_scenario_flow():
    """Exercise Streamlit What-If scenario comparison and metric delta rendering."""
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("⚡ What-If Scenario Analysis").run()
    assert len(at.exception) == 0

    # Click scenario computation button
    at.button[0].click().run()
    assert len(at.exception) == 0

    # Verify metric cards are populated
    metric_labels = [m.label for m in at.metric]
    assert any("Baseline Projection" in l for l in metric_labels)
    assert any("Scenario Projection" in l for l in metric_labels)


