"""Automated unit and integration tests for feature engineering transformations."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from employee_performance.features import (
    ALL_CANDIDATE_FEATURES,
    CANDIDATE_FEATURE_A,
    CANDIDATE_FEATURE_B,
    CANDIDATE_FEATURE_C,
    CANDIDATE_FEATURE_D,
    FeatureEngineer,
    calculate_completion_attendance_interaction,
    calculate_experience_adjusted_tenure,
    calculate_prev_perf_training_interaction,
    calculate_quality_to_productivity_ratio,
)
from employee_performance.data_generator import generate_synthetic_data
from employee_performance.train import ALL_PERMITTED_FEATURES, prepare_temporal_split


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Fixture providing a known valid feature DataFrame."""
    df = generate_synthetic_data(num_employees=10, num_periods=2, random_seed=42)
    split = prepare_temporal_split(df, train_periods=(1,), test_period=2)
    return split.X_train_full.copy()


def test_calculate_experience_adjusted_tenure_known_values():
    """Verify experience_adjusted_tenure calculation on known numeric cases."""
    # 5.0 years experience = 60 career months. Tenure 24 months -> 24 / 60 = 0.40
    tenure = pd.Series([24, 60, 12])
    exp = pd.Series([5.0, 5.0, 1.0])

    ratio = calculate_experience_adjusted_tenure(tenure, exp)
    assert pytest.approx(ratio.iloc[0], 0.001) == 0.40
    assert pytest.approx(ratio.iloc[1], 0.001) == 1.00
    assert pytest.approx(ratio.iloc[2], 0.001) == 1.00


def test_experience_adjusted_tenure_zero_denominator_policy():
    """Verify defensible denominator policy when years_experience is exactly zero."""
    # Brand new worker: 0 years experience, 0 tenure -> denom floored to min_career_months=1.0 -> 0.0
    tenure = pd.Series([0, 6])
    exp = pd.Series([0.0, 0.0])

    ratio = calculate_experience_adjusted_tenure(tenure, exp, min_career_months=1.0)
    assert not np.isnan(ratio).any()
    assert not np.isinf(ratio).any()
    assert ratio.iloc[0] == 0.0
    # Tenure 6 with 0 experience clips at 1.0 (tenure cannot exceed career)
    assert ratio.iloc[1] == 1.0


def test_experience_adjusted_tenure_near_zero_denominator():
    """Verify defensible denominator policy when years_experience approaches zero."""
    tenure = pd.Series([1, 2])
    exp = pd.Series([0.0001, 0.01])  # 0.0001*12 = 0.0012 months -> floored to 1.0

    ratio = calculate_experience_adjusted_tenure(tenure, exp, min_career_months=1.0)
    assert not np.isnan(ratio).any()
    assert not np.isinf(ratio).any()
    assert (ratio >= 0.0).all()
    assert (ratio <= 1.0).all()


def test_calculate_quality_to_productivity_ratio_known_values():
    """Verify quality_to_productivity_ratio matches exact arithmetic values."""
    quality = pd.Series([80.0, 90.0, 50.0])
    prod = pd.Series([80.0, 45.0, 100.0])

    ratio = calculate_quality_to_productivity_ratio(quality, prod)
    assert pytest.approx(ratio.iloc[0], 0.001) == 1.00
    assert pytest.approx(ratio.iloc[1], 0.001) == 2.00
    assert pytest.approx(ratio.iloc[2], 0.001) == 0.50


def test_quality_to_productivity_ratio_zero_denominator():
    """Verify quality_to_productivity_ratio safely avoids division by zero."""
    quality = pd.Series([75.0, 0.0])
    prod = pd.Series([0.0, 0.0])

    ratio = calculate_quality_to_productivity_ratio(quality, prod, min_productivity=1e-4)
    assert not np.isnan(ratio).any()
    assert not np.isinf(ratio).any()
    assert ratio.iloc[0] == 75.0 / 1e-4
    assert ratio.iloc[1] == 0.0


def test_calculate_completion_attendance_interaction():
    """Verify task completion and attendance rate interaction product."""
    comp = pd.Series([0.90, 0.80, 1.00])
    att = pd.Series([0.95, 0.50, 1.00])

    prod = calculate_completion_attendance_interaction(comp, att)
    assert pytest.approx(prod.iloc[0], 0.001) == 0.855
    assert pytest.approx(prod.iloc[1], 0.001) == 0.400
    assert pytest.approx(prod.iloc[2], 0.001) == 1.000


def test_calculate_prev_perf_training_interaction():
    """Verify previous performance score and training hours interaction product."""
    score = pd.Series([2.0, 4.0, 5.0])
    hours = pd.Series([20.0, 10.0, 0.0])

    prod = calculate_prev_perf_training_interaction(score, hours)
    assert pytest.approx(prod.iloc[0], 0.001) == 40.0
    assert pytest.approx(prod.iloc[1], 0.001) == 40.0
    assert pytest.approx(prod.iloc[2], 0.001) == 0.0


def test_feature_engineer_transformer_fit_transform(sample_feature_df: pd.DataFrame):
    """Verify FeatureEngineer transformer produces all requested columns and preserves rows."""
    fe = FeatureEngineer()
    out = fe.fit_transform(sample_feature_df)

    assert len(out) == len(sample_feature_df)
    for col in ALL_CANDIDATE_FEATURES:
        assert col in out.columns
        assert not out[col].isna().any()
        assert np.all(np.isfinite(out[col]))

    # Original columns must be preserved
    for col in sample_feature_df.columns:
        assert col in out.columns
        pd.testing.assert_series_equal(out[col], sample_feature_df[col])


def test_feature_engineer_selective_features(sample_feature_df: pd.DataFrame):
    """Verify requesting a single candidate feature adds only that feature."""
    fe = FeatureEngineer(features_to_add=[CANDIDATE_FEATURE_A])
    out = fe.fit_transform(sample_feature_df)

    assert CANDIDATE_FEATURE_A in out.columns
    assert CANDIDATE_FEATURE_B not in out.columns
    assert CANDIDATE_FEATURE_C not in out.columns
    assert CANDIDATE_FEATURE_D not in out.columns


def test_feature_engineer_invalid_feature_name():
    """Negative test: specifying non-existent candidate feature raises ValueError."""
    with pytest.raises(ValueError, match="Unknown candidate features"):
        FeatureEngineer(features_to_add=["non_existent_feature"])


def test_feature_engineer_fails_on_forbidden_target_or_id(sample_feature_df: pd.DataFrame):
    """Negative test: passing employee_id, period, or target raises ValueError (leakage guard)."""
    fe = FeatureEngineer()

    bad_df1 = sample_feature_df.copy()
    bad_df1["employee_id"] = "EMP-0001"
    with pytest.raises(ValueError, match="forbidden identifier or target columns"):
        fe.fit(bad_df1)

    bad_df2 = sample_feature_df.copy()
    bad_df2["future_performance_score"] = 3.5
    with pytest.raises(ValueError, match="forbidden identifier or target columns"):
        fe.transform(bad_df2)


def test_feature_engineer_fails_on_missing_required_column(sample_feature_df: pd.DataFrame):
    """Negative test: missing base column raises ValueError."""
    fe = FeatureEngineer()
    bad_df = sample_feature_df.drop(columns=["tenure_months"])
    with pytest.raises(ValueError, match="missing required base columns"):
        fe.transform(bad_df)


def test_feature_engineer_input_immutability(sample_feature_df: pd.DataFrame):
    """Verify transformer does not mutate input DataFrame in place."""
    original_cols = list(sample_feature_df.columns)
    fe = FeatureEngineer()
    _ = fe.transform(sample_feature_df)

    assert list(sample_feature_df.columns) == original_cols
