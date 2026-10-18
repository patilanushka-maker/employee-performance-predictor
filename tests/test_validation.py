"""Automated unit tests for DataValidator including positive and negative cases."""

import numpy as np
import pandas as pd
import pytest

from employee_performance.data_generator import generate_synthetic_data
from employee_performance.validation import DataValidator, validate_dataset


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Fixture providing a known valid small synthetic dataframe."""
    return generate_synthetic_data(num_employees=20, num_periods=3, random_seed=42)


def test_validation_passes_on_valid_data(valid_dataframe: pd.DataFrame):
    """Verify that a properly generated dataset passes validation with zero errors."""
    result = validate_dataset(valid_dataframe)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.metrics["row_count"] == 60


def test_validation_fails_on_missing_column(valid_dataframe: pd.DataFrame):
    """Negative test: dropping a required column must fail validation."""
    invalid_df = valid_dataframe.drop(columns=["future_performance_score"])
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("Missing required columns" in err for err in result.errors)


def test_validation_fails_on_null_values(valid_dataframe: pd.DataFrame):
    """Negative test: introducing null/NaN values must fail validation."""
    invalid_df = valid_dataframe.copy()
    invalid_df.loc[2, "quality_score_previous_period"] = np.nan
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("quality_score_previous_period" in err and "missing" in err for err in result.errors)


def test_validation_fails_on_invalid_dtype(valid_dataframe: pd.DataFrame):
    """Negative test: non-numeric type in numeric column must fail validation."""
    invalid_df = valid_dataframe.copy()
    invalid_df["attendance_rate_previous_period"] = invalid_df["attendance_rate_previous_period"].astype(str)
    invalid_df.loc[0, "attendance_rate_previous_period"] = "high"
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("attendance_rate_previous_period" in err and "expected to be numeric" in err for err in result.errors)


def test_validation_fails_on_duplicate_rows(valid_dataframe: pd.DataFrame):
    """Negative test: exact duplicate rows must trigger validation error."""
    invalid_df = pd.concat([valid_dataframe, valid_dataframe.iloc[[0]]], ignore_index=True)
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("duplicate rows" in err for err in result.errors)


def test_validation_fails_on_duplicate_keys(valid_dataframe: pd.DataFrame):
    """Negative test: duplicate (employee_id, period) keys must trigger error."""
    invalid_df = valid_dataframe.copy()
    # Duplicate employee_id and period with different feature values
    new_row = invalid_df.iloc[[0]].copy()
    new_row["quality_score_previous_period"] = 99.0
    invalid_df = pd.concat([invalid_df, new_row], ignore_index=True)
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("duplicate (employee_id, period) composite keys" in err for err in result.errors)


def test_validation_fails_on_invalid_period_value(valid_dataframe: pd.DataFrame):
    """Negative test: non-positive period values must trigger error."""
    invalid_df = valid_dataframe.copy()
    invalid_df.loc[0, "period"] = 0
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("Period values must be positive" in err for err in result.errors)


def test_validation_fails_on_out_of_bounds_numerical(valid_dataframe: pd.DataFrame):
    """Negative test: numerical values outside allowed domain boundaries must fail."""
    # Test attendance > 1.0
    invalid_df1 = valid_dataframe.copy()
    invalid_df1.loc[0, "attendance_rate_previous_period"] = 1.25
    result1 = validate_dataset(invalid_df1)
    assert result1.is_valid is False
    assert any("attendance_rate_previous_period" in err for err in result1.errors)

    # Test future_performance_score > 5.0
    invalid_df2 = valid_dataframe.copy()
    invalid_df2.loc[1, "future_performance_score"] = 6.2
    result2 = validate_dataset(invalid_df2)
    assert result2.is_valid is False
    assert any("future_performance_score" in err for err in result2.errors)


def test_validation_fails_on_invalid_department(valid_dataframe: pd.DataFrame):
    """Negative test: unauthorized department string must fail validation."""
    invalid_df = valid_dataframe.copy()
    invalid_df.loc[0, "department"] = "Astronaut Division"
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("Invalid departments detected" in err for err in result.errors)


def test_validation_fails_on_invalid_job_role_for_department(valid_dataframe: pd.DataFrame):
    """Negative test: mismatch between department and job role must fail."""
    invalid_df = valid_dataframe.copy()
    # Assign HR role to Engineering department
    invalid_df.loc[invalid_df["department"] == "Engineering", "job_role"] = "HR Specialist"
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("contains invalid job roles" in err for err in result.errors)


def test_validation_fails_on_employee_changing_department(valid_dataframe: pd.DataFrame):
    """Negative test: employee switching department across periods must fail."""
    invalid_df = valid_dataframe.copy()
    # Change department for an employee in period 2
    emp_id = invalid_df.loc[0, "employee_id"]
    mask = (invalid_df["employee_id"] == emp_id) & (invalid_df["period"] == 2)
    invalid_df.loc[mask, "department"] = "Sales"
    invalid_df.loc[mask, "job_role"] = "Sales Executive"
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("changed department across observation periods" in err for err in result.errors)


def test_validation_fails_on_decreasing_tenure(valid_dataframe: pd.DataFrame):
    """Negative test: employee tenure decreasing over time must fail."""
    invalid_df = valid_dataframe.copy()
    emp_id = invalid_df.loc[0, "employee_id"]
    mask = (invalid_df["employee_id"] == emp_id) & (invalid_df["period"] == 2)
    invalid_df.loc[mask, "tenure_months"] = 0
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("tenure decreased over time" in err for err in result.errors)


def test_validation_fails_on_zero_target_variance(valid_dataframe: pd.DataFrame):
    """Negative test: constant zero-variance target column must fail."""
    invalid_df = valid_dataframe.copy()
    invalid_df["future_performance_score"] = 3.0
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("zero variance" in err for err in result.errors)


def test_validation_fails_on_non_sequential_period_gaps(valid_dataframe: pd.DataFrame):
    """Negative test: an employee with non-consecutive periods (e.g. 1 and 3) must fail."""
    invalid_df = valid_dataframe.copy()
    emp_id = invalid_df.loc[0, "employee_id"]
    # Drop period 2 for this employee to introduce a gap [1, 3]
    drop_idx = invalid_df[(invalid_df["employee_id"] == emp_id) & (invalid_df["period"] == 2)].index
    invalid_df = invalid_df.drop(index=drop_idx).reset_index(drop=True)
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("non-sequential period jumps/gaps" in err for err in result.errors)


def test_validation_fails_on_decreasing_experience(valid_dataframe: pd.DataFrame):
    """Negative test: employee experience decreasing over time must fail."""
    invalid_df = valid_dataframe.copy()
    emp_id = invalid_df.loc[0, "employee_id"]
    mask = (invalid_df["employee_id"] == emp_id) & (invalid_df["period"] == 2)
    invalid_df.loc[mask, "years_experience"] = 0.5
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("experience decreased over time" in err for err in result.errors)


def test_validation_fails_on_lag_score_mismatch(valid_dataframe: pd.DataFrame):
    """Negative test: previous_performance_score at period t not matching period t-1 future score must fail."""
    invalid_df = valid_dataframe.copy()
    emp_id = invalid_df.loc[0, "employee_id"]
    mask = (invalid_df["employee_id"] == emp_id) & (invalid_df["period"] == 2)
    invalid_df.loc[mask, "previous_performance_score"] = 4.95  # Arbitrary corrupted lag
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("temporal lag inconsistencies" in err for err in result.errors)


def test_validation_fails_on_tenure_exceeding_career(valid_dataframe: pd.DataFrame):
    """Negative test: tenure_months exceeding lifetime career months must fail."""
    invalid_df = valid_dataframe.copy()
    invalid_df.loc[0, "years_experience"] = 2.0  # 24 months total career
    invalid_df.loc[0, "tenure_months"] = 60      # 60 months at company impossible
    result = validate_dataset(invalid_df)

    assert result.is_valid is False
    assert any("tenure_months exceeds lifetime career months" in err for err in result.errors)


def test_validation_emits_warning_on_low_attendance(valid_dataframe: pd.DataFrame):
    """Warning test: low attendance (<0.75) generates descriptive warning."""
    warn_df = valid_dataframe.copy()
    warn_df.loc[0, "attendance_rate_previous_period"] = 0.70
    result = validate_dataset(warn_df)

    # Hard validation should still pass, but warning list must be populated
    assert result.is_valid is True
    assert len(result.warnings) > 0
    assert any("attendance rates below 0.75" in warn for warn in result.warnings)

