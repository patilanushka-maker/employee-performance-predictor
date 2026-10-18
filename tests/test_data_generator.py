"""Automated unit and integration tests for SyntheticDataGenerator."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from employee_performance.data_generator import (
    GeneratorConfig,
    SyntheticDataGenerator,
    generate_synthetic_data,
    DEPARTMENTS,
    DEPARTMENT_ROLES,
)
from employee_performance.validation import REQUIRED_COLUMNS


def test_deterministic_generation_same_seed():
    """Verify that identical random seeds yield byte-identical datasets."""
    config1 = GeneratorConfig(num_employees=30, num_periods=3, random_seed=42)
    config2 = GeneratorConfig(num_employees=30, num_periods=3, random_seed=42)

    df1 = SyntheticDataGenerator(config1).generate()
    df2 = SyntheticDataGenerator(config2).generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_yield_different_data():
    """Verify that different random seeds produce different outputs."""
    df1 = generate_synthetic_data(num_employees=30, num_periods=3, random_seed=42)
    df2 = generate_synthetic_data(num_employees=30, num_periods=3, random_seed=999)

    assert not df1["future_performance_score"].equals(df2["future_performance_score"])


def test_schema_and_column_presence():
    """Verify all 13 required columns are present with expected types."""
    df = generate_synthetic_data(num_employees=20, num_periods=2, random_seed=42)

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df.columns) == 13

    # Ensure no age or salary columns are present
    assert "age" not in df.columns
    assert "salary" not in df.columns


def test_record_counts_and_period_coverage():
    """Verify total record count equals num_employees * num_periods."""
    num_emp = 25
    num_periods = 4
    df = generate_synthetic_data(
        num_employees=num_emp,
        num_periods=num_periods,
        random_seed=42,
    )

    assert len(df) == num_emp * num_periods
    assert df["employee_id"].nunique() == num_emp
    assert sorted(df["period"].unique().tolist()) == [1, 2, 3, 4]


def test_unique_composite_keys():
    """Verify that (employee_id, period) composite keys are strictly unique."""
    df = generate_synthetic_data(num_employees=40, num_periods=3, random_seed=42)
    dups = df.duplicated(subset=["employee_id", "period"]).sum()
    assert dups == 0


def test_employee_stability_across_periods():
    """Verify employee ID, department, and career progression consistency."""
    df = generate_synthetic_data(num_employees=30, num_periods=4, random_seed=42)

    # Department must not change for any employee
    dept_counts = df.groupby("employee_id")["department"].nunique()
    assert (dept_counts == 1).all()

    # Tenure must strictly increase over periods for each employee
    for emp_id, group in df.groupby("employee_id"):
        sorted_group = group.sort_values(by="period")
        tenure_diffs = sorted_group["tenure_months"].diff().dropna()
        assert (tenure_diffs >= 0).all(), f"Employee {emp_id} tenure decreased"


def test_temporal_chaining_and_lag_coherence():
    """Verify temporal design: previous_performance_score at t matches future_performance_score at t-1 across all periods."""
    num_periods = 4
    df = generate_synthetic_data(num_employees=30, num_periods=num_periods, random_seed=42)

    for emp_id, group in df.groupby("employee_id"):
        sorted_group = group.sort_values(by="period").reset_index(drop=True)
        assert len(sorted_group) == num_periods, f"Employee {emp_id} missing periods"
        for t in range(num_periods - 1):
            prior_future = sorted_group.loc[t, "future_performance_score"]
            next_prev = sorted_group.loc[t + 1, "previous_performance_score"]
            assert prior_future == next_prev, (
                f"Employee {emp_id}: period {t+1} future ({prior_future}) != period {t+2} previous ({next_prev})"
            )


def test_valid_numerical_ranges():
    """Verify generated columns respect domain boundaries."""
    df = generate_synthetic_data(num_employees=40, num_periods=3, random_seed=42)

    assert (df["project_completion_rate_previous_period"] >= 0.0).all()
    assert (df["project_completion_rate_previous_period"] <= 1.0).all()

    assert (df["attendance_rate_previous_period"] >= 0.0).all()
    assert (df["attendance_rate_previous_period"] <= 1.0).all()

    assert (df["quality_score_previous_period"] >= 0.0).all()
    assert (df["quality_score_previous_period"] <= 100.0).all()

    assert (df["productivity_score_previous_period"] >= 0.0).all()
    assert (df["productivity_score_previous_period"] <= 100.0).all()

    assert (df["previous_performance_score"] >= 1.0).all()
    assert (df["previous_performance_score"] <= 5.0).all()

    assert (df["future_performance_score"] >= 1.0).all()
    assert (df["future_performance_score"] <= 5.0).all()


def test_valid_categories():
    """Verify department and job role values map correctly."""
    df = generate_synthetic_data(num_employees=30, num_periods=2, random_seed=42)

    for _, row in df.iterrows():
        dept = row["department"]
        role = row["job_role"]
        assert dept in DEPARTMENTS
        assert role in DEPARTMENT_ROLES[dept]


def test_output_path_saving():
    """Verify dataset is correctly persisted to the requested output path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = Path(tmp_dir) / "test_output.csv"
        df = generate_synthetic_data(
            num_employees=10,
            num_periods=2,
            random_seed=42,
            output_path=tmp_csv,
        )

        assert tmp_csv.is_file()
        loaded_df = pd.read_csv(tmp_csv)
        assert len(loaded_df) == 20
        pd.testing.assert_frame_equal(df, loaded_df)
