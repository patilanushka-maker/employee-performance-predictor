"""Automated unit and integration tests for EDA engine and workflow runner."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from employee_performance.data_generator import generate_synthetic_data
from employee_performance.eda import (
    compute_dataset_overview,
    compute_correlations,
    compute_department_metrics,
    compute_job_role_metrics,
    compute_period_trends,
    analyze_training_relationship,
    validate_eda_input,
    REQUIRED_EDA_COLUMNS,
)
from scripts.run_eda import run_eda


@pytest.fixture
def small_eda_df() -> pd.DataFrame:
    """Fixture providing a known small synthetic dataset for testing."""
    return generate_synthetic_data(num_employees=15, num_periods=3, random_seed=42)


def test_validate_eda_input_passes_on_valid_data(small_eda_df: pd.DataFrame):
    """Verify validation passes on valid data without exception."""
    validate_eda_input(small_eda_df)


def test_validate_eda_input_fails_on_non_dataframe():
    """Negative test: non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        validate_eda_input("not_a_df")  # type: ignore


def test_validate_eda_input_fails_on_empty_df():
    """Negative test: empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Cannot perform EDA on an empty DataFrame"):
        validate_eda_input(pd.DataFrame())


def test_validate_eda_input_fails_on_missing_column(small_eda_df: pd.DataFrame):
    """Negative test: dropping a required column raises ValueError."""
    bad_df = small_eda_df.drop(columns=["quality_score_previous_period"])
    with pytest.raises(ValueError, match="missing required EDA columns"):
        validate_eda_input(bad_df)


def test_compute_dataset_overview_metrics(small_eda_df: pd.DataFrame):
    """Verify overview dictionary accurately calculates dimension and key counts."""
    overview = compute_dataset_overview(small_eda_df)

    assert overview["row_count"] == 45
    assert overview["unique_employees"] == 15
    assert overview["duplicate_rows"] == 0
    assert overview["duplicate_keys"] == 0
    assert len(overview["period_counts"]) == 3
    assert "future_performance_score" in overview["numeric_stats"]


def test_numerical_summary_matches_known_values():
    """Verify numerical descriptive stats match exact hand-calculated values."""
    toy_df = pd.DataFrame({
        "employee_id": ["E1", "E1", "E2", "E2"],
        "period": [1, 2, 1, 2],
        "department": ["Engineering", "Engineering", "Sales", "Sales"],
        "job_role": ["Software Engineer", "Software Engineer", "Sales Executive", "Sales Executive"],
        "years_experience": [5.0, 5.25, 10.0, 10.25],
        "tenure_months": [12, 15, 24, 27],
        "training_hours_previous_period": [10.0, 12.0, 8.0, 14.0],
        "project_completion_rate_previous_period": [0.8, 0.9, 0.7, 0.85],
        "quality_score_previous_period": [70.0, 80.0, 75.0, 85.0],
        "productivity_score_previous_period": [65.0, 75.0, 70.0, 80.0],
        "attendance_rate_previous_period": [0.95, 0.98, 0.90, 0.92],
        "previous_performance_score": [3.0, 3.5, 3.2, 3.6],
        "future_performance_score": [3.5, 3.8, 3.6, 4.0],
    })

    overview = compute_dataset_overview(toy_df)
    target_stats = overview["numeric_stats"]["future_performance_score"]

    # Known values for [3.5, 3.8, 3.6, 4.0]: mean = 3.725, median = 3.70, min = 3.5, max = 4.0
    assert pytest.approx(target_stats["mean"], 0.001) == 3.725
    assert pytest.approx(target_stats["median"], 0.001) == 3.70
    assert target_stats["min"] == 3.5
    assert target_stats["max"] == 4.0


def test_constant_feature_and_single_observation_group():
    """Verify robust execution when a column is constant or group has 1 element."""
    toy_df = pd.DataFrame({
        "employee_id": ["E1", "E2"],
        "period": [1, 1],
        "department": ["SingleDept", "SingleDept"],
        "job_role": ["SingleRole", "SingleRole"],
        "years_experience": [5.0, 5.0],
        "tenure_months": [10, 10],
        "training_hours_previous_period": [10.0, 10.0],  # Constant
        "project_completion_rate_previous_period": [0.8, 0.8],
        "quality_score_previous_period": [75.0, 75.0],
        "productivity_score_previous_period": [70.0, 70.0],
        "attendance_rate_previous_period": [0.95, 0.95],
        "previous_performance_score": [3.0, 3.0],
        "future_performance_score": [3.0, 3.0],
    })

    # Must complete without crashing
    overview = compute_dataset_overview(toy_df)
    assert overview["numeric_stats"]["training_hours_previous_period"]["std"] == 0.0

    dept_metrics = compute_department_metrics(toy_df)
    assert len(dept_metrics) == 1


def test_compute_correlations_integrity(small_eda_df: pd.DataFrame):
    """Verify correlation matrices are symmetric and bounded in [-1, 1]."""
    corrs = compute_correlations(small_eda_df)
    pearson = corrs["pearson_matrix"]
    spearman = corrs["spearman_matrix"]

    assert pearson.shape == spearman.shape
    assert (pearson.values >= -1.0 - 1e-6).all()
    assert (pearson.values <= 1.0 + 1e-6).all()
    # Diagonal must be 1.0
    for col in pearson.columns:
        assert pytest.approx(pearson.loc[col, col], 0.001) == 1.0


def test_training_hours_selection_bias_analysis(small_eda_df: pd.DataFrame):
    """Verify training hours analysis stratifies by prior performance tiers."""
    analysis = analyze_training_relationship(small_eda_df)

    assert "marginal_correlation" in analysis
    assert "tier_analysis" in analysis
    assert len(analysis["tier_analysis"]) > 0


def test_period_trends_and_within_employee_dynamics(small_eda_df: pd.DataFrame):
    """Verify quarterly trends and delta calculation consistency."""
    trends = compute_period_trends(small_eda_df)

    assert "period_summary" in trends
    assert "within_employee_metrics" in trends

    we = trends["within_employee_metrics"]
    # 15 employees observed over 3 periods -> 15 * 2 = 30 transitions
    assert we["total_transitions"] == 30
    assert pytest.approx(we["improved_share"] + we["declined_share"] + we["stable_share"], 0.01) == 1.0


def test_run_eda_script_in_temporary_directory(small_eda_df: pd.DataFrame):
    """Integration test: run full run_eda pipeline in temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "test_data.csv"
        fig_dir = tmp_path / "test_figures"
        report_file = tmp_path / "test_report.md"

        small_eda_df.to_csv(csv_file, index=False)

        exit_code = run_eda(
            data_path=csv_file,
            figures_dir=fig_dir,
            report_path=report_file,
        )

        assert exit_code == 0
        assert report_file.is_file()
        assert fig_dir.is_dir()

        # Check that figures were generated
        figures = list(fig_dir.glob("*.png"))
        assert len(figures) == 8
        for fig in figures:
            assert fig.stat().st_size > 0

        # Check report content
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "# Exploratory Data Analysis (EDA) Report" in content
            assert "Dataset Overview" in content


def test_run_eda_fails_on_nonexistent_file():
    """Negative integration test: nonexistent file returns exit code 1."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        exit_code = run_eda(
            data_path=tmp_path / "nonexistent.csv",
            figures_dir=tmp_path / "figs",
            report_path=tmp_path / "report.md",
        )
        assert exit_code == 1


def test_run_eda_does_not_modify_permanent_docs_eda_report(small_eda_df: pd.DataFrame):
    """Regression test: verify running EDA on a fixture does NOT overwrite permanent docs/eda_report.md."""
    project_root = Path(__file__).resolve().parent.parent
    permanent_docs_report = project_root / "docs" / "eda_report.md"
    permanent_outputs_report = project_root / "outputs" / "reports" / "eda_report.md"

    # Capture initial states
    docs_content_before = permanent_docs_report.read_text(encoding="utf-8") if permanent_docs_report.exists() else None
    docs_mtime_before = permanent_docs_report.stat().st_mtime if permanent_docs_report.exists() else None

    outputs_content_before = permanent_outputs_report.read_text(encoding="utf-8") if permanent_outputs_report.exists() else None
    outputs_mtime_before = permanent_outputs_report.stat().st_mtime if permanent_outputs_report.exists() else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "test_fixture.csv"
        fig_dir = tmp_path / "figs"
        report_file = tmp_path / "fixture_report.md"

        small_eda_df.to_csv(csv_file, index=False)

        exit_code = run_eda(
            data_path=csv_file,
            figures_dir=fig_dir,
            report_path=report_file,
            docs_copy_path=None,
        )

        assert exit_code == 0
        assert report_file.exists()

    # Verify permanent reports were untouched
    if permanent_docs_report.exists():
        assert permanent_docs_report.read_text(encoding="utf-8") == docs_content_before
        assert permanent_docs_report.stat().st_mtime == docs_mtime_before

    if permanent_outputs_report.exists():
        assert permanent_outputs_report.read_text(encoding="utf-8") == outputs_content_before
        assert permanent_outputs_report.stat().st_mtime == outputs_mtime_before

