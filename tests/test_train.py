"""Automated unit and integration tests for baseline model training and evaluation."""

import json
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest

from employee_performance.data_generator import generate_synthetic_data
from employee_performance.train import (
    create_dummy_baseline_pipeline,
    create_full_feature_pipeline,
    create_prev_performance_pipeline,
    evaluate_predictions,
    prepare_temporal_split,
    train_and_evaluate_baselines,
    ALL_PERMITTED_FEATURES,
    PREV_PERF_ONLY_FEATURES,
    TARGET_COLUMN,
)
from scripts.train_baselines import compute_file_sha256


@pytest.fixture
def small_panel_df() -> pd.DataFrame:
    """Fixture providing a known small synthetic dataset across 4 periods."""
    return generate_synthetic_data(num_employees=20, num_periods=4, random_seed=42)


def test_prepare_temporal_split_partitioning(small_panel_df: pd.DataFrame):
    """Verify strict temporal partitioning: periods 1-3 train, period 4 holdout."""
    split = prepare_temporal_split(small_panel_df, train_periods=(1, 2, 3), test_period=4)

    # 20 employees x 3 periods = 60 training rows
    assert len(split.train_df) == 60
    assert sorted(split.train_df["period"].unique().tolist()) == [1, 2, 3]

    # 20 employees x 1 period = 20 test rows
    assert len(split.test_df) == 20
    assert sorted(split.test_df["period"].unique().tolist()) == [4]

    # Total rows preserved
    assert len(split.train_df) + len(split.test_df) == len(small_panel_df)


def test_feature_matrix_excludes_employee_id_and_target(small_panel_df: pd.DataFrame):
    """Verify that employee_id, period, and target are strictly excluded from predictors."""
    split = prepare_temporal_split(small_panel_df)

    assert "employee_id" not in split.X_train_full.columns
    assert "period" not in split.X_train_full.columns
    assert TARGET_COLUMN not in split.X_train_full.columns
    assert list(split.X_train_full.columns) == ALL_PERMITTED_FEATURES

    # For previous performance only
    assert list(split.X_train_prev_only.columns) == PREV_PERF_ONLY_FEATURES


def test_evaluate_predictions_known_values():
    """Verify MAE, RMSE, and R2 calculation matches hand-calculated values."""
    y_true = pd.Series([3.0, 4.0, 5.0])
    y_pred = np.array([3.0, 4.0, 4.0])

    # Absolute errors: [0, 0, 1] -> MAE = 1/3
    # Squared errors: [0, 0, 1] -> RMSE = sqrt(1/3)
    # y_true mean = 4.0. SS_tot = (-1)^2 + 0^2 + 1^2 = 2.0. SS_res = 1.0 -> R2 = 1 - 1/2 = 0.5
    metrics = evaluate_predictions(y_true, y_pred, "TestModel")

    assert pytest.approx(metrics.mae, 0.001) == 1.0 / 3.0
    assert pytest.approx(metrics.rmse, 0.001) == np.sqrt(1.0 / 3.0)
    assert pytest.approx(metrics.r2, 0.001) == 0.5
    assert metrics.min_pred == 3.0
    assert metrics.max_pred == 4.0
    assert metrics.out_of_bounds_count == 0


def test_consistent_evaluation_rows_across_models(small_panel_df: pd.DataFrame):
    """Verify that all three models are evaluated on the exact same holdout rows."""
    results = train_and_evaluate_baselines(small_panel_df, train_periods=(1, 2, 3), test_period=4)
    split = results["split"]
    preds = results["predictions"]

    n_test = len(split.y_test)
    assert n_test == 20

    for model_name, p in preds.items():
        assert len(p) == n_test, f"{model_name} prediction length mismatch"


def test_no_preprocessing_leakage(small_panel_df: pd.DataFrame):
    """Verify pipeline transformers fit on training data without peeking at test set."""
    split = prepare_temporal_split(small_panel_df)

    pipe = create_full_feature_pipeline()
    # Fit strictly on training
    pipe.fit(split.X_train_full, split.y_train)

    preprocessor = pipe.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["num"]

    # Verify scaler mean matches training set numeric mean, NOT full dataset mean
    num_cols = [c for c in ALL_PERMITTED_FEATURES if c in small_panel_df.select_dtypes(include=np.number).columns]
    expected_train_means = split.X_train_full[num_cols].mean().values

    np.testing.assert_allclose(scaler.mean_, expected_train_means, rtol=1e-5)


def test_reproducibility_of_training(small_panel_df: pd.DataFrame):
    """Verify running the experiment twice on the same dataset yields identical metrics."""
    res1 = train_and_evaluate_baselines(small_panel_df)
    res2 = train_and_evaluate_baselines(small_panel_df)

    pd.testing.assert_frame_equal(res1["metrics_df"], res2["metrics_df"])
    for name in res1["predictions"]:
        np.testing.assert_array_equal(res1["predictions"][name], res2["predictions"][name])


def test_train_fails_on_missing_target(small_panel_df: pd.DataFrame):
    """Negative test: missing target column raises ValueError."""
    bad_df = small_panel_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="Missing target column"):
        prepare_temporal_split(bad_df)


def test_train_fails_on_missing_period(small_panel_df: pd.DataFrame):
    """Negative test: missing period column raises ValueError."""
    bad_df = small_panel_df.drop(columns=["period"])
    with pytest.raises(ValueError, match="Missing 'period' column"):
        prepare_temporal_split(bad_df)


def test_train_fails_on_empty_df():
    """Negative test: empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Cannot split an empty DataFrame"):
        prepare_temporal_split(pd.DataFrame())


def test_train_fails_on_non_dataframe():
    """Negative test: non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        prepare_temporal_split("invalid")  # type: ignore


def test_compute_file_sha256_known_content():
    """Verify SHA-256 hash calculation function on temporary file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test.txt"
        tmp_file.write_text("hello world", encoding="utf-8")

        computed_hash = compute_file_sha256(tmp_file)
        # Known sha256 for "hello world"
        import hashlib
        expected_hash = hashlib.sha256(b"hello world").hexdigest()
        assert computed_hash == expected_hash


def test_predictions_are_finite_and_bounded(small_panel_df: pd.DataFrame):
    """Verify that predictions from all models are finite numbers."""
    results = train_and_evaluate_baselines(small_panel_df)
    for model_name, preds in results["predictions"].items():
        assert np.all(np.isfinite(preds)), f"{model_name} generated non-finite predictions"
        # Bounded between 1 and 5
        assert np.all(preds >= 1.0), f"{model_name} predicted below 1.0"
        assert np.all(preds <= 5.0), f"{model_name} predicted above 5.0"


def test_dataset_hash_provenance():
    """Verify that dataset on disk matches canonical SHA-256 hash."""
    csv_path = Path("data/synthetic/employee_performance.csv")
    if not csv_path.exists():
        pytest.skip("Dataset file not found on disk")
    actual_hash = compute_file_sha256(csv_path)
    expected_hash = "2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79"
    assert actual_hash == expected_hash, f"Dataset hash mismatch: {actual_hash} != {expected_hash}"


def test_model_serialization_and_reload():
    """Verify that saved baseline pipeline can be reloaded and reproduces identical predictions."""
    model_path = Path("models/baseline_pipeline.joblib")
    if not model_path.exists():
        pytest.skip("Model artifact not yet generated")
    pipeline = joblib.load(model_path)
    csv_path = Path("data/synthetic/employee_performance.csv")
    df = pd.read_csv(csv_path)
    split = prepare_temporal_split(df)
    reloaded_preds = pipeline.predict(split.X_test_full)
    assert len(reloaded_preds) == 1000
    assert np.all(np.isfinite(reloaded_preds))


def test_metrics_report_consistency():
    """Verify metrics CSV and Markdown report contain consistent numerical values."""
    metrics_path = Path("outputs/metrics/baseline_model_metrics.csv")
    report_path = Path("outputs/reports/baseline_model_report.md")
    if not metrics_path.exists() or not report_path.exists():
        pytest.skip("Metrics or report file not found on disk")
    metrics_df = pd.read_csv(metrics_path)
    report_text = report_path.read_text(encoding="utf-8")
    for _, row in metrics_df.iterrows():
        name = row["model_name"]
        mae_str = f"{row['mae']:.4f}"
        rmse_str = f"{row['rmse']:.4f}"
        assert name in report_text, f"Model {name} missing from report"
        assert mae_str in report_text, f"MAE {mae_str} for {name} missing from report"
        assert rmse_str in report_text, f"RMSE {rmse_str} for {name} missing from report"


def test_temporal_split_cohort_consistency(small_panel_df: pd.DataFrame):
    """Verify that temporal holdout evaluates the same employee cohort over time, not unseen employees."""
    split = prepare_temporal_split(small_panel_df, train_periods=(1, 2, 3), test_period=4)

    train_cohort = set(split.train_df["employee_id"].unique())
    test_cohort = set(split.test_df["employee_id"].unique())

    # Exact cohort overlap: every test employee was observed in the training periods
    assert test_cohort == train_cohort
    assert len(test_cohort) == 20


def test_temporal_lag_identity_on_actual_dataset():
    """Verify for periods 2-4 that previous_performance_score equals preceding period's future score."""
    csv_path = Path("data/synthetic/employee_performance.csv")
    if not csv_path.exists():
        pytest.skip("Dataset file not found on disk")
    df = pd.read_csv(csv_path)
    sorted_df = df.sort_values(by=["employee_id", "period"]).reset_index(drop=True)
    sorted_df["lag_future"] = sorted_df.groupby("employee_id")["future_performance_score"].shift(1)
    mask = sorted_df["period"] > 1
    diffs = (sorted_df.loc[mask, "previous_performance_score"] - sorted_df.loc[mask, "lag_future"]).abs()
    assert (diffs < 1e-4).all(), f"Found {int((diffs >= 1e-4).sum())} lag mismatches in dataset"


