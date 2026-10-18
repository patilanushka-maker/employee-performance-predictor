"""Automated unit and integration tests for advanced modeling and temporal validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from employee_performance.data_generator import generate_synthetic_data
from employee_performance.modeling import (
    FoldMetric,
    ModelComparisonResult,
    ValidationFold,
    build_pipeline,
    create_temporal_validation_folds,
    evaluate_final_holdout_period_4,
    evaluate_pipeline_on_temporal_folds,
    evaluate_unseen_employee_generalization,
    run_candidate_model_exploration,
)
from employee_performance.train import ALL_PERMITTED_FEATURES, TARGET_COLUMN


@pytest.fixture
def sample_longitudinal_df() -> pd.DataFrame:
    """Fixture providing a known small synthetic dataset across 4 periods."""
    return generate_synthetic_data(num_employees=30, num_periods=4, random_seed=42)


def test_create_temporal_validation_folds_structure(sample_longitudinal_df: pd.DataFrame):
    """Verify expanding-window temporal validation folds structure and isolation."""
    folds = create_temporal_validation_folds(sample_longitudinal_df)

    assert len(folds) == 2

    # Fold 1: train P1 (30 rows), val P2 (30 rows)
    f1 = folds[0]
    assert f1.fold_id == 1
    assert f1.train_periods == (1,)
    assert f1.val_period == 2
    assert f1.train_count == 30
    assert f1.val_count == 30
    assert list(f1.X_train.columns) == ALL_PERMITTED_FEATURES
    assert list(f1.X_val.columns) == ALL_PERMITTED_FEATURES
    assert "period" not in f1.X_train.columns
    assert "employee_id" not in f1.X_train.columns
    assert TARGET_COLUMN not in f1.X_train.columns

    # Fold 2: train P1, P2 (60 rows), val P3 (30 rows)
    f2 = folds[1]
    assert f2.fold_id == 2
    assert f2.train_periods == (1, 2)
    assert f2.val_period == 3
    assert f2.train_count == 60
    assert f2.val_count == 30

    # Period 4 must not appear anywhere in validation folds
    for f in folds:
        assert 4 not in f.train_periods
        assert f.val_period != 4


def test_preprocessing_isolation_per_fold(sample_longitudinal_df: pd.DataFrame):
    """Verify that transformers fit strictly on the training partition of each fold."""
    folds = create_temporal_validation_folds(sample_longitudinal_df)
    pipe = build_pipeline("ridge", hyperparameters={"alpha": 1.0})

    for fold in folds:
        pipe.fit(fold.X_train, fold.y_train)
        scaler = pipe.named_steps["preprocessor"].named_transformers_["num"]
        # Numeric columns mean should match training fold exactly
        num_cols = pipe.named_steps["preprocessor"].transformers[0][2]
        expected_means = fold.X_train[num_cols].mean().values
        np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)


def test_build_pipeline_supported_architectures():
    """Verify all supported model types construct functional pipelines."""
    architectures = [
        ("dummy", {}),
        ("prev_perf_only", {}),
        ("ridge", {"alpha": 1.0}),
        ("random_forest", {"n_estimators": 10, "max_depth": 3}),
        ("hist_gradient_boosting", {"max_iter": 20}),
    ]

    for model_type, params in architectures:
        pipe = build_pipeline(model_type, hyperparameters=params)
        assert isinstance(pipe, Pipeline)
        assert "regressor" in pipe.named_steps


def test_build_pipeline_invalid_model_type():
    """Negative test: unsupported model type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_pipeline("unsupported_type")


def test_evaluate_pipeline_on_temporal_folds(sample_longitudinal_df: pd.DataFrame):
    """Verify complete evaluation across expanding temporal folds produces valid metrics."""
    folds = create_temporal_validation_folds(sample_longitudinal_df)
    pipe = build_pipeline("ridge", hyperparameters={"alpha": 1.0})

    result = evaluate_pipeline_on_temporal_folds(
        pipeline=pipe,
        folds=folds,
        model_name="Test Ridge",
        feature_set_name="Baseline Features",
        hyperparameters={"alpha": 1.0},
    )

    assert isinstance(result, ModelComparisonResult)
    assert len(result.fold_metrics) == 2
    assert result.mean_mae > 0.0
    assert result.mean_rmse > 0.0
    assert -1.0 <= result.mean_r2 <= 1.0
    assert result.std_mae >= 0.0

    # Ensure fold metrics details
    for fm in result.fold_metrics:
        assert isinstance(fm, FoldMetric)
        assert np.isfinite(fm.mae)
        assert np.isfinite(fm.rmse)
        assert np.isfinite(fm.r2)


def test_unseen_employee_generalization_group_isolation(sample_longitudinal_df: pd.DataFrame):
    """Verify that GroupKFold strictly prevents employee leakage across splits."""
    results = evaluate_unseen_employee_generalization(sample_longitudinal_df, n_splits=3, random_seed=42)

    assert len(results) == 3
    for r in results:
        assert r.mae_mean > 0.0
        assert r.rmse_mean > 0.0
        assert len(r.fold_maes) == 3

    # Notice that New Hire profile without previous score has higher error than full features
    new_hire = results[0]
    full_cohort = results[2]
    assert new_hire.mae_mean > full_cohort.mae_mean, "New hire without prior score should have higher error"


def test_temporal_validation_consistent_validation_rows(sample_longitudinal_df: pd.DataFrame):
    """Verify all candidate pipelines are evaluated on identical validation rows within each fold."""
    folds = create_temporal_validation_folds(sample_longitudinal_df)
    pipe_ridge = build_pipeline("ridge", hyperparameters={"alpha": 1.0})
    pipe_dummy = build_pipeline("dummy")

    res_ridge = evaluate_pipeline_on_temporal_folds(pipe_ridge, folds, "Ridge", "Base", {})
    res_dummy = evaluate_pipeline_on_temporal_folds(pipe_dummy, folds, "Dummy", "Base", {})

    assert len(res_ridge.fold_metrics) == len(res_dummy.fold_metrics)
    for m_r, m_d in zip(res_ridge.fold_metrics, res_dummy.fold_metrics):
        assert m_r.val_count == m_d.val_count
        assert m_r.val_period == m_d.val_period


def test_unseen_employee_zero_id_overlap_across_all_folds(sample_longitudinal_df: pd.DataFrame):
    """Verify that GroupKFold strictly enforces 0 employee_id overlap between train and val."""
    from sklearn.model_selection import GroupKFold
    train_df = sample_longitudinal_df[sample_longitudinal_df["period"].isin([1, 2, 3])].reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)

    for train_idx, val_idx in gkf.split(train_df, groups=train_df["employee_id"]):
        train_emps = set(train_df.iloc[train_idx]["employee_id"].unique())
        val_emps = set(train_df.iloc[val_idx]["employee_id"].unique())
        assert len(train_emps.intersection(val_emps)) == 0, "Found employee_id overlap across GroupKFold splits!"


def test_model_selection_uses_only_periods_1_to_3(sample_longitudinal_df: pd.DataFrame):
    """Verify candidate model selection strictly ranks models by mean MAE on Periods 1-3 without Period 4."""
    results, winner = run_candidate_model_exploration(sample_longitudinal_df)
    # Winner must match the model with minimum mean_mae
    assert winner.mean_mae == min(r.mean_mae for r in results)
    # Ensure no fold metric has period 4
    for r in results:
        for fm in r.fold_metrics:
            assert fm.val_period in [2, 3]
            assert 4 not in fm.train_periods


def test_evaluate_final_holdout_period_4(sample_longitudinal_df: pd.DataFrame):
    """Verify final holdout evaluation protocol on Period 4."""
    pipe = build_pipeline("ridge", hyperparameters={"alpha": 1.0})
    holdout_res = evaluate_final_holdout_period_4(pipe, sample_longitudinal_df)

    assert holdout_res["holdout_period"] == 4
    # 30 employees x 3 train periods = 90 train obs, 30 test obs
    assert holdout_res["train_observations"] == 90
    assert holdout_res["test_observations"] == 30
    assert holdout_res["mae"] > 0.0
    assert holdout_res["rmse"] > 0.0
    assert "mae_diff" in holdout_res
    assert "mae_pct_change" in holdout_res
    assert "r2_diff" in holdout_res
    assert holdout_res["out_of_bounds_count"] == 0


def test_advanced_model_serialization_and_reload():
    """Verify saved advanced model artifact reloads and reproduces exact predictions."""
    import joblib
    from pathlib import Path

    model_path = Path("models/advanced_pipeline.joblib")
    if not model_path.exists():
        pytest.skip("Advanced model artifact not found on disk")

    pipe = joblib.load(model_path)
    csv_path = Path("data/synthetic/employee_performance.csv")
    df = pd.read_csv(csv_path)

    test_df = df[df["period"] == 4]
    preds = pipe.predict(test_df[ALL_PERMITTED_FEATURES])

    assert len(preds) == 1000
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 1.0)
    assert np.all(preds <= 5.0)


def test_advanced_metrics_and_report_consistency():
    """Verify agreement across metrics CSV, metadata sidecar, and Markdown report."""
    import json
    from pathlib import Path

    metrics_path = Path("outputs/metrics/advanced_model_metrics.csv")
    metadata_path = Path("models/advanced_pipeline_metadata.json")
    report_path = Path("outputs/reports/advanced_model_exploration_report.md")

    if not metrics_path.exists() or not metadata_path.exists() or not report_path.exists():
        pytest.skip("Phase 6 artifacts not yet generated")

    metrics_df = pd.read_csv(metrics_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")

    # Verify metadata dataset hash
    assert metadata["dataset_sha256"] == "2df006a243b4bbb43299275073d91802f4fd524da6a5674b44092f0358f68e79"

    # Verify holdout metrics match between metadata and CSV
    selected_row = metrics_df[metrics_df["model_name"] == metadata["selected_model_name"]].iloc[0]
    assert pytest.approx(metadata["period_4_holdout_evaluation"]["mae"], 0.0001) == selected_row["mae"]
    assert pytest.approx(metadata["period_4_holdout_evaluation"]["rmse"], 0.0001) == selected_row["rmse"]
    assert pytest.approx(metadata["period_4_holdout_evaluation"]["r2"], 0.0001) == selected_row["r2"]

    # Verify report contains the winning model and numbers
    assert metadata["selected_model_name"] in report_text
    assert f"{selected_row['mae']:.4f}" in report_text

