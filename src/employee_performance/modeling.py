"""Advanced modeling, temporal validation, and model exploration module.

Implements expanding-window temporal cross-validation (Periods 1-3),
scikit-learn pipelines for regularized linear and ensemble architectures,
isolated unseen-employee generalization evaluation, and holdout evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from employee_performance.features import (
    ALL_CANDIDATE_FEATURES,
    CANDIDATE_FEATURE_A,
    CANDIDATE_FEATURE_B,
    CANDIDATE_FEATURE_C,
    CANDIDATE_FEATURE_D,
    FeatureEngineer,
)
from employee_performance.train import (
    ALL_PERMITTED_FEATURES,
    FULL_FEATURE_CATEGORICAL,
    FULL_FEATURE_NUMERIC,
    PREV_PERF_ONLY_FEATURES,
    TARGET_COLUMN,
    ModelMetrics,
    evaluate_predictions,
)


@dataclass
class ValidationFold:
    """Represents a single temporal expanding-window validation split."""

    fold_id: int
    train_periods: Tuple[int, ...]
    val_period: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    train_count: int
    val_count: int


@dataclass
class FoldMetric:
    """Evaluation metrics for a model on a single validation fold."""

    fold_id: int
    train_periods: Tuple[int, ...]
    val_period: int
    train_count: int
    val_count: int
    mae: float
    rmse: float
    r2: float
    min_pred: float
    max_pred: float
    out_of_bounds_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert fold metric to dictionary."""
        return {
            "fold_id": self.fold_id,
            "train_periods": list(self.train_periods),
            "val_period": self.val_period,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "r2": round(self.r2, 4),
            "min_pred": round(self.min_pred, 4),
            "max_pred": round(self.max_pred, 4),
            "out_of_bounds_count": self.out_of_bounds_count,
        }


@dataclass
class ModelComparisonResult:
    """Aggregated temporal validation evaluation for a candidate model."""

    model_name: str
    feature_set_name: str
    features_used: List[str]
    hyperparameters: Dict[str, Any]
    fold_metrics: List[FoldMetric]
    mean_mae: float
    mean_rmse: float
    mean_r2: float
    std_mae: float
    warnings_or_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary to dictionary."""
        return {
            "model_name": self.model_name,
            "feature_set_name": self.feature_set_name,
            "hyperparameters": str(self.hyperparameters),
            "mean_mae": round(self.mean_mae, 4),
            "mean_rmse": round(self.mean_rmse, 4),
            "mean_r2": round(self.mean_r2, 4),
            "std_mae": round(self.std_mae, 4),
            "warnings_or_notes": self.warnings_or_notes,
        }


def create_temporal_validation_folds(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> List[ValidationFold]:
    """Create expanding-window temporal validation folds strictly from Periods 1-3.

    Fold 1:
        - Training: Period 1 (1,000 observations)
        - Validation: Period 2 (1,000 observations)
    Fold 2:
        - Training: Periods 1 and 2 (2,000 observations)
        - Validation: Period 3 (1,000 observations)

    Ensures zero temporal look-ahead:
        Each fold trains only on strictly earlier periods and validates on the
        immediately following sequential period. Period 4 is completely sequestered.

    Args:
        df: Input DataFrame containing longitudinal records.
        target_col: Prediction target column name.

    Returns:
        List of ValidationFold instances.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    if "period" not in df.columns:
        raise ValueError("Missing 'period' column required for temporal validation.")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'.")

    # Define expanding window configurations
    fold_configs = [
        (1, (1,), 2),
        (2, (1, 2), 3),
    ]

    folds: List[ValidationFold] = []
    for fold_id, train_periods, val_period in fold_configs:
        train_mask = df["period"].isin(train_periods)
        val_mask = df["period"] == val_period

        train_df = df[train_mask]
        val_df = df[val_mask]

        if train_df.empty or val_df.empty:
            raise ValueError(f"Empty split for fold {fold_id} (train={train_periods}, val={val_period})")

        X_train = train_df[ALL_PERMITTED_FEATURES].copy()
        y_train = train_df[target_col].copy()

        X_val = val_df[ALL_PERMITTED_FEATURES].copy()
        y_val = val_df[target_col].copy()

        folds.append(
            ValidationFold(
                fold_id=fold_id,
                train_periods=train_periods,
                val_period=val_period,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                train_count=len(train_df),
                val_count=len(val_df),
            )
        )

    return folds


def build_pipeline(
    model_type: str,
    features_to_add: Optional[List[str]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a self-contained scikit-learn Pipeline with feature engineering, preprocessing, and model.

    All preprocessing transformers are contained within the Pipeline and fitted
    strictly during `.fit()`, eliminating any data leakage.

    Supported model types:
        - 'dummy': DummyRegressor(strategy='mean')
        - 'prev_perf_only': LinearRegression on previous_performance_score only
        - 'ridge': Ridge regression with StandardScaler and OneHotEncoder
        - 'random_forest': RandomForestRegressor with OneHotEncoder
        - 'hist_gradient_boosting': HistGradientBoostingRegressor with OneHotEncoder

    Args:
        model_type: Architecture identifier string.
        features_to_add: Optional candidate features to generate via FeatureEngineer.
        hyperparameters: Estimator hyperparameter dictionary.

    Returns:
        Pipeline: Complete executable scikit-learn pipeline.
    """
    params = hyperparameters or {}
    added_feats = features_to_add or []

    # 1. Dummy baseline
    if model_type == "dummy":
        return Pipeline(steps=[("regressor", DummyRegressor(strategy="mean"))])

    # 2. Previous-Performance-Only baseline
    if model_type == "prev_perf_only":
        return Pipeline(steps=[
            ("selector", ColumnTransformer([
                ("prev", StandardScaler(), PREV_PERF_ONLY_FEATURES),
            ], remainder="drop")),
            ("regressor", LinearRegression()),
        ])

    # 3. Multivariable pipelines
    numeric_cols = list(FULL_FEATURE_NUMERIC) + list(added_feats)
    categorical_cols = list(FULL_FEATURE_CATEGORICAL)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    steps: List[Tuple[str, Any]] = []
    if added_feats:
        steps.append(("engineer", FeatureEngineer(features_to_add=added_feats)))
    steps.append(("preprocessor", preprocessor))

    if model_type == "ridge":
        alpha = params.get("alpha", 1.0)
        steps.append(("regressor", Ridge(alpha=alpha, random_state=42)))
    elif model_type == "random_forest":
        n_est = params.get("n_estimators", 100)
        max_d = params.get("max_depth", None)
        min_samples = params.get("min_samples_leaf", 2)
        steps.append(("regressor", RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_leaf=min_samples,
            random_state=42,
            n_jobs=-1,
        )))
    elif model_type == "hist_gradient_boosting":
        max_iter = params.get("max_iter", 100)
        max_leaf = params.get("max_leaf_nodes", 31)
        l2_reg = params.get("l2_regularization", 0.0)
        lr = params.get("learning_rate", 0.1)
        steps.append(("regressor", HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_leaf_nodes=max_leaf,
            l2_regularization=l2_reg,
            learning_rate=lr,
            random_state=42,
        )))
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    return Pipeline(steps=steps)


def evaluate_pipeline_on_temporal_folds(
    pipeline: Pipeline,
    folds: List[ValidationFold],
    model_name: str,
    feature_set_name: str,
    hyperparameters: Dict[str, Any],
) -> ModelComparisonResult:
    """Train and evaluate a pipeline across expanding-window temporal folds.

    Preprocessing and feature engineering are fitted strictly on each fold's training slice.

    Args:
        pipeline: Unfitted scikit-learn Pipeline template.
        folds: List of ValidationFold instances.
        model_name: Identifier name for the model.
        feature_set_name: Description of feature set.
        hyperparameters: Dictionary of configured hyperparameters.

    Returns:
        ModelComparisonResult aggregating fold-level and summary metrics.
    """
    fold_metrics: List[FoldMetric] = []

    for fold in folds:
        # Clone pipeline for clean separation across folds
        pipe = clone(pipeline)

        # Fit strictly on training fold
        pipe.fit(fold.X_train, fold.y_train)

        # Predict strictly on validation fold
        preds = pipe.predict(fold.X_val)
        y_val_arr = np.asarray(fold.y_val).ravel()
        preds_arr = np.asarray(preds).ravel()

        mae = float(mean_absolute_error(y_val_arr, preds_arr))
        rmse = float(np.sqrt(mean_squared_error(y_val_arr, preds_arr)))
        r2 = float(r2_score(y_val_arr, preds_arr))
        min_p = float(preds_arr.min())
        max_p = float(preds_arr.max())
        oob = int(((preds_arr < 1.0) | (preds_arr > 5.0)).sum())

        fold_metrics.append(
            FoldMetric(
                fold_id=fold.fold_id,
                train_periods=fold.train_periods,
                val_period=fold.val_period,
                train_count=fold.train_count,
                val_count=fold.val_count,
                mae=mae,
                rmse=rmse,
                r2=r2,
                min_pred=min_p,
                max_pred=max_p,
                out_of_bounds_count=oob,
            )
        )

    maes = [fm.mae for fm in fold_metrics]
    rmses = [fm.rmse for fm in fold_metrics]
    r2s = [fm.r2 for fm in fold_metrics]

    return ModelComparisonResult(
        model_name=model_name,
        feature_set_name=feature_set_name,
        features_used=ALL_PERMITTED_FEATURES,
        hyperparameters=hyperparameters,
        fold_metrics=fold_metrics,
        mean_mae=float(np.mean(maes)),
        mean_rmse=float(np.mean(rmses)),
        mean_r2=float(np.mean(r2s)),
        std_mae=float(np.std(maes)),
    )


def run_candidate_model_exploration(
    df: pd.DataFrame,
) -> Tuple[List[ModelComparisonResult], ModelComparisonResult]:
    """Execute complete temporal validation benchmark across all candidate architectures.

    Benchmarks:
        1. DummyRegressor(mean)
        2. Previous-Performance-Only
        3. Full-Feature Ridge (Baseline, alpha=1.0)
        4. Ridge + Feature A (Quality/Productivity ratio)
        5. Ridge + Feature B (Completion*Attendance interaction)
        6. Ridge + Feature C (PrevPerf*Training interaction)
        7. Ridge + Feature D (Experience-adjusted tenure)
        8. Ridge + All Engineered Features (Tuned alpha)
        9. RandomForestRegressor (Tuned depth and min_samples_leaf)
        10. HistGradientBoostingRegressor (Tuned max_iter and regularization)

    Model selection uses MAE as primary metric across expanding-window folds (Periods 1-3 only).
    Period 4 is never touched.

    Args:
        df: Input DataFrame containing longitudinal records.

    Returns:
        Tuple: (List of all ModelComparisonResults sorted by mean_mae, winning ModelComparisonResult).
    """
    folds = create_temporal_validation_folds(df)
    results: List[ModelComparisonResult] = []

    # 1. DummyRegressor
    p_dummy = build_pipeline("dummy")
    results.append(evaluate_pipeline_on_temporal_folds(
        p_dummy, folds, "DummyRegressor (Mean)", "Baseline (None)", {"strategy": "mean"}
    ))

    # 2. Previous-Performance-Only
    p_prev = build_pipeline("prev_perf_only")
    results.append(evaluate_pipeline_on_temporal_folds(
        p_prev, folds, "Previous-Performance-Only", "Single Predictor", {}
    ))

    # 3. Existing Full-Feature Ridge baseline (alpha=1.0)
    p_ridge_base = build_pipeline("ridge", hyperparameters={"alpha": 1.0})
    results.append(evaluate_pipeline_on_temporal_folds(
        p_ridge_base, folds, "Ridge Baseline", "Baseline (10 features)", {"alpha": 1.0}
    ))

    # 4-7. Ridge with individual candidate features
    indiv_candidates = [
        ("Ridge + Ratio (Quality/Prod)", [CANDIDATE_FEATURE_A]),
        ("Ridge + Task Interaction", [CANDIDATE_FEATURE_B]),
        ("Ridge + Training Interaction", [CANDIDATE_FEATURE_C]),
        ("Ridge + Adjusted Tenure", [CANDIDATE_FEATURE_D]),
    ]
    for name, feat_list in indiv_candidates:
        pipe = build_pipeline("ridge", features_to_add=feat_list, hyperparameters={"alpha": 1.0})
        results.append(evaluate_pipeline_on_temporal_folds(
            pipe, folds, name, f"Baseline + {feat_list[0]}", {"alpha": 1.0}
        ))

    # 8. Ridge with all engineered features across alpha grid
    for alpha in [0.1, 1.0, 10.0]:
        pipe = build_pipeline("ridge", features_to_add=ALL_CANDIDATE_FEATURES, hyperparameters={"alpha": alpha})
        results.append(evaluate_pipeline_on_temporal_folds(
            pipe, folds, f"Ridge (All Eng Feats, alpha={alpha})", "Baseline + 4 Candidate Features", {"alpha": alpha}
        ))

    # 9. RandomForestRegressor grid
    rf_grid = [
        {"max_depth": 6, "min_samples_leaf": 5, "n_estimators": 100},
        {"max_depth": 10, "min_samples_leaf": 2, "n_estimators": 100},
    ]
    for params in rf_grid:
        pipe = build_pipeline("random_forest", features_to_add=ALL_CANDIDATE_FEATURES, hyperparameters=params)
        results.append(evaluate_pipeline_on_temporal_folds(
            pipe, folds, f"RandomForest (depth={params['max_depth']}, leaf={params['min_samples_leaf']})",
            "Baseline + 4 Candidate Features", params
        ))

    # 10. HistGradientBoostingRegressor grid
    hgb_grid = [
        {"max_iter": 50, "max_leaf_nodes": 15, "l2_regularization": 1.0, "learning_rate": 0.08},
        {"max_iter": 100, "max_leaf_nodes": 31, "l2_regularization": 0.1, "learning_rate": 0.1},
    ]
    for params in hgb_grid:
        pipe = build_pipeline("hist_gradient_boosting", features_to_add=ALL_CANDIDATE_FEATURES, hyperparameters=params)
        results.append(evaluate_pipeline_on_temporal_folds(
            pipe, folds, f"HistGradientBoosting (iter={params['max_iter']}, leaf={params['max_leaf_nodes']})",
            "Baseline + 4 Candidate Features", params
        ))

    # Sort results by mean MAE ascending (primary metric)
    results_sorted = sorted(results, key=lambda x: x.mean_mae)
    winner = results_sorted[0]

    return results_sorted, winner


@dataclass
class UnseenEmployeeEvaluationResult:
    """Results for separate unseen-employee generalization evaluation."""

    feature_set_name: str
    features_used: List[str]
    mae_mean: float
    rmse_mean: float
    r2_mean: float
    fold_maes: List[float]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "feature_set_name": self.feature_set_name,
            "features_used": ", ".join(self.features_used),
            "mae_mean": round(self.mae_mean, 4),
            "rmse_mean": round(self.rmse_mean, 4),
            "r2_mean": round(self.r2_mean, 4),
            "notes": self.notes,
        }


def evaluate_unseen_employee_generalization(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[UnseenEmployeeEvaluationResult]:
    """Execute separate group-aware cross-validation on training data (Periods 1-3).

    Methodological Distinction:
        - Temporal holdout measures future-period performance forecasting for the SAME employee cohort.
        - This experiment evaluates prediction for ENTIRELY UNSEEN employees where historical
          appraisal scores may be unavailable (new hire onboarding use case).

    Features compared:
        Scenario 1: Truly New Hire (No historical appraisal rating; profile & tenure only).
            Features: department, job_role, years_experience, tenure_months.
        Scenario 2: Transferred / Experienced Hire (Operational task metrics available, but no firm appraisal).
            Features: profile + quality, productivity, completion, attendance, training.
        Scenario 3: Unseen Employee with Prior External Score (Full baseline feature set).

    Args:
        df: Input DataFrame.
        n_splits: Number of employee groups (default: 5).
        random_seed: Seed for reproducibility.

    Returns:
        List of UnseenEmployeeEvaluationResult objects.
    """
    # Evaluate strictly on training periods 1-3
    train_mask = df["period"].isin([1, 2, 3])
    train_df = df[train_mask].copy().reset_index(drop=True)

    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df["employee_id"]

    feature_scenarios = [
        (
            "New Hire Profile (No Appraisal Score)",
            ["department", "job_role", "years_experience", "tenure_months"],
            ["years_experience", "tenure_months"],
            ["department", "job_role"],
            "Operational reality: new employees have zero prior appraisal ratings.",
        ),
        (
            "Experienced Hire (Operational Metrics, No Appraisal)",
            [
                "department", "job_role", "years_experience", "tenure_months",
                "quality_score_previous_period", "productivity_score_previous_period",
                "project_completion_rate_previous_period", "attendance_rate_previous_period",
                "training_hours_previous_period",
            ],
            [
                "years_experience", "tenure_months",
                "quality_score_previous_period", "productivity_score_previous_period",
                "project_completion_rate_previous_period", "attendance_rate_previous_period",
                "training_hours_previous_period",
            ],
            ["department", "job_role"],
            "Operational indicators known from probation period, but formal annual appraisal missing.",
        ),
        (
            "Cohort Benchmarking (Full Baseline Features Across Unseen Groups)",
            ALL_PERMITTED_FEATURES,
            FULL_FEATURE_NUMERIC,
            FULL_FEATURE_CATEGORICAL,
            "Contrasts group generalization against temporal cohort forecasting.",
        ),
    ]

    scenario_results: List[UnseenEmployeeEvaluationResult] = []

    for name, all_cols, num_cols, cat_cols, notes in feature_scenarios:
        fold_maes: List[float] = []
        fold_rmses: List[float] = []
        fold_r2s: List[float] = []

        for train_idx, val_idx in gkf.split(train_df, groups=groups):
            X_tr, y_tr = train_df.iloc[train_idx][all_cols], train_df.iloc[train_idx][TARGET_COLUMN]
            X_va, y_val = train_df.iloc[val_idx][all_cols], train_df.iloc[val_idx][TARGET_COLUMN]

            pipe = Pipeline([
                ("prep", ColumnTransformer([
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ], remainder="drop")),
                ("reg", Ridge(alpha=1.0, random_state=random_seed)),
            ])

            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_va)

            fold_maes.append(float(mean_absolute_error(y_val, preds)))
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
            fold_r2s.append(float(r2_score(y_val, preds)))

        scenario_results.append(
            UnseenEmployeeEvaluationResult(
                feature_set_name=name,
                features_used=all_cols,
                mae_mean=float(np.mean(fold_maes)),
                rmse_mean=float(np.mean(fold_rmses)),
                r2_mean=float(np.mean(fold_r2s)),
                fold_maes=fold_maes,
                notes=notes,
            )
        )

    return scenario_results


def evaluate_final_holdout_period_4(
    selected_pipeline: Pipeline,
    df: pd.DataFrame,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Refit the selected model on Periods 1-3 and evaluate on Period 4 holdout.

    Compares test performance against the full-feature Ridge baseline:
        - Baseline MAE: 0.2192
        - Baseline RMSE: 0.2713
        - Baseline R2: 0.7885

    Args:
        selected_pipeline: The unfitted Pipeline chosen during temporal validation.
        df: Full longitudinal dataset.
        baseline_metrics: Optional reference baseline metrics dictionary.

    Returns:
        Dictionary containing holdout metrics, comparisons, and boundary checks.
    """
    ref_baseline = baseline_metrics or {
        "mae": 0.2192,
        "rmse": 0.2713,
        "r2": 0.7885,
    }

    train_mask = df["period"].isin([1, 2, 3])
    test_mask = df["period"] == 4

    X_train = df[train_mask][ALL_PERMITTED_FEATURES].copy()
    y_train = df[train_mask][TARGET_COLUMN].copy()

    X_test = df[test_mask][ALL_PERMITTED_FEATURES].copy()
    y_test = df[test_mask][TARGET_COLUMN].copy()

    # Fit selected pipeline strictly on training data
    pipeline = clone(selected_pipeline)
    pipeline.fit(X_train, y_train)

    # Predict on unseen Period 4
    preds = pipeline.predict(X_test)
    y_test_arr = np.asarray(y_test).ravel()
    preds_arr = np.asarray(preds).ravel()

    mae = float(mean_absolute_error(y_test_arr, preds_arr))
    rmse = float(np.sqrt(mean_squared_error(y_test_arr, preds_arr)))
    r2 = float(r2_score(y_test_arr, preds_arr))
    min_p = float(preds_arr.min())
    max_p = float(preds_arr.max())
    oob = int(((preds_arr < 1.0) | (preds_arr > 5.0)).sum())

    # Calculate absolute and relative differences vs baseline
    mae_diff = mae - ref_baseline["mae"]
    mae_pct = (mae_diff / ref_baseline["mae"]) * 100.0

    rmse_diff = rmse - ref_baseline["rmse"]
    rmse_pct = (rmse_diff / ref_baseline["rmse"]) * 100.0

    r2_diff = r2 - ref_baseline["r2"]

    return {
        "holdout_period": 4,
        "train_observations": len(X_train),
        "test_observations": len(X_test),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "min_pred": round(min_p, 4),
        "max_pred": round(max_p, 4),
        "out_of_bounds_count": oob,
        "baseline_mae": ref_baseline["mae"],
        "baseline_rmse": ref_baseline["rmse"],
        "baseline_r2": ref_baseline["r2"],
        "mae_diff": round(mae_diff, 4),
        "mae_pct_change": round(mae_pct, 2),
        "rmse_diff": round(rmse_diff, 4),
        "rmse_pct_change": round(rmse_pct, 2),
        "r2_diff": round(r2_diff, 4),
        "fitted_pipeline": pipeline,
        "predictions": preds_arr,
    }
