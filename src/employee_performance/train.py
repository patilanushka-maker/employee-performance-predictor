"""Baseline modeling and temporal evaluation module for Employee Performance Predictor.

Implements strict temporal train/test partitioning (Periods 1-3 train, Period 4 holdout),
leakage-free scikit-learn pipelines (Dummy, Previous-Performance-Only, Full-Feature Ridge),
and comprehensive metric evaluation (MAE, RMSE, R²).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FULL_FEATURE_NUMERIC: List[str] = [
    "previous_performance_score",
    "quality_score_previous_period",
    "productivity_score_previous_period",
    "project_completion_rate_previous_period",
    "attendance_rate_previous_period",
    "training_hours_previous_period",
    "years_experience",
    "tenure_months",
]

FULL_FEATURE_CATEGORICAL: List[str] = [
    "department",
    "job_role",
]

ALL_PERMITTED_FEATURES: List[str] = FULL_FEATURE_NUMERIC + FULL_FEATURE_CATEGORICAL
PREV_PERF_ONLY_FEATURES: List[str] = ["previous_performance_score"]
TARGET_COLUMN: str = "future_performance_score"


@dataclass
class ModelMetrics:
    """Stores quantitative evaluation metrics for a single model on test set."""

    model_name: str
    mae: float
    rmse: float
    r2: float
    min_pred: float
    max_pred: float
    out_of_bounds_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "model_name": self.model_name,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "r2": round(self.r2, 4),
            "min_pred": round(self.min_pred, 4),
            "max_pred": round(self.max_pred, 4),
            "out_of_bounds_count": self.out_of_bounds_count,
        }


@dataclass
class TemporalDataSplit:
    """Holds partitioned training and test dataframes and series."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train_full: pd.DataFrame
    X_test_full: pd.DataFrame
    X_train_prev_only: pd.DataFrame
    X_test_prev_only: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_periods: Tuple[int, ...]
    test_period: int


def prepare_temporal_split(
    df: pd.DataFrame,
    train_periods: Tuple[int, ...] = (1, 2, 3),
    test_period: int = 4,
    target_col: str = TARGET_COLUMN,
) -> TemporalDataSplit:
    """Partition longitudinal dataset temporally into training and holdout test sets.

    Ensures zero temporal leakage: Period 4 is held out strictly for final evaluation.
    employee_id and period are excluded from feature matrices.

    Args:
        df: Input longitudinal dataset.
        train_periods: Tuple of period integers for training (default: (1, 2, 3)).
        test_period: Period integer for test evaluation (default: 4).
        target_col: Target column name.

    Returns:
        TemporalDataSplit with cleanly isolated feature matrices and target series.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, received {type(df)}")
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if "period" not in df.columns:
        raise ValueError("Missing 'period' column required for temporal split.")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in dataset.")

    # Check for required features
    missing_feats = [c for c in ALL_PERMITTED_FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required feature columns: {missing_feats}")

    # Temporal partition
    train_mask = df["period"].isin(train_periods)
    test_mask = df["period"] == test_period

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if train_df.empty:
        raise ValueError(f"No records found for training periods {train_periods}")
    if test_df.empty:
        raise ValueError(f"No records found for test period {test_period}")

    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()

    # Isolate feature matrices (strictly excluding employee_id, period, and target)
    X_train_full = train_df[ALL_PERMITTED_FEATURES].copy()
    X_test_full = test_df[ALL_PERMITTED_FEATURES].copy()

    X_train_prev_only = train_df[PREV_PERF_ONLY_FEATURES].copy()
    X_test_prev_only = test_df[PREV_PERF_ONLY_FEATURES].copy()

    return TemporalDataSplit(
        train_df=train_df,
        test_df=test_df,
        X_train_full=X_train_full,
        X_test_full=X_test_full,
        X_train_prev_only=X_train_prev_only,
        X_test_prev_only=X_test_prev_only,
        y_train=y_train,
        y_test=y_test,
        train_periods=train_periods,
        test_period=test_period,
    )


def create_dummy_baseline_pipeline() -> Pipeline:
    """Build DummyRegressor pipeline predicting target mean on training set."""
    return Pipeline(steps=[
        ("regressor", DummyRegressor(strategy="mean"))
    ])


def create_prev_performance_pipeline() -> Pipeline:
    """Build single-predictor Linear Regression pipeline using previous_performance_score."""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])


def create_full_feature_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    alpha: float = 1.0,
) -> Pipeline:
    """Build full-feature ColumnTransformer + Ridge regression pipeline.

    Applies StandardScaler to continuous predictors and OneHotEncoder with
    handle_unknown='ignore' to categorical variables.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        alpha: L2 regularization strength for Ridge regression.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    num_cols = numeric_features or FULL_FEATURE_NUMERIC
    cat_cols = categorical_features or FULL_FEATURE_CATEGORICAL

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha)),
    ])


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> ModelMetrics:
    """Calculate MAE, RMSE, R², and boundary conformity on holdout predictions.

    Args:
        y_true: Ground truth target series.
        y_pred: Array of predicted values.
        model_name: Human-readable model identifier.

    Returns:
        ModelMetrics dataclass with rounded evaluation metrics.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true length {len(y_t)} != y_pred length {len(y_p)}")

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    min_p = float(y_p.min())
    max_p = float(y_p.max())
    out_bounds = int(((y_p < 1.0) | (y_p > 5.0)).sum())

    return ModelMetrics(
        model_name=model_name,
        mae=mae,
        rmse=rmse,
        r2=r2,
        min_pred=min_p,
        max_pred=max_p,
        out_of_bounds_count=out_bounds,
    )


def train_and_evaluate_baselines(
    df: pd.DataFrame,
    train_periods: Tuple[int, ...] = (1, 2, 3),
    test_period: int = 4,
) -> Dict[str, Any]:
    """Execute complete training and evaluation protocol across all 3 baseline models.

    Args:
        df: Input longitudinal dataset.
        train_periods: Periods for training set.
        test_period: Holdout test period.

    Returns:
        Dictionary containing trained pipelines, predictions, metrics, and dataset split info.
    """
    split = prepare_temporal_split(
        df,
        train_periods=train_periods,
        test_period=test_period,
    )

    models: Dict[str, Pipeline] = {
        "DummyRegressor (Mean)": create_dummy_baseline_pipeline(),
        "Previous-Performance-Only": create_prev_performance_pipeline(),
        "Full-Feature Ridge Model": create_full_feature_pipeline(),
    }

    predictions: Dict[str, np.ndarray] = {}
    metrics_list: List[ModelMetrics] = []

    # 1. Fit and evaluate Dummy
    models["DummyRegressor (Mean)"].fit(split.X_train_full, split.y_train)
    p_dummy = models["DummyRegressor (Mean)"].predict(split.X_test_full)
    predictions["DummyRegressor (Mean)"] = p_dummy
    metrics_list.append(evaluate_predictions(split.y_test, p_dummy, "DummyRegressor (Mean)"))

    # 2. Fit and evaluate Previous-Performance-Only
    models["Previous-Performance-Only"].fit(split.X_train_prev_only, split.y_train)
    p_prev = models["Previous-Performance-Only"].predict(split.X_test_prev_only)
    predictions["Previous-Performance-Only"] = p_prev
    metrics_list.append(evaluate_predictions(split.y_test, p_prev, "Previous-Performance-Only"))

    # 3. Fit and evaluate Full-Feature Ridge
    models["Full-Feature Ridge Model"].fit(split.X_train_full, split.y_train)
    p_full = models["Full-Feature Ridge Model"].predict(split.X_test_full)
    predictions["Full-Feature Ridge Model"] = p_full
    metrics_list.append(evaluate_predictions(split.y_test, p_full, "Full-Feature Ridge Model"))

    metrics_df = pd.DataFrame([m.to_dict() for m in metrics_list])

    return {
        "split": split,
        "models": models,
        "predictions": predictions,
        "metrics_df": metrics_df,
        "metrics_list": metrics_list,
    }
