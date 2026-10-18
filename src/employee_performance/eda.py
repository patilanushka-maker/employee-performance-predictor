"""Exploratory Data Analysis (EDA) analytical engine for Employee Performance Predictor.

Provides modular, non-mutating analytical routines to compute descriptive overviews,
correlation matrices, subgroup aggregations, longitudinal period trends, and
within-employee performance dynamics.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_EDA_COLUMNS: List[str] = [
    "employee_id",
    "period",
    "department",
    "job_role",
    "years_experience",
    "tenure_months",
    "training_hours_previous_period",
    "project_completion_rate_previous_period",
    "quality_score_previous_period",
    "productivity_score_previous_period",
    "attendance_rate_previous_period",
    "previous_performance_score",
    "future_performance_score",
]

NUMERIC_PREDICTORS: List[str] = [
    "years_experience",
    "tenure_months",
    "training_hours_previous_period",
    "project_completion_rate_previous_period",
    "quality_score_previous_period",
    "productivity_score_previous_period",
    "attendance_rate_previous_period",
    "previous_performance_score",
]

FEATURE_SPECS: Dict[str, Dict[str, str]] = {
    "employee_id": {"scale": "Identifier", "unit": "String", "role": "Cluster / Panel Key"},
    "period": {"scale": "1 to 4", "unit": "Integer Quarter", "role": "Temporal Index"},
    "department": {"scale": "7 Categories", "unit": "Categorical", "role": "Business Unit"},
    "job_role": {"scale": "19 Categories", "unit": "Categorical", "role": "Functional Title"},
    "years_experience": {"scale": "0.5 to 30.0", "unit": "Years", "role": "Continuous Predictor"},
    "tenure_months": {"scale": "1 to 300", "unit": "Months", "role": "Discrete Predictor"},
    "training_hours_previous_period": {"scale": "0 to 60", "unit": "Hours", "role": "Continuous Predictor"},
    "project_completion_rate_previous_period": {"scale": "0.0 to 1.0", "unit": "Fraction (0-1)", "role": "Ratio Predictor"},
    "quality_score_previous_period": {"scale": "0 to 100", "unit": "Audit Index (0-100)", "role": "Continuous Predictor"},
    "productivity_score_previous_period": {"scale": "0 to 100", "unit": "Output Index (0-100)", "role": "Continuous Predictor"},
    "attendance_rate_previous_period": {"scale": "0.0 to 1.0", "unit": "Fraction (0-1)", "role": "Ratio Predictor"},
    "previous_performance_score": {"scale": "1.0 to 5.0", "unit": "Appraisal Rating", "role": "Continuous Predictor"},
    "future_performance_score": {"scale": "1.0 to 5.0", "unit": "Appraisal Rating", "role": "Prediction Target (y)"},
}


def validate_eda_input(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
    """Validate that the input DataFrame meets structural contracts for EDA.

    Args:
        df: Input DataFrame.
        required_columns: List of columns required for analysis.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If DataFrame is empty or missing required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, received {type(df)}")
    if df.empty:
        raise ValueError("Cannot perform EDA on an empty DataFrame (0 rows).")

    cols = required_columns or REQUIRED_EDA_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required EDA columns: {missing}")


def compute_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level dataset metrics, integrity counts, and descriptive stats.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with dataset dimension metrics and descriptive stats.
    """
    validate_eda_input(df)

    row_count = int(len(df))
    col_count = int(df.shape[1])
    unique_emp = int(df["employee_id"].nunique())
    period_counts = df["period"].value_counts().sort_index().to_dict()
    missing_counts = df[REQUIRED_EDA_COLUMNS].isna().sum().to_dict()
    dup_rows = int(df.duplicated().sum())
    dup_keys = int(df.duplicated(subset=["employee_id", "period"]).sum())

    # Parametric and non-parametric numeric summaries
    numeric_cols = NUMERIC_PREDICTORS + ["future_performance_score"]
    numeric_stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        n = len(series)
        mean_val = float(series.mean())
        std_val = float(series.std(ddof=1)) if n > 1 else 0.0
        median_val = float(series.median())
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        min_val = float(series.min())
        max_val = float(series.max())
        skew_val = float(stats.skew(series)) if (n > 2 and std_val > 0) else 0.0
        kurt_val = float(stats.kurtosis(series)) if (n > 3 and std_val > 0) else 0.0

        numeric_stats[col] = {
            "count": n,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "q25": q25,
            "median": median_val,
            "q75": q75,
            "max": max_val,
            "skewness": skew_val,
            "kurtosis": kurt_val,
        }

    return {
        "row_count": row_count,
        "col_count": col_count,
        "unique_employees": unique_emp,
        "period_counts": period_counts,
        "missing_counts": missing_counts,
        "duplicate_rows": dup_rows,
        "duplicate_keys": dup_keys,
        "department_counts": df["department"].value_counts().to_dict(),
        "job_role_counts": df["job_role"].value_counts().to_dict(),
        "numeric_stats": numeric_stats,
    }


def compute_correlations(
    df: pd.DataFrame,
    target_col: str = "future_performance_score",
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation matrices and rankings with target.

    Args:
        df: Input DataFrame.
        target_col: Name of the dependent variable.
        numeric_cols: Features to include in correlation matrix.

    Returns:
        Dictionary with full correlation matrices and target ranking pairs.
    """
    validate_eda_input(df)
    cols = numeric_cols or (NUMERIC_PREDICTORS + [target_col])

    # Filter to existing numeric columns
    valid_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # Pearson matrix
    pearson_df = df[valid_cols].corr(method="pearson").round(4)
    # Spearman matrix
    spearman_df = df[valid_cols].corr(method="spearman").round(4)

    # Specific correlations with target
    target_pearson = (
        pearson_df[target_col]
        .drop(target_col, errors="ignore")
        .sort_values(ascending=False)
        .to_dict()
    )
    target_spearman = (
        spearman_df[target_col]
        .drop(target_col, errors="ignore")
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "pearson_matrix": pearson_df,
        "spearman_matrix": spearman_df,
        "target_pearson": target_pearson,
        "target_spearman": target_spearman,
    }


def compute_department_metrics(
    df: pd.DataFrame,
    target_col: str = "future_performance_score",
) -> pd.DataFrame:
    """Compute summary distribution metrics of target broken down by department.

    Args:
        df: Input DataFrame.
        target_col: Performance score column.

    Returns:
        DataFrame indexed by department with count, mean, std, median, and IQR.
    """
    validate_eda_input(df)

    grouped = df.groupby("department")[target_col]
    summary = grouped.agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        min="min",
        max="max",
    ).round(3)
    summary["iqr"] = (summary["q75"] - summary["q25"]).round(3)
    return summary.sort_values(by="mean", ascending=False)


def compute_job_role_metrics(
    df: pd.DataFrame,
    target_col: str = "future_performance_score",
) -> pd.DataFrame:
    """Compute performance score distribution summary per job role.

    Args:
        df: Input DataFrame.
        target_col: Performance score column.

    Returns:
        DataFrame indexed by job_role with parent department, count, mean, and std.
    """
    validate_eda_input(df)

    summary = (
        df.groupby(["job_role", "department"])[target_col]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            min="min",
            max="max",
        )
        .round(3)
        .reset_index()
    )
    return summary.sort_values(by=["department", "mean"], ascending=[True, False])


def compute_period_trends(
    df: pd.DataFrame,
    target_col: str = "future_performance_score",
) -> Dict[str, Any]:
    """Compute quarterly macroeconomic trend and individual within-employee dynamics.

    Args:
        df: Input DataFrame.
        target_col: Target variable.

    Returns:
        Dictionary with period-level metrics and within-employee transition stats.
    """
    validate_eda_input(df)

    # 1. Macro period-level trends
    period_summary = (
        df.groupby("period")[target_col]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            min="min",
            max="max",
        )
        .round(3)
        .to_dict(orient="index")
    )

    # 2. Within-employee longitudinal changes
    sorted_df = df.sort_values(by=["employee_id", "period"]).copy()
    sorted_df["delta_target"] = sorted_df.groupby("employee_id")[target_col].diff()

    deltas = sorted_df["delta_target"].dropna()
    total_transitions = int(len(deltas))

    mean_delta = float(deltas.mean()) if total_transitions > 0 else 0.0
    std_delta = float(deltas.std(ddof=1)) if total_transitions > 1 else 0.0
    median_delta = float(deltas.median()) if total_transitions > 0 else 0.0

    improved_count = int((deltas > 0.05).sum())
    declined_count = int((deltas < -0.05).sum())
    stable_count = total_transitions - improved_count - declined_count

    within_emp_metrics = {
        "total_transitions": total_transitions,
        "mean_change": round(mean_delta, 4),
        "std_change": round(std_delta, 4),
        "median_change": round(median_delta, 4),
        "improved_share": round(improved_count / total_transitions, 4) if total_transitions > 0 else 0.0,
        "declined_share": round(declined_count / total_transitions, 4) if total_transitions > 0 else 0.0,
        "stable_share": round(stable_count / total_transitions, 4) if total_transitions > 0 else 0.0,
    }

    return {
        "period_summary": period_summary,
        "within_employee_metrics": within_emp_metrics,
    }


def analyze_training_relationship(df: pd.DataFrame) -> Dict[str, Any]:
    """Investigate the negative marginal correlation of training hours vs conditional effect.

    Demonstrates confounding by indication: lower performers get assigned remedial training,
    inducing a negative overall correlation despite positive structural utility.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary explaining the marginal correlation vs subgroup correlations.
    """
    validate_eda_input(df)

    overall_corr = float(
        df["training_hours_previous_period"].corr(df["future_performance_score"])
    )

    # Segment employees by previous performance tier
    df_bins = df.copy()
    df_bins["perf_tier"] = pd.cut(
        df_bins["previous_performance_score"],
        bins=[0.0, 2.5, 3.5, 5.0],
        labels=["Low Tier (<=2.5)", "Mid Tier (2.5-3.5)", "High Tier (>3.5)"],
    )

    tier_stats = {}
    for tier, group in df_bins.groupby("perf_tier", observed=False):
        mean_train = float(group["training_hours_previous_period"].mean())
        mean_future = float(group["future_performance_score"].mean())
        if (
            len(group) >= 2
            and group["training_hours_previous_period"].std() > 0
            and group["future_performance_score"].std() > 0
        ):
            corr_val = float(
                group["training_hours_previous_period"].corr(group["future_performance_score"])
            )
        else:
            corr_val = 0.0

        tier_stats[str(tier)] = {
            "count": int(len(group)),
            "mean_training_hours": round(mean_train, 2),
            "mean_future_perf": round(mean_future, 2),
            "within_tier_correlation": round(corr_val, 4),
        }

    return {
        "marginal_correlation": round(overall_corr, 4),
        "tier_analysis": tier_stats,
    }
