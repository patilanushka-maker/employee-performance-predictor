"""Data validation suite for Employee Performance Predictor datasets.

Verifies schema adherence, column data types, missing values, duplicate records,
key uniqueness, numerical bounds, categorical validity, employee temporal stability,
and prediction target integrity without mutating data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from employee_performance.data_generator import DEPARTMENT_ROLES, DEPARTMENTS

REQUIRED_COLUMNS: List[str] = [
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

NUMERIC_COLUMNS: List[str] = [
    "period",
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

NUMERICAL_BOUNDS: Dict[str, tuple] = {
    "years_experience": (0.0, 50.0),
    "tenure_months": (0, 600),
    "training_hours_previous_period": (0.0, 200.0),
    "project_completion_rate_previous_period": (0.0, 1.0),
    "quality_score_previous_period": (0.0, 100.0),
    "productivity_score_previous_period": (0.0, 100.0),
    "attendance_rate_previous_period": (0.0, 1.0),
    "previous_performance_score": (1.0, 5.0),
    "future_performance_score": (1.0, 5.0),
}


@dataclass
class ValidationResult:
    """Holds structured outcome of a dataset validation run."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Format human-readable summary of the validation outcome."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Dataset Validation Status: {status}",
            f"Total Errors Found: {len(self.errors)}",
            f"Total Warnings: {len(self.warnings)}",
        ]
        if self.metrics:
            lines.append("Dataset Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"  - {k}: {v}")
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  [ERROR] {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  [WARNING] {warn}")
        return "\n".join(lines)


class DataValidator:
    """Validates employee performance datasets against enterprise data contracts."""

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        numeric_bounds: Optional[Dict[str, tuple]] = None,
    ):
        self.required_columns = required_columns or REQUIRED_COLUMNS
        self.numeric_bounds = numeric_bounds or NUMERICAL_BOUNDS

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Perform comprehensive data integrity and schema validation on a DataFrame.

        The validator never modifies or imputes data.

        Args:
            df: The DataFrame to inspect.

        Returns:
            ValidationResult with boolean validity, errors, warnings, and summary metrics.
        """
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        if not isinstance(df, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                errors=[f"Input must be a pandas DataFrame, received {type(df)}"],
            )

        metrics["row_count"] = int(len(df))
        metrics["column_count"] = int(df.shape[1])

        if df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["Dataset is empty (0 rows)."],
                metrics=metrics,
            )

        # 1. Required Columns Check
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            # If critical columns are missing, return early to prevent secondary exceptions
            return ValidationResult(is_valid=False, errors=errors, metrics=metrics)

        # 2. Data Types & Null Values Check
        for col in self.required_columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                errors.append(
                    f"Column '{col}' contains {null_count} missing (NaN/null) values."
                )

        # Check numeric conversions
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(
                        f"Column '{col}' is expected to be numeric, found dtype '{df[col].dtype}'."
                    )

        # 3. Duplicate Records Check
        duplicate_rows = int(df.duplicated().sum())
        metrics["duplicate_rows"] = duplicate_rows
        if duplicate_rows > 0:
            errors.append(f"Found {duplicate_rows} exact duplicate rows.")

        # 4. Composite Key Uniqueness Check: (employee_id, period)
        if "employee_id" in df.columns and "period" in df.columns:
            key_dups = int(df.duplicated(subset=["employee_id", "period"]).sum())
            metrics["duplicate_employee_period_keys"] = key_dups
            if key_dups > 0:
                errors.append(
                    f"Found {key_dups} duplicate (employee_id, period) composite keys."
                )

        # 5. Period Integrity & Ordering Check
        if "period" in df.columns and pd.api.types.is_numeric_dtype(df["period"]):
            unique_periods = sorted(df["period"].unique().tolist())
            metrics["unique_periods"] = unique_periods
            if any(p <= 0 for p in unique_periods):
                errors.append(f"Period values must be positive integers: found {unique_periods}")

        # 6. Numerical Range Validations
        for col, (low, high) in self.numeric_bounds.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                out_of_bounds = df[(df[col] < low) | (df[col] > high)]
                if len(out_of_bounds) > 0:
                    errors.append(
                        f"Column '{col}' has {len(out_of_bounds)} values outside "
                        f"expected bounds [{low}, {high}]. Min: {df[col].min()}, Max: {df[col].max()}."
                    )
                metrics[f"{col}_min"] = float(df[col].min())
                metrics[f"{col}_max"] = float(df[col].max())

        # 7. Categorical Domain Validations
        if "department" in df.columns:
            invalid_depts = df[~df["department"].isin(DEPARTMENTS)]["department"].unique()
            if len(invalid_depts) > 0:
                errors.append(f"Invalid departments detected: {invalid_depts.tolist()}")

        if "department" in df.columns and "job_role" in df.columns:
            for dept, allowed_roles in DEPARTMENT_ROLES.items():
                dept_mask = df["department"] == dept
                invalid_roles = df[dept_mask & (~df["job_role"].isin(allowed_roles))][
                    "job_role"
                ].unique()
                if len(invalid_roles) > 0:
                    errors.append(
                        f"Department '{dept}' contains invalid job roles: {invalid_roles.tolist()}"
                    )

        # 8. Employee Longitudinal Stability & Coverage
        if "employee_id" in df.columns:
            unique_employees = int(df["employee_id"].nunique())
            metrics["unique_employees"] = unique_employees

            # Department stability per employee
            dept_changes = df.groupby("employee_id")["department"].nunique()
            unstable_depts = dept_changes[dept_changes > 1]
            if len(unstable_depts) > 0:
                errors.append(
                    f"{len(unstable_depts)} employees changed department across observation periods."
                )

            # Monotonic tenure and experience checks per employee
            if "period" in df.columns:
                sorted_df = df.sort_values(by=["employee_id", "period"])

                # Period continuity check (periods must not have gaps, e.g. [1, 3])
                period_diffs = sorted_df.groupby("employee_id")["period"].diff().dropna()
                gapped_periods = (period_diffs != 1).sum()
                if gapped_periods > 0:
                    errors.append(
                        f"Detected {gapped_periods} non-sequential period jumps/gaps for employees."
                    )

                # Monotonic tenure check
                if "tenure_months" in df.columns:
                    tenure_diffs = sorted_df.groupby("employee_id")["tenure_months"].diff().dropna()
                    if (tenure_diffs < 0).any():
                        errors.append("Detected non-monotonic tenure (tenure decreased over time for some employees).")

                # Monotonic experience check
                if "years_experience" in df.columns:
                    exp_diffs = sorted_df.groupby("employee_id")["years_experience"].diff().dropna()
                    if (exp_diffs < 0).any():
                        errors.append("Detected non-monotonic experience (experience decreased over time for some employees).")

                # Longitudinal lag coherence check: row[t].previous_performance_score == row[t-1].future_performance_score
                if (
                    "previous_performance_score" in df.columns
                    and "future_performance_score" in df.columns
                    and pd.api.types.is_numeric_dtype(df["previous_performance_score"])
                    and pd.api.types.is_numeric_dtype(df["future_performance_score"])
                ):
                    sorted_df = sorted_df.copy()
                    sorted_df["lag_target"] = sorted_df.groupby("employee_id")["future_performance_score"].shift(1)
                    # For rows where period > min_period, previous_performance_score must match lag_target
                    lag_mask = sorted_df["lag_target"].notna()
                    if lag_mask.any():
                        score_diffs = (sorted_df.loc[lag_mask, "previous_performance_score"] - sorted_df.loc[lag_mask, "lag_target"]).abs()
                        mismatches = int((score_diffs > 1e-3).sum())
                        metrics["lag_score_mismatches"] = mismatches
                        if mismatches > 0:
                            errors.append(
                                f"Detected {mismatches} temporal lag inconsistencies where previous_performance_score "
                                f"in period t does not match future_performance_score in period t-1."
                            )

            # Career tenure plausibility check (tenure cannot exceed total career length)
            if (
                "tenure_months" in df.columns
                and "years_experience" in df.columns
                and pd.api.types.is_numeric_dtype(df["tenure_months"])
                and pd.api.types.is_numeric_dtype(df["years_experience"])
            ):
                max_allowed_tenure = (df["years_experience"] * 12) + 2.0  # allow 2 months grace for rounding
                excessive_tenure = (df["tenure_months"] > max_allowed_tenure).sum()
                if excessive_tenure > 0:
                    errors.append(
                        f"Detected {excessive_tenure} records where tenure_months exceeds lifetime career months (years_experience * 12)."
                    )

            # Descriptive distribution warnings
            if (
                "attendance_rate_previous_period" in df.columns
                and pd.api.types.is_numeric_dtype(df["attendance_rate_previous_period"])
            ):
                low_att = int((df["attendance_rate_previous_period"] < 0.75).sum())
                if low_att > 0:
                    warnings.append(f"{low_att} records have attendance rates below 0.75 (flagged for review).")

        # 9. Target Validity & Statistical Health
        if "future_performance_score" in df.columns and pd.api.types.is_numeric_dtype(
            df["future_performance_score"]
        ):
            target_std = float(df["future_performance_score"].std())
            metrics["target_std"] = target_std
            if target_std == 0.0:
                errors.append(
                    "Target 'future_performance_score' has zero variance (constant across all rows)."
                )

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """Convenience functional wrapper to run full validation on a DataFrame.

    Args:
        df: DataFrame to validate.

    Returns:
        ValidationResult with validation details.
    """
    validator = DataValidator()
    return validator.validate(df)
