"""Feature engineering module for Employee Performance Predictor.

Provides scikit-learn compatible transformers and functional utilities for
creating domain-justified operational interactions and ratio predictors
with strict prediction-time alignment and zero target leakage.
"""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CANDIDATE_FEATURE_A = "quality_to_productivity_ratio"
CANDIDATE_FEATURE_B = "completion_attendance_interaction"
CANDIDATE_FEATURE_C = "prev_perf_training_interaction"
CANDIDATE_FEATURE_D = "experience_adjusted_tenure"

ALL_CANDIDATE_FEATURES: List[str] = [
    CANDIDATE_FEATURE_A,
    CANDIDATE_FEATURE_B,
    CANDIDATE_FEATURE_C,
    CANDIDATE_FEATURE_D,
]

REQUIRED_BASE_COLUMNS: Set[str] = {
    "quality_score_previous_period",
    "productivity_score_previous_period",
    "project_completion_rate_previous_period",
    "attendance_rate_previous_period",
    "previous_performance_score",
    "training_hours_previous_period",
    "years_experience",
    "tenure_months",
}

FORBIDDEN_TRANSFORM_COLUMNS: Set[str] = {
    "employee_id",
    "period",
    "future_performance_score",
}


def calculate_experience_adjusted_tenure(
    tenure_months: pd.Series,
    years_experience: pd.Series,
    min_career_months: float = 1.0,
) -> pd.Series:
    """Calculate experience-adjusted tenure: tenure_months / (12 * years_experience).

    Denominator Policy:
        In the synthetic dataset, years_experience ranges from 0.92 to 28.46 years,
        meaning denominators are strictly positive (> 11 months).
        However, for arbitrary or entry-level profiles where years_experience approaches
        or equals 0.0, dividing by 0 yields Inf or NaN.

        Defensible Denominator Policy:
        1. Career months is calculated as `12.0 * years_experience`.
        2. To prevent division by zero or near-zero singularities without fabricating arbitrary
           unscaled constants, the denominator is bounded from below by `min_career_months`
           (default: 1.0 month, representing the minimum discrete unit of employment tenure).
        3. Since an employee's organizational tenure cannot logically exceed their total lifetime
           career tenure, the resulting ratio is clipped to [0.0, 1.0].

    Args:
        tenure_months: Cumulative tenure at the employer in months.
        years_experience: Cumulative lifetime career experience in years.
        min_career_months: Floor applied to career months denominator (default: 1.0).

    Returns:
        pd.Series: Ratio bounded within [0.0, 1.0].
    """
    career_months = 12.0 * np.asarray(years_experience, dtype=float)
    # Enforce defensible denominator policy
    denom = np.maximum(career_months, min_career_months)
    tenure = np.asarray(tenure_months, dtype=float)
    ratio = tenure / denom
    # Clip to valid domain [0.0, 1.0]
    return pd.Series(np.clip(ratio, 0.0, 1.0), index=tenure_months.index, name=CANDIDATE_FEATURE_D)


def calculate_quality_to_productivity_ratio(
    quality_score: pd.Series,
    productivity_score: pd.Series,
    min_productivity: float = 1e-4,
) -> pd.Series:
    """Calculate quality-to-productivity ratio: quality / productivity.

    Denominator Policy:
        In the synthetic dataset, productivity scores span [40.0, 99.5].
        For general safety against zero productivity scores, the denominator is
        bounded from below by `min_productivity` (1e-4).

    Args:
        quality_score: Quality score in preceding period (0-100 scale).
        productivity_score: Productivity score in preceding period (0-100 scale).
        min_productivity: Denominator floor to avoid division by zero.

    Returns:
        pd.Series: Quality-to-productivity ratio.
    """
    denom = np.maximum(np.asarray(productivity_score, dtype=float), min_productivity)
    ratio = np.asarray(quality_score, dtype=float) / denom
    return pd.Series(ratio, index=quality_score.index, name=CANDIDATE_FEATURE_A)


def calculate_completion_attendance_interaction(
    project_completion_rate: pd.Series,
    attendance_rate: pd.Series,
) -> pd.Series:
    """Calculate task delivery efficiency interaction: completion_rate * attendance_rate.

    Args:
        project_completion_rate: Project completion rate in preceding period ([0.0, 1.0]).
        attendance_rate: Attendance rate in preceding period ([0.0, 1.0]).

    Returns:
        pd.Series: Interaction product ([0.0, 1.0]).
    """
    prod = np.asarray(project_completion_rate, dtype=float) * np.asarray(attendance_rate, dtype=float)
    return pd.Series(prod, index=project_completion_rate.index, name=CANDIDATE_FEATURE_B)


def calculate_prev_perf_training_interaction(
    previous_performance_score: pd.Series,
    training_hours: pd.Series,
) -> pd.Series:
    """Calculate prior performance and training hours interaction.

    Justification:
        EDA revealed confounding by indication: struggling performers receive higher
        volumes of mandatory remedial training. This interaction term allows models
        to capture the conditional relationship between training hours and prior appraisal rating.

    Args:
        previous_performance_score: Performance rating at t-1 ([1.0, 5.0]).
        training_hours: Professional training hours completed at t-1 ([0.0, 60.0]).

    Returns:
        pd.Series: Interaction product.
    """
    prod = np.asarray(previous_performance_score, dtype=float) * np.asarray(training_hours, dtype=float)
    return pd.Series(prod, index=previous_performance_score.index, name=CANDIDATE_FEATURE_C)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for controlled feature engineering.

    Computes justified candidate features from historical predictor columns
    available strictly before the start of period t.

    Performance Momentum Clarification:
        Quarter-over-quarter score delta (momentum: score_{t-1} - score_{t-2})
        fundamentally requires two prior observation quarters. In this longitudinal dataset,
        Period 1 contains only t-1 baseline indicators (no t-2 score exists).
        Attempting to calculate multi-period momentum produces 1,000 missing values
        (100% of Period 1), making expanding-window temporal validation unviable on Fold 1.
        Therefore, multi-period momentum is excluded because it is not uniformly available
        at the intended prediction moment across all observation quarters.
    """

    def __init__(
        self,
        features_to_add: Optional[List[str]] = None,
        min_career_months: float = 1.0,
        min_productivity: float = 1e-4,
    ):
        """Initialize FeatureEngineer.

        Args:
            features_to_add: List of candidate feature names to engineer.
                Defaults to ALL_CANDIDATE_FEATURES if None.
            min_career_months: Denominator floor for experience-adjusted tenure.
            min_productivity: Denominator floor for quality-to-productivity ratio.
        """
        self.features_to_add = features_to_add if features_to_add is not None else list(ALL_CANDIDATE_FEATURES)
        self.min_career_months = min_career_months
        self.min_productivity = min_productivity

        # Validate candidate names
        invalid_feats = [f for f in self.features_to_add if f not in ALL_CANDIDATE_FEATURES]
        if invalid_feats:
            raise ValueError(f"Unknown candidate features: {invalid_feats}. Allowed: {ALL_CANDIDATE_FEATURES}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit transformer (stateless; verifies schema compatibility).

        Args:
            X: Input feature DataFrame.
            y: Ignored (zero target leakage).

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        # Verify no forbidden target/id columns are passed as features
        forbidden_present = FORBIDDEN_TRANSFORM_COLUMNS.intersection(X.columns)
        if forbidden_present:
            raise ValueError(
                f"Feature matrix X must not contain forbidden identifier or target columns: {forbidden_present}"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations.

        Args:
            X: Input feature DataFrame.

        Returns:
            pd.DataFrame: New DataFrame containing original features plus requested candidate features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        forbidden_present = FORBIDDEN_TRANSFORM_COLUMNS.intersection(X.columns)
        if forbidden_present:
            raise ValueError(
                f"Feature matrix X must not contain forbidden identifier or target columns: {forbidden_present}"
            )

        # Verify required source columns
        missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in X.columns]
        if missing_base:
            raise ValueError(f"X is missing required base columns for feature engineering: {missing_base}")

        out = X.copy()

        # A. Quality to productivity ratio
        if CANDIDATE_FEATURE_A in self.features_to_add:
            out[CANDIDATE_FEATURE_A] = calculate_quality_to_productivity_ratio(
                quality_score=X["quality_score_previous_period"],
                productivity_score=X["productivity_score_previous_period"],
                min_productivity=self.min_productivity,
            )

        # B. Completion attendance interaction
        if CANDIDATE_FEATURE_B in self.features_to_add:
            out[CANDIDATE_FEATURE_B] = calculate_completion_attendance_interaction(
                project_completion_rate=X["project_completion_rate_previous_period"],
                attendance_rate=X["attendance_rate_previous_period"],
            )

        # C. Previous performance training interaction
        if CANDIDATE_FEATURE_C in self.features_to_add:
            out[CANDIDATE_FEATURE_C] = calculate_prev_perf_training_interaction(
                previous_performance_score=X["previous_performance_score"],
                training_hours=X["training_hours_previous_period"],
            )

        # D. Experience-adjusted tenure
        if CANDIDATE_FEATURE_D in self.features_to_add:
            out[CANDIDATE_FEATURE_D] = calculate_experience_adjusted_tenure(
                tenure_months=X["tenure_months"],
                years_experience=X["years_experience"],
                min_career_months=self.min_career_months,
            )

        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return output feature names for scikit-learn pipeline inspection."""
        base_names = list(input_features) if input_features is not None else list(REQUIRED_BASE_COLUMNS)
        return base_names + [f for f in self.features_to_add if f not in base_names]
