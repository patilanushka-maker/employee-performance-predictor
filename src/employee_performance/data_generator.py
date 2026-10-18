"""Longitudinal synthetic employee data generator for Employee Performance Predictor.

Simulates workforce performance metrics across ordered observation periods with
realistic correlations, latent ability, autoregressive persistence, and stochastic noise.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Canonical categorical mappings for departments and job roles
DEPARTMENT_ROLES: Dict[str, List[str]] = {
    "Engineering": [
        "Software Engineer",
        "Senior Software Engineer",
        "QA Engineer",
        "DevOps Engineer",
    ],
    "Sales": [
        "Sales Executive",
        "Account Manager",
        "Business Development Representative",
    ],
    "Marketing": [
        "Marketing Specialist",
        "Content Strategist",
        "SEO Analyst",
    ],
    "Customer Support": [
        "Support Specialist",
        "Customer Success Manager",
    ],
    "Finance": [
        "Financial Analyst",
        "Senior Accountant",
    ],
    "Human Resources": [
        "HR Specialist",
        "Recruiter",
        "HR Business Partner",
    ],
    "Product": [
        "Product Manager",
        "UX Designer",
    ],
}

DEPARTMENTS = list(DEPARTMENT_ROLES.keys())
DEPARTMENT_PROBS = [0.28, 0.20, 0.14, 0.14, 0.08, 0.08, 0.08]


@dataclass
class GeneratorConfig:
    """Configuration parameters for synthetic data generation."""

    num_employees: int = 1000
    num_periods: int = 4
    random_seed: int = 42
    output_path: Optional[Union[str, Path]] = None


class SyntheticDataGenerator:
    """Generates longitudinally consistent synthetic employee performance records.

    Temporal Design:
    Each record represents an observation at the start of `period` t.
    All predictors ('..._previous_period' and 'previous_performance_score') reflect
    historical performance realized prior to period t. The target 'future_performance_score'
    represents the performance rating evaluated during period t.
    No future-period information is ever leaked into historical feature vectors.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate(self) -> pd.DataFrame:
        """Generate the full longitudinal dataset across all employees and periods.

        Returns:
            pd.DataFrame: Longitudinal dataset containing all configured periods.
        """
        num_emp = self.config.num_employees
        num_periods = self.config.num_periods

        if num_emp <= 0:
            raise ValueError(f"num_employees must be positive, got {num_emp}")
        if num_periods <= 0:
            raise ValueError(f"num_periods must be positive, got {num_periods}")

        # 1. Generate persistent baseline attributes per employee
        employee_ids = [f"EMP-{i + 1:04d}" for i in range(num_emp)]
        departments = self.rng.choice(DEPARTMENTS, size=num_emp, p=DEPARTMENT_PROBS)

        job_roles = [
            self.rng.choice(DEPARTMENT_ROLES[dept]) for dept in departments
        ]

        # Latent persistent ability trait: N(0, 1)
        base_ability = self.rng.normal(loc=0.0, scale=1.0, size=num_emp)

        # Baseline years of professional experience (clamped to 0.5 - 28.0)
        base_experience = np.clip(
            self.rng.gamma(shape=3.0, scale=2.2, size=num_emp) + 0.5,
            0.5,
            28.0,
        )

        # Initial tenure in months (must not exceed total career months)
        max_possible_tenure = (base_experience * 12).astype(int)
        initial_tenure = np.array([
            self.rng.integers(1, max(2, max_possible_tenure[i]))
            for i in range(num_emp)
        ])
        initial_tenure = np.clip(initial_tenure, 1, 300)

        # 2. Longitudinal simulation across periods
        records = []

        # Current state trackers for longitudinal evolution
        current_prev_perf = np.clip(
            3.0 + 0.45 * base_ability + self.rng.normal(0, 0.35, num_emp),
            1.0,
            5.0,
        )
        current_quality = np.clip(
            75.0 + 7.5 * base_ability + 0.2 * base_experience + self.rng.normal(0, 5.0, num_emp),
            40.0,
            100.0,
        )
        current_productivity = np.clip(
            72.0 + 7.0 * base_ability + self.rng.normal(0, 6.0, num_emp),
            40.0,
            100.0,
        )
        current_project_completion = np.clip(
            0.80 + 0.06 * base_ability + self.rng.normal(0, 0.08, num_emp),
            0.40,
            1.00,
        )
        current_attendance = np.clip(
            0.94 + 0.02 * base_ability + self.rng.normal(0, 0.03, num_emp),
            0.65,
            1.00,
        )
        current_training_hours = np.clip(
            self.rng.poisson(lam=12.0, size=num_emp) + self.rng.uniform(0, 4.0, num_emp),
            0.0,
            60.0,
        )

        for period in range(1, num_periods + 1):
            for i in range(num_emp):
                # Experience advances slightly each quarter (+0.25 years)
                exp_t = round(float(base_experience[i] + (period - 1) * 0.25), 2)
                # Tenure advances by 3 months per period
                tenure_t = int(initial_tenure[i] + (period - 1) * 3)

                # Predictor values (historical indicators available before period t)
                train_hrs = round(float(current_training_hours[i]), 1)
                proj_comp = round(float(current_project_completion[i]), 3)
                qual_score = round(float(current_quality[i]), 1)
                prod_score = round(float(current_productivity[i]), 1)
                att_rate = round(float(current_attendance[i]), 3)
                prev_perf = round(float(current_prev_perf[i]), 2)

                # Future target calculation for period t
                # Governed by autoregressive performance, quality, productivity, attendance,
                # training impact, latent ability, and stochastic innovation shock
                target_latent = (
                    3.15
                    + 0.40 * (prev_perf - 3.0)
                    + 0.014 * (qual_score - 75.0)
                    + 0.011 * (prod_score - 72.0)
                    + 0.70 * (proj_comp - 0.80)
                    + 0.008 * (train_hrs - 12.0)
                    + 0.90 * (att_rate - 0.94)
                    + 0.20 * base_ability[i]
                    + self.rng.normal(loc=0.0, scale=0.25)
                )
                future_perf = round(float(np.clip(target_latent, 1.0, 5.0)), 2)

                records.append({
                    "employee_id": employee_ids[i],
                    "period": period,
                    "department": departments[i],
                    "job_role": job_roles[i],
                    "years_experience": exp_t,
                    "tenure_months": tenure_t,
                    "training_hours_previous_period": train_hrs,
                    "project_completion_rate_previous_period": proj_comp,
                    "quality_score_previous_period": qual_score,
                    "productivity_score_previous_period": prod_score,
                    "attendance_rate_previous_period": att_rate,
                    "previous_performance_score": prev_perf,
                    "future_performance_score": future_perf,
                })

                # Longitudinal transition for period t + 1
                # Next period's historical indicators inherit realized performance
                current_prev_perf[i] = future_perf
                current_quality[i] = np.clip(
                    75.0 + 6.0 * (future_perf - 3.0) + self.rng.normal(0, 4.0),
                    40.0,
                    100.0,
                )
                current_productivity[i] = np.clip(
                    72.0 + 5.5 * (future_perf - 3.0) + self.rng.normal(0, 5.0),
                    40.0,
                    100.0,
                )
                current_project_completion[i] = np.clip(
                    0.80 + 0.05 * (future_perf - 3.0) + self.rng.normal(0, 0.06),
                    0.40,
                    1.00,
                )
                current_attendance[i] = np.clip(
                    0.94 + 0.015 * (future_perf - 3.0) + self.rng.normal(0, 0.025),
                    0.65,
                    1.00,
                )
                current_training_hours[i] = np.clip(
                    self.rng.poisson(lam=10.0 + max(0.0, (3.5 - future_perf) * 4.0)),
                    0.0,
                    60.0,
                )

        df = pd.DataFrame(records)

        # Sort explicitly by employee_id and period
        df = df.sort_values(by=["employee_id", "period"]).reset_index(drop=True)

        # Save to output path if specified
        if self.config.output_path:
            save_path = Path(self.config.output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)

        return df


def generate_synthetic_data(
    num_employees: int = 1000,
    num_periods: int = 4,
    random_seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Convenience functional wrapper to generate synthetic employee data.

    Args:
        num_employees: Total unique employees to simulate.
        num_periods: Number of ordered sequential observation periods.
        random_seed: RNG seed for reproducible generation.
        output_path: Optional file path to persist the CSV output.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    config = GeneratorConfig(
        num_employees=num_employees,
        num_periods=num_periods,
        random_seed=random_seed,
        output_path=output_path,
    )
    generator = SyntheticDataGenerator(config=config)
    return generator.generate()
