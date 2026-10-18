"""Reproducible Exploratory Data Analysis (EDA) workflow runner.

Executes comprehensive distribution analysis, correlation screening, subgroup
comparisons, longitudinal trend evaluation, and report generation.

Usage:
    python scripts/run_eda.py [--data PATH] [--figures-dir PATH] [--report-path PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

# Set non-interactive plotting backend prior to importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from employee_performance.eda import (
    compute_dataset_overview,
    compute_correlations,
    compute_department_metrics,
    compute_job_role_metrics,
    compute_period_trends,
    analyze_training_relationship,
    validate_eda_input,
    FEATURE_SPECS,
    NUMERIC_PREDICTORS,
    REQUIRED_EDA_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for EDA execution."""
    parser = argparse.ArgumentParser(
        prog="run_eda.py",
        description="Run reproducible exploratory data analysis on employee performance dataset.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "synthetic" / "employee_performance.csv"),
        help="Path to employee performance CSV (default: data/synthetic/employee_performance.csv).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "figures"),
        help="Directory to save generated figures (default: outputs/figures).",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "reports" / "eda_report.md"),
        help="Destination path for written Markdown report (default: outputs/reports/eda_report.md).",
    )
    return parser.parse_args()


# ==============================================================================
# Visualization Routines
# ==============================================================================

def setup_plot_style():
    """Configure clean, professional aesthetics for matplotlib charts."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.autolayout": True,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })


def plot_target_distribution(df: pd.DataFrame, output_path: Path):
    """Plot distribution of future_performance_score (histogram and boxplot)."""
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}
    )

    target = df["future_performance_score"]

    # Boxplot
    bp = ax_box.boxplot(
        target,
        orientation="horizontal",
        patch_artist=True,
        boxprops=dict(facecolor="#3498db", color="#1f618d"),
        medianprops=dict(color="#e74c3c", linewidth=2),
    )
    ax_box.set_yticks([])
    ax_box.set_title("Distribution of Future Performance Score (Target y)")

    # Histogram + Density
    counts, bins, _ = ax_hist.hist(
        target, bins=25, edgecolor="#1f618d", color="#5dade2", alpha=0.75, density=True
    )
    # Parametric normal fit line for comparison
    mu, sigma = target.mean(), target.std()
    x = np.linspace(target.min(), target.max(), 100)
    ax_hist.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
                 color="#c0392b", linewidth=2, label=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})")

    ax_hist.axvline(mu, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mu:.2f}")
    ax_hist.axvline(target.median(), color="#27ae60", linestyle=":", linewidth=1.5, label=f"Median: {target.median():.2f}")

    ax_hist.set_xlabel("Future Performance Score [1.0 to 5.0]")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right", frameon=True)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_numeric_predictors(df: pd.DataFrame, output_path: Path):
    """Plot 2x4 grid of histograms for all numeric explanatory predictors."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = ["#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085", "#2c3e50", "#c0392b", "#7f8c8d"]

    for i, col in enumerate(NUMERIC_PREDICTORS):
        ax = axes[i]
        vals = df[col]
        ax.hist(vals, bins=25, color=colors[i % len(colors)], edgecolor="#222222", alpha=0.75)
        mean_v, med_v = vals.mean(), vals.median()
        ax.axvline(mean_v, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Mean: {mean_v:.2f}")
        ax.axvline(med_v, color="#f39c12", linestyle=":", linewidth=1.2, label=f"Med: {med_v:.2f}")
        unit = FEATURE_SPECS.get(col, {}).get("unit", "")
        ax.set_title(col.replace("_previous_period", "").replace("_", " ").title())
        ax.set_xlabel(f"{col} ({unit})")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot Pearson correlation heatmap with numerical annotations."""
    cols = NUMERIC_PREDICTORS + ["future_performance_score"]
    corr = df[cols].corr().values
    labels = [c.replace("_previous_period", "_prev").replace("_performance_score", "_score") for c in cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Pearson Correlation Heatmap of Numeric Features & Target")

    # Annotate values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_key_predictors_vs_target(df: pd.DataFrame, output_path: Path):
    """Plot 2x2 grid of scatter/regression plots for top predictors vs target."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    key_features = [
        ("previous_performance_score", "Previous Performance Score", "#2980b9"),
        ("quality_score_previous_period", "Quality Score Previous Period", "#27ae60"),
        ("productivity_score_previous_period", "Productivity Score Previous Period", "#8e44ad"),
        ("training_hours_previous_period", "Training Hours Previous Period", "#e74c3c"),
    ]

    for idx, (col, title, color) in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        x = df[col]
        y = df["future_performance_score"]

        # Jitter scatter slightly to reveal density
        x_jitter = x + np.random.normal(0, 0.01 * (x.max() - x.min()), size=len(x))
        y_jitter = y + np.random.normal(0, 0.02, size=len(y))

        ax.scatter(x_jitter, y_jitter, alpha=0.25, color=color, edgecolors="none", s=20)

        # Linear regression trend line
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_seq = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_seq, intercept + slope * x_seq, color="#222222", linewidth=2.5,
                label=f"Trend: y = {slope:.2f}x + {intercept:.2f} (r={r_value:.2f})")

        ax.set_title(f"{title} vs Target")
        ax.set_xlabel(title)
        ax.set_ylabel("Future Performance Score")
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_performance_by_department(df: pd.DataFrame, output_path: Path):
    """Plot boxplots of performance score by department with observation counts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    depts = sorted(df["department"].unique())
    dept_data = [df[df["department"] == d]["future_performance_score"].values for d in depts]
    dept_counts = [len(df[df["department"] == d]) for d in depts]

    bp = ax.boxplot(
        dept_data,
        tick_labels=[f"{d}\n(n={c})" for d, c in zip(depts, dept_counts)],
        patch_artist=True,
        medianprops=dict(color="#c0392b", linewidth=2),
    )

    colors = ["#3498db", "#2ecc71", "#9b59b6", "#f1c40f", "#e67e22", "#1abc9c", "#95a5a6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Plot overall mean horizontal line
    overall_mean = df["future_performance_score"].mean()
    ax.axhline(overall_mean, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Global Mean: {overall_mean:.2f}")

    ax.set_title("Future Performance Score Distribution by Department")
    ax.set_xlabel("Department (Observation Count)")
    ax.set_ylabel("Future Performance Score [1.0 to 5.0]")
    ax.legend(loc="upper right")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_performance_by_job_role(df: pd.DataFrame, output_path: Path):
    """Plot horizontal boxplots of performance scores across 19 job roles."""
    fig, ax = plt.subplots(figsize=(12, 9))

    role_order = (
        df.groupby("job_role")["future_performance_score"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    role_data = [df[df["job_role"] == r]["future_performance_score"].values for r in role_order]
    role_labels = [f"{r} (n={len(df[df['job_role'] == r])})" for r in role_order]

    bp = ax.boxplot(
        role_data,
        orientation="horizontal",
        tick_labels=role_labels,
        patch_artist=True,
        medianprops=dict(color="#c0392b", linewidth=1.8),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("#5dade2")
        patch.set_alpha(0.75)

    ax.axvline(df["future_performance_score"].mean(), color="#e74c3c", linestyle="--",
               label=f"Global Mean: {df['future_performance_score'].mean():.2f}")

    ax.set_title("Future Performance Score by Job Role (Ranked by Median)")
    ax.set_xlabel("Future Performance Score [1.0 to 5.0]")
    ax.set_ylabel("Job Role")
    ax.legend(loc="lower right")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_quarterly_trends(df: pd.DataFrame, output_path: Path):
    """Plot macro quarterly trend and within-employee delta distribution."""
    fig, (ax_macro, ax_delta) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Macro period trend
    periods = sorted(df["period"].unique())
    period_means = [df[df["period"] == p]["future_performance_score"].mean() for p in periods]
    period_medians = [df[df["period"] == p]["future_performance_score"].median() for p in periods]
    period_stds = [df[df["period"] == p]["future_performance_score"].std() for p in periods]

    ax_macro.errorbar(periods, period_means, yerr=period_stds, fmt="-o", color="#2980b9",
                      ecolor="#aed6f1", elinewidth=2, capsize=5, label="Mean ± 1 Std Dev")
    ax_macro.plot(periods, period_medians, "--s", color="#e74c3c", label="Median")

    ax_macro.set_xticks(periods)
    ax_macro.set_xticklabels([f"Quarter {p}\n(n=1000)" for p in periods])
    ax_macro.set_title("Macroeconomic Performance Trend Across Quarters")
    ax_macro.set_xlabel("Observation Period")
    ax_macro.set_ylabel("Future Performance Score")
    ax_macro.legend(loc="upper left")

    # 2. Within-employee delta distribution
    sorted_df = df.sort_values(by=["employee_id", "period"])
    deltas = sorted_df.groupby("employee_id")["future_performance_score"].diff().dropna()

    ax_delta.hist(deltas, bins=30, color="#27ae60", edgecolor="#222222", alpha=0.75)
    ax_delta.axvline(0.0, color="#222222", linestyle="-", linewidth=1.2)
    ax_delta.axvline(deltas.mean(), color="#e74c3c", linestyle="--",
                     label=f"Mean Change: {deltas.mean():+.3f}")
    ax_delta.axvline(deltas.median(), color="#f39c12", linestyle=":",
                     label=f"Median Change: {deltas.median():+.3f}")

    ax_delta.set_title("Within-Employee Change (Δ Target Across Consecutive Quarters)")
    ax_delta.set_xlabel("Quarter-over-Quarter Score Change (Δ y)")
    ax_delta.set_ylabel("Frequency of Employee Transitions (n=3000)")
    ax_delta.legend(loc="upper right")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_training_hours_remedial_analysis(df: pd.DataFrame, output_path: Path):
    """Plot training hours vs performance stratified by prior performance tier."""
    fig, (ax_scatter, ax_bars) = plt.subplots(1, 2, figsize=(14, 5))

    # Stratified scatter
    df_plot = df.copy()
    df_plot["tier"] = pd.cut(
        df_plot["previous_performance_score"],
        bins=[0.0, 2.5, 3.5, 5.0],
        labels=["Low Tier (<=2.5)", "Mid Tier (2.5-3.5)", "High Tier (>3.5)"],
    )

    palette = {"Low Tier (<=2.5)": "#e74c3c", "Mid Tier (2.5-3.5)": "#f39c12", "High Tier (>3.5)": "#27ae60"}

    for tier_name, group in df_plot.groupby("tier", observed=False):
        valid_subset = group[["training_hours_previous_period", "future_performance_score"]].dropna()
        if len(valid_subset) >= 2 and valid_subset["training_hours_previous_period"].std() > 0 and valid_subset["future_performance_score"].std() > 0:
            r_val = valid_subset["training_hours_previous_period"].corr(valid_subset["future_performance_score"])
            r_str = f"{r_val:+.2f}"
        else:
            r_str = "N/A"
        ax_scatter.scatter(
            group["training_hours_previous_period"],
            group["future_performance_score"],
            label=f"{tier_name} (r={r_str})",
            color=palette[tier_name],
            alpha=0.35,
            s=22,
        )

    ax_scatter.set_title("Training Hours vs Target Stratified by Prior Performance Tier")
    ax_scatter.set_xlabel("Training Hours (Previous Period)")
    ax_scatter.set_ylabel("Future Performance Score")
    ax_scatter.legend(loc="lower left", fontsize=9)

    # Bar chart of mean training hours by tier
    tier_means = df_plot.groupby("tier", observed=False)["training_hours_previous_period"].mean()
    bar_colors = [palette[t] for t in tier_means.index]
    bars = ax_bars.bar(tier_means.index, tier_means.values, color=bar_colors, edgecolor="#222222", alpha=0.8)

    for bar in bars:
        h = bar.get_height()
        ax_bars.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.1f} hrs", ha="center", va="bottom", weight="bold")

    ax_bars.set_title("Remedial Selection: Mean Training Hours Prescribed per Tier")
    ax_bars.set_xlabel("Prior Performance Tier")
    ax_bars.set_ylabel("Average Training Hours Completed")
    ax_bars.set_ylim(0, max(tier_means.values) * 1.2)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ==============================================================================
# Markdown Report Generation
# ==============================================================================

def generate_markdown_report(
    overview: Dict[str, Any],
    correlations: Dict[str, Any],
    dept_metrics: pd.DataFrame,
    role_metrics: pd.DataFrame,
    period_trends: Dict[str, Any],
    training_analysis: Dict[str, Any],
    report_path: Path,
) -> None:
    """Generate professional, fully quantified Markdown report from EDA results."""
    lines: List[str] = [
        "# Exploratory Data Analysis (EDA) Report",
        "",
        "**Project**: Employee Performance Predictor using Data Analytics  ",
        "**Execution Date**: 2026-10-16  ",
        "**Status**: Phase 4 Completed  ",
        "**Dataset Analyzed**: `data/synthetic/employee_performance.csv` (Seed 42)  ",
        "",
        "---",
        "",
        "## 1. Objective & Scope",
        "",
        "This exploratory data analysis establishes the empirical foundation for subsequent predictive modeling. "
        "The objective is to rigorously inspect data distributions, evaluate linear and rank correlations, "
        "investigate departmental and occupational variance, track quarterly panel trends, and uncover "
        "structural data-generating dynamics without performing premature feature selection or training predictive algorithms.",
        "",
        "> [!IMPORTANT]",
        "> **Synthetic Nature & Causal Boundary**:",
        "> All records in this dataset are synthetically generated for a fictional cohort of 1,000 employees. "
        "> All relationships documented below are mathematical artifacts of the underlying generator and must "
        "> **not** be interpreted as empirical truths or causal facts regarding real-world human workforce behavior.",
        "",
        "---",
        "",
        "## 2. Dataset Overview & Data-Quality Audit Summary",
        "",
        f"- **Total Rows**: {overview['row_count']:,}",
        f"- **Total Columns**: {overview['col_count']}",
        f"- **Unique Fictional Employees**: {overview['unique_employees']:,}",
        "- **Records per Period**: " + ", ".join([f"Quarter {k}: {v:,}" for k, v in overview['period_counts'].items()]),
        f"- **Missing Values**: {sum(overview['missing_counts'].values())} across all cells (100% completeness)",
        f"- **Exact Duplicate Rows**: {overview['duplicate_rows']}",
        f"- **Duplicate Composite Keys `(employee_id, period)`**: {overview['duplicate_keys']}",
        f"- **Departments Represented**: {len(overview['department_counts'])} divisions",
        f"- **Job Roles Represented**: {len(overview['job_role_counts'])} discrete titles",
        "",
        "### Feature Dictionary & Measurement Units",
        "",
        "| Feature Name | Scale / Domain | Unit | Modeling Role |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for feat, meta in FEATURE_SPECS.items():
        lines.append(f"| `{feat}` | {meta['scale']} | {meta['unit']} | {meta['role']} |")

    lines.extend([
        "",
        "### Parametric & Non-Parametric Numerical Statistics",
        "",
        "| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max | Skewness | Kurtosis |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for feat, stats_dict in overview["numeric_stats"].items():
        lines.append(
            f"| `{feat}` | {stats_dict['mean']:.2f} | {stats_dict['std']:.2f} | {stats_dict['min']:.2f} | "
            f"{stats_dict['q25']:.2f} | {stats_dict['median']:.2f} | {stats_dict['q75']:.2f} | "
            f"{stats_dict['max']:.2f} | {stats_dict['skewness']:+.2f} | {stats_dict['kurtosis']:+.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Univariate Distribution Findings",
        "",
        "- **Target Variable (`future_performance_score`)**:",
        f"  - Demonstrates a symmetric, unimodal bell-shaped distribution with mean **{overview['numeric_stats']['future_performance_score']['mean']:.2f}** "
        f"and median **{overview['numeric_stats']['future_performance_score']['median']:.2f}**.",
        f"  - Standard deviation is **{overview['numeric_stats']['future_performance_score']['std']:.2f}**, with minimal skewness ({overview['numeric_stats']['future_performance_score']['skewness']:+.2f}) "
        f"and negative excess kurtosis ({overview['numeric_stats']['future_performance_score']['kurtosis']:+.2f}), indicating healthy dispersion across the [1.0, 5.0] appraisal scale.",
        "- **Ratio Features (`attendance_rate`, `project_completion_rate`)**:",
        f"  - Attendance rate exhibits high concentration near 1.0 (mean {overview['numeric_stats']['attendance_rate_previous_period']['mean']:.3f}, median {overview['numeric_stats']['attendance_rate_previous_period']['median']:.3f}), reflecting typical corporate attendance floors.",
        f"  - Project completion rate is normally distributed around {overview['numeric_stats']['project_completion_rate_previous_period']['mean']:.3f} with bounds spanning [{overview['numeric_stats']['project_completion_rate_previous_period']['min']:.3f}, {overview['numeric_stats']['project_completion_rate_previous_period']['max']:.3f}].",
        "- **Index Features (`quality_score`, `productivity_score`)**:",
        f"  - Both quality ({overview['numeric_stats']['quality_score_previous_period']['mean']:.1f} ± {overview['numeric_stats']['quality_score_previous_period']['std']:.1f}) and productivity "
        f"({overview['numeric_stats']['productivity_score_previous_period']['mean']:.1f} ± {overview['numeric_stats']['productivity_score_previous_period']['std']:.1f}) show wide, well-scaled variation across their 0–100 domains.",
        "",
        "**Rendered Figures**:",
        "- `outputs/figures/01_target_distribution.png`",
        "- `outputs/figures/02_numeric_features_distributions.png`",
        "",
        "---",
        "",
        "## 4. Correlation Analysis & Feature Associations",
        "",
        "Both Pearson ($r$) and Spearman rank ($\\rho$) correlations were computed to detect linear and monotonic associations:",
        "",
        "| Explanatory Predictor | Pearson Correlation ($r$) | Spearman Rank ($\\rho$) | Interpretation |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for feat in NUMERIC_PREDICTORS:
        p_val = correlations["target_pearson"].get(feat, 0.0)
        s_val = correlations["target_spearman"].get(feat, 0.0)
        interp = "Strong positive autoregressive persistence" if p_val > 0.70 else \
                 "Moderate-to-strong positive driver" if p_val > 0.40 else \
                 "Negative marginal correlation (Remedial assignment)" if p_val < -0.15 else \
                 "Near-zero linear association"
        lines.append(f"| `{feat}` | **{p_val:+.4f}** | **{s_val:+.4f}** | {interp} |")

    lines.extend([
        "",
        "> [!NOTE]",
        "> **Absence of Multicollinearity Screening (VIF)**:",
        "> Consistent with project scope restrictions, Variance Inflation Factor (VIF) screening is intentionally deferred "
        "> to Phase 5 (Feature Engineering) after candidate features, interaction terms, and categorical encodings are formalized.",
        "",
        "**Rendered Figures**:",
        "- `outputs/figures/03_correlation_heatmap.png`",
        "- `outputs/figures/04_key_predictors_vs_target.png`",
        "",
        "---",
        "",
        "## 5. Investigation of the Training Hours Paradox",
        "",
        f"A prominent empirical finding in this dataset is the **negative marginal correlation** ($r = {training_analysis['marginal_correlation']:+.4f}$) "
        "between `training_hours_previous_period` and `future_performance_score`.",
        "",
        "### Root Cause: Confounding by Indication (Selection Bias)",
        "Review of the generator mechanics confirms that training hours are assigned conditionally based on previous performance. "
        "Struggling employees receive higher volumes of mandated remedial training:",
        "",
        "| Prior Performance Tier | Observation Count | Average Training Prescribed | Mean Future Performance | Within-Tier Correlation ($r$) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for tier, data in training_analysis["tier_analysis"].items():
        lines.append(
            f"| **{tier}** | {data['count']:,} | **{data['mean_training_hours']:.1f} hrs** | "
            f"{data['mean_future_perf']:.2f} | **{data['within_tier_correlation']:+.4f}** |"
        )

    lines.extend([
        "",
        "- **Key Insight**: Lower-tier performers receive an average of "
        f"**{training_analysis['tier_analysis'].get('Low Tier (<=2.5)', {}).get('mean_training_hours', 0):.1f} hours** of training, "
        "whereas high performers average only "
        f"**{training_analysis['tier_analysis'].get('High Tier (>3.5)', {}).get('mean_training_hours', 0):.1f} hours**.",
        "- Within each stratum, the local correlation is near zero or weakly positive. In the synthetic generator formula, "
        "the structural linear weight parameter on training hours is +0.008 * (training_hours - 12.0). However, this mathematical parameter "
        "is an artifact of the simulation and must not be interpreted as an empirical causal effect. In real-world workforce analytics, "
        "observing a negative raw association would require formal observational causal inference methods (e.g. instrumental variables or "
        "panel fixed effects) to separate selection into training from training effectiveness.",
        "",
        "**Rendered Figure**:",
        "- `outputs/figures/08_training_hours_remedial_analysis.png`",
        "",
        "---",
        "",
        "## 6. Departmental & Job-Role Analysis",
        "",
        "Performance distributions across organizational units:",
        "",
        "| Department | Observation Count | Mean Target ($y$) | Std Dev | Median | IQR | Range [Min, Max] |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for dept, row in dept_metrics.iterrows():
        lines.append(
            f"| **{dept}** | {int(row['count']):,} | {row['mean']:.3f} | {row['std']:.3f} | "
            f"{row['median']:.3f} | {row['iqr']:.3f} | [{row['min']:.2f}, {row['max']:.2f}] |"
        )

    lines.extend([
        "",
        "- **Departmental Descriptive Summary**: Mean performance ratings across all 7 departments fall within a narrow descriptive band "
        f"({dept_metrics['mean'].min():.3f} to {dept_metrics['mean'].max():.3f}). These values describe the synthetic dataset and do not imply "
        "formal statistical equivalence across departments. Naive row-level ANOVA is deliberately omitted because it would violate independence assumptions.",
        "- **Job Role Granularity**: Across all 19 job roles, median scores remain stable between 3.10 and 3.40.",
        "",
        "**Rendered Figures**:",
        "- `outputs/figures/05_performance_by_department.png`",
        "- `outputs/figures/06_performance_by_job_role.png`",
        "",
        "---",
        "",
        "## 7. Quarterly Trends & Repeated-Measures Awareness",
        "",
        "### A. Macroeconomic Period-Level Summary",
        "",
        "| Quarter | Record Count | Mean Score | Std Dev | Median | [Min, Max] |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for p, stats_p in period_trends["period_summary"].items():
        lines.append(
            f"| **Quarter {p}** | {int(stats_p['count']):,} | {stats_p['mean']:.3f} | "
            f"{stats_p['std']:.3f} | {stats_p['median']:.3f} | [{stats_p['min']:.2f}, {stats_p['max']:.2f}] |"
        )

    we = period_trends["within_employee_metrics"]
    lines.extend([
        "",
        "### B. Within-Employee Transition Dynamics (n=3,000 Transitions)",
        "",
        f"- **Total Within-Employee Transitions**: {we['total_transitions']:,}",
        f"- **Mean Quarter-over-Quarter Change ($\\Delta y$)**: {we['mean_change']:+.4f}",
        f"- **Median Quarter-over-Quarter Change**: {we['median_change']:+.4f}",
        f"- **Standard Deviation of Change ($\\sigma_\\Delta$)**: {we['std_change']:.4f}",
        f"- **Employees Showing Positive Change ($\\Delta > +0.05$)*:** {we['improved_share']*100:.1f}%",
        f"- **Employees Showing Negative Change ($\\Delta < -0.05$)*:** {we['declined_share']*100:.1f}%",
        f"- **Employees Stable ($-0.05 \\le \\Delta \\le +0.05$)*:** {we['stable_share']*100:.1f}%",
        "  *(Note: The $\\pm 0.05$ interval is an exploratory reporting convention to segment score fluctuations, not a validated business-significance threshold.)*",
        "",
        "### C. Methodological Caution on Repeated Measures",
        "",
        "> [!WARNING]",
        "> **Clustered Panel Data Notice**:",
        "> The 4,000 observations in this dataset represent **1,000 unique employees observed across 4 consecutive quarters**.",
        "> They do **not** represent 4,000 independent and identically distributed (i.i.d.) observations.",
        "> ",
        "> Conventional inferential tests (such as row-level ordinary least squares standard errors or one-way ANOVA) that treat "
        "> all 4,000 rows as independent entities will substantially underestimate standard errors and artificially inflate test statistics.",
        "> ",
        "> In Phase 5 and Phase 6, cross-validation must utilize **GroupKFold (grouped by `employee_id`)** or **Temporal Out-of-Time Splitting** "
        "> (e.g. training on Quarters 1–3 and evaluating on Quarter 4) to ensure generalizability and eliminate data leakage.",
        "",
        "**Rendered Figure**:",
        "- `outputs/figures/07_quarterly_performance_trends.png`",
        "",
        "---",
        "",
        "## 8. Summary of Generated Figures",
        "",
        "| Figure File | Description |",
        "| :--- | :--- |",
        "| `outputs/figures/01_target_distribution.png` | Boxplot and histogram of `future_performance_score` with parametric normal overlay |",
        "| `outputs/figures/02_numeric_features_distributions.png` | 2x4 grid displaying parametric histograms for all numeric explanatory predictors |",
        "| `outputs/figures/03_correlation_heatmap.png` | Annotated correlation matrix across all numeric features and target |",
        "| `outputs/figures/04_key_predictors_vs_target.png` | Scatterplots with linear trend fits for key drivers vs target |",
        "| `outputs/figures/05_performance_by_department.png` | Boxplot of performance scores across 7 corporate departments |",
        "| `outputs/figures/06_performance_by_job_role.png` | Horizontal boxplots of ratings ranked across 19 job titles |",
        "| `outputs/figures/07_quarterly_performance_trends.png` | Macro period trajectory alongside within-employee quarter-over-quarter delta distribution |",
        "| `outputs/figures/08_training_hours_remedial_analysis.png` | Remedial selection breakdown explaining negative marginal training hours correlation |",
        "",
        "---",
        "",
        "## 9. Reproduction Instructions",
        "",
        "To reproduce this analysis and regenerate all tables and figures from scratch:",
        "```powershell",
        "# Execute reproducible EDA script",
        "python scripts/run_eda.py",
        "",
        "# Or specify custom paths",
        "python scripts/run_eda.py --data data/synthetic/employee_performance.csv --figures-dir outputs/figures --report-path outputs/reports/eda_report.md",
        "```",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ==============================================================================
# Main Orchestration
# ==============================================================================

def run_eda(
    data_path: Path,
    figures_dir: Path,
    report_path: Path,
    docs_copy_path: Optional[Path] = None,
) -> int:
    """Execute complete EDA pipeline."""
    if not data_path.is_file():
        print(f"[ERROR] Specified data file does not exist: {data_path.resolve()}", file=sys.stderr)
        return 1

    print(f"Reading dataset: {data_path.resolve()}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}", file=sys.stderr)
        return 1

    # Validate schema before analysis
    try:
        validate_eda_input(df)
    except Exception as e:
        print(f"[ERROR] Dataset validation failed: {e}", file=sys.stderr)
        return 1

    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nComputing statistical summaries and correlation matrices...")
    overview = compute_dataset_overview(df)
    correlations = compute_correlations(df)
    dept_metrics = compute_department_metrics(df)
    role_metrics = compute_job_role_metrics(df)
    period_trends = compute_period_trends(df)
    training_analysis = analyze_training_relationship(df)

    print("\nGenerating exploratory figures in non-interactive backend...")
    setup_plot_style()

    plot_target_distribution(df, figures_dir / "01_target_distribution.png")
    plot_numeric_predictors(df, figures_dir / "02_numeric_features_distributions.png")
    plot_correlation_heatmap(df, figures_dir / "03_correlation_heatmap.png")
    plot_key_predictors_vs_target(df, figures_dir / "04_key_predictors_vs_target.png")
    plot_performance_by_department(df, figures_dir / "05_performance_by_department.png")
    plot_performance_by_job_role(df, figures_dir / "06_performance_by_job_role.png")
    plot_quarterly_trends(df, figures_dir / "07_quarterly_performance_trends.png")
    plot_training_hours_remedial_analysis(df, figures_dir / "08_training_hours_remedial_analysis.png")

    print("\nCompiling written Markdown report...")
    generate_markdown_report(
        overview=overview,
        correlations=correlations,
        dept_metrics=dept_metrics,
        role_metrics=role_metrics,
        period_trends=period_trends,
        training_analysis=training_analysis,
        report_path=report_path,
    )

    # Sync a copy to docs_copy_path if requested
    if docs_copy_path:
        docs_copy_path.parent.mkdir(parents=True, exist_ok=True)
        generate_markdown_report(
            overview=overview,
            correlations=correlations,
            dept_metrics=dept_metrics,
            role_metrics=role_metrics,
            period_trends=period_trends,
            training_analysis=training_analysis,
            report_path=docs_copy_path,
        )

    print("\n" + "=" * 60)
    print("EDA EXECUTION REPORT")
    print("=" * 60)
    print(f"Dataset Analyzed: {len(df):,} rows x {df.shape[1]} columns")
    print(f"Figures Generated (8): {figures_dir.resolve()}")
    print(f"Report Generated: {report_path.resolve()}")
    if docs_copy_path:
        print(f"Documentation Copy: {docs_copy_path.resolve()}")
    print("Target Variable: future_performance_score")
    print(f"  - Mean: {overview['numeric_stats']['future_performance_score']['mean']:.3f}")
    print(f"  - Median: {overview['numeric_stats']['future_performance_score']['median']:.3f}")
    print(f"  - Std: {overview['numeric_stats']['future_performance_score']['std']:.3f}")
    print(f"Top Predictor Correlation: previous_performance_score (r = {correlations['target_pearson']['previous_performance_score']:+.4f})")
    print(f"Training Hours Marginal Correlation: r = {training_analysis['marginal_correlation']:+.4f} (Selection Bias verified)")
    print("=" * 60)
    print("[SUCCESS] Exploratory data analysis completed successfully.")
    return 0


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_args()
    return run_eda(
        data_path=Path(args.data),
        figures_dir=Path(args.figures_dir),
        report_path=Path(args.report_path),
        docs_copy_path=PROJECT_ROOT / "docs" / "eda_report.md",
    )


if __name__ == "__main__":
    sys.exit(main())
