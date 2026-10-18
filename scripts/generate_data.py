"""Workflow script to generate and validate synthetic employee performance data.

Usage:
    python scripts/generate_data.py [--config CONFIG_PATH] [--seed SEED]
"""

import argparse
import sys
from pathlib import Path
import yaml

# Ensure project src directory is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from employee_performance.data_generator import GeneratorConfig, SyntheticDataGenerator
from employee_performance.validation import DataValidator


def load_config(config_path: Path) -> dict:
    """Load settings from YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate longitudinal synthetic employee performance dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--num-employees",
        type=int,
        default=None,
        help="Override number of employees to generate.",
    )
    parser.add_argument(
        "--num-periods",
        type=int,
        default=None,
        help="Override number of observation periods to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random number generator seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override destination CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute data generation and validation pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    print(f"Loading configuration from: {config_path}")
    raw_config = load_config(config_path)
    gen_config_dict = raw_config.get("data_generation", {})

    num_employees = args.num_employees or gen_config_dict.get("num_employees", 1000)
    num_periods = args.num_periods or gen_config_dict.get("num_periods", 4)
    random_seed = args.seed if args.seed is not None else gen_config_dict.get("random_seed", 42)
    output_path_str = args.output or gen_config_dict.get(
        "output_path", "data/synthetic/employee_performance.csv"
    )

    output_path = PROJECT_ROOT / output_path_str

    print(f"Configuring generator:")
    print(f"  - Employees: {num_employees}")
    print(f"  - Periods: {num_periods}")
    print(f"  - Random Seed: {random_seed}")
    print(f"  - Output Target: {output_path}")

    # Initialize generator
    config = GeneratorConfig(
        num_employees=num_employees,
        num_periods=num_periods,
        random_seed=random_seed,
        output_path=output_path,
    )
    generator = SyntheticDataGenerator(config=config)

    print("\nGenerating synthetic longitudinal dataset...")
    df = generator.generate()

    print("\nRunning comprehensive schema and integrity validation...")
    validator = DataValidator()
    result = validator.validate(df)

    print("\n" + "=" * 60)
    print("DATASET EXECUTION & VALIDATION REPORT")
    print("=" * 60)
    print(f"File Saved: {output_path.resolve()}")
    print(f"Total Rows Generated: {len(df)}")
    print(f"Unique Employees: {df['employee_id'].nunique()}")
    print(f"Period Coverage: {sorted(df['period'].unique().tolist())}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nValidation Summary:")
    print(result.summary())
    print("=" * 60)

    if not result.is_valid:
        print("\n[FAILED] Dataset failed schema and integrity validation!", file=sys.stderr)
        return 1

    print("\n[SUCCESS] Dataset generated and validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
