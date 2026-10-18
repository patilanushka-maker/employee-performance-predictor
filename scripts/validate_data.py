"""Standalone dataset validation script for Employee Performance Predictor.

Usage:
    python scripts/validate_data.py --data data/synthetic/employee_performance.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from employee_performance.validation import DataValidator


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate employee performance CSV against schema requirements."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "synthetic" / "employee_performance.csv"),
        help="Path to CSV dataset to validate.",
    )
    return parser.parse_args()


def main() -> int:
    """Run standalone validation."""
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.is_file():
        print(f"[ERROR] Target dataset file does not exist: {data_path.resolve()}", file=sys.stderr)
        return 1

    print(f"Reading dataset: {data_path.resolve()}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse CSV: {e}", file=sys.stderr)
        return 1

    validator = DataValidator()
    result = validator.validate(df)

    print("\n" + "=" * 60)
    print("STANDALONE DATA VALIDATION REPORT")
    print("=" * 60)
    print(result.summary())
    print("=" * 60)

    if not result.is_valid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
