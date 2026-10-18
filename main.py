"""Main entry point for the Employee Performance Predictor CLI.

Provides command-line coordination for data pipelines, modeling,
and evaluation once implemented.
"""

import argparse
import sys
from pathlib import Path

# Ensure src directory is on sys.path for direct script execution
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml
import employee_performance



def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="employee-performance-predictor",
        description="Employee Performance Predictor using Data Analytics",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {employee_performance.__version__}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current environment, configuration, and project status",
    )
    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}", file=sys.stderr)
        return 1

    if args.status:
        print(f"Project: {config.get('project', {}).get('name')}")
        print(f"Version: {employee_performance.__version__}")
        print(f"Environment configuration loaded successfully from {args.config}")
        return 0

    print(
        f"Employee Performance Predictor (v{employee_performance.__version__})\n"
        f"Run with '--help' for available commands or '--status' to view configuration."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
