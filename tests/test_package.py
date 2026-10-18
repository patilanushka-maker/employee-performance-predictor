"""Test package initialization and configuration loading."""

import os
from pathlib import Path
import pytest
import yaml

import employee_performance


def test_package_metadata():
    """Verify package version is defined."""
    assert hasattr(employee_performance, "__version__")
    assert employee_performance.__version__ == "0.1.0"


def test_config_file_exists():
    """Verify default configuration file is present and valid YAML."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    assert config_path.is_file(), f"Configuration file not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    assert isinstance(config_data, dict)
    assert "project" in config_data
    assert "paths" in config_data
