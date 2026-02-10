"""
Configuration loader for YAML experiment settings.
"""
import yaml
from pathlib import Path


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config