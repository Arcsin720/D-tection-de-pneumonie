"""Utility functions for the project"""

import os
import json
from pathlib import Path


def create_results_dir(path: str = "results") -> str:
    """Create results directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(metrics: dict, filepath: str = "results/metrics.json"):
    """Save metrics to JSON file"""
    create_results_dir()
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved to {filepath}")


def load_metrics(filepath: str = "results/metrics.json") -> dict:
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
