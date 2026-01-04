"""
CISE - Constraint-Induced Structure Explorer

A computational experiment engine for studying how constraints affect
ensemble distributions. Generates baseline ensembles, applies constraints
via energy-based reweighting, and compares resulting distributions with
clear metrics and plots.

This is a neutral, non-ontological experiment framework.
"""

__version__ = "1.0.0"
__author__ = "CISE Contributors"

from cise.config import Config, load_config
from cise.experiments.runner import ExperimentRunner

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "ExperimentRunner",
]
