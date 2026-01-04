"""
Experiments module for CISE.

Provides the main experiment runner that orchestrates sampling, constraint
application, metrics computation, and plotting.
"""

from cise.experiments.runner import ExperimentRunner, generate_summary_report

__all__ = [
    "ExperimentRunner",
    "generate_summary_report",
]
