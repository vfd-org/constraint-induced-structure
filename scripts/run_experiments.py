#!/usr/bin/env python3
"""
Run CISE experiments script.

This script provides a simple way to run CISE experiments without using the CLI.
It's equivalent to: python -m cise run --config configs/default.yaml --out outputs/run_001

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --config configs/custom.yaml --out outputs/my_run
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cise.config import load_config
from cise.experiments.runner import ExperimentRunner


def main():
    """Run CISE experiments."""
    parser = argparse.ArgumentParser(
        description="Run CISE experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --config configs/default.yaml --out outputs/run_001
    python scripts/run_experiments.py --seed 42 --n-samples 5000
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--out", "-o",
        type=str,
        default="outputs/run_001",
        help="Output directory for results (default: outputs/run_001)",
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Override random seed from config",
    )

    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=None,
        help="Override number of samples from config",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CISE - Constraint-Induced Structure Explorer")
    print("=" * 60)
    print()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    print(f"Loading configuration: {args.config}")
    config = load_config(args.config)

    # Apply overrides
    if args.seed is not None:
        config.seed = args.seed
        print(f"Overriding seed: {args.seed}")

    if args.n_samples is not None:
        config.n_samples = args.n_samples
        print(f"Overriding n_samples: {args.n_samples}")

    print(f"Output directory: {args.out}")
    print()

    # Run experiments
    runner = ExperimentRunner(config, output_dir=args.out)
    results = runner.run()

    print()
    print("=" * 60)
    print("Experiment complete!")
    print(f"Results saved to: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
