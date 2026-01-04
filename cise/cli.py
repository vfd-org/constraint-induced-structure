"""
CISE Command Line Interface.

Provides commands for running constraint experiments, beta sweeps,
and generating reports.
"""

import click
import os
import sys
from pathlib import Path


@click.group()
@click.version_option(version="1.0.0", prog_name="cise")
def main():
    """CISE - Constraint-Induced Structure Explorer.

    A computational experiment engine for studying constraint effects
    on ensemble distributions.
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="configs/default.yaml",
    help="Path to configuration YAML file."
)
@click.option(
    "--out", "-o",
    type=click.Path(),
    required=True,
    help="Output directory for results."
)
@click.option(
    "--seed", "-s",
    type=int,
    default=None,
    help="Override random seed from config."
)
def run(config, out, seed):
    """Run baseline + constrained experiments from config.

    Generates baseline ensembles, applies constraints via energy-based
    reweighting, computes metrics, and produces comparison plots.

    Example:
        cise run --config configs/default.yaml --out outputs/run_001
    """
    from cise.config import load_config
    from cise.experiments.runner import ExperimentRunner

    click.echo(f"CISE v1.0.0 - Constraint-Induced Structure Explorer")
    click.echo(f"=" * 55)
    click.echo(f"Loading config: {config}")

    cfg = load_config(config)

    if seed is not None:
        cfg.seed = seed
        click.echo(f"Overriding seed: {seed}")

    click.echo(f"Output directory: {out}")
    click.echo()

    runner = ExperimentRunner(cfg, output_dir=out)
    runner.run()

    click.echo()
    click.echo(f"Experiment complete. Results saved to: {out}")


@main.command()
@click.option(
    "--beta", "-b",
    type=float,
    multiple=True,
    default=(0.0, 0.5, 1.0, 2.0),
    help="Beta values for sweep (can specify multiple)."
)
@click.option(
    "--out", "-o",
    type=click.Path(),
    required=True,
    help="Output directory for results."
)
@click.option(
    "--n-samples", "-n",
    type=int,
    default=10000,
    help="Number of samples per ensemble."
)
@click.option(
    "--seed", "-s",
    type=int,
    default=1337,
    help="Random seed."
)
@click.option(
    "--dim", "-d",
    type=int,
    default=32,
    help="Vector dimension."
)
@click.option(
    "--matrix-size", "-k",
    type=int,
    default=6,
    help="Matrix size (k x k)."
)
def sweep(beta, out, n_samples, seed, dim, matrix_size):
    """Run beta sweep with default constraints.

    Convenience command for running experiments across multiple beta values
    using default constraint configurations.

    Example:
        cise sweep --beta 0 0.5 1 2 --out outputs/run_002
    """
    from cise.config import Config
    from cise.experiments.runner import ExperimentRunner

    click.echo(f"CISE v1.0.0 - Beta Sweep")
    click.echo(f"=" * 55)
    click.echo(f"Beta values: {list(beta)}")
    click.echo(f"Samples: {n_samples}, Seed: {seed}")
    click.echo(f"Vector dim: {dim}, Matrix size: {matrix_size}x{matrix_size}")
    click.echo(f"Output directory: {out}")
    click.echo()

    cfg = Config(
        seed=seed,
        n_samples=n_samples,
        vector_dim=dim,
        matrix_size=matrix_size,
        beta_values=list(beta),
        vector_constraints=["smoothness", "l1"],
        matrix_constraints=["lowrank", "l1"],
        samplers=["gaussian", "uniform"],
    )

    runner = ExperimentRunner(cfg, output_dir=out)
    runner.run()

    click.echo()
    click.echo(f"Sweep complete. Results saved to: {out}")


@main.command()
@click.option(
    "--in", "-i", "input_dir",
    type=click.Path(exists=True),
    required=True,
    help="Input directory containing metrics.json."
)
def report(input_dir):
    """Regenerate summary.md from existing metrics.json.

    Reads metrics from a previous run and generates an updated
    summary report.

    Example:
        cise report --in outputs/run_001
    """
    import json
    from cise.experiments.runner import generate_summary_report

    click.echo(f"CISE v1.0.0 - Report Generator")
    click.echo(f"=" * 55)
    click.echo(f"Input directory: {input_dir}")

    metrics_path = Path(input_dir) / "metrics.json"
    if not metrics_path.exists():
        click.echo(f"Error: metrics.json not found in {input_dir}", err=True)
        sys.exit(1)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    summary_path = Path(input_dir) / "summary.md"
    generate_summary_report(metrics, summary_path, Path(input_dir))

    click.echo(f"Summary report regenerated: {summary_path}")


if __name__ == "__main__":
    main()
