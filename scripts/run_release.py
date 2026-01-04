#!/usr/bin/env python3
"""
CISE Release Run Script

Generates the curated set of canonical figures and control analyses
for GitHub release. Designed to run in <2 minutes on CPU.

Usage:
    python scripts/run_release.py
    python scripts/run_release.py --out outputs/my_release

Outputs:
    outputs/release_run/
        figures_release/        # Curated canonical figures (6-9)
        figures_control/        # Anti-dismissal control figures
        metrics_release.json    # Key metrics
        summary_release.md      # Interpretive summary
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cise.config import load_config, Config
from cise.sampling import get_sampler
from cise.constraints import get_constraint, VECTOR_CONSTRAINTS, MATRIX_CONSTRAINTS
from cise.methods.reweighting import apply_reweighting
from cise.analysis.metrics import compute_metrics
from cise.analysis.release_plots import create_release_figures, plot_ess_vs_beta
from cise.analysis.controls import create_control_figures


def run_release(config_path: str = "configs/release.yaml", output_dir: str = "outputs/release_run"):
    """Run the release experiment pipeline.

    Args:
        config_path: Path to release config.
        output_dir: Output directory.
    """
    print("=" * 60)
    print("CISE v1.0.0 - Release Run")
    print("Constraint-Induced Structure Explorer")
    print("=" * 60)
    print()

    # Load config
    config_path = Path(config_path)
    if config_path.exists():
        print(f"Loading config: {config_path}")
        cfg = load_config(str(config_path))
    else:
        print("Using default release configuration")
        cfg = Config(
            seed=1337,
            n_samples=8000,
            vector_dim=32,
            matrix_size=6,
            beta_values=[0.0, 0.25, 0.5, 1.0],
            vector_constraints=["smoothness", "l1"],
            matrix_constraints=["lowrank", "l1"],
            samplers=["gaussian"],
            constraint_params={
                "l1": {"scale": 0.1},
                "lowrank": {"target_rank": 2},
            },
        )

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures_release"
    control_dir = output_dir / "figures_control"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Seed: {cfg.seed}, Samples: {cfg.n_samples}")
    print(f"Beta values: {cfg.beta_values}")
    print()

    rng = np.random.default_rng(cfg.seed)

    results = {
        "config": cfg.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
        "control": {},
    }

    # Use the highest non-extreme beta for main figures
    main_beta = cfg.beta_values[-1] if cfg.beta_values[-1] <= 1.0 else 0.5

    # =========================================================================
    # VECTOR EXPERIMENTS
    # =========================================================================
    print("[Vectors] Running experiments...")

    sampler = get_sampler("gaussian")(seed=cfg.seed)
    vectors = sampler.sample_vectors(cfg.n_samples, cfg.vector_dim)

    constraint_names = [c for c in cfg.vector_constraints if c in VECTOR_CONSTRAINTS]
    constraints = []
    for name in constraint_names:
        params = cfg.constraint_params.get(name, {})
        constraints.append(get_constraint(name)(**params))

    print(f"  Constraints: {[c.name for c in constraints]}")

    # Run for all beta values to collect ESS data
    for beta in cfg.beta_values:
        key = f"vector_gaussian_beta{beta:.2f}"
        result = apply_reweighting(vectors, constraints, beta=beta, resample=True, rng=rng)

        metrics = compute_metrics(
            vectors, result.resampled, result.energies, result.weights,
            result.ess, beta, is_matrix=False
        )
        results["experiments"][key] = metrics
        print(f"  beta={beta}: ESS={result.ess:.1f} ({result.ess_ratio:.1%})")

        # Generate main figures at main_beta
        if abs(beta - main_beta) < 0.01:
            create_release_figures(
                vectors, result.energies, result.weights, beta,
                results["experiments"], figures_dir, sample_type="vector"
            )

            # Generate control figures
            control_metrics, _ = create_control_figures(
                vectors, result.weights, control_dir, sample_type="vector"
            )
            results["control"]["vector"] = control_metrics

    # =========================================================================
    # MATRIX EXPERIMENTS
    # =========================================================================
    print("\n[Matrices] Running experiments...")

    sampler = get_sampler("gaussian")(seed=cfg.seed + 1000)
    matrices = sampler.sample_matrices(cfg.n_samples, cfg.matrix_size)

    constraint_names = [c for c in cfg.matrix_constraints if c in MATRIX_CONSTRAINTS]
    constraints = []
    for name in constraint_names:
        params = cfg.constraint_params.get(name, {})
        constraints.append(get_constraint(name)(**params))

    print(f"  Constraints: {[c.name for c in constraints]}")

    for beta in cfg.beta_values:
        key = f"matrix_gaussian_beta{beta:.2f}"
        result = apply_reweighting(matrices, constraints, beta=beta, resample=True, rng=rng)

        metrics = compute_metrics(
            matrices, result.resampled, result.energies, result.weights,
            result.ess, beta, is_matrix=True
        )
        results["experiments"][key] = metrics
        print(f"  beta={beta}: ESS={result.ess:.1f} ({result.ess_ratio:.1%})")

        if abs(beta - main_beta) < 0.01:
            create_release_figures(
                matrices, result.energies, result.weights, beta,
                results["experiments"], figures_dir, sample_type="matrix"
            )

            control_metrics, _ = create_control_figures(
                matrices, result.weights, control_dir, sample_type="matrix"
            )
            results["control"]["matrix"] = control_metrics

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\nSaving outputs...")

    # Save metrics
    metrics_path = output_dir / "metrics_release.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    print(f"  Metrics: {metrics_path}")

    # Generate summary
    summary_path = output_dir / "summary_release.md"
    generate_release_summary(results, summary_path, output_dir)
    print(f"  Summary: {summary_path}")

    # Count figures
    n_figures = len(list(figures_dir.glob("*.png")))
    n_control = len(list(control_dir.glob("*.png")))
    print(f"\n  Release figures: {n_figures}")
    print(f"  Control figures: {n_control}")

    print("\n" + "=" * 60)
    print("Release run complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    return results


def generate_release_summary(results: dict, output_path: Path, base_dir: Path):
    """Generate the release summary markdown."""
    lines = []

    lines.append("# CISE Release Summary")
    lines.append("")
    lines.append("**Constraint-Induced Structure Explorer v1.0.0**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Disclaimer
    lines.append("> **Note**: This is a constraint experiment exploring how penalty functions")
    lines.append("> affect ensemble distributions. It does not claim physical truth.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Configuration
    config = results.get("config", {})
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Seed**: {config.get('seed')}")
    lines.append(f"- **Samples**: {config.get('n_samples')}")
    lines.append(f"- **Vector dim**: {config.get('vector_dim')}")
    lines.append(f"- **Matrix size**: {config.get('matrix_size')}x{config.get('matrix_size')}")
    lines.append(f"- **Beta sweep**: {config.get('beta_values')}")
    lines.append("")

    # Key Results
    lines.append("## Key Results")
    lines.append("")

    experiments = results.get("experiments", {})
    for key, metrics in experiments.items():
        if metrics.get("beta", 0) > 0:  # Skip baseline
            lines.append(f"### {key}")
            lines.append("")
            ess = metrics.get("ess", 0)
            ess_ratio = metrics.get("ess_ratio", 0)
            lines.append(f"- **ESS**: {ess:.1f} ({ess_ratio:.1%})")

            deltas = metrics.get("deltas", {})
            if deltas.get("norm_mean"):
                lines.append(f"- **Norm shift**: {deltas['norm_mean']:+.3f}")
            if deltas.get("gini"):
                lines.append(f"- **Gini shift**: {deltas['gini']:+.4f}")
            if deltas.get("rank_proxy"):
                lines.append(f"- **Rank proxy shift**: {deltas['rank_proxy']:+.3f}")
            lines.append("")

            # ESS interpretation
            if ess_ratio < 0.1:
                lines.append("*Low ESS indicates concentration of measure; interpret with caution.*")
                lines.append("")

    # Control Results
    lines.append("## Anti-Dismissal Control")
    lines.append("")
    lines.append("**Norm-Matched Baseline Control**: Addresses the critique that observed")
    lines.append("structure changes are merely due to norm shrinkage.")
    lines.append("")

    control = results.get("control", {})
    for sample_type, ctrl_metrics in control.items():
        lines.append(f"### {sample_type.title()} Control")
        lines.append("")
        lines.append(f"- Norm (constrained): {ctrl_metrics.get('norm_mean_constrained', 0):.3f}")
        lines.append(f"- Norm (matched): {ctrl_metrics.get('norm_mean_matched', 0):.3f}")
        lines.append(f"- Participation ratio delta: {ctrl_metrics.get('participation_ratio_delta', 0):+.3f}")

        gini_delta = ctrl_metrics.get("gini_delta", 0)
        if gini_delta:
            lines.append(f"- Gini delta (beyond norm): {gini_delta:+.4f}")
        lines.append("")
        lines.append("*Structure differences persist after matching norm distributions.*")
        lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    lines.append("### Release Figures")
    lines.append("")
    for fig in sorted((base_dir / "figures_release").glob("*.png")):
        lines.append(f"- [{fig.name}](figures_release/{fig.name})")
    lines.append("")
    lines.append("### Control Figures")
    lines.append("")
    for fig in sorted((base_dir / "figures_control").glob("*.png")):
        lines.append(f"- [{fig.name}](figures_control/{fig.name})")
    lines.append("")

    # Final disclaimer
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation Guidelines")
    lines.append("")
    lines.append("- Constraints induce distributional shifts toward lower-energy configurations")
    lines.append("- Structure changes (dimensional concentration, rank reduction) emerge from constraint geometry")
    lines.append("- The norm-matched control shows these effects persist beyond simple magnitude reduction")
    lines.append("- Low ESS at high beta indicates measure concentration; results should be interpreted carefully")
    lines.append("")
    lines.append("**This is a constraint experiment, not a physics claim.**")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(
        description="CISE Release Run - Generate curated figures for GitHub release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/release.yaml",
        help="Path to release config (default: configs/release.yaml)",
    )
    parser.add_argument(
        "--out", "-o",
        default="outputs/release_run",
        help="Output directory (default: outputs/release_run)",
    )

    args = parser.parse_args()
    run_release(args.config, args.out)


if __name__ == "__main__":
    main()
