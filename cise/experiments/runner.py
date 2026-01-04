"""
Experiment runner for CISE.

Orchestrates the full experiment pipeline: sampling, constraint application,
metrics computation, plotting, and report generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from cise.config import Config
from cise.sampling import get_sampler, SampleType
from cise.constraints import get_constraint, VECTOR_CONSTRAINTS, MATRIX_CONSTRAINTS
from cise.methods.reweighting import apply_reweighting
from cise.analysis.metrics import compute_metrics
from cise.analysis.embeddings import (
    compute_pca_explained_variance,
    compute_intrinsic_dimension,
    compute_clustering_tendency,
)
from cise.analysis.plots import create_all_plots


class ExperimentRunner:
    """Main experiment runner for CISE.

    Orchestrates the complete experiment pipeline:
    1. Generate baseline ensembles
    2. Apply constraints via energy-based reweighting
    3. Compute metrics comparing baseline vs constrained
    4. Generate comparison plots
    5. Write summary report
    """

    def __init__(self, config: Config, output_dir: str):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration.
            output_dir: Directory for output files.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.rng = np.random.default_rng(config.seed)

        # Results storage
        self.results: Dict[str, Any] = {
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "experiments": {},
        }

    def run(self) -> Dict[str, Any]:
        """Run all experiments.

        Returns:
            Complete results dictionary.
        """
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running experiments with seed={self.config.seed}")
        print(f"Beta values: {self.config.beta_values}")
        print()

        # Run vector experiments
        for sampler_name in self.config.samplers:
            self._run_vector_experiments(sampler_name)

        # Run matrix experiments
        for sampler_name in self.config.samplers:
            self._run_matrix_experiments(sampler_name)

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.results, f, indent=2, default=_json_serializer)
        print(f"\nMetrics saved to: {metrics_path}")

        # Generate summary report
        summary_path = self.output_dir / "summary.md"
        generate_summary_report(self.results, summary_path, self.output_dir)
        print(f"Summary saved to: {summary_path}")

        return self.results

    def _run_vector_experiments(self, sampler_name: str) -> None:
        """Run experiments on vectors.

        Args:
            sampler_name: Name of sampler to use.
        """
        print(f"[Vectors] Sampler: {sampler_name}")

        # Initialize sampler
        sampler_class = get_sampler(sampler_name)
        sampler = sampler_class(seed=self.config.seed)

        # Generate baseline samples
        samples = sampler.sample_vectors(self.config.n_samples, self.config.vector_dim)
        print(f"  Generated {self.config.n_samples} vectors of dim {self.config.vector_dim}")

        # Get applicable constraints
        constraint_names = [
            c for c in self.config.vector_constraints if c in VECTOR_CONSTRAINTS
        ]
        if not constraint_names:
            print("  No valid constraints, skipping")
            return

        # Initialize constraints
        constraints = []
        for name in constraint_names:
            constraint_class = get_constraint(name)
            params = self.config.constraint_params.get(name, {})
            constraints.append(constraint_class(**params))

        print(f"  Constraints: {[c.name for c in constraints]}")

        # Run for each beta value
        for beta in self.config.beta_values:
            key = f"vector_{sampler_name}_beta{beta:.2f}"
            print(f"  β={beta}: ", end="")

            # Apply reweighting
            result = apply_reweighting(
                samples,
                constraints,
                beta=beta,
                resample=True,
                rng=self.rng,
            )

            # Compute metrics
            metrics = compute_metrics(
                samples,
                result.resampled,
                result.energies,
                result.weights,
                result.ess,
                beta,
                is_matrix=False,
            )

            # Add embedding analysis
            metrics["pca_variance"] = compute_pca_explained_variance(samples)
            metrics["intrinsic_dimension"] = compute_intrinsic_dimension(samples)
            metrics["baseline_clustering"] = compute_clustering_tendency(samples)
            metrics["constrained_clustering"] = compute_clustering_tendency(
                samples, weights=result.weights
            )

            self.results["experiments"][key] = metrics
            print(f"ESS={result.ess:.1f} ({result.ess_ratio:.1%})")

            # Generate plots
            prefix = f"vector_{sampler_name}"
            create_all_plots(
                samples,
                result.energies,
                result.weights,
                beta,
                self.figures_dir,
                prefix=prefix,
                is_matrix=False,
            )

    def _run_matrix_experiments(self, sampler_name: str) -> None:
        """Run experiments on matrices.

        Args:
            sampler_name: Name of sampler to use.
        """
        print(f"[Matrices] Sampler: {sampler_name}")

        # Initialize sampler
        sampler_class = get_sampler(sampler_name)
        sampler = sampler_class(seed=self.config.seed + 1000)  # Different seed for matrices

        # Generate baseline samples
        samples = sampler.sample_matrices(self.config.n_samples, self.config.matrix_size)
        print(
            f"  Generated {self.config.n_samples} matrices of size "
            f"{self.config.matrix_size}x{self.config.matrix_size}"
        )

        # Get applicable constraints
        constraint_names = [
            c for c in self.config.matrix_constraints if c in MATRIX_CONSTRAINTS
        ]
        if not constraint_names:
            print("  No valid constraints, skipping")
            return

        # Initialize constraints
        constraints = []
        for name in constraint_names:
            constraint_class = get_constraint(name)
            params = self.config.constraint_params.get(name, {})
            constraints.append(constraint_class(**params))

        print(f"  Constraints: {[c.name for c in constraints]}")

        # Run for each beta value
        for beta in self.config.beta_values:
            key = f"matrix_{sampler_name}_beta{beta:.2f}"
            print(f"  β={beta}: ", end="")

            # Apply reweighting
            result = apply_reweighting(
                samples,
                constraints,
                beta=beta,
                resample=True,
                rng=self.rng,
            )

            # Compute metrics
            metrics = compute_metrics(
                samples,
                result.resampled,
                result.energies,
                result.weights,
                result.ess,
                beta,
                is_matrix=True,
            )

            # Add embedding analysis (flatten matrices for PCA)
            n_samples = samples.shape[0]
            samples_flat = samples.reshape(n_samples, -1)
            metrics["pca_variance"] = compute_pca_explained_variance(samples_flat)
            metrics["intrinsic_dimension"] = compute_intrinsic_dimension(samples_flat)
            metrics["baseline_clustering"] = compute_clustering_tendency(samples_flat)
            metrics["constrained_clustering"] = compute_clustering_tendency(
                samples_flat, weights=result.weights
            )

            self.results["experiments"][key] = metrics
            print(f"ESS={result.ess:.1f} ({result.ess_ratio:.1%})")

            # Generate plots
            prefix = f"matrix_{sampler_name}"
            create_all_plots(
                samples,
                result.energies,
                result.weights,
                beta,
                self.figures_dir,
                prefix=prefix,
                is_matrix=True,
            )


def generate_summary_report(
    results: Dict[str, Any],
    output_path: Path,
    base_dir: Path,
) -> None:
    """Generate markdown summary report.

    Args:
        results: Complete results dictionary.
        output_path: Path for summary.md.
        base_dir: Base output directory (for relative paths).
    """
    lines = []

    lines.append("# CISE Experiment Summary")
    lines.append("")
    lines.append("Constraint-Induced Structure Explorer - Experiment Results")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    config = results.get("config", {})
    lines.append(f"- **Seed**: {config.get('seed', 'N/A')}")
    lines.append(f"- **Samples**: {config.get('n_samples', 'N/A')}")
    lines.append(f"- **Vector dimension**: {config.get('vector_dim', 'N/A')}")
    lines.append(f"- **Matrix size**: {config.get('matrix_size', 'N/A')}x{config.get('matrix_size', 'N/A')}")
    lines.append(f"- **Beta values**: {config.get('beta_values', [])}")
    lines.append(f"- **Vector constraints**: {config.get('vector_constraints', [])}")
    lines.append(f"- **Matrix constraints**: {config.get('matrix_constraints', [])}")
    lines.append(f"- **Samplers**: {config.get('samplers', [])}")
    lines.append("")
    lines.append(f"*Generated: {results.get('timestamp', 'N/A')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Key Findings
    lines.append("## Key Findings")
    lines.append("")

    experiments = results.get("experiments", {})

    for exp_name, metrics in experiments.items():
        lines.append(f"### {exp_name}")
        lines.append("")

        beta = metrics.get("beta", "N/A")
        ess = metrics.get("ess", "N/A")
        ess_ratio = metrics.get("ess_ratio", 0)

        lines.append(f"**Beta**: {beta}")
        lines.append(f"**ESS**: {ess:.1f} ({ess_ratio:.1%})" if isinstance(ess, float) else f"**ESS**: {ess}")
        lines.append("")

        # Deltas
        deltas = metrics.get("deltas", {})
        if deltas:
            lines.append("**Distributional shifts (constrained - baseline)**:")
            lines.append("")
            for key, value in deltas.items():
                if value is not None:
                    lines.append(f"- {key}: {value:+.4f}")
            lines.append("")

        # Interpretation
        lines.append("**Interpretation**:")
        lines.append("")

        norm_delta = deltas.get("norm_mean", 0)
        gini_delta = deltas.get("gini", 0)

        if abs(norm_delta) > 0.1:
            direction = "decrease" if norm_delta < 0 else "increase"
            lines.append(f"- Constraints induce a {direction} in sample norms")

        if abs(gini_delta) > 0.01:
            direction = "increase" if gini_delta > 0 else "decrease"
            lines.append(f"- Gini coefficient shows {direction} in value inequality")

        if "rank_proxy" in deltas:
            rank_delta = deltas["rank_proxy"]
            if rank_delta is not None and abs(rank_delta) > 0.1:
                direction = "decrease" if rank_delta < 0 else "increase"
                lines.append(f"- Effective rank shows {direction}")

        if ess_ratio < 0.1:
            lines.append("- Low ESS indicates strong constraint effect (few samples dominate)")
        elif ess_ratio > 0.5:
            lines.append("- High ESS indicates weak constraint effect (weights are uniform)")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    lines.append("Generated figures are located in `figures/`:")
    lines.append("")

    figures_dir = base_dir / "figures"
    if figures_dir.exists():
        for fig_path in sorted(figures_dir.glob("*.png")):
            rel_path = fig_path.relative_to(base_dir)
            lines.append(f"- [{fig_path.name}]({rel_path})")
    else:
        lines.append("*(No figures generated)*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Disclaimer
    lines.append("## Disclaimer")
    lines.append("")
    lines.append(
        "This is a constraint experiment exploring how penalty functions affect "
        "ensemble distributions. Results describe distributional shifts under "
        "constraint-induced reweighting and should not be interpreted as physical "
        "predictions or claims about any underlying system."
    )
    lines.append("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
