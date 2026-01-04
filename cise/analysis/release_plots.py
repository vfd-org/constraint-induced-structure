"""
Release plotting functions for CISE.

Generates the curated set of canonical figures for GitHub release.
These are the 6-9 key visualizations that tell the constraint story clearly.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def plot_ess_vs_beta(
    results: Dict[str, Any],
    save_path: Path,
    sample_type: str = "vector",
) -> None:
    """Plot ESS ratio vs beta value (key diagnostic).

    Args:
        results: Dictionary with experiment results by beta.
        save_path: Path to save figure.
        sample_type: 'vector' or 'matrix'.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    betas = []
    ess_ratios = []

    for key, metrics in results.items():
        if sample_type in key and "gaussian" in key:
            betas.append(metrics["beta"])
            ess_ratios.append(metrics["ess_ratio"] * 100)

    if not betas:
        plt.close(fig)
        return

    # Sort by beta
    sorted_pairs = sorted(zip(betas, ess_ratios))
    betas, ess_ratios = zip(*sorted_pairs)

    ax.plot(betas, ess_ratios, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.fill_between(betas, ess_ratios, alpha=0.2, color="steelblue")

    ax.set_xlabel(r"Constraint strength ($\beta$)", fontsize=12)
    ax.set_ylabel("Effective Sample Size (%)", fontsize=12)
    ax.set_title("ESS vs Constraint Strength", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Add interpretation zone
    ax.axhline(10, color="orange", linestyle="--", alpha=0.7, label="Caution zone (<10%)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_histogram_overlay(
    energies: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
) -> None:
    """Plot energy histogram: baseline vs constrained overlay.

    Args:
        energies: Energy values.
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Compute shared bins
    bins = np.linspace(np.percentile(energies, 1), np.percentile(energies, 99), 40)

    ax.hist(
        energies, bins=bins, alpha=0.5, label="Baseline", density=True, color="steelblue"
    )
    ax.hist(
        energies,
        bins=bins,
        weights=weights,
        alpha=0.5,
        label=f"Constrained",
        density=True,
        color="darkorange",
    )

    ax.set_xlabel("Constraint Energy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Energy Distribution Shift ($\\beta$={beta})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_norm_distribution_overlay(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
) -> None:
    """Plot norm distribution: baseline vs constrained overlay.

    Args:
        samples: Sample array.
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
    """
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)
    norms = np.linalg.norm(samples_flat, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(np.percentile(norms, 1), np.percentile(norms, 99), 40)

    ax.hist(norms, bins=bins, alpha=0.5, label="Baseline", density=True, color="steelblue")
    ax.hist(
        norms,
        bins=bins,
        weights=weights,
        alpha=0.5,
        label="Constrained",
        density=True,
        color="darkorange",
    )

    # Add means
    baseline_mean = np.mean(norms)
    constrained_mean = np.sum(weights * norms) / np.sum(weights)

    ax.axvline(baseline_mean, color="steelblue", linestyle="--", alpha=0.8)
    ax.axvline(constrained_mean, color="darkorange", linestyle="--", alpha=0.8)

    ax.set_xlabel("Sample Norm", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Norm Distribution Shift ($\\beta$={beta})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gini_distribution(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
) -> None:
    """Plot Gini coefficient distribution: baseline vs constrained.

    Args:
        samples: Sample array.
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
    """
    from cise.analysis.metrics import compute_gini_coefficient

    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    # Compute Gini for each sample
    ginis = np.array([compute_gini_coefficient(s) for s in samples_flat])

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, 1, 40)

    ax.hist(ginis, bins=bins, alpha=0.5, label="Baseline", density=True, color="steelblue")
    ax.hist(
        ginis,
        bins=bins,
        weights=weights,
        alpha=0.5,
        label="Constrained",
        density=True,
        color="darkorange",
    )

    ax.set_xlabel("Gini Coefficient (value inequality)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Gini Distribution Shift ($\\beta$={beta})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance_comparison(
    baseline_samples: np.ndarray,
    weights: np.ndarray,
    save_path: Path,
    n_components: int = 15,
) -> None:
    """Plot PCA explained variance curves: baseline vs constrained.

    Args:
        baseline_samples: Baseline samples.
        weights: Importance weights.
        save_path: Path to save figure.
        n_components: Number of components.
    """
    from sklearn.decomposition import PCA

    n_samples = baseline_samples.shape[0]
    samples_flat = baseline_samples.reshape(n_samples, -1)
    n_components = min(n_components, n_samples - 1, samples_flat.shape[1])

    # Baseline PCA
    pca_baseline = PCA(n_components=n_components)
    pca_baseline.fit(samples_flat)

    # Constrained PCA (via weighted resampling)
    rng = np.random.default_rng(42)
    weights_norm = weights / np.sum(weights)
    indices = rng.choice(n_samples, size=n_samples, replace=True, p=weights_norm)
    resampled = samples_flat[indices]

    pca_constrained = PCA(n_components=n_components)
    pca_constrained.fit(resampled)

    fig, ax = plt.subplots(figsize=(8, 5))

    components = np.arange(1, n_components + 1)

    ax.plot(
        components,
        np.cumsum(pca_baseline.explained_variance_ratio_) * 100,
        "o-",
        label="Baseline",
        color="steelblue",
        markersize=5,
        linewidth=2,
    )

    ax.plot(
        components,
        np.cumsum(pca_constrained.explained_variance_ratio_) * 100,
        "s-",
        label="Constrained",
        color="darkorange",
        markersize=5,
        linewidth=2,
    )

    ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

    ax.set_xlabel("Number of Principal Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
    ax.set_title("Dimensional Concentration Under Constraints", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter_comparison(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
) -> None:
    """Plot 2D PCA scatter: baseline vs constrained side-by-side.

    Args:
        samples: Sample array.
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
    """
    from sklearn.decomposition import PCA

    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(samples_flat)

    rng = np.random.default_rng(42)
    weights_norm = weights / np.sum(weights)
    n_show = min(1500, n_samples)
    baseline_idx = rng.choice(n_samples, size=n_show, replace=False)
    constrained_idx = rng.choice(n_samples, size=n_show, replace=True, p=weights_norm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline
    axes[0].scatter(
        projected[baseline_idx, 0],
        projected[baseline_idx, 1],
        alpha=0.4,
        s=12,
        c="steelblue",
        edgecolors="none",
    )
    axes[0].set_xlabel("PC1", fontsize=11)
    axes[0].set_ylabel("PC2", fontsize=11)
    axes[0].set_title("Baseline", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Constrained
    axes[1].scatter(
        projected[constrained_idx, 0],
        projected[constrained_idx, 1],
        alpha=0.4,
        s=12,
        c="darkorange",
        edgecolors="none",
    )
    axes[1].set_xlabel("PC1", fontsize=11)
    axes[1].set_ylabel("PC2", fontsize=11)
    axes[1].set_title("Constrained", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Match axes
    all_x = projected[:, 0]
    all_y = projected[:, 1]
    margin = 0.1
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    xlim = (all_x.min() - margin * x_range, all_x.max() + margin * x_range)
    ylim = (all_y.min() - margin * y_range, all_y.max() + margin * y_range)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(f"PCA Projection ($\\beta$={beta})", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sv_spectrum(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
) -> None:
    """Plot singular value spectrum: baseline vs constrained.

    Args:
        samples: Matrix samples (n_samples, k, k).
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
    """
    if samples.ndim != 3:
        return

    n_samples = samples.shape[0]
    k = samples.shape[1]

    svs = np.zeros((n_samples, k))
    for i in range(n_samples):
        try:
            svs[i] = np.linalg.svd(samples[i], compute_uv=False)
        except np.linalg.LinAlgError:
            svs[i] = np.zeros(k)

    # Baseline stats
    sv_mean_baseline = np.mean(svs, axis=0)
    sv_std_baseline = np.std(svs, axis=0)

    # Constrained stats
    weights_norm = weights / np.sum(weights)
    sv_mean_constrained = np.sum(weights_norm[:, None] * svs, axis=0)
    sv_var = np.sum(weights_norm[:, None] * (svs - sv_mean_constrained) ** 2, axis=0)
    sv_std_constrained = np.sqrt(sv_var)

    fig, ax = plt.subplots(figsize=(8, 5))

    indices = np.arange(1, k + 1)

    ax.errorbar(
        indices - 0.08,
        sv_mean_baseline,
        yerr=sv_std_baseline,
        fmt="o-",
        label="Baseline",
        color="steelblue",
        capsize=4,
        markersize=7,
        linewidth=2,
    )

    ax.errorbar(
        indices + 0.08,
        sv_mean_constrained,
        yerr=sv_std_constrained,
        fmt="s-",
        label="Constrained",
        color="darkorange",
        capsize=4,
        markersize=7,
        linewidth=2,
    )

    # Mark target rank boundary
    ax.axvline(2.5, color="gray", linestyle="--", alpha=0.6, label="Target rank (r=2)")

    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Singular Value (mean ± std)", fontsize=12)
    ax.set_title(f"Singular Value Spectrum ($\\beta$={beta})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(indices)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hierarchy_curve(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
    is_matrix: bool = False,
) -> None:
    """Plot hierarchy curve: sorted magnitudes baseline vs constrained.

    Args:
        samples: Sample array.
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
        is_matrix: Whether samples are matrices.
    """
    n_samples = samples.shape[0]

    if is_matrix:
        k = samples.shape[1]
        magnitudes = np.zeros((n_samples, k))
        for i in range(n_samples):
            try:
                magnitudes[i] = np.linalg.svd(samples[i], compute_uv=False)
            except np.linalg.LinAlgError:
                magnitudes[i] = np.zeros(k)
    else:
        magnitudes = np.abs(samples)

    sorted_mags = np.sort(magnitudes, axis=1)[:, ::-1]

    mean_baseline = np.mean(sorted_mags, axis=0)
    weights_norm = weights / np.sum(weights)
    mean_constrained = np.sum(weights_norm[:, None] * sorted_mags, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    indices = np.arange(1, len(mean_baseline) + 1)

    ax.semilogy(
        indices,
        mean_baseline + 1e-10,
        "o-",
        label="Baseline",
        color="steelblue",
        markersize=4,
        linewidth=2,
    )

    ax.semilogy(
        indices,
        mean_constrained + 1e-10,
        "s-",
        label="Constrained",
        color="darkorange",
        markersize=4,
        linewidth=2,
    )

    ax.set_xlabel("Rank (sorted by magnitude)", fontsize=12)
    ax.set_ylabel("Mean Magnitude (log scale)", fontsize=12)
    ax.set_title(f"Hierarchical Structure ($\\beta$={beta})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_release_figures(
    samples: np.ndarray,
    energies: np.ndarray,
    weights: np.ndarray,
    beta: float,
    all_results: Dict[str, Any],
    output_dir: Path,
    sample_type: str = "vector",
) -> List[Path]:
    """Create the curated set of release figures.

    Args:
        samples: Sample array.
        energies: Energy values.
        weights: Importance weights.
        beta: Beta value.
        all_results: All experiment results (for ESS plot).
        output_dir: Output directory.
        sample_type: 'vector' or 'matrix'.

    Returns:
        List of created figure paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    is_matrix = sample_type == "matrix"

    # 1. ESS vs beta (once per sample type)
    path = output_dir / f"{sample_type}_ess_vs_beta.png"
    if not path.exists():
        plot_ess_vs_beta(all_results, path, sample_type)
        created.append(path)

    # 2. Energy histogram
    path = output_dir / f"{sample_type}_energy_histogram.png"
    plot_energy_histogram_overlay(energies, weights, beta, path)
    created.append(path)

    # 3. Norm distribution
    path = output_dir / f"{sample_type}_norm_distribution.png"
    plot_norm_distribution_overlay(samples, weights, beta, path)
    created.append(path)

    # 4. Gini distribution
    path = output_dir / f"{sample_type}_gini_distribution.png"
    plot_gini_distribution(samples, weights, beta, path)
    created.append(path)

    # 5. PCA variance
    path = output_dir / f"{sample_type}_pca_variance.png"
    plot_pca_variance_comparison(samples, weights, path)
    created.append(path)

    # 6. PCA scatter
    path = output_dir / f"{sample_type}_pca_scatter.png"
    plot_pca_scatter_comparison(samples, weights, beta, path)
    created.append(path)

    # 7. Hierarchy curve
    path = output_dir / f"{sample_type}_hierarchy.png"
    plot_hierarchy_curve(samples, weights, beta, path, is_matrix=is_matrix)
    created.append(path)

    # 8. SV spectrum (matrices only)
    if is_matrix:
        path = output_dir / f"{sample_type}_sv_spectrum.png"
        plot_sv_spectrum(samples, weights, beta, path)
        created.append(path)

    return created
