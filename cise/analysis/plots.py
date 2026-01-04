"""
Plotting functions for CISE.

Generates comparison plots for baseline vs constrained ensembles.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless use


def plot_energy_histogram(
    energies: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot energy histogram comparing baseline (uniform) vs constrained (weighted).

    Args:
        energies: Energy values for all samples.
        weights: Importance weights.
        beta: Beta value used.
        save_path: Path to save figure.
        title: Optional title override.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline histogram (unweighted)
    ax.hist(
        energies,
        bins=50,
        alpha=0.6,
        label="Baseline",
        density=True,
        color="steelblue",
    )

    # Constrained histogram (weighted)
    ax.hist(
        energies,
        bins=50,
        weights=weights,
        alpha=0.6,
        label=f"Constrained (β={beta})",
        density=True,
        color="darkorange",
    )

    ax.set_xlabel("Energy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title or f"Energy Distribution (β={beta})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_norm_distributions(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot norm distributions comparing baseline vs constrained.

    Args:
        samples: Samples array.
        weights: Importance weights.
        beta: Beta value used.
        save_path: Path to save figure.
        title: Optional title override.
    """
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)
    norms = np.linalg.norm(samples_flat, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline histogram
    ax.hist(
        norms,
        bins=50,
        alpha=0.6,
        label="Baseline",
        density=True,
        color="steelblue",
    )

    # Constrained histogram
    ax.hist(
        norms,
        bins=50,
        weights=weights,
        alpha=0.6,
        label=f"Constrained (β={beta})",
        density=True,
        color="darkorange",
    )

    ax.set_xlabel("Norm (L2/Frobenius)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title or f"Norm Distribution (β={beta})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(
    baseline_samples: np.ndarray,
    constrained_samples: np.ndarray,
    weights: np.ndarray,
    save_path: Path,
    n_components: int = 20,
    title: Optional[str] = None,
) -> None:
    """Plot PCA explained variance curves.

    Args:
        baseline_samples: Baseline samples.
        constrained_samples: Samples for constrained comparison.
        weights: Importance weights.
        save_path: Path to save figure.
        n_components: Number of components to plot.
        title: Optional title override.
    """
    from sklearn.decomposition import PCA

    n_samples = baseline_samples.shape[0]
    samples_flat = baseline_samples.reshape(n_samples, -1)

    n_components = min(n_components, n_samples - 1, samples_flat.shape[1])

    # Baseline PCA
    pca_baseline = PCA(n_components=n_components)
    pca_baseline.fit(samples_flat)

    # For constrained, use weighted resampling
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
        np.cumsum(pca_baseline.explained_variance_ratio_),
        "o-",
        label="Baseline",
        color="steelblue",
        markersize=5,
    )

    ax.plot(
        components,
        np.cumsum(pca_constrained.explained_variance_ratio_),
        "s-",
        label="Constrained",
        color="darkorange",
        markersize=5,
    )

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title(title or "PCA Explained Variance", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot 2D PCA scatter comparing baseline vs constrained.

    Args:
        samples: Samples array.
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
        title: Optional title override.
    """
    from sklearn.decomposition import PCA

    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    # Fit PCA on all samples
    pca = PCA(n_components=2)
    projected = pca.fit_transform(samples_flat)

    # Resample for constrained visualization
    rng = np.random.default_rng(42)
    weights_norm = weights / np.sum(weights)
    n_show = min(2000, n_samples)  # Limit points for clarity
    baseline_idx = rng.choice(n_samples, size=n_show, replace=False)
    constrained_idx = rng.choice(n_samples, size=n_show, replace=True, p=weights_norm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline scatter
    axes[0].scatter(
        projected[baseline_idx, 0],
        projected[baseline_idx, 1],
        alpha=0.3,
        s=10,
        c="steelblue",
    )
    axes[0].set_xlabel("PC1", fontsize=11)
    axes[0].set_ylabel("PC2", fontsize=11)
    axes[0].set_title("Baseline", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Constrained scatter
    axes[1].scatter(
        projected[constrained_idx, 0],
        projected[constrained_idx, 1],
        alpha=0.3,
        s=10,
        c="darkorange",
    )
    axes[1].set_xlabel("PC1", fontsize=11)
    axes[1].set_ylabel("PC2", fontsize=11)
    axes[1].set_title(f"Constrained (β={beta})", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Match axis limits
    xlim = (
        min(axes[0].get_xlim()[0], axes[1].get_xlim()[0]),
        max(axes[0].get_xlim()[1], axes[1].get_xlim()[1]),
    )
    ylim = (
        min(axes[0].get_ylim()[0], axes[1].get_ylim()[0]),
        max(axes[0].get_ylim()[1], axes[1].get_ylim()[1]),
    )
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(title or f"PCA Projection (β={beta})", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_singular_value_spectra(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot singular value spectra for matrices.

    Args:
        samples: Matrix samples (n_samples, k, k).
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
        title: Optional title override.
    """
    if samples.ndim != 3:
        raise ValueError("Expected 3D array (matrices)")

    n_samples = samples.shape[0]
    k = samples.shape[1]

    # Compute SVs for all samples
    svs = np.zeros((n_samples, k))
    for i in range(n_samples):
        try:
            svs[i] = np.linalg.svd(samples[i], compute_uv=False)
        except np.linalg.LinAlgError:
            svs[i] = np.zeros(k)

    # Baseline statistics
    sv_mean_baseline = np.mean(svs, axis=0)
    sv_std_baseline = np.std(svs, axis=0)

    # Constrained statistics (weighted)
    weights_norm = weights / np.sum(weights)
    sv_mean_constrained = np.sum(weights_norm[:, None] * svs, axis=0)
    sv_var_constrained = np.sum(
        weights_norm[:, None] * (svs - sv_mean_constrained) ** 2, axis=0
    )
    sv_std_constrained = np.sqrt(sv_var_constrained)

    fig, ax = plt.subplots(figsize=(8, 5))

    indices = np.arange(1, k + 1)

    # Baseline
    ax.errorbar(
        indices,
        sv_mean_baseline,
        yerr=sv_std_baseline,
        fmt="o-",
        label="Baseline",
        color="steelblue",
        capsize=3,
        markersize=6,
    )

    # Constrained
    ax.errorbar(
        indices + 0.1,
        sv_mean_constrained,
        yerr=sv_std_constrained,
        fmt="s-",
        label=f"Constrained (β={beta})",
        color="darkorange",
        capsize=3,
        markersize=6,
    )

    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Singular Value", fontsize=12)
    ax.set_title(title or f"Singular Value Spectrum (β={beta})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(indices)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hierarchy(
    samples: np.ndarray,
    weights: np.ndarray,
    beta: float,
    save_path: Path,
    is_matrix: bool = False,
    title: Optional[str] = None,
) -> None:
    """Plot mean sorted absolute values (hierarchy structure).

    Args:
        samples: Samples array.
        weights: Importance weights.
        beta: Beta value.
        save_path: Path to save figure.
        is_matrix: Whether samples are matrices.
        title: Optional title override.
    """
    n_samples = samples.shape[0]

    if is_matrix:
        # Use singular values
        k = samples.shape[1]
        magnitudes = np.zeros((n_samples, k))
        for i in range(n_samples):
            try:
                magnitudes[i] = np.linalg.svd(samples[i], compute_uv=False)
            except np.linalg.LinAlgError:
                magnitudes[i] = np.zeros(k)
    else:
        # Use absolute values of components
        magnitudes = np.abs(samples)

    # Sort each row in descending order
    sorted_mags = np.sort(magnitudes, axis=1)[:, ::-1]

    # Baseline mean
    mean_baseline = np.mean(sorted_mags, axis=0)

    # Constrained mean (weighted)
    weights_norm = weights / np.sum(weights)
    mean_constrained = np.sum(weights_norm[:, None] * sorted_mags, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    indices = np.arange(1, len(mean_baseline) + 1)

    ax.semilogy(
        indices,
        mean_baseline + 1e-10,  # Add epsilon for log scale
        "o-",
        label="Baseline",
        color="steelblue",
        markersize=4,
    )

    ax.semilogy(
        indices,
        mean_constrained + 1e-10,
        "s-",
        label=f"Constrained (β={beta})",
        color="darkorange",
        markersize=4,
    )

    ax.set_xlabel("Rank (sorted by magnitude)", fontsize=12)
    ax.set_ylabel("Mean Magnitude (log scale)", fontsize=12)
    ax.set_title(title or f"Hierarchy Structure (β={beta})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_all_plots(
    samples: np.ndarray,
    energies: np.ndarray,
    weights: np.ndarray,
    beta: float,
    output_dir: Path,
    prefix: str = "",
    is_matrix: bool = False,
) -> List[Path]:
    """Create all comparison plots.

    Args:
        samples: Samples array.
        energies: Energy values.
        weights: Importance weights.
        beta: Beta value.
        output_dir: Output directory for figures.
        prefix: Prefix for filenames.
        is_matrix: Whether samples are matrices.

    Returns:
        List of created figure paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    prefix_str = f"{prefix}_" if prefix else ""
    beta_str = f"beta{beta:.1f}".replace(".", "p")

    # Energy histogram
    path = output_dir / f"{prefix_str}energy_histogram_{beta_str}.png"
    plot_energy_histogram(energies, weights, beta, path)
    created.append(path)

    # Norm distributions
    path = output_dir / f"{prefix_str}norm_distribution_{beta_str}.png"
    plot_norm_distributions(samples, weights, beta, path)
    created.append(path)

    # PCA variance
    path = output_dir / f"{prefix_str}pca_variance_{beta_str}.png"
    plot_pca_variance(samples, samples, weights, path)
    created.append(path)

    # PCA scatter
    path = output_dir / f"{prefix_str}pca_scatter_{beta_str}.png"
    plot_pca_scatter(samples, weights, beta, path)
    created.append(path)

    # Hierarchy
    path = output_dir / f"{prefix_str}hierarchy_{beta_str}.png"
    plot_hierarchy(samples, weights, beta, path, is_matrix=is_matrix)
    created.append(path)

    # Singular value spectra (matrices only)
    if is_matrix:
        path = output_dir / f"{prefix_str}sv_spectrum_{beta_str}.png"
        plot_singular_value_spectra(samples, weights, beta, path)
        created.append(path)

    return created
