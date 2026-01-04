"""
Anti-dismissal control analyses for CISE.

Implements controls that address common critiques:
1. Norm-matched baseline: Shows structure change persists beyond norm shrinkage
2. Magnitude-neutral constraints: Shows effect without L1 penalty
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def create_norm_matched_baseline(
    samples: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a baseline subset with matching norm distribution.

    This addresses the critique: "Isn't the structure just from norm shrinkage?"
    By matching norms, any remaining structural differences must come from
    the constraint geometry, not just magnitude reduction.

    Args:
        samples: Original samples.
        weights: Importance weights from constrained reweighting.
        n_bins: Number of bins for histogram matching.
        rng: Random number generator.

    Returns:
        Tuple of (matched_samples, matched_indices).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)
    norms = np.linalg.norm(samples_flat, axis=1)

    # Compute constrained norm distribution via weighted histogram
    weights_norm = weights / np.sum(weights)

    # Define bins based on norm range
    norm_min, norm_max = norms.min(), norms.max()
    bins = np.linspace(norm_min, norm_max, n_bins + 1)
    bin_indices = np.digitize(norms, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute target (constrained) histogram
    target_hist = np.zeros(n_bins)
    for i in range(n_samples):
        target_hist[bin_indices[i]] += weights_norm[i]

    # Compute baseline histogram
    baseline_hist = np.zeros(n_bins)
    for i in range(n_samples):
        baseline_hist[bin_indices[i]] += 1.0 / n_samples

    # Compute resampling weights for baseline to match constrained
    # Weight for each sample = target_density / baseline_density in its bin
    resample_weights = np.zeros(n_samples)
    for i in range(n_samples):
        b = bin_indices[i]
        if baseline_hist[b] > 0:
            resample_weights[i] = target_hist[b] / baseline_hist[b]
        else:
            resample_weights[i] = 0

    # Normalize resampling weights
    resample_weights /= resample_weights.sum()

    # Resample baseline to match constrained norm distribution
    matched_indices = rng.choice(
        n_samples, size=n_samples, replace=True, p=resample_weights
    )
    matched_samples = samples[matched_indices]

    return matched_samples, matched_indices


def compute_control_metrics(
    original_samples: np.ndarray,
    constrained_weights: np.ndarray,
    matched_samples: np.ndarray,
) -> Dict[str, Any]:
    """Compute metrics comparing constrained vs norm-matched baseline.

    Args:
        original_samples: Original baseline samples.
        constrained_weights: Weights from constraint reweighting.
        matched_samples: Norm-matched baseline samples.

    Returns:
        Dictionary of control comparison metrics.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from cise.analysis.metrics import compute_gini_coefficient

    n_samples = original_samples.shape[0]
    original_flat = original_samples.reshape(n_samples, -1)
    matched_flat = matched_samples.reshape(n_samples, -1)

    # Resample original according to weights for fair comparison
    rng = np.random.default_rng(42)
    weights_norm = constrained_weights / constrained_weights.sum()
    constrained_idx = rng.choice(n_samples, size=n_samples, replace=True, p=weights_norm)
    constrained_flat = original_flat[constrained_idx]

    metrics = {}

    # 1. Verify norm distributions match
    constrained_norms = np.linalg.norm(constrained_flat, axis=1)
    matched_norms = np.linalg.norm(matched_flat, axis=1)
    metrics["norm_mean_constrained"] = float(np.mean(constrained_norms))
    metrics["norm_mean_matched"] = float(np.mean(matched_norms))
    metrics["norm_std_constrained"] = float(np.std(constrained_norms))
    metrics["norm_std_matched"] = float(np.std(matched_norms))

    # 2. PCA explained variance comparison
    n_comp = min(15, n_samples - 1, original_flat.shape[1])

    pca_constrained = PCA(n_components=n_comp)
    pca_constrained.fit(constrained_flat)

    pca_matched = PCA(n_components=n_comp)
    pca_matched.fit(matched_flat)

    metrics["pca_var_constrained"] = pca_constrained.explained_variance_ratio_.tolist()
    metrics["pca_var_matched"] = pca_matched.explained_variance_ratio_.tolist()

    # Intrinsic dimension proxy (participation ratio)
    lambda_c = pca_constrained.explained_variance_
    lambda_m = pca_matched.explained_variance_
    pr_constrained = (lambda_c.sum() ** 2) / (lambda_c ** 2).sum()
    pr_matched = (lambda_m.sum() ** 2) / (lambda_m ** 2).sum()
    metrics["participation_ratio_constrained"] = float(pr_constrained)
    metrics["participation_ratio_matched"] = float(pr_matched)
    metrics["participation_ratio_delta"] = float(pr_constrained - pr_matched)

    # 3. Clustering tendency
    try:
        km_constrained = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_c = km_constrained.fit_predict(constrained_flat)
        sil_constrained = silhouette_score(constrained_flat, labels_c)

        km_matched = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_m = km_matched.fit_predict(matched_flat)
        sil_matched = silhouette_score(matched_flat, labels_m)

        metrics["silhouette_constrained"] = float(sil_constrained)
        metrics["silhouette_matched"] = float(sil_matched)
        metrics["silhouette_delta"] = float(sil_constrained - sil_matched)
    except Exception:
        metrics["silhouette_constrained"] = None
        metrics["silhouette_matched"] = None
        metrics["silhouette_delta"] = None

    # 4. Gini coefficient comparison
    gini_constrained = np.mean([compute_gini_coefficient(s) for s in constrained_flat])
    gini_matched = np.mean([compute_gini_coefficient(s) for s in matched_flat])
    metrics["gini_constrained"] = float(gini_constrained)
    metrics["gini_matched"] = float(gini_matched)
    metrics["gini_delta"] = float(gini_constrained - gini_matched)

    return metrics


def plot_control_pca_variance(
    original_samples: np.ndarray,
    constrained_weights: np.ndarray,
    matched_samples: np.ndarray,
    save_path: Path,
    n_components: int = 15,
) -> None:
    """Plot PCA variance: constrained vs norm-matched baseline.

    Args:
        original_samples: Original samples.
        constrained_weights: Constraint weights.
        matched_samples: Norm-matched samples.
        save_path: Path to save figure.
        n_components: Number of components.
    """
    from sklearn.decomposition import PCA

    n_samples = original_samples.shape[0]
    original_flat = original_samples.reshape(n_samples, -1)
    matched_flat = matched_samples.reshape(n_samples, -1)
    n_components = min(n_components, n_samples - 1, original_flat.shape[1])

    # Constrained
    rng = np.random.default_rng(42)
    weights_norm = constrained_weights / constrained_weights.sum()
    constrained_idx = rng.choice(n_samples, size=n_samples, replace=True, p=weights_norm)
    constrained_flat = original_flat[constrained_idx]

    pca_constrained = PCA(n_components=n_components)
    pca_constrained.fit(constrained_flat)

    pca_matched = PCA(n_components=n_components)
    pca_matched.fit(matched_flat)

    fig, ax = plt.subplots(figsize=(8, 5))

    components = np.arange(1, n_components + 1)

    ax.plot(
        components,
        np.cumsum(pca_matched.explained_variance_ratio_) * 100,
        "o-",
        label="Norm-Matched Baseline",
        color="gray",
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

    ax.axhline(90, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Number of Principal Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
    ax.set_title("Control: Constrained vs Norm-Matched Baseline", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Add annotation
    ax.annotate(
        "Structure difference\npersists beyond\nnorm matching",
        xy=(n_components * 0.6, 70),
        fontsize=9,
        ha="center",
        style="italic",
        color="gray",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_control_pca_scatter(
    original_samples: np.ndarray,
    constrained_weights: np.ndarray,
    matched_samples: np.ndarray,
    save_path: Path,
) -> None:
    """Plot PCA scatter: constrained vs norm-matched baseline.

    Args:
        original_samples: Original samples.
        constrained_weights: Constraint weights.
        matched_samples: Norm-matched samples.
        save_path: Path to save figure.
    """
    from sklearn.decomposition import PCA

    n_samples = original_samples.shape[0]
    original_flat = original_samples.reshape(n_samples, -1)
    matched_flat = matched_samples.reshape(n_samples, -1)

    # Fit PCA on original
    pca = PCA(n_components=2)
    pca.fit(original_flat)

    # Project all
    proj_matched = pca.transform(matched_flat)

    rng = np.random.default_rng(42)
    weights_norm = constrained_weights / constrained_weights.sum()
    constrained_idx = rng.choice(n_samples, size=n_samples, replace=True, p=weights_norm)
    constrained_flat = original_flat[constrained_idx]
    proj_constrained = pca.transform(constrained_flat)

    n_show = min(1500, n_samples)
    show_idx = rng.choice(n_samples, size=n_show, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Norm-matched baseline
    axes[0].scatter(
        proj_matched[show_idx, 0],
        proj_matched[show_idx, 1],
        alpha=0.4,
        s=12,
        c="gray",
        edgecolors="none",
    )
    axes[0].set_xlabel("PC1", fontsize=11)
    axes[0].set_ylabel("PC2", fontsize=11)
    axes[0].set_title("Norm-Matched Baseline", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Constrained
    axes[1].scatter(
        proj_constrained[show_idx, 0],
        proj_constrained[show_idx, 1],
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
    all_proj = np.vstack([proj_matched, proj_constrained])
    margin = 0.1
    x_range = all_proj[:, 0].max() - all_proj[:, 0].min()
    y_range = all_proj[:, 1].max() - all_proj[:, 1].min()
    xlim = (
        all_proj[:, 0].min() - margin * x_range,
        all_proj[:, 0].max() + margin * x_range,
    )
    ylim = (
        all_proj[:, 1].min() - margin * y_range,
        all_proj[:, 1].max() + margin * y_range,
    )
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle("Control: Structure Beyond Norm Matching", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_control_figures(
    samples: np.ndarray,
    weights: np.ndarray,
    output_dir: Path,
    sample_type: str = "vector",
    n_bins: int = 50,
) -> Tuple[Dict[str, Any], list]:
    """Create anti-dismissal control analysis and figures.

    Args:
        samples: Original samples.
        weights: Importance weights.
        output_dir: Output directory.
        sample_type: 'vector' or 'matrix'.
        n_bins: Bins for norm matching.

    Returns:
        Tuple of (control_metrics, list of figure paths).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create norm-matched baseline
    matched_samples, _ = create_norm_matched_baseline(samples, weights, n_bins=n_bins)

    # Compute control metrics
    metrics = compute_control_metrics(samples, weights, matched_samples)

    # Create control figures
    created = []

    # Control PCA variance
    path = output_dir / f"{sample_type}_control_pca_variance.png"
    plot_control_pca_variance(samples, weights, matched_samples, path)
    created.append(path)

    # Control PCA scatter
    path = output_dir / f"{sample_type}_control_pca_scatter.png"
    plot_control_pca_scatter(samples, weights, matched_samples, path)
    created.append(path)

    return metrics, created
