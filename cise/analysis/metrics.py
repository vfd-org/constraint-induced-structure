"""
Metrics computation for CISE.

Computes various statistics comparing baseline and constrained ensembles.
"""

from typing import Dict, Any, Optional, List
import numpy as np


def compute_norm_stats(
    samples: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute norm statistics for samples.

    For vectors: L2 norm ||x||_2
    For matrices: Frobenius norm ||M||_F

    Args:
        samples: Array of samples (n_samples, ...).
        weights: Optional importance weights for weighted statistics.

    Returns:
        Dictionary with norm statistics.
    """
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    # Compute L2/Frobenius norms
    norms = np.linalg.norm(samples_flat, axis=1)

    if weights is not None:
        weights_norm = weights / np.sum(weights)
        mean = np.sum(weights_norm * norms)
        var = np.sum(weights_norm * (norms - mean) ** 2)
        std = np.sqrt(var)
    else:
        mean = np.mean(norms)
        std = np.std(norms)

    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "median": float(np.median(norms)),
    }


def compute_energy_stats(energies: np.ndarray) -> Dict[str, float]:
    """Compute energy distribution statistics.

    Args:
        energies: Array of energy values.

    Returns:
        Dictionary with energy statistics.
    """
    return {
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies)),
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "median": float(np.median(energies)),
        "percentile_25": float(np.percentile(energies, 25)),
        "percentile_75": float(np.percentile(energies, 75)),
    }


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient of absolute values.

    Gini coefficient measures inequality in distribution.
    0 = perfect equality, 1 = maximum inequality.

    Args:
        values: Array of values.

    Returns:
        Gini coefficient between 0 and 1.
    """
    abs_values = np.abs(values.flatten())
    if len(abs_values) == 0:
        return 0.0

    # Sort values
    sorted_values = np.sort(abs_values)
    n = len(sorted_values)

    # Compute Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    return float(max(0, gini))


def compute_compressibility(
    samples: np.ndarray,
    epsilon: float = 0.1,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute compressibility proxies for samples.

    Args:
        samples: Array of samples.
        epsilon: Threshold for "near zero" values.
        weights: Optional importance weights.

    Returns:
        Dictionary with compressibility metrics.
    """
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)
    n_elements = samples_flat.shape[1]

    # Fraction near zero
    near_zero = np.abs(samples_flat) < epsilon
    fraction_near_zero = np.mean(near_zero, axis=1)  # per sample

    # Gini coefficients per sample
    gini_per_sample = np.array([compute_gini_coefficient(s) for s in samples_flat])

    if weights is not None:
        weights_norm = weights / np.sum(weights)
        mean_fraction = np.sum(weights_norm * fraction_near_zero)
        mean_gini = np.sum(weights_norm * gini_per_sample)
    else:
        mean_fraction = np.mean(fraction_near_zero)
        mean_gini = np.mean(gini_per_sample)

    return {
        "fraction_near_zero": float(mean_fraction),
        "gini_coefficient": float(mean_gini),
        "n_elements": int(n_elements),
        "epsilon": float(epsilon),
    }


def compute_matrix_metrics(
    samples: np.ndarray,
    weights: Optional[np.ndarray] = None,
    rank_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Compute matrix-specific metrics.

    Args:
        samples: Array of matrices (n_samples, k, k).
        weights: Optional importance weights.
        rank_threshold: Threshold for counting significant singular values.

    Returns:
        Dictionary with matrix metrics.
    """
    if samples.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {samples.shape}")

    n_samples = samples.shape[0]
    k = samples.shape[1]

    # Compute singular values for all matrices
    singular_values = np.zeros((n_samples, k))
    condition_numbers = np.zeros(n_samples)
    rank_proxies = np.zeros(n_samples)

    for i in range(n_samples):
        try:
            svs = np.linalg.svd(samples[i], compute_uv=False)
            singular_values[i] = svs
            # Condition number (avoid division by zero)
            if svs[-1] > 1e-10:
                condition_numbers[i] = svs[0] / svs[-1]
            else:
                condition_numbers[i] = np.inf
            # Rank proxy: number of singular values above threshold * max
            threshold = rank_threshold * svs[0] if svs[0] > 0 else rank_threshold
            rank_proxies[i] = np.sum(svs > threshold)
        except np.linalg.LinAlgError:
            singular_values[i] = np.zeros(k)
            condition_numbers[i] = np.inf
            rank_proxies[i] = 0

    # Compute statistics
    if weights is not None:
        weights_norm = weights / np.sum(weights)
        sv_mean = np.sum(weights_norm[:, None] * singular_values, axis=0)
        sv_std = np.sqrt(
            np.sum(weights_norm[:, None] * (singular_values - sv_mean) ** 2, axis=0)
        )

        finite_cond = np.isfinite(condition_numbers)
        if np.any(finite_cond):
            cond_mean = np.sum(weights_norm[finite_cond] * condition_numbers[finite_cond])
        else:
            cond_mean = np.inf

        rank_mean = np.sum(weights_norm * rank_proxies)
    else:
        sv_mean = np.mean(singular_values, axis=0)
        sv_std = np.std(singular_values, axis=0)

        finite_cond = np.isfinite(condition_numbers)
        cond_mean = np.mean(condition_numbers[finite_cond]) if np.any(finite_cond) else np.inf

        rank_mean = np.mean(rank_proxies)

    return {
        "singular_value_mean": sv_mean.tolist(),
        "singular_value_std": sv_std.tolist(),
        "condition_number_mean": float(cond_mean) if np.isfinite(cond_mean) else None,
        "rank_proxy_mean": float(rank_mean),
        "rank_threshold": float(rank_threshold),
    }


def compute_metrics(
    baseline_samples: np.ndarray,
    constrained_samples: np.ndarray,
    energies: np.ndarray,
    weights: np.ndarray,
    ess: float,
    beta: float,
    is_matrix: bool = False,
) -> Dict[str, Any]:
    """Compute all metrics comparing baseline and constrained ensembles.

    Args:
        baseline_samples: Original baseline samples.
        constrained_samples: Samples with importance weights applied (or resampled).
        energies: Energy values for each sample.
        weights: Importance weights.
        ess: Effective sample size.
        beta: Beta value used.
        is_matrix: Whether samples are matrices.

    Returns:
        Comprehensive metrics dictionary.
    """
    metrics = {
        "beta": float(beta),
        "n_samples": int(baseline_samples.shape[0]),
        "ess": float(ess),
        "ess_ratio": float(ess / baseline_samples.shape[0]),
    }

    # Norm statistics
    metrics["baseline_norms"] = compute_norm_stats(baseline_samples)
    metrics["constrained_norms"] = compute_norm_stats(baseline_samples, weights=weights)

    # Energy statistics
    metrics["energy_stats"] = compute_energy_stats(energies)

    # Compressibility
    metrics["baseline_compressibility"] = compute_compressibility(baseline_samples)
    metrics["constrained_compressibility"] = compute_compressibility(
        baseline_samples, weights=weights
    )

    # Matrix-specific metrics
    if is_matrix:
        metrics["baseline_matrix"] = compute_matrix_metrics(baseline_samples)
        metrics["constrained_matrix"] = compute_matrix_metrics(baseline_samples, weights=weights)

    # Compute deltas for key metrics
    metrics["deltas"] = {
        "norm_mean": metrics["constrained_norms"]["mean"] - metrics["baseline_norms"]["mean"],
        "gini": (
            metrics["constrained_compressibility"]["gini_coefficient"]
            - metrics["baseline_compressibility"]["gini_coefficient"]
        ),
        "fraction_near_zero": (
            metrics["constrained_compressibility"]["fraction_near_zero"]
            - metrics["baseline_compressibility"]["fraction_near_zero"]
        ),
    }

    if is_matrix and metrics.get("constrained_matrix"):
        metrics["deltas"]["rank_proxy"] = (
            metrics["constrained_matrix"]["rank_proxy_mean"]
            - metrics["baseline_matrix"]["rank_proxy_mean"]
        )

    return metrics
