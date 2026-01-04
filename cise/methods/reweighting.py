"""
Energy-based reweighting method for CISE.

Applies constraints via Boltzmann weights: w(z) = exp(-beta * E(z))
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from cise.constraints.base import BaseConstraint


@dataclass
class ReweightingResult:
    """Result of energy-based reweighting.

    Attributes:
        samples: Original samples.
        weights: Normalized importance weights.
        raw_weights: Unnormalized weights (exp(-beta * E)).
        energies: Energy values for each sample.
        ess: Effective sample size.
        ess_ratio: ESS / n_samples ratio.
        beta: Beta value used.
        weight_stats: Statistics about the weights.
        resampled: Optional resampled ensemble.
        resample_indices: Indices used for resampling.
    """

    samples: np.ndarray
    weights: np.ndarray
    raw_weights: np.ndarray
    energies: np.ndarray
    ess: float
    ess_ratio: float
    beta: float
    weight_stats: Dict[str, float]
    resampled: Optional[np.ndarray] = None
    resample_indices: Optional[np.ndarray] = None


def compute_weights(
    samples: np.ndarray,
    constraints: List[BaseConstraint],
    beta: float,
    normalize: bool = True,
) -> tuple:
    """Compute importance weights from energy-based reweighting.

    Given samples z and energy E(z), computes:
        w(z) = exp(-beta * E(z))

    Args:
        samples: Array of samples.
        constraints: List of constraint objects.
        beta: Inverse temperature parameter.
        normalize: Whether to normalize weights to sum to 1.

    Returns:
        Tuple of (weights, total_energies, individual_energies).
            - weights: Normalized (or raw) importance weights, shape (n_samples,).
            - total_energies: Total energy per sample, shape (n_samples,).
            - individual_energies: Dict mapping constraint names to energy arrays.
    """
    n_samples = samples.shape[0]
    total_energies = np.zeros(n_samples)
    individual_energies = {}

    # Compute energies from all constraints
    for constraint in constraints:
        energies = constraint.energy(samples)
        individual_energies[constraint.name] = energies
        total_energies += energies

    # Compute Boltzmann weights
    # Use numerically stable computation by subtracting max
    scaled_energies = beta * total_energies
    max_energy = np.max(scaled_energies)
    raw_weights = np.exp(-(scaled_energies - max_energy))

    if normalize:
        weights = raw_weights / np.sum(raw_weights)
    else:
        weights = raw_weights

    return weights, total_energies, individual_energies


def compute_ess(weights: np.ndarray) -> float:
    """Compute effective sample size from importance weights.

    ESS = 1 / sum(w_i^2)

    where w_i are normalized weights summing to 1.

    Args:
        weights: Normalized importance weights (sum to 1).

    Returns:
        Effective sample size.
    """
    # Ensure weights are normalized
    weights_normalized = weights / np.sum(weights)

    # ESS = 1 / sum(w^2)
    ess = 1.0 / np.sum(weights_normalized ** 2)
    return ess


def importance_resample(
    samples: np.ndarray,
    weights: np.ndarray,
    n_resample: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Resample from ensemble using importance weights.

    Performs multinomial resampling to create a new ensemble where
    samples are drawn proportionally to their weights.

    Args:
        samples: Original samples.
        weights: Importance weights (will be normalized).
        n_resample: Number of samples to draw (default: same as original).
        rng: Random number generator.

    Returns:
        Tuple of (resampled_samples, indices).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = samples.shape[0]
    if n_resample is None:
        n_resample = n_samples

    # Normalize weights
    weights_normalized = weights / np.sum(weights)

    # Resample indices
    indices = rng.choice(n_samples, size=n_resample, replace=True, p=weights_normalized)

    # Get resampled samples
    resampled = samples[indices]

    return resampled, indices


def apply_reweighting(
    samples: np.ndarray,
    constraints: List[BaseConstraint],
    beta: float,
    resample: bool = False,
    n_resample: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> ReweightingResult:
    """Apply energy-based reweighting to samples.

    This is the main interface for Method A: Energy-based reweighting.

    Args:
        samples: Array of samples.
        constraints: List of constraint objects.
        beta: Inverse temperature parameter.
        resample: Whether to create resampled ensemble.
        n_resample: Number of samples for resampling (default: same as original).
        rng: Random number generator.

    Returns:
        ReweightingResult with weights, energies, ESS, and optional resampled ensemble.
    """
    n_samples = samples.shape[0]

    # Compute weights
    weights, total_energies, individual_energies = compute_weights(
        samples, constraints, beta, normalize=True
    )

    # Compute raw weights for reference
    scaled_energies = beta * total_energies
    max_energy = np.max(scaled_energies)
    raw_weights = np.exp(-(scaled_energies - max_energy))

    # Compute ESS
    ess = compute_ess(weights)
    ess_ratio = ess / n_samples

    # Compute weight statistics
    weight_stats = {
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "max_min_ratio": float(np.max(weights) / (np.min(weights) + 1e-10)),
        "normalization": float(np.sum(raw_weights)),
    }

    result = ReweightingResult(
        samples=samples,
        weights=weights,
        raw_weights=raw_weights,
        energies=total_energies,
        ess=ess,
        ess_ratio=ess_ratio,
        beta=beta,
        weight_stats=weight_stats,
    )

    # Optionally resample
    if resample:
        resampled, indices = importance_resample(
            samples, weights, n_resample=n_resample, rng=rng
        )
        result.resampled = resampled
        result.resample_indices = indices

    return result
