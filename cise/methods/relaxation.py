"""
Relaxation method for CISE (optional Method B).

Applies constraints via iterative relaxation / gradient descent.
This is an optional extension for v1, only to be used after
core reweighting functionality is complete.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np

from cise.constraints.base import BaseConstraint


@dataclass
class RelaxationResult:
    """Result of relaxation optimization.

    Attributes:
        initial_samples: Samples before relaxation.
        relaxed_samples: Samples after relaxation.
        initial_energies: Energy values before relaxation.
        final_energies: Energy values after relaxation.
        n_iterations: Number of iterations performed.
        converged: Whether optimization converged.
    """

    initial_samples: np.ndarray
    relaxed_samples: np.ndarray
    initial_energies: np.ndarray
    final_energies: np.ndarray
    n_iterations: int
    converged: bool


def numerical_gradient(
    sample: np.ndarray,
    energy_fn: Callable[[np.ndarray], float],
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient of energy function.

    Args:
        sample: Single sample.
        energy_fn: Function that computes energy for a sample.
        epsilon: Step size for finite differences.

    Returns:
        Gradient array with same shape as sample.
    """
    gradient = np.zeros_like(sample)
    flat_sample = sample.flatten()
    flat_grad = gradient.flatten()

    for i in range(len(flat_sample)):
        sample_plus = flat_sample.copy()
        sample_minus = flat_sample.copy()
        sample_plus[i] += epsilon
        sample_minus[i] -= epsilon

        e_plus = energy_fn(sample_plus.reshape(sample.shape))
        e_minus = energy_fn(sample_minus.reshape(sample.shape))

        flat_grad[i] = (e_plus - e_minus) / (2 * epsilon)

    return flat_grad.reshape(sample.shape)


def relax_sample(
    sample: np.ndarray,
    constraints: List[BaseConstraint],
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple:
    """Relax a single sample to minimize constraint energy.

    Args:
        sample: Initial sample.
        constraints: List of constraints.
        learning_rate: Step size for gradient descent.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence threshold for energy change.

    Returns:
        Tuple of (relaxed_sample, final_energy, n_iterations, converged).
    """

    def total_energy(s: np.ndarray) -> float:
        s_batch = s.reshape(1, *s.shape)
        total = 0.0
        for c in constraints:
            total += float(c.energy(s_batch)[0])
        return total

    current = sample.copy()
    prev_energy = total_energy(current)

    for iteration in range(max_iterations):
        # Compute gradient
        grad = numerical_gradient(current, total_energy)

        # Gradient descent step
        current = current - learning_rate * grad

        # Check convergence
        current_energy = total_energy(current)
        energy_change = abs(prev_energy - current_energy)

        if energy_change < tolerance:
            return current, current_energy, iteration + 1, True

        prev_energy = current_energy

    return current, prev_energy, max_iterations, False


def apply_relaxation(
    samples: np.ndarray,
    constraints: List[BaseConstraint],
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> RelaxationResult:
    """Apply relaxation to all samples.

    Note: This method is computationally expensive and optional for v1.

    Args:
        samples: Array of samples.
        constraints: List of constraints.
        learning_rate: Step size for gradient descent.
        max_iterations: Maximum iterations per sample.
        tolerance: Convergence threshold.

    Returns:
        RelaxationResult with initial and relaxed samples.
    """
    n_samples = samples.shape[0]
    relaxed = np.zeros_like(samples)
    final_energies = np.zeros(n_samples)
    all_converged = True
    total_iterations = 0

    # Compute initial energies
    initial_energies = np.zeros(n_samples)
    for c in constraints:
        initial_energies += c.energy(samples)

    # Relax each sample
    for i in range(n_samples):
        relaxed[i], final_energies[i], n_iter, converged = relax_sample(
            samples[i],
            constraints,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        total_iterations += n_iter
        all_converged = all_converged and converged

    return RelaxationResult(
        initial_samples=samples,
        relaxed_samples=relaxed,
        initial_energies=initial_energies,
        final_energies=final_energies,
        n_iterations=total_iterations,
        converged=all_converged,
    )
