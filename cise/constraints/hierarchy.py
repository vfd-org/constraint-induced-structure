"""
Hierarchy constraint for CISE.

Penalizes deviation from geometric decay in magnitude hierarchy.
"""

import numpy as np

from cise.constraints.base import BaseConstraint, ConstraintType


class HierarchyConstraint(BaseConstraint):
    """Hierarchy constraint for vectors and matrices.

    Penalizes deviation from geometric decay in sorted absolute magnitudes:

        E_hier(a) = sum_{i=1}^{n-1} (log(a_{i+1}) - log(a_i) - delta)^2

    where a_1 >= a_2 >= ... >= a_n are the sorted absolute magnitudes,
    and delta is the expected log-ratio between consecutive magnitudes
    (negative for decay).

    For vectors: magnitudes are |x_i|
    For matrices: magnitudes are singular values

    This encourages hierarchical structure with consistent decay ratios.
    """

    applicable_types = [ConstraintType.BOTH]

    def __init__(
        self,
        delta: float = -0.5,
        epsilon: float = 1e-10,
        scale: float = 1.0,
        **params
    ):
        """Initialize hierarchy constraint.

        Args:
            delta: Expected log-ratio between consecutive magnitudes (default -0.5).
                   Negative values encourage decay.
            epsilon: Small constant to avoid log(0) (default 1e-10).
            scale: Scaling factor for the energy (default 1.0).
            **params: Additional parameters.
        """
        super().__init__(delta=delta, epsilon=epsilon, scale=scale, **params)
        self.delta = delta
        self.epsilon = epsilon
        self.scale = scale

    def energy(self, samples: np.ndarray) -> np.ndarray:
        """Compute hierarchy energy for samples.

        Args:
            samples: Array of samples.
                - For vectors: shape (n_samples, dim)
                - For matrices: shape (n_samples, k, k)

        Returns:
            Array of energies, shape (n_samples,).
        """
        n_samples = samples.shape[0]
        energies = np.zeros(n_samples)

        for i in range(n_samples):
            magnitudes = self._get_magnitudes(samples[i])
            energies[i] = self._compute_hierarchy_energy(magnitudes)

        return self.scale * energies

    def _get_magnitudes(self, sample: np.ndarray) -> np.ndarray:
        """Extract sorted absolute magnitudes from a sample.

        Args:
            sample: Single sample (vector or matrix).

        Returns:
            Sorted absolute magnitudes in descending order.
        """
        if sample.ndim == 1:
            # Vector: use absolute values of components
            magnitudes = np.abs(sample)
        elif sample.ndim == 2:
            # Matrix: use singular values
            try:
                magnitudes = np.linalg.svd(sample, compute_uv=False)
            except np.linalg.LinAlgError:
                magnitudes = np.zeros(min(sample.shape))
        else:
            raise ValueError(f"Unexpected sample dimension: {sample.ndim}")

        # Sort in descending order
        magnitudes = np.sort(magnitudes)[::-1]
        return magnitudes

    def _compute_hierarchy_energy(self, magnitudes: np.ndarray) -> float:
        """Compute hierarchy energy for sorted magnitudes.

        Args:
            magnitudes: Sorted absolute magnitudes (descending order).

        Returns:
            Hierarchy energy value.
        """
        if len(magnitudes) < 2:
            return 0.0

        # Add epsilon to avoid log(0)
        log_mags = np.log(magnitudes + self.epsilon)

        # Compute log-ratios between consecutive magnitudes
        log_ratios = np.diff(log_mags)  # log(a_{i+1}) - log(a_i)

        # Penalize deviation from expected delta
        deviations = log_ratios - self.delta
        energy = np.sum(deviations ** 2)

        return energy
