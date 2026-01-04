"""
L1 constraint for CISE.

Penalizes non-sparse samples using L1 norm.
"""

import numpy as np

from cise.constraints.base import BaseConstraint, ConstraintType


class L1Constraint(BaseConstraint):
    """L1 simplicity constraint for vectors and matrices.

    Penalizes samples with large absolute values, encouraging sparsity:

        For vectors: E_L1(x) = sum_i |x_i|
        For matrices: E_L1(M) = sum_{ij} |M_{ij}|

    This encourages simpler representations with fewer large components.
    """

    applicable_types = [ConstraintType.BOTH]

    def __init__(self, scale: float = 1.0, **params):
        """Initialize L1 constraint.

        Args:
            scale: Scaling factor for the energy (default 1.0).
            **params: Additional parameters.
        """
        super().__init__(scale=scale, **params)
        self.scale = scale

    def energy(self, samples: np.ndarray) -> np.ndarray:
        """Compute L1 energy for samples.

        Args:
            samples: Array of samples.
                - For vectors: shape (n_samples, dim)
                - For matrices: shape (n_samples, k, k)

        Returns:
            Array of energies, shape (n_samples,).
        """
        n_samples = samples.shape[0]

        # Flatten each sample and compute L1 norm
        samples_flat = samples.reshape(n_samples, -1)
        energies = np.sum(np.abs(samples_flat), axis=1)

        return self.scale * energies
