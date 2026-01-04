"""
Smoothness constraint for CISE.

Penalizes large differences between adjacent vector components.
"""

import numpy as np

from cise.constraints.base import BaseConstraint, ConstraintType


class SmoothnessConstraint(BaseConstraint):
    """Smoothness constraint for vectors.

    Penalizes non-smooth vectors by measuring squared differences
    between adjacent components:

        E_smooth(x) = sum_{i=1}^{d-1} (x_{i+1} - x_i)^2

    This encourages vectors with gradual transitions between components.
    """

    applicable_types = [ConstraintType.VECTOR]

    def __init__(self, scale: float = 1.0, **params):
        """Initialize smoothness constraint.

        Args:
            scale: Scaling factor for the energy (default 1.0).
            **params: Additional parameters.
        """
        super().__init__(scale=scale, **params)
        self.scale = scale

    def energy(self, samples: np.ndarray) -> np.ndarray:
        """Compute smoothness energy for vector samples.

        Args:
            samples: Array of shape (n_samples, dim).

        Returns:
            Array of energies, shape (n_samples,).

        Raises:
            ValueError: If samples don't have expected shape.
        """
        if samples.ndim != 2:
            raise ValueError(
                f"Smoothness constraint expects 2D array (n_samples, dim), "
                f"got shape {samples.shape}"
            )

        # Compute differences between adjacent components
        diffs = np.diff(samples, axis=1)  # shape (n_samples, dim-1)

        # Sum of squared differences
        energies = np.sum(diffs ** 2, axis=1)  # shape (n_samples,)

        return self.scale * energies
