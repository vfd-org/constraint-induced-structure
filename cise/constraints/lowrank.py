"""
Low-rank constraint for CISE.

Penalizes matrices that deviate from low-rank structure.
"""

import numpy as np

from cise.constraints.base import BaseConstraint, ConstraintType


class LowRankConstraint(BaseConstraint):
    """Low-rank constraint for matrices.

    Penalizes matrices with significant singular values beyond a target rank:

        E_rank(M) = sum_{j=r+1}^{k} sigma_j^2

    where sigma_j are the singular values of M sorted in descending order,
    and r is the target rank.

    This encourages matrices with approximate low-rank structure.
    """

    applicable_types = [ConstraintType.MATRIX]

    def __init__(self, target_rank: int = 2, scale: float = 1.0, **params):
        """Initialize low-rank constraint.

        Args:
            target_rank: Target rank r (default 2).
            scale: Scaling factor for the energy (default 1.0).
            **params: Additional parameters.
        """
        super().__init__(target_rank=target_rank, scale=scale, **params)
        self.target_rank = target_rank
        self.scale = scale

    def energy(self, samples: np.ndarray) -> np.ndarray:
        """Compute low-rank energy for matrix samples.

        Args:
            samples: Array of shape (n_samples, k, k).

        Returns:
            Array of energies, shape (n_samples,).

        Raises:
            ValueError: If samples don't have expected shape.
        """
        if samples.ndim != 3:
            raise ValueError(
                f"Low-rank constraint expects 3D array (n_samples, k, k), "
                f"got shape {samples.shape}"
            )

        n_samples = samples.shape[0]
        energies = np.zeros(n_samples)

        for i in range(n_samples):
            # Compute singular values
            try:
                singular_values = np.linalg.svd(samples[i], compute_uv=False)
            except np.linalg.LinAlgError:
                # Handle numerical issues gracefully
                singular_values = np.zeros(samples.shape[1])

            # Sum squared singular values beyond target rank
            if len(singular_values) > self.target_rank:
                energies[i] = np.sum(singular_values[self.target_rank:] ** 2)

        return self.scale * energies

    def get_singular_values(self, samples: np.ndarray) -> np.ndarray:
        """Get singular values for all samples (utility method).

        Args:
            samples: Array of shape (n_samples, k, k).

        Returns:
            Array of shape (n_samples, k) with singular values.
        """
        n_samples = samples.shape[0]
        k = min(samples.shape[1], samples.shape[2])
        svs = np.zeros((n_samples, k))

        for i in range(n_samples):
            try:
                svs[i] = np.linalg.svd(samples[i], compute_uv=False)
            except np.linalg.LinAlgError:
                svs[i] = np.zeros(k)

        return svs
