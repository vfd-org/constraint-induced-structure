"""
Gaussian sampler for CISE.

Generates samples from standard normal distribution.
"""

from typing import Optional
import numpy as np

from cise.sampling.base import BaseSampler


class GaussianSampler(BaseSampler):
    """Gaussian (normal) distribution sampler.

    Generates:
    - Vectors: x ~ N(0, I), each component independently from N(0, 1)
    - Matrices: M_ij ~ N(0, 1), each entry independently from N(0, 1)
    """

    def __init__(self, seed: Optional[int] = None, mean: float = 0.0, std: float = 1.0):
        """Initialize Gaussian sampler.

        Args:
            seed: Random seed for reproducibility.
            mean: Mean of the distribution (default 0).
            std: Standard deviation of the distribution (default 1).
        """
        super().__init__(seed)
        self.mean = mean
        self.std = std

    def sample_vectors(self, n_samples: int, dim: int) -> np.ndarray:
        """Generate vector samples from N(mean, std^2 * I).

        Args:
            n_samples: Number of vectors to generate.
            dim: Dimension of each vector.

        Returns:
            Array of shape (n_samples, dim) with Gaussian entries.
        """
        return self._rng.normal(self.mean, self.std, size=(n_samples, dim))

    def sample_matrices(self, n_samples: int, size: int) -> np.ndarray:
        """Generate matrix samples with Gaussian entries.

        Args:
            n_samples: Number of matrices to generate.
            size: Size of each square matrix.

        Returns:
            Array of shape (n_samples, size, size) with Gaussian entries.
        """
        return self._rng.normal(self.mean, self.std, size=(n_samples, size, size))

    def __repr__(self) -> str:
        return f"GaussianSampler(seed={self.seed}, mean={self.mean}, std={self.std})"
