"""
Uniform sampler for CISE.

Generates samples from uniform distribution over [-1, 1].
"""

from typing import Optional
import numpy as np

from cise.sampling.base import BaseSampler


class UniformSampler(BaseSampler):
    """Uniform distribution sampler.

    Generates:
    - Vectors: x_i ~ U(-1, 1), each component independently
    - Matrices: M_ij ~ U(-1, 1), each entry independently
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        low: float = -1.0,
        high: float = 1.0
    ):
        """Initialize Uniform sampler.

        Args:
            seed: Random seed for reproducibility.
            low: Lower bound of uniform distribution (default -1).
            high: Upper bound of uniform distribution (default 1).
        """
        super().__init__(seed)
        self.low = low
        self.high = high

    def sample_vectors(self, n_samples: int, dim: int) -> np.ndarray:
        """Generate vector samples from U(low, high).

        Args:
            n_samples: Number of vectors to generate.
            dim: Dimension of each vector.

        Returns:
            Array of shape (n_samples, dim) with uniform entries.
        """
        return self._rng.uniform(self.low, self.high, size=(n_samples, dim))

    def sample_matrices(self, n_samples: int, size: int) -> np.ndarray:
        """Generate matrix samples with uniform entries.

        Args:
            n_samples: Number of matrices to generate.
            size: Size of each square matrix.

        Returns:
            Array of shape (n_samples, size, size) with uniform entries.
        """
        return self._rng.uniform(self.low, self.high, size=(n_samples, size, size))

    def __repr__(self) -> str:
        return f"UniformSampler(seed={self.seed}, low={self.low}, high={self.high})"
