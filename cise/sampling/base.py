"""
Base sampler interface for CISE.

Defines the abstract interface that all samplers must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class SampleType(Enum):
    """Types of samples that can be generated."""
    VECTOR = "vector"
    MATRIX = "matrix"


class BaseSampler(ABC):
    """Abstract base class for ensemble samplers.

    All samplers must implement methods to generate vectors and matrices
    with reproducible seeding.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize sampler with optional seed.

        Args:
            seed: Random seed for reproducibility. If None, uses random state.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None):
        """Reset random state.

        Args:
            seed: New seed to use. If None, uses original seed.
        """
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        """Return sampler name."""
        return self.__class__.__name__.replace("Sampler", "").lower()

    @abstractmethod
    def sample_vectors(self, n_samples: int, dim: int) -> np.ndarray:
        """Generate vector samples.

        Args:
            n_samples: Number of vectors to generate.
            dim: Dimension of each vector.

        Returns:
            Array of shape (n_samples, dim).
        """
        pass

    @abstractmethod
    def sample_matrices(self, n_samples: int, size: int) -> np.ndarray:
        """Generate matrix samples.

        Args:
            n_samples: Number of matrices to generate.
            size: Size of each square matrix (size x size).

        Returns:
            Array of shape (n_samples, size, size).
        """
        pass

    def sample(
        self,
        n_samples: int,
        sample_type: SampleType,
        dim: Optional[int] = None,
        size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate samples of specified type.

        Args:
            n_samples: Number of samples to generate.
            sample_type: Type of sample (VECTOR or MATRIX).
            dim: Vector dimension (required for VECTOR type).
            size: Matrix size (required for MATRIX type).

        Returns:
            Array of samples.
        """
        if sample_type == SampleType.VECTOR:
            if dim is None:
                raise ValueError("dim is required for vector samples")
            return self.sample_vectors(n_samples, dim)
        elif sample_type == SampleType.MATRIX:
            if size is None:
                raise ValueError("size is required for matrix samples")
            return self.sample_matrices(n_samples, size)
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed})"
