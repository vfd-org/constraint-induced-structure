"""
Base constraint interface for CISE.

Defines the abstract interface for constraint penalty functions.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List
import numpy as np


class ConstraintType(Enum):
    """Types of samples a constraint can apply to."""
    VECTOR = "vector"
    MATRIX = "matrix"
    BOTH = "both"


class BaseConstraint(ABC):
    """Abstract base class for constraint penalty functions.

    Constraints define energy functions E(z) that are used in
    energy-based reweighting: w(z) = exp(-beta * E(z)).
    """

    # Subclasses should override this to indicate applicable types
    applicable_types: List[ConstraintType] = [ConstraintType.BOTH]

    def __init__(self, **params):
        """Initialize constraint with parameters.

        Args:
            **params: Constraint-specific parameters.
        """
        self.params = params

    @property
    def name(self) -> str:
        """Return constraint name."""
        return self.__class__.__name__.replace("Constraint", "").lower()

    @abstractmethod
    def energy(self, samples: np.ndarray) -> np.ndarray:
        """Compute penalty energy for each sample.

        Args:
            samples: Array of samples.
                - For vectors: shape (n_samples, dim)
                - For matrices: shape (n_samples, k, k)

        Returns:
            Array of energies, shape (n_samples,).
        """
        pass

    def can_apply_to(self, sample_type: ConstraintType) -> bool:
        """Check if constraint can apply to a sample type.

        Args:
            sample_type: Type to check.

        Returns:
            True if constraint is applicable.
        """
        if ConstraintType.BOTH in self.applicable_types:
            return True
        return sample_type in self.applicable_types

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """Compute energy (alias for energy method).

        Args:
            samples: Array of samples.

        Returns:
            Array of energies.
        """
        return self.energy(samples)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
