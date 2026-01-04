"""
Configuration management for CISE experiments.

Handles loading, validation, and defaults for experiment configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class Config:
    """Configuration for CISE experiments.

    Attributes:
        seed: Random seed for reproducibility.
        n_samples: Number of samples per ensemble.
        vector_dim: Dimension of vectors (d).
        matrix_size: Size of square matrices (k x k).
        beta_values: List of beta values for reweighting sweep.
        vector_constraints: List of constraint names for vectors.
        matrix_constraints: List of constraint names for matrices.
        samplers: List of sampler names to use.
        constraint_params: Optional parameters for constraints.
    """

    seed: int = 1337
    n_samples: int = 10000
    vector_dim: int = 32
    matrix_size: int = 6
    beta_values: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    vector_constraints: List[str] = field(default_factory=lambda: ["smoothness", "l1"])
    matrix_constraints: List[str] = field(default_factory=lambda: ["lowrank", "l1"])
    samplers: List[str] = field(default_factory=lambda: ["gaussian", "uniform"])
    constraint_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_samples < 100:
            raise ValueError("n_samples must be at least 100")
        if self.vector_dim < 2:
            raise ValueError("vector_dim must be at least 2")
        if self.matrix_size < 2:
            raise ValueError("matrix_size must be at least 2")
        if not self.beta_values:
            raise ValueError("beta_values must not be empty")

        # Set default constraint parameters
        default_params = {
            "lowrank": {"target_rank": 2},
            "hierarchy": {"delta": -0.5, "epsilon": 1e-10},
            "l1": {"scale": 1.0},
            "smoothness": {"scale": 1.0},
        }
        for key, val in default_params.items():
            if key not in self.constraint_params:
                self.constraint_params[key] = val

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            seed=data.get("seed", 1337),
            n_samples=data.get("n_samples", 10000),
            vector_dim=data.get("vector_dim", data.get("vector", {}).get("dim", 32)),
            matrix_size=data.get("matrix_size", data.get("matrix", {}).get("size", 6)),
            beta_values=data.get("beta_values", data.get("beta_sweep", [0.0, 0.5, 1.0, 2.0])),
            vector_constraints=data.get(
                "vector_constraints",
                data.get("constraints", {}).get("vectors", ["smoothness", "l1"])
            ),
            matrix_constraints=data.get(
                "matrix_constraints",
                data.get("constraints", {}).get("matrices", ["lowrank", "l1"])
            ),
            samplers=data.get("samplers", ["gaussian", "uniform"]),
            constraint_params=data.get("constraint_params", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "seed": self.seed,
            "n_samples": self.n_samples,
            "vector_dim": self.vector_dim,
            "matrix_size": self.matrix_size,
            "beta_values": self.beta_values,
            "vector_constraints": self.vector_constraints,
            "matrix_constraints": self.matrix_constraints,
            "samplers": self.samplers,
            "constraint_params": self.constraint_params,
        }


def load_config(path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Config object with loaded settings.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)
