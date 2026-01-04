"""
Sampling module for CISE.

Provides baseline ensemble samplers for vectors and matrices.
"""

from cise.sampling.base import BaseSampler, SampleType
from cise.sampling.gaussian import GaussianSampler
from cise.sampling.uniform import UniformSampler

SAMPLERS = {
    "gaussian": GaussianSampler,
    "uniform": UniformSampler,
}


def get_sampler(name: str) -> type:
    """Get sampler class by name.

    Args:
        name: Sampler name ('gaussian' or 'uniform').

    Returns:
        Sampler class.

    Raises:
        ValueError: If sampler name is unknown.
    """
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler: {name}. Available: {list(SAMPLERS.keys())}")
    return SAMPLERS[name]


__all__ = [
    "BaseSampler",
    "SampleType",
    "GaussianSampler",
    "UniformSampler",
    "get_sampler",
    "SAMPLERS",
]
