"""
Smoke tests for CISE.

These tests verify basic functionality:
- Samplers generate correct shapes
- Constraints return finite values
- Reweighting returns weights and ESS
- Runner produces expected output files
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cise.config import Config, load_config
from cise.sampling import GaussianSampler, UniformSampler, get_sampler
from cise.constraints import (
    SmoothnessConstraint,
    L1Constraint,
    LowRankConstraint,
    HierarchyConstraint,
    get_constraint,
)
from cise.methods.reweighting import compute_weights, compute_ess, apply_reweighting
from cise.experiments.runner import ExperimentRunner


class TestSamplers:
    """Test sampler functionality."""

    def test_gaussian_sampler_vectors(self):
        """Gaussian sampler generates correct vector shapes."""
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_vectors(n_samples=100, dim=32)

        assert samples.shape == (100, 32)
        assert np.isfinite(samples).all()
        assert samples.dtype == np.float64

    def test_gaussian_sampler_matrices(self):
        """Gaussian sampler generates correct matrix shapes."""
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_matrices(n_samples=100, size=6)

        assert samples.shape == (100, 6, 6)
        assert np.isfinite(samples).all()

    def test_uniform_sampler_vectors(self):
        """Uniform sampler generates correct vector shapes."""
        sampler = UniformSampler(seed=42)
        samples = sampler.sample_vectors(n_samples=100, dim=32)

        assert samples.shape == (100, 32)
        assert np.isfinite(samples).all()
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_uniform_sampler_matrices(self):
        """Uniform sampler generates correct matrix shapes."""
        sampler = UniformSampler(seed=42)
        samples = sampler.sample_matrices(n_samples=100, size=6)

        assert samples.shape == (100, 6, 6)
        assert np.isfinite(samples).all()
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_get_sampler(self):
        """get_sampler returns correct classes."""
        assert get_sampler("gaussian") == GaussianSampler
        assert get_sampler("uniform") == UniformSampler

        with pytest.raises(ValueError):
            get_sampler("unknown")

    def test_sampler_reproducibility(self):
        """Samplers are reproducible with same seed."""
        sampler1 = GaussianSampler(seed=42)
        sampler2 = GaussianSampler(seed=42)

        samples1 = sampler1.sample_vectors(100, 32)
        samples2 = sampler2.sample_vectors(100, 32)

        np.testing.assert_array_equal(samples1, samples2)


class TestConstraints:
    """Test constraint functionality."""

    def test_smoothness_constraint(self):
        """Smoothness constraint returns finite values."""
        constraint = SmoothnessConstraint()
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_vectors(100, 32)

        energies = constraint.energy(samples)

        assert energies.shape == (100,)
        assert np.isfinite(energies).all()
        assert (energies >= 0).all()

    def test_l1_constraint_vectors(self):
        """L1 constraint returns finite values for vectors."""
        constraint = L1Constraint()
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_vectors(100, 32)

        energies = constraint.energy(samples)

        assert energies.shape == (100,)
        assert np.isfinite(energies).all()
        assert (energies >= 0).all()

    def test_l1_constraint_matrices(self):
        """L1 constraint returns finite values for matrices."""
        constraint = L1Constraint()
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_matrices(100, 6)

        energies = constraint.energy(samples)

        assert energies.shape == (100,)
        assert np.isfinite(energies).all()
        assert (energies >= 0).all()

    def test_lowrank_constraint(self):
        """Low-rank constraint returns finite values."""
        constraint = LowRankConstraint(target_rank=2)
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_matrices(100, 6)

        energies = constraint.energy(samples)

        assert energies.shape == (100,)
        assert np.isfinite(energies).all()
        assert (energies >= 0).all()

    def test_hierarchy_constraint_vectors(self):
        """Hierarchy constraint returns finite values for vectors."""
        constraint = HierarchyConstraint(delta=-0.5)
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_vectors(100, 32)

        energies = constraint.energy(samples)

        assert energies.shape == (100,)
        assert np.isfinite(energies).all()
        assert (energies >= 0).all()

    def test_hierarchy_constraint_matrices(self):
        """Hierarchy constraint returns finite values for matrices."""
        constraint = HierarchyConstraint(delta=-0.5)
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_matrices(100, 6)

        energies = constraint.energy(samples)

        assert energies.shape == (100,)
        assert np.isfinite(energies).all()
        assert (energies >= 0).all()

    def test_get_constraint(self):
        """get_constraint returns correct classes."""
        assert get_constraint("smoothness") == SmoothnessConstraint
        assert get_constraint("l1") == L1Constraint
        assert get_constraint("lowrank") == LowRankConstraint
        assert get_constraint("hierarchy") == HierarchyConstraint

        with pytest.raises(ValueError):
            get_constraint("unknown")


class TestReweighting:
    """Test reweighting functionality."""

    def test_compute_weights(self):
        """compute_weights returns valid weights."""
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_vectors(100, 32)
        constraints = [SmoothnessConstraint(), L1Constraint()]

        weights, energies, _ = compute_weights(samples, constraints, beta=1.0)

        assert weights.shape == (100,)
        assert energies.shape == (100,)
        assert np.isfinite(weights).all()
        assert np.isfinite(energies).all()
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()

    def test_compute_ess(self):
        """compute_ess returns valid ESS."""
        # Uniform weights should give ESS = n
        uniform_weights = np.ones(100) / 100
        ess = compute_ess(uniform_weights)
        assert np.isclose(ess, 100.0)

        # Single dominant weight should give ESS close to 1
        dominant_weights = np.zeros(100)
        dominant_weights[0] = 1.0
        ess = compute_ess(dominant_weights)
        assert np.isclose(ess, 1.0)

    def test_apply_reweighting(self):
        """apply_reweighting returns complete result."""
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_vectors(100, 32)
        constraints = [SmoothnessConstraint(), L1Constraint()]

        result = apply_reweighting(samples, constraints, beta=1.0, resample=True)

        assert result.samples.shape == (100, 32)
        assert result.weights.shape == (100,)
        assert result.energies.shape == (100,)
        assert 0 < result.ess <= 100
        assert 0 < result.ess_ratio <= 1
        assert result.resampled is not None
        assert result.resampled.shape == (100, 32)

    def test_beta_zero_uniform_weights(self):
        """Beta=0 should give uniform weights."""
        sampler = GaussianSampler(seed=42)
        samples = sampler.sample_vectors(100, 32)
        constraints = [SmoothnessConstraint()]

        result = apply_reweighting(samples, constraints, beta=0.0)

        # All weights should be equal
        expected = 1.0 / 100
        np.testing.assert_allclose(result.weights, expected, rtol=1e-10)


class TestRunner:
    """Test experiment runner functionality."""

    def test_runner_produces_outputs(self):
        """Runner produces expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                seed=42,
                n_samples=500,  # Small for fast test
                vector_dim=16,
                matrix_size=4,
                beta_values=[0.0, 1.0],
                vector_constraints=["smoothness"],
                matrix_constraints=["l1"],
                samplers=["gaussian"],
            )

            runner = ExperimentRunner(config, output_dir=tmpdir)
            results = runner.run()

            # Check output files exist
            output_path = Path(tmpdir)
            assert (output_path / "metrics.json").exists()
            assert (output_path / "summary.md").exists()
            assert (output_path / "figures").is_dir()

            # Check figures were created
            figures = list((output_path / "figures").glob("*.png"))
            assert len(figures) >= 6  # At least 6 figures

            # Check results structure
            assert "config" in results
            assert "experiments" in results
            assert len(results["experiments"]) > 0


class TestConfig:
    """Test configuration functionality."""

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = Config()

        assert config.seed == 1337
        assert config.n_samples == 10000
        assert config.vector_dim == 32
        assert config.matrix_size == 6
        assert 0.0 in config.beta_values
        assert len(config.beta_values) > 0

    def test_config_validation(self):
        """Config validates parameters."""
        with pytest.raises(ValueError):
            Config(n_samples=10)  # Too few samples

        with pytest.raises(ValueError):
            Config(vector_dim=1)  # Dimension too small

        with pytest.raises(ValueError):
            Config(beta_values=[])  # Empty beta values

    def test_config_from_dict(self):
        """Config can be created from dict."""
        data = {
            "seed": 42,
            "n_samples": 5000,
            "vector_dim": 64,
            "matrix_size": 8,
        }
        config = Config.from_dict(data)

        assert config.seed == 42
        assert config.n_samples == 5000
        assert config.vector_dim == 64
        assert config.matrix_size == 8

    def test_config_to_dict(self):
        """Config can be converted to dict."""
        config = Config(seed=42, n_samples=5000)
        data = config.to_dict()

        assert data["seed"] == 42
        assert data["n_samples"] == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
