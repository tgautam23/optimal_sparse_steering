"""Tests for convex optimal steering (SOCP) with ConceptSubspace."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np

from src.steering.convex_optimal import ConvexOptimalSteering
from src.probes.concept_subspace import ConceptSubspace


def _make_concept_subspace(d_model=32, n_samples=200):
    """Create a fitted ConceptSubspace from synthetic linearly separable data."""
    np.random.seed(42)
    n_per_class = n_samples // 2

    # Class 0: centered at -1 along first axis
    X0 = np.random.randn(n_per_class, d_model) * 0.5
    X0[:, 0] -= 1.0

    # Class 1: centered at +1 along first axis
    X1 = np.random.randn(n_per_class, d_model) * 0.5
    X1[:, 0] += 1.0

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    cs = ConceptSubspace(n_components=10, n_select=3)
    cs.fit(X, y)
    return cs


class TestConvexOptimalSteering:
    def _make_problem(self, d_model=32, d_sae=100):
        """Create a small synthetic SOCP problem with ConceptSubspace."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Random decoder matrix
        D = torch.randn(d_sae, d_model)
        # Normalize rows
        D = D / D.norm(dim=1, keepdim=True)

        # Concept subspace
        cs = _make_concept_subspace(d_model=d_model)

        # Current activation (low projection, so we need to steer positive)
        h = torch.randn(d_model) * 0.1

        # Fake SAE features (all active)
        sae_features = torch.abs(torch.randn(d_sae)) + 0.1

        return h, cs, D, sae_features

    def test_solves_successfully(self):
        h, cs, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(
            epsilon=10.0, solver="SCS", prefilter_topk=100
        )
        vec = method.compute_steering_vector(
            h=h, D=D, concept_subspace=cs,
            sae_features=sae_features, target_class=1,
        )

        assert method.solve_status in ("optimal", "optimal_inaccurate")
        assert vec.shape == (32,)

    def test_delta_is_sparse(self):
        h, cs, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(
            epsilon=10.0, solver="SCS", prefilter_topk=100
        )
        method.compute_steering_vector(
            h=h, D=D, concept_subspace=cs,
            sae_features=sae_features, target_class=1,
        )

        if method.solve_status in ("optimal", "optimal_inaccurate"):
            delta = method.delta
            l0 = (delta > 1e-6).sum()
            # L1 minimization should produce sparse solution
            assert l0 < 100  # much less than d_sae=100

    def test_delta_nonnegative(self):
        h, cs, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(
            epsilon=10.0, solver="SCS", prefilter_topk=100
        )
        method.compute_steering_vector(
            h=h, D=D, concept_subspace=cs,
            sae_features=sae_features, target_class=1,
        )

        if method.solve_status in ("optimal", "optimal_inaccurate"):
            assert (method.delta >= -1e-8).all()

    def test_coherence_constraint(self):
        h, cs, D, sae_features = self._make_problem()
        epsilon = 5.0

        method = ConvexOptimalSteering(
            epsilon=epsilon, solver="SCS", prefilter_topk=100
        )
        vec = method.compute_steering_vector(
            h=h, D=D, concept_subspace=cs,
            sae_features=sae_features, target_class=1,
        )

        if method.solve_status in ("optimal", "optimal_inaccurate"):
            # ||D^T delta||_2 should be <= epsilon (with small tolerance)
            residual_norm = vec.norm().item()
            assert residual_norm <= epsilon + 0.5  # solver tolerance

    def test_summary(self):
        h, cs, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(epsilon=10.0)
        method.compute_steering_vector(
            h=h, D=D, concept_subspace=cs,
            sae_features=sae_features, target_class=1,
        )

        summary = method.summary()
        assert "method" in summary
        assert summary["method"] == "convex_optimal"
        assert "solve_status" in summary
        assert "epsilon" in summary

    def test_not_solved_raises(self):
        method = ConvexOptimalSteering()
        with pytest.raises(RuntimeError):
            _ = method.delta
        with pytest.raises(RuntimeError):
            _ = method.active_features

    def test_batch_steering(self):
        d_model, d_sae = 32, 100
        np.random.seed(42)
        torch.manual_seed(42)

        D = torch.randn(d_sae, d_model)
        cs = _make_concept_subspace(d_model=d_model)
        activations = torch.randn(3, d_model) * 0.1
        sae_batch = torch.abs(torch.randn(3, d_sae)) + 0.1

        method = ConvexOptimalSteering(
            epsilon=10.0, prefilter_topk=100
        )
        results = method.compute_batch_steering(
            activations=activations, D=D, concept_subspace=cs,
            sae_features_batch=sae_batch, target_class=1,
        )

        assert len(results) == 3
        for vec in results:
            assert vec.shape == (d_model,)
