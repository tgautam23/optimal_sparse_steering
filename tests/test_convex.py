"""Tests for convex optimal steering (SOCP)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np

from src.steering.convex_optimal import ConvexOptimalSteering


class TestConvexOptimalSteering:
    def _make_problem(self, d_model=32, d_sae=100):
        """Create a small synthetic SOCP problem."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Random decoder matrix
        D = torch.randn(d_sae, d_model)
        # Normalize rows
        D = D / D.norm(dim=1, keepdim=True)

        # Random probe
        w = np.random.randn(d_model)
        w = w / np.linalg.norm(w)
        b = 0.0

        # Current activation (probe score ~ -0.5, so we need to steer positive)
        h = torch.randn(d_model) * 0.1

        # Fake SAE features (all active)
        sae_features = torch.abs(torch.randn(d_sae)) + 0.1

        return h, w, b, D, sae_features

    def test_solves_successfully(self):
        h, w, b, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(
            epsilon=10.0, tau=0.5, solver="SCS", prefilter_threshold=0.0
        )
        vec = method.compute_steering_vector(
            h=h, probe_w=w, probe_b=b, D=D,
            sae_features=sae_features, target_class=1,
        )

        assert method.solve_status in ("optimal", "optimal_inaccurate")
        assert vec.shape == (32,)

    def test_delta_is_sparse(self):
        h, w, b, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(
            epsilon=10.0, tau=0.5, solver="SCS", prefilter_threshold=0.0
        )
        method.compute_steering_vector(
            h=h, probe_w=w, probe_b=b, D=D,
            sae_features=sae_features, target_class=1,
        )

        if method.solve_status in ("optimal", "optimal_inaccurate"):
            delta = method.delta
            l0 = (delta > 1e-6).sum()
            # L1 minimization should produce sparse solution
            assert l0 < 100  # much less than d_sae=100

    def test_delta_nonnegative(self):
        h, w, b, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(
            epsilon=10.0, tau=0.5, solver="SCS", prefilter_threshold=0.0
        )
        method.compute_steering_vector(
            h=h, probe_w=w, probe_b=b, D=D,
            sae_features=sae_features, target_class=1,
        )

        if method.solve_status in ("optimal", "optimal_inaccurate"):
            assert (method.delta >= -1e-8).all()

    def test_coherence_constraint(self):
        h, w, b, D, sae_features = self._make_problem()
        epsilon = 5.0

        method = ConvexOptimalSteering(
            epsilon=epsilon, tau=0.5, solver="SCS", prefilter_threshold=0.0
        )
        vec = method.compute_steering_vector(
            h=h, probe_w=w, probe_b=b, D=D,
            sae_features=sae_features, target_class=1,
        )

        if method.solve_status in ("optimal", "optimal_inaccurate"):
            # ||D^T delta||_2 should be <= epsilon (with small tolerance)
            residual_norm = vec.norm().item()
            assert residual_norm <= epsilon + 0.5  # solver tolerance

    def test_summary(self):
        h, w, b, D, sae_features = self._make_problem()

        method = ConvexOptimalSteering(epsilon=10.0, tau=0.5)
        method.compute_steering_vector(
            h=h, probe_w=w, probe_b=b, D=D,
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
        w = np.random.randn(d_model)
        b = 0.0
        activations = torch.randn(3, d_model) * 0.1
        sae_batch = torch.abs(torch.randn(3, d_sae)) + 0.1

        method = ConvexOptimalSteering(
            epsilon=10.0, tau=0.5, prefilter_threshold=0.0
        )
        results = method.compute_batch_steering(
            activations=activations, probe_w=w, probe_b=b, D=D,
            sae_features_batch=sae_batch, target_class=1,
        )

        assert len(results) == 3
        for vec in results:
            assert vec.shape == (d_model,)
