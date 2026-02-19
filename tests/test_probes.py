"""Tests for linear probe."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.probes.linear_probe import LinearProbe


class TestLinearProbe:
    def test_fit_and_predict(self):
        """Test probe on linearly separable data."""
        np.random.seed(42)
        d_model = 64
        n = 100

        # Create linearly separable data
        X_pos = np.random.randn(n // 2, d_model) + 1.0
        X_neg = np.random.randn(n // 2, d_model) - 1.0
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        probe = LinearProbe(d_model=d_model)
        probe.fit(X, y)

        # Should achieve high accuracy on training data
        acc = probe.score(X, y)
        assert acc > 0.9

    def test_weight_vector_shape(self):
        np.random.seed(42)
        d_model = 32
        X = np.random.randn(50, d_model)
        y = np.random.randint(0, 2, 50)

        probe = LinearProbe(d_model=d_model)
        probe.fit(X, y)

        assert probe.weight_vector.shape == (d_model,)
        assert isinstance(probe.bias, float)

    def test_not_fitted_raises(self):
        probe = LinearProbe(d_model=32)
        with pytest.raises(RuntimeError):
            _ = probe.weight_vector
        with pytest.raises(RuntimeError):
            _ = probe.bias

    def test_predict_proba(self):
        np.random.seed(42)
        X = np.random.randn(20, 16)
        y = np.random.randint(0, 2, 20)

        probe = LinearProbe(d_model=16)
        probe.fit(X, y)

        proba = probe.predict_proba(X)
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
