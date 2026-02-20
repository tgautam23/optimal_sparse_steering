"""Tests for ConceptSubspace class."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from src.probes.concept_subspace import ConceptSubspace


def _make_separable_data(d_model=64, n_per_class=100, separation=2.0):
    """Create synthetic linearly separable data."""
    np.random.seed(42)

    # Class 0: centered at -separation/2 along first axis
    X0 = np.random.randn(n_per_class, d_model) * 0.5
    X0[:, 0] -= separation / 2

    # Class 1: centered at +separation/2 along first axis
    X1 = np.random.randn(n_per_class, d_model) * 0.5
    X1[:, 0] += separation / 2

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


class TestConceptSubspaceFit:
    def test_fit_returns_self(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=3)
        result = cs.fit(X, y)
        assert result is cs

    def test_directions_shape(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=3)
        cs.fit(X, y)
        assert cs.directions.shape == (3, 64)

    def test_n_directions(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=5)
        cs.fit(X, y)
        assert cs.n_directions == 5

    def test_directions_unit_norm(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=20, n_select=5)
        cs.fit(X, y)

        norms = np.linalg.norm(cs.directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_directions_sign_corrected(self):
        """After sign correction, w_j^T mu_diff should be positive for all j."""
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=20, n_select=5)
        cs.fit(X, y)

        # class_separations should all be positive after sign correction
        assert (cs.class_separations > 0).all()

    def test_class_separations_sorted_descending(self):
        """Selected directions should be sorted by |s_j| descending."""
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=20, n_select=5)
        cs.fit(X, y)

        # Since we select top-k by |s_j|, separations should be descending
        seps = cs.class_separations
        assert all(seps[i] >= seps[i+1] for i in range(len(seps) - 1))


class TestConceptSubspaceConstraintHelpers:
    def test_compute_rhs_shape(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=3)
        cs.fit(X, y)

        h = np.random.randn(64)
        rhs = cs.compute_rhs(h)
        assert rhs.shape == (3,)

    def test_compute_thresholds_shape(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=3)
        cs.fit(X, y)

        thresholds = cs.compute_thresholds()
        assert thresholds.shape == (3,)

    def test_compute_rhs_consistency(self):
        """rhs(h) = thresholds - W @ h."""
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=3)
        cs.fit(X, y)

        h = np.random.randn(64)
        rhs = cs.compute_rhs(h)
        thresholds = cs.compute_thresholds()
        proj_h = cs.directions @ h

        np.testing.assert_allclose(rhs, thresholds - proj_h, atol=1e-10)

    def test_compute_prefilter_relevance_shape(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=3)
        cs.fit(X, y)

        d_sae = 100
        D = np.random.randn(d_sae, 64)
        relevance = cs.compute_prefilter_relevance(D)
        assert relevance.shape == (d_sae,)

    def test_compute_prefilter_relevance_nonneg(self):
        X, y = _make_separable_data()
        cs = ConceptSubspace(n_components=10, n_select=3)
        cs.fit(X, y)

        D = np.random.randn(100, 64)
        relevance = cs.compute_prefilter_relevance(D)
        assert (relevance >= 0).all()


class TestConceptSubspaceNotFitted:
    def test_directions_raises(self):
        cs = ConceptSubspace()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = cs.directions

    def test_class_separations_raises(self):
        cs = ConceptSubspace()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = cs.class_separations

    def test_midpoints_raises(self):
        cs = ConceptSubspace()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = cs.midpoints

    def test_n_directions_raises(self):
        cs = ConceptSubspace()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = cs.n_directions

    def test_compute_rhs_raises(self):
        cs = ConceptSubspace()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cs.compute_rhs(np.zeros(10))

    def test_compute_thresholds_raises(self):
        cs = ConceptSubspace()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cs.compute_thresholds()

    def test_compute_prefilter_relevance_raises(self):
        cs = ConceptSubspace()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cs.compute_prefilter_relevance(np.zeros((10, 5)))
