"""Tests for steering methods."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock


class TestNoSteering:
    def test_zero_vector(self):
        from src.steering.no_steering import NoSteering
        method = NoSteering()
        vec = method.compute_steering_vector(d_model=768)
        assert vec.shape == (768,)
        assert torch.allclose(vec, torch.zeros(768))

    def test_noop_hook(self):
        from src.steering.no_steering import NoSteering
        method = NoSteering()
        method.compute_steering_vector(d_model=64)
        hook_fn = method.get_hook_fn(alpha=5.0)
        acts = torch.randn(2, 10, 64)
        result = hook_fn(acts.clone(), None)
        assert torch.allclose(result, acts)


class TestRandomDirection:
    def test_unit_norm(self):
        from src.steering.random_direction import RandomDirection
        method = RandomDirection(seed=42)
        vec = method.compute_steering_vector(d_model=768)
        assert vec.shape == (768,)
        assert abs(vec.norm().item() - 1.0) < 1e-5

    def test_deterministic(self):
        from src.steering.random_direction import RandomDirection
        v1 = RandomDirection(seed=42).compute_steering_vector(d_model=768)
        v2 = RandomDirection(seed=42).compute_steering_vector(d_model=768)
        assert torch.allclose(v1, v2)


class TestCAAMeanDiff:
    def test_computes_difference(self):
        from src.steering.caa import CAAMeanDiff
        method = CAAMeanDiff()
        pos = torch.ones(10, 64) * 2
        neg = torch.ones(10, 64) * -2
        vec = method.compute_steering_vector(activations_pos=pos, activations_neg=neg)
        assert vec.shape == (64,)
        assert vec.norm().item() > 0.99  # unit norm


class TestTopKFeatures:
    def test_selects_features(self):
        from src.steering.topk_features import TopKFeatures
        np.random.seed(42)

        d_sae, d_model, n = 100, 32, 50
        features = np.random.randn(n, d_sae).astype(np.float32)
        labels = np.random.randint(0, 2, n)
        # Make feature 0 highly correlated with labels
        features[:, 0] = labels * 10.0

        mock_wrapper = MagicMock()
        mock_sae = MagicMock()
        mock_sae.W_dec = torch.randn(d_sae, d_model)
        mock_wrapper.get_sae.return_value = mock_sae

        method = TopKFeatures(topk=5)
        vec = method.compute_steering_vector(
            sae_features=torch.tensor(features),
            labels=labels,
            model_wrapper=mock_wrapper,
            layer=0,
        )
        assert vec.shape == (d_model,)
        assert 0 in method.selected_features  # feature 0 should be selected


class TestRegistry:
    def test_get_known_method(self):
        from src.steering.registry import get_steering_method
        method = get_steering_method("no_steering")
        assert method.name == "no_steering"

    def test_unknown_method_raises(self):
        from src.steering.registry import get_steering_method
        with pytest.raises(ValueError):
            get_steering_method("nonexistent_method")
