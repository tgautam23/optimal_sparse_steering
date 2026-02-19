"""Abstract base class for all steering methods."""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import numpy as np


class SteeringMethod(ABC):
    """Base class for steering methods.

    All steering methods compute a steering vector in residual stream space
    that can be added to model activations at a specific layer.
    """

    def __init__(self, name: str):
        self.name = name
        self._steering_vector: Optional[torch.Tensor] = None

    @abstractmethod
    def compute_steering_vector(self, **kwargs) -> torch.Tensor:
        """Compute the steering vector.

        Returns:
            Steering vector of shape (d_model,) in residual stream space.
        """
        pass

    @property
    def steering_vector(self) -> torch.Tensor:
        if self._steering_vector is None:
            raise RuntimeError(f"{self.name}: steering vector not yet computed. Call compute_steering_vector first.")
        return self._steering_vector

    def get_hook_fn(self, alpha: float = 1.0):
        """Return a TransformerLens hook function that adds the steering vector.

        The hook adds alpha * steering_vector to ALL token positions.
        """
        vec = self.steering_vector

        def hook_fn(activations, hook):
            # activations shape: (batch, seq_len, d_model)
            activations = activations + alpha * vec.to(activations.device)
            return activations

        return hook_fn

    def summary(self) -> dict:
        """Return a summary dict with method name and steering vector stats."""
        info = {"method": self.name}
        if self._steering_vector is not None:
            v = self._steering_vector
            info["norm"] = float(v.norm().item())
            info["d_model"] = v.shape[0]
        return info
