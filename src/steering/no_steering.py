"""No steering (control baseline)."""

import torch
from .base import SteeringMethod


class NoSteering(SteeringMethod):
    """Control: zero steering vector."""

    def __init__(self):
        super().__init__("no_steering")

    def compute_steering_vector(self, d_model: int = 768, **kwargs) -> torch.Tensor:
        self._steering_vector = torch.zeros(d_model)
        return self._steering_vector

    def get_hook_fn(self, alpha: float = 1.0):
        # No-op hook for efficiency
        def hook_fn(activations, hook):
            return activations
        return hook_fn
