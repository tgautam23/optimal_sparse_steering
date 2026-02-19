"""Random direction steering (null hypothesis baseline)."""

import torch
from .base import SteeringMethod


class RandomDirection(SteeringMethod):
    """Null hypothesis: random unit vector in residual stream space."""

    def __init__(self, seed: int = 42):
        super().__init__("random_direction")
        self.seed = seed

    def compute_steering_vector(self, d_model: int = 768, **kwargs) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed)
        vec = torch.randn(d_model, generator=generator)
        vec = vec / vec.norm()
        self._steering_vector = vec
        return self._steering_vector
