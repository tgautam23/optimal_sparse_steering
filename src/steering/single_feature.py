"""Single SAE feature steering — manual feature selection."""

import logging
import torch
from .base import SteeringMethod

logger = logging.getLogger(__name__)


class SingleFeature(SteeringMethod):
    """Steer using a single SAE feature's decoder direction.

    The user specifies a feature index (e.g., from Neuronpedia inspection).
    The steering vector is the SAE decoder direction for that feature.
    """

    def __init__(self, feature_idx: int = 0):
        super().__init__("single_feature")
        self.feature_idx = feature_idx

    def compute_steering_vector(
        self,
        model_wrapper=None,
        layer: int = 0,
        feature_idx: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute steering vector from a single SAE decoder direction.

        Args:
            model_wrapper: ModelWrapper with SAE access.
            layer: Layer to get SAE from.
            feature_idx: Override feature index (if not set in constructor).
        """
        if feature_idx is not None:
            self.feature_idx = feature_idx

        sae = model_wrapper.get_sae(layer)
        # Get decoder direction for this feature: W_dec[feature_idx] shape (d_model,)
        direction = sae.W_dec[self.feature_idx].detach().clone().float().cpu()
        direction = direction / direction.norm()

        logger.info(f"SingleFeature: using feature {self.feature_idx} at layer {layer}")
        self._steering_vector = direction
        return self._steering_vector
