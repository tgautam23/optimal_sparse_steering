"""Top-k correlated SAE features steering."""

import logging
import torch
import numpy as np
from .base import SteeringMethod

logger = logging.getLogger(__name__)


class TopKFeatures(SteeringMethod):
    """Steer using a weighted combination of the top-k most correlated SAE features.

    Computes Pearson correlation between each SAE feature's activation and binary labels,
    selects the top-k features, and constructs a weighted sum of their decoder directions.
    """

    def __init__(self, topk: int = 10):
        super().__init__("topk_features")
        self.topk = topk
        self.selected_features: list[int] = []
        self.correlations: np.ndarray | None = None

    def compute_steering_vector(
        self,
        sae_features: torch.Tensor = None,
        labels: np.ndarray = None,
        model_wrapper=None,
        layer: int = 0,
        topk: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute steering vector from top-k correlated SAE features.

        Args:
            sae_features: SAE feature activations, shape (n_samples, d_sae).
            labels: Binary labels, shape (n_samples,).
            model_wrapper: ModelWrapper with SAE access.
            layer: Layer to get SAE decoder from.
            topk: Override k value.
        """
        if topk is not None:
            self.topk = topk

        # Convert to numpy for correlation computation
        if isinstance(sae_features, torch.Tensor):
            feat_np = sae_features.detach().cpu().numpy()
        else:
            feat_np = np.asarray(sae_features)
        labels_np = np.asarray(labels, dtype=np.float64)

        # Compute Pearson correlation for each feature
        n = len(labels_np)
        feat_centered = feat_np - feat_np.mean(axis=0, keepdims=True)
        labels_centered = labels_np - labels_np.mean()

        # Correlation = (X^T y) / (||X|| * ||y||)
        numerator = feat_centered.T @ labels_centered  # (d_sae,)
        feat_std = np.sqrt((feat_centered ** 2).sum(axis=0) + 1e-10)  # (d_sae,)
        label_std = np.sqrt((labels_centered ** 2).sum() + 1e-10)
        correlations = numerator / (feat_std * label_std)
        self.correlations = correlations

        # Select top-k by absolute correlation
        top_indices = np.argsort(np.abs(correlations))[::-1][:self.topk]
        self.selected_features = top_indices.tolist()

        logger.info(f"TopKFeatures: selected features {self.selected_features}")
        for idx in self.selected_features:
            logger.info(f"  Feature {idx}: correlation = {correlations[idx]:.4f}")

        # Get SAE decoder matrix
        sae = model_wrapper.get_sae(layer)
        W_dec = sae.W_dec.detach().float().cpu()  # (d_sae, d_model)

        # Weighted sum of decoder directions (weighted by correlation sign and magnitude)
        steering_vec = torch.zeros(W_dec.shape[1])
        for idx in self.selected_features:
            weight = correlations[idx]
            steering_vec += weight * W_dec[idx]

        # Normalize to unit length
        norm = steering_vec.norm()
        if norm > 1e-8:
            steering_vec = steering_vec / norm

        self._steering_vector = steering_vec
        return self._steering_vector
