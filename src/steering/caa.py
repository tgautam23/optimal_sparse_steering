"""Contrastive Activation Addition (CAA) steering methods.

Implements three variants of CAA for computing steering vectors from
contrastive activation patterns:

1. CAAMeanDiff — Raw mean difference between two sets of pre-extracted
   activations (e.g., class-conditional splits).
2. CAAContrastive — Mean difference from contrastive prompt pairs run
   through the model at a specified layer.
3. CAARepE — Representation Engineering approach that prepends persona
   prefixes to neutral queries to create contrastive activation pairs.

References:
    - Turner et al. (2023), "Activation Addition"
    - Zou et al. (2023), "Representation Engineering"
"""

import logging
from typing import Any

import torch

from .base import SteeringMethod
from src.data.preprocessing import extract_activations

logger = logging.getLogger(__name__)


class CAAMeanDiff(SteeringMethod):
    """Contrastive Activation Addition via raw mean difference.

    Given two sets of pre-extracted activations (e.g., one for a target
    class and one for a contrasting class), computes a unit-norm steering
    vector as the normalized difference of their means.
    """

    def __init__(self):
        super().__init__("caa_mean_diff")

    def compute_steering_vector(
        self,
        activations_pos: torch.Tensor,
        activations_neg: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute steering vector as normalized mean difference.

        Args:
            activations_pos: Activations for the target class, shape
                (n_pos, d_model).
            activations_neg: Activations for the contrasting class, shape
                (n_neg, d_model).

        Returns:
            Unit-norm steering vector of shape (d_model,).
        """
        with torch.no_grad():
            mean_pos = activations_pos.mean(dim=0)
            mean_neg = activations_neg.mean(dim=0)

            diff = mean_pos - mean_neg
            norm = diff.norm()
            if norm < 1e-8:
                logger.warning(
                    "CAAMeanDiff: mean difference has near-zero norm (%.2e); "
                    "steering vector may be degenerate.",
                    norm.item(),
                )
            steering_vec = diff / norm

        self._steering_vector = steering_vec
        logger.info(
            "CAAMeanDiff: computed steering vector (d_model=%d) from "
            "%d positive and %d negative activations.",
            steering_vec.shape[0],
            activations_pos.shape[0],
            activations_neg.shape[0],
        )
        return self._steering_vector


class CAAContrastive(SteeringMethod):
    """Contrastive Activation Addition from paired positive/negative prompts.

    For each contrastive pair (positive_text, negative_text), both texts
    are run through the model and their residual-stream activations at the
    specified layer are extracted. The steering vector is the normalized
    mean of the per-pair activation differences.
    """

    def __init__(self):
        super().__init__("caa_contrastive")

    def compute_steering_vector(
        self,
        model_wrapper: Any,
        layer: int,
        contrastive_pairs: list[tuple[str, str]],
        batch_size: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        """Compute steering vector from contrastive prompt pairs.

        Args:
            model_wrapper: A ModelWrapper instance providing the model and
                ``run_with_cache`` method.
            layer: Transformer layer index from which to extract activations.
            contrastive_pairs: List of (positive_text, negative_text) tuples.
                Each pair should express contrasting behaviors or attributes.
            batch_size: Number of texts per forward-pass batch.

        Returns:
            Unit-norm steering vector of shape (d_model,).
        """
        positive_texts = [pair[0] for pair in contrastive_pairs]
        negative_texts = [pair[1] for pair in contrastive_pairs]

        with torch.no_grad():
            activations_pos = extract_activations(
                positive_texts, model_wrapper, layer, batch_size=batch_size
            )
            activations_neg = extract_activations(
                negative_texts, model_wrapper, layer, batch_size=batch_size
            )

            # Per-pair differences, then average
            diff = (activations_pos - activations_neg).mean(dim=0)
            norm = diff.norm()
            if norm < 1e-8:
                logger.warning(
                    "CAAContrastive: mean difference has near-zero norm (%.2e); "
                    "steering vector may be degenerate.",
                    norm.item(),
                )
            steering_vec = diff / norm

        self._steering_vector = steering_vec
        logger.info(
            "CAAContrastive: computed steering vector (d_model=%d) from "
            "%d contrastive pairs at layer %d.",
            steering_vec.shape[0],
            len(contrastive_pairs),
            layer,
        )
        return self._steering_vector


class CAARepE(SteeringMethod):
    """Representation Engineering / persona-prefix CAA.

    Prepends a positive and a negative persona prefix to each neutral
    query, extracts activations for both sets, and computes a steering
    vector from the normalized mean difference. This follows the RepE
    (Representation Engineering) paradigm where system-prompt-level
    persona instructions create contrastive internal representations.
    """

    def __init__(self):
        super().__init__("caa_repe")

    def compute_steering_vector(
        self,
        model_wrapper: Any,
        layer: int,
        neutral_queries: list[str],
        positive_prefix: str,
        negative_prefix: str,
        batch_size: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        """Compute steering vector via persona-prefix contrast.

        Args:
            model_wrapper: A ModelWrapper instance providing the model and
                ``run_with_cache`` method.
            layer: Transformer layer index from which to extract activations.
            neutral_queries: List of neutral prompts that do not inherently
                exhibit the target behavior.
            positive_prefix: Persona prefix that encourages the target
                behavior (e.g., "You are an extremely helpful assistant. ").
            negative_prefix: Persona prefix that encourages the opposite
                behavior (e.g., "You are an unhelpful assistant. ").
            batch_size: Number of texts per forward-pass batch.

        Returns:
            Unit-norm steering vector of shape (d_model,).
        """
        positive_texts = [positive_prefix + query for query in neutral_queries]
        negative_texts = [negative_prefix + query for query in neutral_queries]

        with torch.no_grad():
            activations_pos = extract_activations(
                positive_texts, model_wrapper, layer, batch_size=batch_size
            )
            activations_neg = extract_activations(
                negative_texts, model_wrapper, layer, batch_size=batch_size
            )

            # Mean difference across all queries
            diff = (activations_pos - activations_neg).mean(dim=0)
            norm = diff.norm()
            if norm < 1e-8:
                logger.warning(
                    "CAARepE: mean difference has near-zero norm (%.2e); "
                    "steering vector may be degenerate.",
                    norm.item(),
                )
            steering_vec = diff / norm

        self._steering_vector = steering_vec
        logger.info(
            "CAARepE: computed steering vector (d_model=%d) from "
            "%d neutral queries with persona prefixes at layer %d.",
            steering_vec.shape[0],
            len(neutral_queries),
            layer,
        )
        return self._steering_vector
