"""ModelWrapper around HookedSAETransformer with SAE caching."""

from __future__ import annotations

import torch
from transformer_lens import HookedSAETransformer
from sae_lens import SAE

from configs.base import ModelConfig
from .sae_utils import encode_activations, decode_features


class ModelWrapper:
    """Thin wrapper around a TransformerLens model with lazy-loaded SAEs.

    Parameters
    ----------
    config : ModelConfig
        Configuration dataclass that specifies the model name, SAE release,
        hook templates, device, dtype, etc.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
            config.tl_name,
            device=config.device,
            dtype=getattr(torch, config.dtype),
        )
        # Lazy-loaded SAE cache: layer -> SAE
        self._saes: dict[int, SAE] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        """Return the stored ModelConfig."""
        return self._config

    @property
    def model(self) -> HookedSAETransformer:
        """Return the underlying HookedSAETransformer."""
        return self._model

    @property
    def tokenizer(self):
        """Return the tokenizer associated with the model."""
        return self._model.tokenizer

    # ------------------------------------------------------------------
    # SAE management
    # ------------------------------------------------------------------

    def get_sae(self, layer: int) -> SAE:
        """Lazy-load and cache an SAE for the given layer.

        Parameters
        ----------
        layer : int
            Transformer layer index.

        Returns
        -------
        SAE
            The loaded sae-lens SAE, placed on the same device as the model.
        """
        if layer not in self._saes:
            sae_id = self._config.sae_id_template.format(layer=layer)
            sae, _, _ = SAE.from_pretrained(
                release=self._config.sae_release,
                sae_id=sae_id,
            )
            sae = sae.to(self._config.device)
            self._saes[layer] = sae
        return self._saes[layer]

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_with_cache(
        self,
        tokens: torch.Tensor,
        names_filter=None,
    ) -> tuple:
        """Run the model and return (logits, cache).

        Parameters
        ----------
        tokens : torch.Tensor
            Input token ids of shape ``(batch, seq)``.
        names_filter : str | list[str] | callable | None
            Passed directly to ``HookedSAETransformer.run_with_cache``.

        Returns
        -------
        tuple
            ``(logits, cache)`` as returned by the underlying model.
        """
        return self._model.run_with_cache(tokens, names_filter=names_filter)

    @torch.no_grad()
    def encode_with_sae(
        self,
        activations: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Encode residual-stream activations into SAE feature activations.

        Parameters
        ----------
        activations : torch.Tensor
            Residual-stream activations, typically of shape ``(batch, seq, d_model)``.
        layer : int
            Layer whose SAE should be used.

        Returns
        -------
        torch.Tensor
            SAE feature activations.
        """
        sae = self.get_sae(layer)
        return encode_activations(sae, activations)

    @torch.no_grad()
    def decode_with_sae(
        self,
        features: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Decode SAE feature activations back to residual-stream space.

        Parameters
        ----------
        features : torch.Tensor
            SAE feature activations.
        layer : int
            Layer whose SAE should be used.

        Returns
        -------
        torch.Tensor
            Reconstructed residual-stream activations.
        """
        sae = self.get_sae(layer)
        return decode_features(sae, features)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_hook_name(self, layer: int) -> str:
        """Return the TransformerLens hook-point name for *layer*.

        Parameters
        ----------
        layer : int
            Transformer layer index.

        Returns
        -------
        str
            Hook name, e.g. ``"blocks.7.hook_resid_pre"``.
        """
        return self._config.hook_template.format(layer=layer)
