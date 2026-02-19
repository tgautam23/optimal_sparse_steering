"""SAE encode/decode helpers and decoder matrix extraction."""

from __future__ import annotations

import torch
from sae_lens import SAE


def encode_activations(sae: SAE, activations: torch.Tensor) -> torch.Tensor:
    """Run activations through the SAE encoder.

    Parameters
    ----------
    sae : SAE
        A loaded sae-lens SAE.
    activations : torch.Tensor
        Residual-stream activations, typically of shape ``(batch, seq, d_model)``.

    Returns
    -------
    torch.Tensor
        SAE feature activations (post-encoder, post-activation-function).
    """
    return sae.encode(activations)


def decode_features(sae: SAE, features: torch.Tensor) -> torch.Tensor:
    """Decode SAE feature activations back through the SAE decoder.

    Parameters
    ----------
    sae : SAE
        A loaded sae-lens SAE.
    features : torch.Tensor
        SAE feature activations.

    Returns
    -------
    torch.Tensor
        Reconstructed residual-stream activations.
    """
    return sae.decode(features)


def get_decoder_matrix(sae: SAE) -> torch.Tensor:
    """Extract the SAE decoder weight matrix.

    Parameters
    ----------
    sae : SAE
        A loaded sae-lens SAE.

    Returns
    -------
    torch.Tensor
        Decoder weight matrix of shape ``(d_sae, d_model)``, detached and on
        CPU.
    """
    return sae.W_dec.detach().cpu()


def get_encoder_matrix(sae: SAE) -> torch.Tensor:
    """Extract the SAE encoder weight matrix.

    Parameters
    ----------
    sae : SAE
        A loaded sae-lens SAE.

    Returns
    -------
    torch.Tensor
        Encoder weight matrix, detached and on CPU.
    """
    return sae.W_enc.detach().cpu()


def get_feature_direction(sae: SAE, feature_idx: int) -> torch.Tensor:
    """Get the unit-norm decoder direction for a single feature.

    Parameters
    ----------
    sae : SAE
        A loaded sae-lens SAE.
    feature_idx : int
        Index of the SAE feature.

    Returns
    -------
    torch.Tensor
        Unit-length decoder direction vector of shape ``(d_model,)``.
    """
    direction = sae.W_dec[feature_idx].detach().cpu().float()
    return direction / direction.norm()
