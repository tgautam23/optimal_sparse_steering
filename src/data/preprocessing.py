"""Tokenization and batched activation extraction.

Provides utilities for tokenizing text, extracting residual-stream
activations at a specified layer, and extracting SAE feature activations
for use in probe training and steering vector computation.
"""

import logging
from typing import Any

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def tokenize_texts(
    texts: list[str],
    model: Any,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    """Tokenize a list of texts using the model's tokenizer.

    Applies left-padding so that the last non-padding token is always
    the final meaningful position (consistent with causal LM convention).

    Args:
        texts: Raw input strings to tokenize.
        model: A HookedSAETransformer (or compatible) with a `.tokenizer`
            attribute.
        max_seq_len: Maximum sequence length; sequences are truncated
            to this length.

    Returns:
        Dictionary with:
            - "input_ids": torch.LongTensor of shape (batch, seq_len)
            - "attention_mask": torch.LongTensor of shape (batch, seq_len)
    """
    tokenizer = model.tokenizer

    # Store original padding side to restore after tokenization
    original_padding_side = getattr(tokenizer, "padding_side", "right")

    # Ensure pad token is set (common for GPT-style models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use left-padding for causal LMs so the last token is always meaningful
    tokenizer.padding_side = "left"

    try:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
    finally:
        # Restore original padding side
        tokenizer.padding_side = original_padding_side

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def _get_last_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Find the index of the last non-padding token for each sequence.

    Args:
        attention_mask: Binary mask of shape (batch, seq_len) where 1 indicates
            a real token and 0 indicates padding.

    Returns:
        LongTensor of shape (batch,) with the index of the last attended
        position in each sequence.
    """
    # Sum attention mask per row and subtract 1 to get the last valid index
    return attention_mask.sum(dim=1).long() - 1


def extract_activations(
    texts: list[str],
    model_wrapper: Any,
    layer: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract residual-stream activations at a given layer for each text.

    Uses the last non-padding token position as the representative
    activation for each input sequence (standard for causal LMs).

    Args:
        texts: Input texts to process.
        model_wrapper: A ModelWrapper instance providing `.model` (the
            HookedSAETransformer) and `.run_with_cache(tokens, names_filter)`.
        layer: Transformer layer index to extract activations from.
        batch_size: Number of texts to process in each forward pass.

    Returns:
        Tensor of shape (n_texts, d_model) containing the residual-stream
        activation at the last token position for each input.
    """
    model = model_wrapper.model
    hook_name = f"blocks.{layer}.hook_resid_pre"
    device = next(model.parameters()).device

    all_activations: list[torch.Tensor] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Extracting activations (layer {layer})",
        leave=False,
    ):
        batch_texts = texts[start : start + batch_size]
        tokenized = tokenize_texts(
            batch_texts,
            model,
            max_seq_len=128,
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model_wrapper.run_with_cache(
                input_ids,
                names_filter=hook_name,
            )

        # cache[hook_name] has shape (batch, seq_len, d_model)
        hidden_states = cache[hook_name]

        # Gather the last non-padding token activation for each sequence
        last_positions = _get_last_token_positions(attention_mask)
        # last_positions: (batch,) -> index into seq_len dimension
        batch_indices = torch.arange(hidden_states.size(0), device=device)
        batch_activations = hidden_states[batch_indices, last_positions]

        all_activations.append(batch_activations.cpu())

    result = torch.cat(all_activations, dim=0)
    logger.info(
        "Extracted activations: shape %s from layer %d",
        tuple(result.shape),
        layer,
    )
    return result


def extract_sae_features(
    texts: list[str],
    model_wrapper: Any,
    layer: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract SAE feature activations at a given layer for each text.

    First extracts residual-stream activations, then encodes them through
    the SAE to obtain sparse feature activations.

    Args:
        texts: Input texts to process.
        model_wrapper: A ModelWrapper instance providing `.model`,
            `.run_with_cache(tokens, names_filter)`, and
            `.encode_with_sae(activations, layer)`.
        layer: Transformer layer index to extract and encode from.
        batch_size: Number of texts to process in each forward pass.

    Returns:
        Tensor of shape (n_texts, d_sae) containing the SAE feature
        activations for each input.
    """
    model = model_wrapper.model
    hook_name = f"blocks.{layer}.hook_resid_pre"
    device = next(model.parameters()).device

    all_sae_features: list[torch.Tensor] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Extracting SAE features (layer {layer})",
        leave=False,
    ):
        batch_texts = texts[start : start + batch_size]
        tokenized = tokenize_texts(
            batch_texts,
            model,
            max_seq_len=128,
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model_wrapper.run_with_cache(
                input_ids,
                names_filter=hook_name,
            )

        # cache[hook_name] has shape (batch, seq_len, d_model)
        hidden_states = cache[hook_name]

        # Gather the last non-padding token activation for each sequence
        last_positions = _get_last_token_positions(attention_mask)
        batch_indices = torch.arange(hidden_states.size(0), device=device)
        batch_activations = hidden_states[batch_indices, last_positions]

        # Encode through the SAE
        with torch.no_grad():
            sae_features = model_wrapper.encode_with_sae(
                batch_activations, layer
            )

        all_sae_features.append(sae_features.cpu())

    result = torch.cat(all_sae_features, dim=0)
    logger.info(
        "Extracted SAE features: shape %s from layer %d",
        tuple(result.shape),
        layer,
    )
    return result
