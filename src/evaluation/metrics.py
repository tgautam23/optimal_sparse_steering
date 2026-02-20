"""Evaluation metrics for steering."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def compute_subspace_score(concept_subspace, activations: np.ndarray,
                           target_class: int = 1) -> float:
    """Mean fraction of subspace constraints satisfied across samples.

    For each sample, checks each direction against the target class
    mean projection threshold.

    Args:
        concept_subspace: A fitted ConceptSubspace instance.
        activations: Array of shape (n_samples, d_model).
        target_class: Which class the constraints target (0 or 1).

    Returns:
        Float in [0, 1] -- mean fraction of constraints satisfied.
    """
    activations = np.asarray(activations, dtype=np.float64)
    dirs = concept_subspace.get_constraint_directions(target_class)  # (k, d_model)
    thresholds = concept_subspace.compute_thresholds(target_class)  # (k,)
    projections = activations @ dirs.T  # (n, k)
    satisfied = (projections >= thresholds[None, :]).astype(float)
    return float(satisfied.mean())


def compute_probe_score(
    probe, activations: np.ndarray, target_class: int = 1
) -> float:
    """Compute mean predicted probability for target_class across all samples.

    Args:
        probe: A fitted sklearn-style classifier with predict_proba method.
        activations: Array of shape (n_samples, n_features).
        target_class: Class index to compute probability for.

    Returns:
        Mean predicted probability for target_class.
    """
    return float(probe.predict_proba(activations)[:, target_class].mean())


def compute_perplexity(
    model, texts: list[str], batch_size: int = 4
) -> float:
    """Compute perplexity of texts under the model.

    Args:
        model: A HookedSAETransformer with .tokenizer attribute.
        texts: List of text strings to evaluate.
        batch_size: Number of texts to process at once.

    Returns:
        Perplexity (exp of mean cross-entropy loss).
    """
    tokenizer = model.tokenizer
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encodings["input_ids"].to(
                next(model.parameters()).device
            )
            attention_mask = encodings["attention_mask"].to(input_ids.device)

            # Forward pass: model returns logits of shape (batch, seq, vocab)
            logits = model(input_ids)

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Compute per-token cross-entropy loss
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())

            # Mask out padding tokens
            masked_loss = loss_per_token * shift_mask
            total_loss += masked_loss.sum().item()
            total_tokens += shift_mask.sum().item()

    if total_tokens == 0:
        return float("inf")

    mean_loss = total_loss / total_tokens
    return math.exp(mean_loss)


def compute_kl_divergence(
    logits_steered: torch.Tensor, logits_base: torch.Tensor
) -> float:
    """Compute KL(steered || base) averaged over sequence positions.

    Args:
        logits_steered: Logits from steered model, shape (batch, seq, vocab).
        logits_base: Logits from base model, shape (batch, seq, vocab).

    Returns:
        Mean KL divergence as a float.
    """
    log_probs_steered = F.log_softmax(logits_steered, dim=-1)
    log_probs_base = F.log_softmax(logits_base, dim=-1)

    # kl_div expects input in log-space, target in log-space with log_target=True
    kl = F.kl_div(
        log_probs_base,
        log_probs_steered,
        reduction="batchmean",
        log_target=True,
    )
    return float(kl.item())


def compute_l0(delta: np.ndarray, threshold: float = 1e-6) -> int:
    """Count nonzero entries in the sparse feature vector.

    Args:
        delta: Sparse feature vector (1-D or N-D array).
        threshold: Absolute value below which entries are treated as zero.

    Returns:
        Number of entries with absolute value above threshold.
    """
    return int(np.sum(np.abs(delta) > threshold))


def compute_all_metrics(
    probe,
    steered_activations: np.ndarray,
    base_activations: np.ndarray,
    delta: np.ndarray,
    target_class: int,
    logits_steered: torch.Tensor,
    logits_base: torch.Tensor,
) -> dict:
    """Compute all evaluation metrics and return them as a dict.

    Args:
        probe: Fitted sklearn-style classifier.
        steered_activations: Activations from steered model.
        base_activations: Activations from base model.
        delta: Sparse steering vector.
        target_class: Class index for probe score computation.
        logits_steered: Logits from steered forward pass.
        logits_base: Logits from base forward pass.

    Returns:
        Dictionary with keys: probe_score_steered, probe_score_base,
        probe_score_delta, kl_divergence, l0.
    """
    probe_steered = compute_probe_score(probe, steered_activations, target_class)
    probe_base = compute_probe_score(probe, base_activations, target_class)
    kl_div = compute_kl_divergence(logits_steered, logits_base)
    l0 = compute_l0(delta)

    return {
        "probe_score_steered": probe_steered,
        "probe_score_base": probe_base,
        "probe_score_delta": probe_steered - probe_base,
        "kl_divergence": kl_div,
        "l0": l0,
    }
