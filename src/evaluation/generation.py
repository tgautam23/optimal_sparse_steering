"""Steered text generation with hooks."""

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def steered_generation(
    model_wrapper,
    prompts: list[str],
    steering_method,
    layer: int,
    alpha: float = 5.0,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[str]:
    """Generate text with a steering hook applied.

    Args:
        model_wrapper: Wrapper around a HookedSAETransformer, providing
            .model, .tokenizer, and .get_hook_name(layer).
        prompts: List of prompt strings.
        steering_method: Object with .get_hook_fn(alpha) returning a hook
            function compatible with HookedSAETransformer.add_hook.
        layer: Transformer layer index at which to apply steering.
        alpha: Steering strength multiplier.
        max_new_tokens: Maximum number of new tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.

    Returns:
        List of generated text strings (one per prompt).
    """
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    hook_name = model_wrapper.get_hook_name(layer)
    hook_fn = steering_method.get_hook_fn(alpha)
    device = next(model.parameters()).device

    generated_texts = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Attach the steering hook
        model.add_hook(hook_name, hook_fn)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
        finally:
            # Always clean up hooks
            model.reset_hooks()

        generated_text = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        generated_texts.append(generated_text)

    return generated_texts


def steered_forward(
    model_wrapper,
    tokens: torch.Tensor,
    steering_method,
    layer: int,
    alpha: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass with a steering hook attached.

    This is used for computing KL divergence and steered probe scores
    without generating text.

    Args:
        model_wrapper: Wrapper around a HookedSAETransformer, providing
            .model, .tokenizer, and .get_hook_name(layer).
        tokens: Input token tensor of shape (batch, seq_len).
        steering_method: Object with .get_hook_fn(alpha) returning a hook
            function compatible with HookedSAETransformer.add_hook.
        layer: Transformer layer index at which to apply steering.
        alpha: Steering strength multiplier.

    Returns:
        Tuple of (logits, steered_activations_at_layer).
        logits has shape (batch, seq_len, vocab_size).
        steered_activations_at_layer has shape (batch, seq_len, d_model).
    """
    model = model_wrapper.model
    hook_name = model_wrapper.get_hook_name(layer)
    hook_fn = steering_method.get_hook_fn(alpha)

    # Storage for capturing activations at the steered layer
    captured_activations = {}

    def capture_hook(value, hook):
        """Hook that captures activations after steering is applied."""
        captured_activations["activations"] = value.detach().clone()
        return value

    # Attach steering hook and capture hook
    model.add_hook(hook_name, hook_fn)
    model.add_hook(hook_name, capture_hook)

    try:
        with torch.no_grad():
            logits = model(tokens)
    finally:
        model.reset_hooks()

    steered_activations = captured_activations["activations"]

    return logits, steered_activations


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample one token from logits at the last position.

    Args:
        logits: Shape (batch, seq_len, vocab_size).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        Token id tensor of shape (batch, 1).
    """
    next_logits = logits[:, -1, :] / max(temperature, 1e-8)

    # Nucleus (top-p) sampling
    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens above the top-p threshold
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[mask] = -float("inf")
    # Scatter back
    next_logits.scatter_(1, sorted_indices, sorted_logits)

    probs = F.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


# ---------------------------------------------------------------------------
# Static optimal generation
# ---------------------------------------------------------------------------

def optimal_steered_generation_static(
    model_wrapper,
    prompts: list[str],
    steering_method,
    layer: int,
    concept_subspace,
    D: torch.Tensor,
    target_class: int = 1,
    alpha: float = 5.0,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[dict]:
    """Generate text with a steering vector solved once on the initial prompt.

    For each prompt:
      1. Forward pass to extract last-token activation and SAE features
      2. Solve the optimization (QP/SOCP) once
      3. Token-by-token generation with the fixed steering vector

    Args:
        model_wrapper: ModelWrapper instance.
        prompts: List of prompt strings.
        steering_method: QPOptimalSteering or ConvexOptimalSteering instance.
        layer: Transformer layer index for steering.
        concept_subspace: Fitted ConceptSubspace instance.
        D: SAE decoder matrix, shape (d_sae, d_model).
        target_class: Target class (0 or 1).
        alpha: Steering strength multiplier.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        List of dicts per prompt with keys: 'text', 'solve_time', 'l0',
        'n_tokens', 'total_time'.
    """
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    hook_name = model_wrapper.get_hook_name(layer)
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        t_start = time.time()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Step 1: Forward pass to get last-token activation + SAE features
        with torch.no_grad():
            _, cache = model_wrapper.run_with_cache(
                input_ids, names_filter=hook_name,
            )
        h = cache[hook_name][0, -1, :].cpu()  # (d_model,)
        sae_feats = model_wrapper.encode_with_sae(h.unsqueeze(0), layer).squeeze(0)

        # Step 2: Solve optimization once
        sv = steering_method.compute_steering_vector(
            h=h, D=D, concept_subspace=concept_subspace,
            sae_features=sae_feats, target_class=target_class,
        )
        solve_time = steering_method.solve_time
        l0 = int((steering_method.delta > 1e-6).sum())

        # Step 3: Token-by-token generation with fixed hook
        sv_device = sv.to(device)

        def static_hook(activations, hook):
            return activations + alpha * sv_device

        eos_id = tokenizer.eos_token_id
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            model.add_hook(hook_name, static_hook)
            try:
                with torch.no_grad():
                    logits = model(generated_ids)
            finally:
                model.reset_hooks()

            next_token = _sample_next_token(logits, temperature, top_p)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        total_time = time.time() - t_start

        results.append({
            "text": text,
            "solve_time": solve_time,
            "l0": l0,
            "n_tokens": generated_ids.shape[1] - input_ids.shape[1],
            "total_time": total_time,
        })
        logger.info(
            f"Static generation: {results[-1]['n_tokens']} tokens, "
            f"solve={solve_time:.3f}s, total={total_time:.2f}s"
        )

    return results


# ---------------------------------------------------------------------------
# Adaptive optimal generation
# ---------------------------------------------------------------------------

def optimal_steered_generation_adaptive(
    model_wrapper,
    prompts: list[str],
    steering_method,
    layer: int,
    concept_subspace,
    D: torch.Tensor,
    target_class: int = 1,
    alpha: float = 5.0,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[dict]:
    """Generate text re-solving the optimization at each token step.

    For each generated token:
      1. Forward pass (unsteered) to extract current last-token activation
      2. Solve the optimization for this activation
      3. Forward pass with the new steering vector to get steered logits
      4. Sample next token

    This is more expensive (two forward passes + one solve per token) but
    produces an input-adaptive steering vector at every step.

    Args:
        model_wrapper: ModelWrapper instance.
        prompts: List of prompt strings.
        steering_method: QPOptimalSteering or ConvexOptimalSteering instance.
        layer: Transformer layer index for steering.
        concept_subspace: Fitted ConceptSubspace instance.
        D: SAE decoder matrix, shape (d_sae, d_model).
        target_class: Target class (0 or 1).
        alpha: Steering strength multiplier.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        List of dicts per prompt with keys: 'text', 'total_solve_time',
        'solve_times', 'l0s', 'n_tokens', 'total_time'.
    """
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    hook_name = model_wrapper.get_hook_name(layer)
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        t_start = time.time()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        eos_id = tokenizer.eos_token_id
        generated_ids = input_ids.clone()
        solve_times = []
        l0s = []

        for _ in range(max_new_tokens):
            # Step 1: Unsteered forward pass to get current activation
            with torch.no_grad():
                _, cache = model_wrapper.run_with_cache(
                    generated_ids, names_filter=hook_name,
                )
            h = cache[hook_name][0, -1, :].cpu()  # (d_model,)
            sae_feats = model_wrapper.encode_with_sae(
                h.unsqueeze(0), layer
            ).squeeze(0)

            # Step 2: Solve optimization for current state
            sv = steering_method.compute_steering_vector(
                h=h, D=D, concept_subspace=concept_subspace,
                sae_features=sae_feats, target_class=target_class,
            )
            solve_times.append(steering_method.solve_time)
            l0s.append(int((steering_method.delta > 1e-6).sum()))

            # Step 3: Steered forward pass to get logits
            sv_device = sv.to(device)

            def adaptive_hook(activations, hook, _sv=sv_device):
                return activations + alpha * _sv

            model.add_hook(hook_name, adaptive_hook)
            try:
                with torch.no_grad():
                    logits = model(generated_ids)
            finally:
                model.reset_hooks()

            # Step 4: Sample next token
            next_token = _sample_next_token(logits, temperature, top_p)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        total_time = time.time() - t_start
        total_solve = sum(solve_times)
        n_tokens = generated_ids.shape[1] - input_ids.shape[1]

        results.append({
            "text": text,
            "total_solve_time": total_solve,
            "solve_times": solve_times,
            "l0s": l0s,
            "n_tokens": n_tokens,
            "total_time": total_time,
        })
        logger.info(
            f"Adaptive generation: {n_tokens} tokens, "
            f"solve={total_solve:.2f}s ({total_solve/max(n_tokens,1):.3f}s/tok), "
            f"total={total_time:.2f}s"
        )

    return results
