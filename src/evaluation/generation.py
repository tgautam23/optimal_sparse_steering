"""Steered text generation with hooks."""

import torch


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
