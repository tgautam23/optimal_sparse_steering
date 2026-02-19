"""Cross-validated probe accuracy across all transformer layers.

Provides utilities to sweep a linear probe over every residual-stream
layer, identify the layer that best linearly separates a target concept,
and return structured results for downstream use.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from src.data.preprocessing import extract_activations


def layer_sweep(
    texts: list[str],
    labels: list[int],
    model_wrapper: Any,
    n_layers: int,
    batch_size: int = 8,
    cv_folds: int = 5,
) -> dict[int, float]:
    """Run cross-validated linear-probe evaluation on every layer.

    For each layer in ``0 .. n_layers - 1`` the function:

    1. Extracts residual-stream activations using
       :func:`src.data.preprocessing.extract_activations`.
    2. Trains a logistic regression classifier with ``cv_folds``-fold
       cross-validation.
    3. Records the mean accuracy across folds.

    Args:
        texts: Input text samples.
        labels: Binary labels corresponding to *texts*.
        model_wrapper: A ``ModelWrapper`` instance (see
            :mod:`src.models.wrapper`).
        n_layers: Number of transformer layers to sweep over.
        batch_size: Batch size for activation extraction.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dictionary mapping layer index to mean cross-validated accuracy.
    """
    y = np.asarray(labels)
    results: dict[int, float] = {}

    for layer in tqdm(range(n_layers), desc="Layer sweep"):
        # Extract activations for this layer -- returns torch.Tensor
        activations = extract_activations(
            texts,
            model_wrapper,
            layer=layer,
            batch_size=batch_size,
        )
        X = activations.numpy()  # (n_texts, d_model)

        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
        )
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
        mean_acc = float(scores.mean())
        results[layer] = mean_acc

    # Report best layer
    best_layer = max(results, key=results.get)  # type: ignore[arg-type]
    print(
        f"Best layer: {best_layer} with accuracy {results[best_layer]:.4f}"
    )

    return results


def find_best_layer(sweep_results: dict[int, float]) -> int:
    """Return the layer index with the highest probe accuracy.

    Args:
        sweep_results: Mapping of layer index to mean accuracy, as returned
            by :func:`layer_sweep`.

    Returns:
        Layer index that achieved the highest cross-validated accuracy.
    """
    return max(sweep_results, key=sweep_results.get)  # type: ignore[arg-type]
