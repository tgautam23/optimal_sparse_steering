"""Concept concentration sweep across all transformer layers.

Measures how well class-relevant structure concentrates into a few PCA
directions at each layer, using the R^2_k metric:

    R^2_k = sum_{j in top-k} s_j^2  /  ||mu_diff||^2

where s_j = v_j^T mu_diff is the class separation along PCA direction v_j.
High R^2_k means the concept lives in a low-dimensional subspace at that
layer -- exactly what sparse steering needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.preprocessing import extract_activations


def concept_concentration_sweep(
    texts: list[str],
    labels: list[int],
    model_wrapper: Any,
    n_layers: int,
    n_components: int = 50,
    n_select: int = 5,
    batch_size: int = 8,
) -> dict[int, dict]:
    """Sweep all layers and compute concept concentration R^2_k at each.

    For each layer in ``0 .. n_layers - 1``:

    1. Extract residual-stream activations.
    2. Compute class means ``mu_0``, ``mu_1``, and ``mu_diff = mu_1 - mu_0``.
    3. SVD on centered activations to get top PCA directions.
    4. Project ``mu_diff`` onto each direction: ``s_j = v_j^T mu_diff``.
    5. Select top ``n_select`` directions by ``|s_j|`` and compute
       ``R^2_k = sum(s_j^2) / ||mu_diff||^2``.

    Args:
        texts: Input text samples.
        labels: Binary labels corresponding to *texts*.
        model_wrapper: A ``ModelWrapper`` instance.
        n_layers: Number of transformer layers to sweep over.
        n_components: Number of PCA components to compute (intermediate).
        n_select: Number of top directions to keep for R^2_k.
        batch_size: Batch size for activation extraction.

    Returns:
        Dictionary mapping layer index to a dict with keys:
            - ``r2_k``: Concept concentration score.
            - ``top_separation``: Largest absolute class separation ``|s_1|``.
    """
    y = np.asarray(labels)
    results: dict[int, dict] = {}

    for layer in tqdm(range(n_layers), desc="Layer sweep (concept concentration)"):
        activations = extract_activations(
            texts, model_wrapper, layer=layer, batch_size=batch_size,
        )
        X = activations.numpy().astype(np.float64)
        n_samples, d_model = X.shape

        # Class means and difference
        mask0 = y == 0
        mask1 = y == 1
        mu_0 = X[mask0].mean(axis=0)
        mu_1 = X[mask1].mean(axis=0)
        mu_diff = mu_1 - mu_0
        mu_diff_sq = float(np.dot(mu_diff, mu_diff))

        if mu_diff_sq < 1e-12:
            results[layer] = {"r2_k": 0.0, "top_separation": 0.0}
            continue

        # PCA via truncated SVD on centered activations
        mean_all = X.mean(axis=0)
        X_centered = X - mean_all
        nc = min(n_components, n_samples, d_model)

        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        V = Vt[:nc]  # (nc, d_model)

        # Class separation per PC: s_j = v_j^T mu_diff
        separations = V @ mu_diff  # (nc,)

        # Select top n_select by |s_j|
        ns = min(n_select, nc)
        top_idx = np.argsort(np.abs(separations))[::-1][:ns]
        top_s = separations[top_idx]

        r2_k = float(np.sum(top_s ** 2) / mu_diff_sq)
        top_separation = float(np.max(np.abs(separations)))

        results[layer] = {"r2_k": r2_k, "top_separation": top_separation}

    # Print summary table
    print(f"\n{'Layer':>5}  {'R^2_k':>8}  {'|s_1|':>10}")
    print("-" * 28)
    best_layer = max(results, key=lambda l: results[l]["r2_k"])
    for layer in sorted(results):
        r = results[layer]
        marker = " <-- best" if layer == best_layer else ""
        print(f"{layer:5d}  {r['r2_k']:8.4f}  {r['top_separation']:10.4f}{marker}")

    return results


def find_best_layer(sweep_results: dict[int, dict]) -> int:
    """Return the layer index with the highest concept concentration R^2_k.

    Args:
        sweep_results: Mapping of layer index to result dict (with ``r2_k``
            key), as returned by :func:`concept_concentration_sweep`.

    Returns:
        Layer index that achieved the highest R^2_k.
    """
    return max(sweep_results, key=lambda l: sweep_results[l]["r2_k"])
