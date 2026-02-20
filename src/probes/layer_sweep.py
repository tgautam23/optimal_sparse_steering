"""Concept concentration sweep across all transformer layers.

Measures how well class-relevant structure concentrates into a few PCA
directions at each layer.  Two metrics are reported:

    R^2_k = sum_{j in top-k} s_j^2  /  ||mu_diff||^2

where s_j = v_j^T mu_diff is the class separation along PCA direction v_j.
High R^2_k means the concept lives in a low-dimensional subspace.

    explained_var = sum_{j in top-k} s_j^2

The absolute explained class separation in the subspace.  Layer selection
uses ``explained_var`` rather than R^2_k because R^2_k can be trivially
high at early layers where ||mu_diff|| is near zero.
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
            - ``r2_k``: Concept concentration ratio (fraction of ||mu_diff||^2
              explained by the top-k subspace).
            - ``explained_var``: Absolute explained separation
              (sum of top-k s_j^2).  Used for layer ranking.
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
            results[layer] = {"r2_k": 0.0, "explained_var": 0.0, "top_separation": 0.0}
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

        explained_var = float(np.sum(top_s ** 2))
        r2_k = explained_var / mu_diff_sq
        top_separation = float(np.max(np.abs(separations)))

        results[layer] = {"r2_k": r2_k, "explained_var": explained_var, "top_separation": top_separation}

    # Print summary table (best marker uses default skip_last_n=2)
    selected = find_best_layer(results)
    print(f"\n{'Layer':>5}  {'R^2_k':>8}  {'Σs_j²':>12}  {'|s_1|':>10}")
    print("-" * 42)
    for layer in sorted(results):
        r = results[layer]
        marker = " <-- best" if layer == selected else ""
        print(f"{layer:5d}  {r['r2_k']:8.4f}  {r['explained_var']:12.4f}  {r['top_separation']:10.4f}{marker}")

    return results


def find_best_layer(sweep_results: dict[int, dict], skip_last_n: int = 2) -> int:
    """Return the layer with the highest absolute explained separation.

    Ranks by ``explained_var`` (sum of top-k s_j^2) rather than R^2_k,
    because R^2_k can be trivially high at early layers where
    ``||mu_diff||`` is near zero.

    The last ``skip_last_n`` layers are excluded because interventions
    there have little downstream computation to propagate through,
    making them poor choices for steering despite strong concept signal.

    Args:
        sweep_results: Mapping of layer index to result dict (with
            ``explained_var`` key), as returned by
            :func:`concept_concentration_sweep`.
        skip_last_n: Number of final layers to exclude from selection.

    Returns:
        Layer index that achieved the highest explained_var among
        eligible layers.
    """
    max_layer = max(sweep_results)
    cutoff = max_layer - skip_last_n + 1
    eligible = {l: v for l, v in sweep_results.items() if l < cutoff}
    if not eligible:
        eligible = sweep_results
    return max(eligible, key=lambda l: eligible[l]["explained_var"])
