"""SAE-concept alignment sweep across all transformer layers.

For each layer, measures how well individual SAE decoder directions
align with the concept subspace directions.  The sweep criterion is
the **bottleneck cosine-squared alignment**:

    score = min_j  max_i  cos²(d_i, w_j)

where d_i are (normalized) SAE decoder directions and w_j are the
concept subspace directions.  High score means every concept direction
has at least one well-aligned SAE feature, predicting that the SOCP/QP
sparse steering formulation will be feasible with a modest coherence
budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data.preprocessing import extract_activations
from src.models.sae_utils import get_decoder_matrix
from src.probes.concept_subspace import ConceptSubspace


def alignment_sweep(
    texts: list[str],
    labels: list[int],
    model_wrapper: Any,
    n_layers: int,
    n_components: int = 50,
    n_select: int = 5,
    batch_size: int = 8,
) -> dict[int, dict]:
    """Sweep layers and compute SAE-concept alignment at each.

    For each layer in ``0 .. n_layers - 1``:

    1. Extract residual-stream activations and fit a concept subspace
       (PCA + class separation selection) to get *k* unit-norm
       concept directions.
    2. Load the SAE decoder matrix for that layer.
    3. Compute ``cos²(d_i, w_j)`` for all SAE features *i* and
       concept directions *j*.
    4. For each concept direction *j*, find the best single-feature
       alignment: ``max_i cos²(d_i, w_j)``.
    5. The layer score is the **bottleneck** (worst-covered direction):
       ``min_j max_i cos²(d_i, w_j)``.

    Args:
        texts: Input text samples.
        labels: Binary labels corresponding to *texts*.
        model_wrapper: A ``ModelWrapper`` instance.
        n_layers: Number of transformer layers to sweep.
        n_components: PCA components for concept subspace.
        n_select: Top directions to keep for concept subspace.
        batch_size: Batch size for activation extraction.

    Returns:
        Dictionary mapping layer index to a dict with:
            - ``alignment``: Bottleneck cos² (min over concept directions).
            - ``mean_alignment``: Mean cos² across concept directions.
            - ``per_direction``: List of max cos² per concept direction.
    """
    y = np.asarray(labels)
    results: dict[int, dict] = {}

    for layer in tqdm(range(n_layers), desc="Layer sweep (SAE-concept alignment)"):
        # 1. Extract activations and fit concept subspace
        activations = extract_activations(
            texts, model_wrapper, layer=layer, batch_size=batch_size,
        )
        X = activations.numpy().astype(np.float64)

        cs = ConceptSubspace(n_components=n_components, n_select=n_select)
        cs.fit(X, y)
        W = cs.directions  # (k, d_model), unit-norm rows

        # 2. Load SAE decoder
        sae = model_wrapper.get_sae(layer)
        D = get_decoder_matrix(sae)  # (d_sae, d_model)
        if isinstance(D, torch.Tensor):
            D_np = D.detach().cpu().numpy().astype(np.float64)
        else:
            D_np = np.asarray(D, dtype=np.float64)

        # 3. Normalize decoder rows to unit norm
        D_norms = np.linalg.norm(D_np, axis=1, keepdims=True)
        D_hat = D_np / np.maximum(D_norms, 1e-8)  # (d_sae, d_model)

        # 4. cos(d_i, w_j) = D_hat @ W^T since W rows are unit norm
        cos_matrix = D_hat @ W.T  # (d_sae, k)
        cos_sq = cos_matrix ** 2  # (d_sae, k)

        # 5. For each concept direction j, best single-feature alignment
        max_cos_sq = cos_sq.max(axis=0)  # (k,)
        bottleneck = float(max_cos_sq.min())
        mean_align = float(max_cos_sq.mean())

        results[layer] = {
            "alignment": bottleneck,
            "mean_alignment": mean_align,
            "per_direction": max_cos_sq.tolist(),
        }

    # Print summary table
    selected = find_best_layer(results)
    print(f"\n{'Layer':>5}  {'min cos²':>10}  {'mean cos²':>10}")
    print("-" * 30)
    for layer in sorted(results):
        r = results[layer]
        marker = " <-- best" if layer == selected else ""
        print(f"{layer:5d}  {r['alignment']:10.4f}  {r['mean_alignment']:10.4f}{marker}")

    return results


def find_best_layer(sweep_results: dict[int, dict], skip_last_n: int = 2) -> int:
    """Return the layer with the highest bottleneck SAE-concept alignment.

    The last ``skip_last_n`` layers are excluded because interventions
    there have little downstream computation to propagate through.

    Args:
        sweep_results: As returned by :func:`alignment_sweep`.
        skip_last_n: Number of final layers to exclude.

    Returns:
        Layer index with the highest alignment score among eligible layers.
    """
    max_layer = max(sweep_results)
    cutoff = max_layer - skip_last_n + 1
    eligible = {l: v for l, v in sweep_results.items() if l < cutoff}
    if not eligible:
        eligible = sweep_results
    return max(eligible, key=lambda l: eligible[l]["alignment"])
