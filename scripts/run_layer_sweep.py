#!/usr/bin/env python3
"""CLI: concept concentration sweep across all layers."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base import load_config
from src.data.loader import load_dataset_splits
from src.models.wrapper import ModelWrapper
from src.probes.layer_sweep import concept_concentration_sweep, find_best_layer


def main():
    parser = argparse.ArgumentParser(description="Sweep concept concentration across layers")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--n-components", type=int, default=None, help="PCA components (default: from config)")
    parser.add_argument("--n-select", type=int, default=None, help="Top-k directions for R^2_k (default: from config)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.batch_size is not None:
        config.model.batch_size = args.batch_size

    n_components = args.n_components or config.steering.subspace_n_components
    n_select = args.n_select or config.steering.subspace_n_select

    # Load model
    model_wrapper = ModelWrapper(config.model)

    # Load data
    data = load_dataset_splits(config.data)
    all_texts = data["train_texts"] + data["test_texts"]
    all_labels = data["train_labels"] + data["test_labels"]

    # Run sweep
    results = concept_concentration_sweep(
        texts=all_texts,
        labels=all_labels,
        model_wrapper=model_wrapper,
        n_layers=config.model.n_layers,
        n_components=n_components,
        n_select=n_select,
        batch_size=config.model.batch_size,
    )

    best = find_best_layer(results)
    print(f"\nBest layer: {best} (R^2_k = {results[best]['r2_k']:.4f})")

    # Save results
    results_dir = Path(config.eval.results_dir) / "layer_sweeps"
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / f"{config.experiment_name}_layer_sweep.json"
    with open(filepath, "w") as f:
        json.dump(
            {
                "results": {str(k): v for k, v in results.items()},
                "best_layer": best,
                "n_components": n_components,
                "n_select": n_select,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    main()
