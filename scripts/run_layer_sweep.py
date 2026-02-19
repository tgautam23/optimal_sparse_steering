#!/usr/bin/env python3
"""CLI: probe accuracy sweep across all layers."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base import load_config
from src.data.loader import load_dataset_splits
from src.models.wrapper import ModelWrapper
from src.probes.layer_sweep import layer_sweep, find_best_layer


def main():
    parser = argparse.ArgumentParser(description="Sweep probe accuracy across layers")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.batch_size is not None:
        config.model.batch_size = args.batch_size

    # Load model
    model_wrapper = ModelWrapper(config.model)

    # Load data
    data = load_dataset_splits(config.data)
    all_texts = data["train_texts"] + data["test_texts"]
    all_labels = data["train_labels"] + data["test_labels"]

    # Run sweep
    results = layer_sweep(
        texts=all_texts,
        labels=all_labels,
        model_wrapper=model_wrapper,
        n_layers=config.model.n_layers,
        batch_size=config.model.batch_size,
        cv_folds=args.cv_folds,
    )

    best = find_best_layer(results)
    print(f"\nBest layer: {best} (accuracy: {results[best]:.4f})")

    # Save results
    results_dir = Path(config.eval.results_dir) / "layer_sweeps"
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / f"{config.experiment_name}_layer_sweep.json"
    with open(filepath, "w") as f:
        json.dump({"results": {str(k): v for k, v in results.items()}, "best_layer": best}, f, indent=2)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    main()
