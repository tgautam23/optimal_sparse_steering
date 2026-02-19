#!/usr/bin/env python3
"""CLI: run a single steering experiment from config."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base import load_config
from src.experiment.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run a steering experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--method", type=str, default=None, help="Override steering method name")
    parser.add_argument("--layer", type=int, default=None, help="Override steering layer")
    parser.add_argument("--alpha", type=float, default=None, help="Override steering strength")
    parser.add_argument("--epsilon", type=float, default=None, help="Override coherence budget")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Apply CLI overrides
    if args.method is not None:
        config.steering.method = args.method
    if args.layer is not None:
        config.model.steering_layer = args.layer
    if args.alpha is not None:
        config.steering.alpha = args.alpha
    if args.epsilon is not None:
        config.steering.epsilon = args.epsilon
    if args.seed is not None:
        config.seed = args.seed

    runner = ExperimentRunner(config)
    results = runner.run(method_name=args.method)

    print("\n=== Results ===")
    for k, v in results.items():
        if k != "sample_generations" and k != "classifier_results":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
