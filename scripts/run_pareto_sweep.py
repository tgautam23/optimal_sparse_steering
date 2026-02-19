#!/usr/bin/env python3
"""CLI: epsilon sweep for Pareto frontier of sparsity vs. steering success."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base import load_config
from src.experiment.sweep import epsilon_sweep, method_comparison


def main():
    parser = argparse.ArgumentParser(description="Pareto sweep over epsilon values")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--epsilons", type=float, nargs="+", default=None,
                        help="Epsilon values to sweep (default: 1 3 5 10 20)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Also run method_comparison across all methods")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Epsilon sweep
    print("=== Epsilon Sweep ===")
    sweep_results = epsilon_sweep(config, epsilons=args.epsilons)

    print("\nEpsilon | Probe Delta | L0    | KL Div")
    print("-" * 50)
    for r in sweep_results:
        eps = r.get("epsilon", "?")
        delta = r.get("probe_score_delta", float("nan"))
        l0 = r.get("l0", "N/A")
        kl = r.get("kl_divergence_mean", float("nan"))
        print(f"{eps:7.1f} | {delta:11.4f} | {str(l0):5s} | {kl:.4f}")

    # Optional full comparison
    if args.compare_all:
        print("\n=== Method Comparison ===")
        comp_results = method_comparison(config)
        print(f"\nMethod{'':20s} | Probe Delta | L0    | KL Div")
        print("-" * 65)
        for r in comp_results:
            name = r.get("method", "?")
            delta = r.get("probe_score_delta", float("nan"))
            l0 = r.get("l0", "N/A")
            kl = r.get("kl_divergence_mean", float("nan"))
            print(f"{name:26s} | {delta:11.4f} | {str(l0):5s} | {kl:.4f}")


if __name__ == "__main__":
    main()
