"""Sweep utilities for hyperparameter exploration."""

import json
import logging
from pathlib import Path

import numpy as np

from configs.base import ExperimentConfig
from .runner import ExperimentRunner

logger = logging.getLogger(__name__)


def epsilon_sweep(
    config: ExperimentConfig,
    epsilons: list[float] = None,
    method_name: str = "convex_optimal",
) -> list[dict]:
    """Run steering experiments across multiple epsilon values.

    Args:
        config: Base experiment config.
        epsilons: List of epsilon values to sweep. Default: [1, 3, 5, 10, 20].
        method_name: Steering method to sweep (must support epsilon).

    Returns:
        List of result dicts, one per epsilon value.
    """
    if epsilons is None:
        epsilons = [1.0, 3.0, 5.0, 10.0, 20.0]

    runner = ExperimentRunner(config)
    runner.setup()
    runner.load_data()
    runner.train_probe()

    all_results = []
    for eps in epsilons:
        logger.info(f"Sweep: epsilon = {eps}")
        config.steering.epsilon = eps
        method = runner.compute_steering(method_name)
        eval_results = runner.evaluate(method)
        eval_results["epsilon"] = eps
        all_results.append(eval_results)

    # Save sweep results
    results_dir = Path(config.eval.results_dir) / "sweeps"
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / f"{config.experiment_name}_epsilon_sweep.json"
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Sweep results saved to {filepath}")

    return all_results


def method_comparison(
    config: ExperimentConfig,
    methods: list[str] = None,
) -> list[dict]:
    """Run all steering methods on the same setup for comparison.

    Args:
        config: Base experiment config.
        methods: List of method names. Default: all methods.

    Returns:
        List of result dicts, one per method.
    """
    if methods is None:
        methods = [
            "no_steering", "random_direction",
            "caa_mean_diff", "caa_contrastive", "caa_repe",
            "single_feature", "topk_features", "convex_optimal",
        ]

    runner = ExperimentRunner(config)
    runner.setup()
    runner.load_data()
    runner.train_probe()

    all_results = []
    for method_name in methods:
        logger.info(f"Comparison: method = {method_name}")
        try:
            method = runner.compute_steering(method_name)
            eval_results = runner.evaluate(method)
            all_results.append(eval_results)
            runner.save_results(eval_results)
        except Exception as e:
            logger.error(f"Method {method_name} failed: {e}")
            all_results.append({"method": method_name, "error": str(e)})

    # Save comparison summary
    results_dir = Path(config.eval.results_dir) / "comparisons"
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / f"{config.experiment_name}_comparison.json"
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Comparison results saved to {filepath}")

    return all_results
