"""Unified dataset loading for SST-2, Civil Comments, and TruthfulQA.

Supports balanced subsampling, binarization, and train/test splitting
for downstream probe training and steering evaluation.
"""

import logging
from typing import Any

import numpy as np
from datasets import load_dataset

from configs.base import DataConfig

logger = logging.getLogger(__name__)


def _load_sst2(config: DataConfig) -> dict[str, list]:
    """Load SST-2 sentiment dataset with optional balanced subsampling.

    Uses the validation split by default (since the test split has hidden
    labels).  Labels are already binary: 0 = negative, 1 = positive.
    When ``config.max_samples_per_class`` is set, subsamples each class
    to at most that many examples for a balanced dataset.
    """
    ds = load_dataset(config.hf_path, split=config.split)
    texts_all = ds[config.text_col]
    labels_all = ds[config.label_col]

    logger.info("Loaded SST-2 %s split: %d examples", config.split, len(texts_all))

    # Balanced subsample if dataset is larger than the budget
    class_indices: dict[int, list[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels_all):
        class_indices[label].append(idx)

    rng = np.random.default_rng(seed=42)
    sampled_indices: list[int] = []
    for cls in [0, 1]:
        indices = np.array(class_indices[cls])
        n_sample = min(len(indices), config.max_samples_per_class)
        chosen = rng.choice(indices, size=n_sample, replace=False)
        sampled_indices.extend(chosen.tolist())

    rng.shuffle(sampled_indices)
    texts = [texts_all[i] for i in sampled_indices]
    labels = [labels_all[i] for i in sampled_indices]

    logger.info(
        "SST-2 after balanced subsampling: %d examples (max %d per class)",
        len(texts),
        config.max_samples_per_class,
    )
    return {"texts": texts, "labels": labels}


def _load_civil_comments(config: DataConfig) -> dict[str, list]:
    """Load Civil Comments toxicity dataset with balanced subsampling.

    Binarizes the continuous toxicity score at `config.toxicity_threshold`
    and subsamples to at most `config.max_samples_per_class` per class
    to produce a balanced dataset.
    """
    ds = load_dataset(config.hf_path, split=config.split)
    texts_all: list[str] = ds[config.text_col]
    toxicity_scores: list[float] = ds[config.label_col]

    # Binarize: toxic (1) if score >= threshold, non-toxic (0) otherwise
    labels_all = [
        1 if score >= config.toxicity_threshold else 0
        for score in toxicity_scores
    ]

    # Separate indices by class for balanced subsampling
    class_indices: dict[int, list[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels_all):
        class_indices[label].append(idx)

    logger.info(
        "Civil Comments raw class distribution: class_0=%d, class_1=%d",
        len(class_indices[0]),
        len(class_indices[1]),
    )

    # Balanced subsample
    rng = np.random.default_rng(seed=42)
    sampled_indices: list[int] = []
    for cls in [0, 1]:
        indices = np.array(class_indices[cls])
        n_sample = min(len(indices), config.max_samples_per_class)
        chosen = rng.choice(indices, size=n_sample, replace=False)
        sampled_indices.extend(chosen.tolist())

    # Shuffle the combined indices
    rng.shuffle(sampled_indices)

    texts = [texts_all[i] for i in sampled_indices]
    labels = [labels_all[i] for i in sampled_indices]
    logger.info(
        "Civil Comments after balanced subsampling: %d examples "
        "(max %d per class)",
        len(texts),
        config.max_samples_per_class,
    )
    return {"texts": texts, "labels": labels}


def _load_truthfulqa(config: DataConfig) -> dict[str, list]:
    """Load TruthfulQA dataset as paired classification examples.

    For each question, constructs:
      - Positive pair (label=1): question + best_answer
      - Negative pair (label=0): question + first incorrect_answer

    Only includes questions that have both a best_answer and at least one
    incorrect_answer.
    """
    ds = load_dataset(config.hf_path, config.hf_config, split=config.split)

    texts: list[str] = []
    labels: list[int] = []

    for row in ds:
        question = row["question"]
        best_answer = row.get("best_answer", "")
        incorrect_answers = row.get("incorrect_answers", [])

        if not best_answer or not incorrect_answers:
            continue

        # Positive: question + best_answer
        texts.append(f"{question} {best_answer}")
        labels.append(1)

        # Negative: question + first incorrect_answer
        texts.append(f"{question} {incorrect_answers[0]}")
        labels.append(0)

    logger.info("Loaded TruthfulQA: %d paired examples", len(texts))
    return {"texts": texts, "labels": labels}


# Registry mapping dataset_name -> loader function
# Accepts both dataset identifiers (sst2, civil_comments, truthfulqa) and
# domain names (sentiment, toxicity, truthfulness) for convenience.
_DATASET_LOADERS: dict[str, Any] = {
    "sst2": _load_sst2,
    "sentiment": _load_sst2,
    "civil_comments": _load_civil_comments,
    "toxicity": _load_civil_comments,
    "truthfulqa": _load_truthfulqa,
    "truthfulness": _load_truthfulqa,
}


def _train_test_split(
    texts: list[str],
    labels: list[int],
    train_ratio: float,
    seed: int = 42,
) -> dict[str, list]:
    """Split texts and labels into stratified train/test sets.

    Stratifies by label to preserve class balance across splits.
    """
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(len(texts))

    # Group by label for stratified splitting
    label_arr = np.array(labels)
    unique_labels = np.unique(label_arr)

    train_indices: list[int] = []
    test_indices: list[int] = []

    for lbl in unique_labels:
        lbl_indices = indices[label_arr == lbl]
        rng.shuffle(lbl_indices)
        n_train = int(len(lbl_indices) * train_ratio)
        train_indices.extend(lbl_indices[:n_train].tolist())
        test_indices.extend(lbl_indices[n_train:].tolist())

    # Shuffle within each split
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return {
        "train_texts": [texts[i] for i in train_indices],
        "train_labels": [labels[i] for i in train_indices],
        "test_texts": [texts[i] for i in test_indices],
        "test_labels": [labels[i] for i in test_indices],
    }


def load_dataset_splits(config: DataConfig) -> dict[str, list]:
    """Load and split a dataset according to the provided configuration.

    Args:
        config: DataConfig specifying which dataset to load, preprocessing
            parameters, and train/test split ratio.

    Returns:
        Dictionary with keys:
            - "train_texts": list[str] — training texts
            - "train_labels": list[int] — training labels (0 or 1)
            - "test_texts": list[str] — test texts
            - "test_labels": list[int] — test labels (0 or 1)

    Raises:
        ValueError: If `config.dataset_name` is not a recognized dataset.
    """
    loader_fn = _DATASET_LOADERS.get(config.dataset_name)
    if loader_fn is None:
        supported = ", ".join(sorted(_DATASET_LOADERS.keys()))
        raise ValueError(
            f"Unknown dataset '{config.dataset_name}'. "
            f"Supported datasets: {supported}"
        )

    raw = loader_fn(config)
    splits = _train_test_split(
        texts=raw["texts"],
        labels=raw["labels"],
        train_ratio=config.probe_train_ratio,
    )

    logger.info(
        "Dataset '%s' split: train=%d, test=%d",
        config.dataset_name,
        len(splits["train_texts"]),
        len(splits["test_texts"]),
    )
    return splits
