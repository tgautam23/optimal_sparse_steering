"""Dataclass-based configuration system with YAML override support."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "gpt2-small"
    # TransformerLens model name
    tl_name: str = "gpt2-small"
    # SAE release name (sae-lens)
    sae_release: str = "gpt2-small-res-jb"
    # SAE id template — {layer} is replaced at runtime
    sae_id_template: str = "blocks.{layer}.hook_resid_pre"
    # Hook point template for activation extraction
    hook_template: str = "blocks.{layer}.hook_resid_pre"
    d_model: int = 768
    n_layers: int = 12
    # Layer to use for steering (set by probe sweep or manually)
    steering_layer: int = 7
    dtype: str = "float32"
    device: str = "cuda"
    batch_size: int = 8


@dataclass
class DataConfig:
    dataset_name: str = "sst2"
    # HuggingFace dataset path
    hf_path: str = "stanfordnlp/sst2"
    hf_config: Optional[str] = None
    split: str = "validation"
    text_col: str = "sentence"
    label_col: str = "label"
    # For Civil Comments: binarization threshold
    toxicity_threshold: float = 0.5
    # Max samples per class for balanced subsampling
    max_samples_per_class: int = 500
    # Train/test split ratio for probe training
    probe_train_ratio: float = 0.8
    max_seq_len: int = 128


@dataclass
class SteeringConfig:
    method: str = "convex_optimal"
    # Steering strength multiplier (alpha)
    alpha: float = 5.0
    # SOCP coherence budget (epsilon)
    epsilon: float = 5.0
    # SOCP probe margin (tau')
    tau: float = 0.5
    # Target class for steering (1 = positive sentiment, toxic, truthful)
    target_class: int = 1
    # Top-k for correlation-based method
    topk: int = 10
    # Manual SAE feature index for single_feature method
    manual_feature_idx: Optional[int] = None
    # Pre-filter threshold: only include features with mean activation > this
    prefilter_threshold: float = 0.01
    # QP coherence penalty weight (lambda)
    lam: float = 1.0
    # Pre-filter top-k features for QP formulation
    prefilter_topk: int = 2000
    # CVXPY solver
    solver: str = "SCS"
    solver_max_iters: int = 10000


@dataclass
class EvalConfig:
    # Number of texts to generate per method
    n_generations: int = 50
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    # Off-the-shelf classifier for downstream evaluation
    classifier_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    results_dir: str = "results"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    experiment_name: str = "gpt2-small_sst2"


def _update_dataclass(dc, overrides: dict):
    """Recursively update a dataclass from a dict of overrides."""
    for k, v in overrides.items():
        if hasattr(dc, k):
            current = getattr(dc, k)
            if hasattr(current, "__dataclass_fields__") and isinstance(v, dict):
                _update_dataclass(current, v)
            else:
                setattr(dc, k, v)
    return dc


def load_config(yaml_path: Optional[str] = None) -> ExperimentConfig:
    """Load an ExperimentConfig, optionally applying YAML overrides."""
    config = ExperimentConfig()
    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists():
            with open(path) as f:
                overrides = yaml.safe_load(f) or {}
            _update_dataclass(config, overrides)
    return config


def save_config(config: ExperimentConfig, path: str):
    """Save an ExperimentConfig to YAML."""
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
