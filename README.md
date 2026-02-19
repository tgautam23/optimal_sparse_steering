# Optimal Sparse Steering via Convex Optimization

Finding the **sparsest** SAE-feature intervention that steers language model behavior, formulated as a convex program over Sparse Autoencoder (SAE) feature space.

## Core Idea

Instead of heuristically selecting SAE features for steering (e.g., by correlation), we solve for the **minimum-L1 feature perturbation** that shifts a linear probe's classification past a target margin, while penalising the L2 norm of the perturbation in residual stream space.

**QP Formulation (primary):**

$$\min_{\delta \geq 0} \; \mathbf{1}^\top \delta \;+\; \frac{\lambda}{2} \| D^\top \delta \|_2^2 \qquad \text{s.t.} \quad (Dw)^\top \delta \;\geq\; \tau - w^\top h - b$$

**SOCP Formulation (alternative):**

$$\min_{\delta \geq 0} \; \mathbf{1}^\top \delta \qquad \text{s.t.} \quad (Dw)^\top \delta \;\geq\; \tau - w^\top h - b, \quad \| D^\top \delta \|_2 \leq \epsilon$$

Where:
- $\delta \in \mathbb{R}^{d_\text{sae}}$ &mdash; sparse feature activation perturbation
- $D \in \mathbb{R}^{d_\text{sae} \times d_\text{model}}$ &mdash; SAE decoder matrix
- $w, b$ &mdash; linear probe weight vector and bias (defines the concept direction)
- $h$ &mdash; input's residual stream activation at the steering layer
- $\lambda$ &mdash; coherence penalty (sparsity vs. distribution shift tradeoff)

The QP relaxation moves the hard L2 constraint into the objective as a penalty, enabling QP solvers (OSQP) with warm-starting for efficient batch solving.

## Methods Compared

| Category | Method | Description |
|----------|--------|-------------|
| Control | `no_steering` | Zero vector baseline |
| Control | `random_direction` | Random unit vector |
| CAA | `caa_mean_diff` | Mean difference of class-conditional activations |
| CAA | `caa_contrastive` | Mean difference from paired contrastive prompts |
| CAA | `caa_repe` | Representation Engineering with persona prefixes |
| SAE | `single_feature` | Single most-correlated SAE decoder direction |
| SAE | `topk_features` | Correlation-weighted sum of top-k SAE features |
| **Ours** | `convex_optimal` | **SOCP with hard L2 coherence constraint** |
| **Ours** | `qp_optimal` | **QP with L2 coherence penalty (warm-startable)** |

## Evaluation Metrics

- **Probe score shift** &mdash; does the intervention flip the linear classifier?
- **L0 sparsity** &mdash; how many SAE features are touched?
- **KL divergence** &mdash; how much does the output distribution change?
- **Downstream classifier** &mdash; sentiment/toxicity classifier on generated text

## Models and Tasks

| Model | d_model | SAE | d_sae |
|-------|---------|-----|-------|
| GPT-2 Small | 768 | `gpt2-small-res-jb` (JumpReLU) | ~24k |
| Gemma-2-2B (pretrained) | 2,304 | `gemma-scope-2b-pt-res-canonical` (width 16k) | 16,384 |
| Gemma-2-2B (instruction-tuned) | 2,304 | Same SAEs (transfer experiment) | 16,384 |

| Task | Dataset | Labels |
|------|---------|--------|
| Sentiment | SST-2 | positive / negative |
| Toxicity | Civil Comments | toxic / non-toxic |
| Truthfulness | TruthfulQA | truthful / untruthful |

## Project Structure

```
optimal_sparse_steering/
├── configs/
│   ├── base.py                  # Dataclass configs with YAML override support
│   └── experiments/             # Per-experiment YAML configs
│       ├── gpt2_sst2.yaml
│       ├── gpt2_toxicity.yaml
│       ├── gemma_pt_sst2.yaml
│       └── ...
├── src/
│   ├── data/
│   │   ├── loader.py            # Unified dataset loading (SST-2, Civil Comments, TruthfulQA)
│   │   ├── preprocessing.py     # Activation and SAE feature extraction
│   │   └── prompts.py           # Contrastive pairs, persona prefixes, neutral queries
│   ├── models/
│   │   ├── wrapper.py           # ModelWrapper around HookedSAETransformer
│   │   └── sae_utils.py         # SAE encode/decode/decoder matrix helpers
│   ├── probes/
│   │   ├── linear_probe.py      # Logistic regression probe (feeds into SOCP/QP)
│   │   └── layer_sweep.py       # Cross-validated probe accuracy per layer
│   ├── steering/
│   │   ├── base.py              # Abstract SteeringMethod base class
│   │   ├── registry.py          # Method name -> class registry
│   │   ├── no_steering.py       # Zero vector control
│   │   ├── random_direction.py  # Random unit vector control
│   │   ├── caa.py               # CAAMeanDiff, CAAContrastive, CAARepE
│   │   ├── single_feature.py    # Single SAE feature steering
│   │   ├── topk_features.py     # Top-k correlated features
│   │   ├── convex_optimal.py    # SOCP formulation (hard L2 constraint)
│   │   └── qp_optimal.py        # QP formulation (L2 penalty, warm-start)
│   ├── evaluation/
│   │   ├── metrics.py           # Probe score, KL divergence, L0, perplexity
│   │   ├── generation.py        # Steered text generation with hooks
│   │   └── text_classifier.py   # Off-the-shelf classifiers for evaluation
│   └── experiment/
│       ├── runner.py            # Full experiment pipeline orchestrator
│       └── sweep.py             # Epsilon sweep, method comparison utilities
├── scripts/
│   ├── run_experiment.py        # CLI: single experiment
│   ├── run_layer_sweep.py       # CLI: probe accuracy across layers
│   └── run_pareto_sweep.py      # CLI: Pareto frontier exploration
├── notebooks/
│   ├── 01_caa_baselines.ipynb           # CAA steering methods (GPT-2 → Gemma)
│   ├── 02_sae_feature_steering.ipynb    # SAE feature selection (GPT-2 → Gemma)
│   ├── 03_optimal_sparse_steering.ipynb # SOCP + QP formulations, Pareto sweeps
│   └── 04_full_comparison.ipynb         # All methods head-to-head
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

### Local (conda)

```bash
# Create and activate environment
conda create -n sparse-steering python=3.11 -y
conda activate sparse-steering

# Install PyTorch (adjust for your CUDA version, or use cpu)
# GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install project and dependencies
cd optimal_sparse_steering
pip install -e .

# Additional notebook dependencies
pip install pandas jupyter
```

### Google Colab

The notebooks are designed to run on Colab's free T4 GPU tier. Each notebook includes a setup cell that handles path configuration automatically.

| Notebook | Colab |
|----------|-------|
| CAA Baselines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgautam23/optimal_sparse_steering/blob/main/notebooks/01_caa_baselines.ipynb) |
| SAE Feature Steering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgautam23/optimal_sparse_steering/blob/main/notebooks/02_sae_feature_steering.ipynb) |
| Optimal Sparse Steering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgautam23/optimal_sparse_steering/blob/main/notebooks/03_optimal_sparse_steering.ipynb) |
| Full Comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tgautam23/optimal_sparse_steering/blob/main/notebooks/04_full_comparison.ipynb) |

## Quick Start

### Run a single experiment (CLI)

```bash
python scripts/run_experiment.py --config configs/experiments/gpt2_sst2.yaml --method caa_mean_diff
```

### Run all methods for comparison

```bash
python scripts/run_pareto_sweep.py --config configs/experiments/gpt2_sst2.yaml --compare-all
```

### Use the QP steering programmatically

```python
from src.models.wrapper import ModelWrapper
from src.models.sae_utils import get_decoder_matrix
from src.probes.linear_probe import LinearProbe
from src.steering.qp_optimal import QPOptimalSteering

# After loading model, training probe, and extracting activations:
D = get_decoder_matrix(model_wrapper.get_sae(layer))

qp = QPOptimalSteering(lam=1.0, tau=0.5)
steering_vec = qp.compute_steering_vector(
    h=activation,           # residual stream activation for this input
    probe_w=probe.weight_vector,
    probe_b=probe.bias,
    D=D,
    sae_features=sae_feats, # for pre-filtering
    target_class=1,
)

# steering_vec is shape (d_model,) — add to activations at the steering layer
```

## References

- Turner et al. (2023), "Activation Addition: Steering Language Models Without Optimization"
- Zou et al. (2023), "Representation Engineering: A Top-Down Approach to AI Transparency"
- Templeton et al. (2024), "Scaling Monosemanticity" (Anthropic)
- Bloom (2024), SAE Lens / JumpReLU SAEs
- Lieberum et al. (2024), "Gemma Scope" (Google DeepMind)

## License

MIT
