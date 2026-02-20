"""Experiment runner: orchestrates the full steering evaluation pipeline."""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from configs.base import ExperimentConfig
from src.data.loader import load_dataset_splits
from src.data.preprocessing import extract_activations, extract_sae_features
from src.data.prompts import get_contrastive_pairs, get_persona_prefix, get_neutral_queries
from src.models.wrapper import ModelWrapper
from src.models.sae_utils import get_decoder_matrix
from src.probes.concept_subspace import ConceptSubspace
from src.steering import get_steering_method
from src.evaluation.metrics import compute_subspace_score, compute_kl_divergence, compute_l0
from src.evaluation.generation import steered_generation, steered_forward
from src.evaluation.text_classifier import get_classifier

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run a full steering experiment from config."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_wrapper = None
        self.concept_subspace = None
        self.data = None
        self.results = {}

    def setup(self):
        """Load model and SAE."""
        logger.info(f"Setting up experiment: {self.config.experiment_name}")
        self.model_wrapper = ModelWrapper(self.config.model)
        # Pre-load SAE for steering layer
        self.model_wrapper.get_sae(self.config.model.steering_layer)
        logger.info("Model and SAE loaded.")

    def load_data(self):
        """Load and split dataset."""
        logger.info(f"Loading dataset: {self.config.data.dataset_name}")
        self.data = load_dataset_splits(self.config.data)
        logger.info(f"Train: {len(self.data['train_texts'])}, Test: {len(self.data['test_texts'])}")

    def fit_subspace(self):
        """Extract activations and fit concept subspace."""
        layer = self.config.model.steering_layer
        logger.info(f"Extracting activations at layer {layer}...")

        train_acts = extract_activations(
            self.data["train_texts"], self.model_wrapper, layer,
            batch_size=self.config.model.batch_size,
        )
        test_acts = extract_activations(
            self.data["test_texts"], self.model_wrapper, layer,
            batch_size=self.config.model.batch_size,
        )

        self.train_activations = train_acts.numpy()
        self.test_activations = test_acts.numpy()
        self.train_labels = np.array(self.data["train_labels"])
        self.test_labels = np.array(self.data["test_labels"])

        logger.info("Fitting concept subspace...")
        self.concept_subspace = ConceptSubspace(
            n_components=self.config.steering.subspace_n_components,
            n_select=self.config.steering.subspace_n_select,
        )
        self.concept_subspace.fit(self.train_activations, self.train_labels)

        logger.info(
            f"Concept subspace fitted: k={self.concept_subspace.n_directions} directions, "
            f"class separations = {self.concept_subspace.class_separations}"
        )

        self.results["subspace_n_directions"] = self.concept_subspace.n_directions
        self.results["subspace_class_separations"] = self.concept_subspace.class_separations.tolist()

    def compute_steering(self, method_name: str = None):
        """Compute steering vector for the specified method."""
        if method_name is None:
            method_name = self.config.steering.method

        layer = self.config.model.steering_layer
        logger.info(f"Computing steering vector: {method_name}")

        # Instantiate method with appropriate kwargs
        method_kwargs = {}
        if method_name == "random_direction":
            method_kwargs["seed"] = self.config.seed
        elif method_name == "single_feature":
            method_kwargs["feature_idx"] = self.config.steering.manual_feature_idx or 0
        elif method_name == "topk_features":
            method_kwargs["topk"] = self.config.steering.topk
        elif method_name == "convex_optimal":
            method_kwargs.update({
                "epsilon": self.config.steering.epsilon,
                "solver": self.config.steering.solver,
                "max_iters": self.config.steering.solver_max_iters,
                "prefilter_topk": self.config.steering.prefilter_topk,
            })
        elif method_name == "qp_optimal":
            method_kwargs.update({
                "lam": self.config.steering.lam,
                "solver": self.config.steering.solver,
                "max_iters": self.config.steering.solver_max_iters,
                "prefilter_topk": self.config.steering.prefilter_topk,
            })

        method = get_steering_method(method_name, **method_kwargs)

        # Compute the steering vector based on method type
        if method_name == "no_steering":
            method.compute_steering_vector(d_model=self.config.model.d_model)

        elif method_name == "random_direction":
            method.compute_steering_vector(d_model=self.config.model.d_model)

        elif method_name == "caa_mean_diff":
            # Split activations by class
            pos_mask = self.train_labels == self.config.steering.target_class
            neg_mask = ~pos_mask
            acts_pos = torch.tensor(self.train_activations[pos_mask])
            acts_neg = torch.tensor(self.train_activations[neg_mask])
            method.compute_steering_vector(activations_pos=acts_pos, activations_neg=acts_neg)

        elif method_name == "caa_contrastive":
            pairs = get_contrastive_pairs(self.config.data.dataset_name)
            method.compute_steering_vector(
                model_wrapper=self.model_wrapper, layer=layer,
                contrastive_pairs=pairs, batch_size=self.config.model.batch_size,
            )

        elif method_name == "caa_repe":
            queries = get_neutral_queries(self.config.data.dataset_name)
            pos_prefix = get_persona_prefix(self.config.data.dataset_name, 1)
            neg_prefix = get_persona_prefix(self.config.data.dataset_name, 0)
            method.compute_steering_vector(
                model_wrapper=self.model_wrapper, layer=layer,
                neutral_queries=queries, positive_prefix=pos_prefix,
                negative_prefix=neg_prefix, batch_size=self.config.model.batch_size,
            )

        elif method_name == "single_feature":
            method.compute_steering_vector(
                model_wrapper=self.model_wrapper, layer=layer,
            )

        elif method_name == "topk_features":
            sae_features = extract_sae_features(
                self.data["train_texts"], self.model_wrapper, layer,
                batch_size=self.config.model.batch_size,
            )
            method.compute_steering_vector(
                sae_features=sae_features, labels=self.train_labels,
                model_wrapper=self.model_wrapper, layer=layer,
            )

        elif method_name == "convex_optimal":
            sae_features = extract_sae_features(
                self.data["test_texts"][:1], self.model_wrapper, layer,
                batch_size=1,
            )
            D = get_decoder_matrix(self.model_wrapper.get_sae(layer))
            h = torch.tensor(self.test_activations[0])
            method.compute_steering_vector(
                h=h, D=D, concept_subspace=self.concept_subspace,
                sae_features=sae_features[0],
                target_class=self.config.steering.target_class,
            )

        elif method_name == "qp_optimal":
            sae_features = extract_sae_features(
                self.data["test_texts"][:1], self.model_wrapper, layer,
                batch_size=1,
            )
            D = get_decoder_matrix(self.model_wrapper.get_sae(layer))
            h = torch.tensor(self.test_activations[0])
            method.compute_steering_vector(
                h=h, D=D, concept_subspace=self.concept_subspace,
                sae_features=sae_features[0],
                target_class=self.config.steering.target_class,
            )

        return method

    def evaluate(self, method):
        """Evaluate a steering method on test data."""
        layer = self.config.model.steering_layer
        alpha = self.config.steering.alpha
        method_name = method.name

        logger.info(f"Evaluating: {method_name}")
        eval_results = {"method": method_name}

        # 1. Subspace score on steered activations
        sv = method.steering_vector.numpy()
        steered_acts = self.test_activations + alpha * sv[np.newaxis, :]
        base_score = compute_subspace_score(
            self.concept_subspace, self.test_activations
        )
        steered_score = compute_subspace_score(
            self.concept_subspace, steered_acts
        )
        eval_results["subspace_score_base"] = base_score
        eval_results["subspace_score_steered"] = steered_score
        eval_results["subspace_score_delta"] = steered_score - base_score

        # 2. L0 sparsity (for SAE-based methods)
        if hasattr(method, "_delta") and method._delta is not None:
            eval_results["l0"] = compute_l0(method._delta)
            eval_results["l1"] = float(method._delta.sum())
        else:
            eval_results["l0"] = None
            eval_results["l1"] = None

        # 3. Steering vector norm
        eval_results["steering_norm"] = float(method.steering_vector.norm().item())

        # 4. Generate texts and evaluate with classifier
        try:
            queries = get_neutral_queries(self.config.data.dataset_name)[:self.config.eval.n_generations]
            generated = steered_generation(
                self.model_wrapper, queries, method, layer, alpha=alpha,
                max_new_tokens=self.config.eval.max_new_tokens,
                temperature=self.config.eval.temperature,
                top_p=self.config.eval.top_p,
            )
            eval_results["sample_generations"] = generated[:5]

            # Classify generated text
            try:
                classifier = get_classifier(self.config.data.dataset_name)
                cls_results = classifier.classify(generated)
                eval_results["classifier_results"] = cls_results
            except Exception as e:
                logger.warning(f"Classifier evaluation failed: {e}")
        except Exception as e:
            logger.warning(f"Generation failed: {e}")

        # 5. KL divergence (compute on a few examples)
        try:
            kl_values = []
            test_texts_sample = self.data["test_texts"][:10]
            tokenizer = self.model_wrapper.tokenizer
            for text in test_texts_sample:
                tokens = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=self.config.data.max_seq_len)
                input_ids = tokens["input_ids"].to(self.config.model.device)

                logits_steered, _ = steered_forward(
                    self.model_wrapper, input_ids, method, layer, alpha=alpha,
                )
                with torch.no_grad():
                    logits_base = self.model_wrapper.model(input_ids)

                kl = compute_kl_divergence(logits_steered, logits_base)
                kl_values.append(kl)

            eval_results["kl_divergence_mean"] = float(np.mean(kl_values))
            eval_results["kl_divergence_std"] = float(np.std(kl_values))
        except Exception as e:
            logger.warning(f"KL divergence computation failed: {e}")

        logger.info(f"Results for {method_name}: subspace_delta={eval_results.get('subspace_score_delta', 'N/A'):.4f}")
        return eval_results

    def save_results(self, eval_results: dict):
        """Save results to JSON."""
        results_dir = Path(self.config.eval.results_dir) / "metrics"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.config.experiment_name}_{eval_results['method']}.json"
        filepath = results_dir / filename

        output = {
            "experiment": self.config.experiment_name,
            "model": self.config.model.name,
            "dataset": self.config.data.dataset_name,
            **self.results,
            **eval_results,
            "config": {
                "steering_layer": self.config.model.steering_layer,
                "alpha": self.config.steering.alpha,
                "epsilon": self.config.steering.epsilon,
                "target_class": self.config.steering.target_class,
            },
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def run(self, method_name: str = None):
        """Run the full experiment pipeline."""
        start_time = time.time()

        self.setup()
        self.load_data()
        self.fit_subspace()
        method = self.compute_steering(method_name)
        eval_results = self.evaluate(method)
        filepath = self.save_results(eval_results)

        elapsed = time.time() - start_time
        logger.info(f"Experiment complete in {elapsed:.1f}s. Results: {filepath}")
        return eval_results
