"""QP-based Optimal Sparse Steering.

Reformulates the SOCP coherence constraint as a quadratic penalty in
the objective, yielding a standard Quadratic Program:

    minimize    1^T delta  +  (lambda/2) * ||D^T delta||_2^2
    s.t.        w_j^T D^T delta  >=  gamma_j(h),   j = 1, ..., k
                delta >= 0

Where:
    delta in R^{d_sae}  -- sparse feature activation perturbation
    D in R^{d_sae x d_model}  -- SAE decoder matrix
    W = {w_1, ..., w_k}  -- concept subspace directions
    h in R^{d_model}  -- current residual stream activation
    gamma_j(h)  -- per-direction RHS (gap to class-1 mean projection)
    lambda  -- coherence penalty (trades off sparsity vs residual-space L2)

Advantages over the SOCP formulation (convex_optimal.py):
    - QP solvers (OSQP) natively support warm-starting across sequential solves
    - Single hyperparameter lambda replaces the geometric epsilon budget
    - Faster convergence for moderate problem sizes
    - The Hessian is implicitly D D^T -- never formed explicitly by SCS
"""

import logging
import time
from typing import Optional

import cvxpy as cp
import numpy as np
import torch

from .base import SteeringMethod

logger = logging.getLogger(__name__)


class QPOptimalSteering(SteeringMethod):
    """QP-based optimal sparse steering in SAE feature space.

    Finds the minimum-L1 (sparsest) feature perturbation that shifts the
    concept subspace projections past target margins, penalised by the
    squared L2 norm of the resulting residual-stream perturbation.
    """

    def __init__(
        self,
        lam: float = 1.0,
        solver: str = "SCS",
        max_iters: int = 10000,
        prefilter_topk: int = 2000,
        warm_start: bool = True,
    ):
        super().__init__("qp_optimal")
        self.lam = lam
        self.solver = solver
        self.max_iters = max_iters
        self.prefilter_topk = prefilter_topk
        self.warm_start = warm_start

        # Results from most recent solve
        self._delta: Optional[np.ndarray] = None
        self._solve_status: Optional[str] = None
        self._active_features: Optional[np.ndarray] = None
        self._feature_mask: Optional[np.ndarray] = None
        self._solve_time: Optional[float] = None

        # Warm-start state
        self._prev_delta: Optional[np.ndarray] = None

    def compute_steering_vector(
        self,
        h: torch.Tensor,
        D: torch.Tensor,
        concept_subspace,
        sae_features: torch.Tensor = None,
        target_class: int = 1,
        lam: float = None,
        **kwargs,
    ) -> torch.Tensor:
        """Solve the QP for a single input to find optimal sparse steering.

        Args:
            h: Current residual stream activation, shape (d_model,).
            D: SAE decoder matrix, shape (d_sae, d_model).
            concept_subspace: ConceptSubspace instance providing k concept
                directions and constraint RHS computation.
            sae_features: SAE feature activations for pre-filtering, shape (d_sae,).
            target_class: Which class to steer toward (0 or 1).
            lam: Override coherence penalty weight.

        Returns:
            Steering vector in residual stream space, shape (d_model,).
        """
        if lam is not None:
            self.lam = lam

        # --- Convert inputs to numpy float64 for CVXPY ---
        h_np = (
            h.detach().cpu().numpy().astype(np.float64)
            if isinstance(h, torch.Tensor)
            else np.asarray(h, dtype=np.float64)
        )
        D_np = (
            D.detach().cpu().numpy().astype(np.float64)
            if isinstance(D, torch.Tensor)
            else np.asarray(D, dtype=np.float64)
        )

        d_sae, d_model = D_np.shape

        # --- Multi-direction pre-filtering ---
        subspace_relevance = concept_subspace.compute_prefilter_relevance(D_np)

        n_keep = min(self.prefilter_topk, d_sae)
        n_half = n_keep // 2

        if sae_features is not None:
            feat_np = (
                sae_features.detach().cpu().numpy()
                if isinstance(sae_features, torch.Tensor)
                else np.asarray(sae_features)
            )
            top_by_act = set(np.argsort(np.abs(feat_np))[::-1][:n_half].tolist())
            top_by_rel = set(np.argsort(subspace_relevance)[::-1][:n_half].tolist())
            selected = sorted(top_by_act | top_by_rel)
            mask = np.zeros(d_sae, dtype=bool)
            mask[selected] = True
            self._feature_mask = mask
        else:
            mask = np.ones(d_sae, dtype=bool)
            self._feature_mask = mask

        active_idx = np.where(mask)[0]
        n_active = len(active_idx)
        logger.info(f"QPOptimal: {n_active}/{d_sae} active features after pre-filtering")

        D_sub = D_np[active_idx]  # (n_active, d_model)

        # Build k constraints: w_j^T D_sub^T delta >= rhs_j
        rhs_vec = concept_subspace.compute_rhs(h_np)  # (k,)
        DW = D_sub @ concept_subspace.directions.T  # (n_active, k)

        logger.info(
            f"QPOptimal: k={concept_subspace.n_directions} constraints, "
            f"rhs range = [{rhs_vec.min():.4f}, {rhs_vec.max():.4f}]"
        )

        # --- Define CVXPY QP ---
        delta = cp.Variable(n_active, nonneg=True)

        sparsity_term = cp.sum(delta)
        coherence_term = (self.lam / 2) * cp.sum_squares(D_sub.T @ delta)
        objective = cp.Minimize(sparsity_term + coherence_term)

        constraints = [DW[:, j] @ delta >= rhs_vec[j]
                       for j in range(concept_subspace.n_directions)]

        problem = cp.Problem(objective, constraints)

        # --- Warm-start from previous solve ---
        if self.warm_start and self._prev_delta is not None:
            try:
                prev_mapped = self._prev_delta[active_idx]
                delta.value = np.maximum(prev_mapped, 0)
            except (IndexError, ValueError):
                pass

        # --- Solve ---
        t0 = time.time()
        solver_kwargs = {"verbose": False}

        if self.solver == "OSQP":
            solver_kwargs.update({
                "max_iter": self.max_iters,
                "warm_start": self.warm_start and (delta.value is not None),
                "eps_abs": 1e-5,
                "eps_rel": 1e-5,
            })
        elif self.solver == "SCS":
            solver_kwargs["max_iters"] = self.max_iters

        try:
            problem.solve(solver=self.solver, **solver_kwargs)
        except cp.SolverError:
            fallback = "SCS" if self.solver != "SCS" else "ECOS"
            logger.warning(f"QPOptimal: {self.solver} failed, trying {fallback}")
            try:
                fb_kwargs = {"verbose": False}
                if fallback == "SCS":
                    fb_kwargs["max_iters"] = self.max_iters
                problem.solve(solver=fallback, **fb_kwargs)
            except cp.SolverError:
                logger.error("QPOptimal: all solvers failed")
                self._solve_status = "failed"
                self._solve_time = time.time() - t0
                self._steering_vector = torch.zeros(d_model)
                return self._steering_vector

        self._solve_time = time.time() - t0
        self._solve_status = problem.status
        logger.info(
            f"QPOptimal: status = {problem.status}, "
            f"value = {problem.value:.6f}, "
            f"solve_time = {self._solve_time:.3f}s"
        )

        if problem.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(f"QPOptimal: non-optimal status: {problem.status}")
            self._steering_vector = torch.zeros(d_model)
            self._delta = np.zeros(d_sae)
            return self._steering_vector

        # --- Extract solution ---
        delta_val = np.maximum(delta.value, 0)  # clip numerical noise

        # Map back to full feature space
        full_delta = np.zeros(d_sae)
        full_delta[active_idx] = delta_val
        self._delta = full_delta
        self._prev_delta = full_delta  # store for warm-starting next solve
        self._active_features = active_idx[delta_val > 1e-6]

        l0 = int((full_delta > 1e-6).sum())
        l1 = float(full_delta.sum())
        coherence_l2 = float(np.linalg.norm(D_np.T @ full_delta))
        logger.info(
            f"QPOptimal: L0 = {l0}, L1 = {l1:.4f}, "
            f"||D^T delta||_2 = {coherence_l2:.4f}"
        )

        # Compute steering vector in residual space: D^T @ delta
        steering_np = D_np.T @ full_delta  # (d_model,)
        self._steering_vector = torch.tensor(steering_np, dtype=torch.float32)
        return self._steering_vector

    def compute_batch_steering(
        self,
        activations: torch.Tensor,
        D: torch.Tensor,
        concept_subspace,
        sae_features_batch: torch.Tensor = None,
        target_class: int = 1,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Solve QP for a batch of inputs with warm-starting between solves.

        Args:
            activations: Batch of activations, shape (batch, d_model).
            D: SAE decoder matrix, shape (d_sae, d_model).
            concept_subspace: ConceptSubspace for multi-constraint mode.
            sae_features_batch: SAE features, shape (batch, d_sae). Optional.
            target_class: Target class for steering.

        Returns:
            List of steering vectors, one per input.
        """
        batch_size = activations.shape[0]
        results = []

        # Reset warm-start state at the beginning of each batch
        self._prev_delta = None
        total_time = 0.0

        for i in range(batch_size):
            h_i = activations[i]
            sae_feat_i = (
                sae_features_batch[i] if sae_features_batch is not None else None
            )
            vec = self.compute_steering_vector(
                h=h_i,
                D=D,
                concept_subspace=concept_subspace,
                sae_features=sae_feat_i,
                target_class=target_class,
                **kwargs,
            )
            results.append(vec)
            total_time += self._solve_time or 0.0

        logger.info(
            f"QPOptimal batch: {batch_size} solves in {total_time:.2f}s "
            f"({total_time / batch_size:.3f}s avg)"
        )
        return results

    def compute_shared_steering(
        self,
        activations: torch.Tensor,
        D: torch.Tensor,
        concept_subspace,
        sae_features: torch.Tensor = None,
        target_class: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Solve a single QP that steers the worst-case input in the batch.

        Uses the input with the largest constraint shortfall (hardest to
        satisfy all subspace constraints) to define the RHS, producing one
        shared steering vector for all inputs.

        Args:
            activations: Batch of activations, shape (batch, d_model).
            D: SAE decoder matrix, shape (d_sae, d_model).
            concept_subspace: ConceptSubspace for multi-constraint mode.
            sae_features: SAE features for pre-filtering, shape (d_sae,).
                Uses mean activation across batch if shape (batch, d_sae).
            target_class: Target class for steering.

        Returns:
            Single steering vector, shape (d_model,).
        """
        acts_np = (
            activations.detach().cpu().numpy().astype(np.float64)
            if isinstance(activations, torch.Tensor)
            else np.asarray(activations, dtype=np.float64)
        )

        # Worst-case: input with smallest minimum projection across
        # concept directions (the hardest to satisfy all constraints)
        projections = acts_np @ concept_subspace.directions.T  # (batch, k)
        thresholds = concept_subspace.compute_thresholds()  # (k,)
        # Margin shortfall per input: max over directions of (threshold - proj)
        shortfalls = thresholds[None, :] - projections  # (batch, k)
        worst_per_input = shortfalls.max(axis=1)  # (batch,)
        worst_idx = int(np.argmax(worst_per_input))
        logger.info(
            f"QPOptimal shared: worst-case input {worst_idx} "
            f"(max shortfall = {worst_per_input[worst_idx]:.4f})"
        )

        # Average SAE features for pre-filtering if batch provided
        if sae_features is not None and sae_features.ndim == 2:
            sae_feat_mean = sae_features.mean(dim=0)
        else:
            sae_feat_mean = sae_features

        return self.compute_steering_vector(
            h=activations[worst_idx],
            D=D,
            concept_subspace=concept_subspace,
            sae_features=sae_feat_mean,
            target_class=target_class,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def delta(self) -> np.ndarray:
        """The sparse feature perturbation vector."""
        if self._delta is None:
            raise RuntimeError("QP not yet solved.")
        return self._delta

    @property
    def solve_status(self) -> str:
        return self._solve_status or "not_solved"

    @property
    def solve_time(self) -> float:
        return self._solve_time or 0.0

    @property
    def active_features(self) -> np.ndarray:
        """Indices of features with nonzero perturbation."""
        if self._active_features is None:
            raise RuntimeError("QP not yet solved.")
        return self._active_features

    def summary(self) -> dict:
        info = super().summary()
        info["solve_status"] = self.solve_status
        info["lam"] = self.lam
        info["solver"] = self.solver
        info["solve_time"] = self.solve_time
        if self._delta is not None:
            info["l0"] = int((self._delta > 1e-6).sum())
            info["l1"] = float(self._delta.sum())
            info["coherence_l2"] = float(
                np.linalg.norm(self._delta)  # approximate; full needs D
            )
            info["n_active_features"] = (
                len(self._active_features) if self._active_features is not None else 0
            )
        return info
