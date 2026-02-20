"""Convex Optimal Sparse Steering via SOCP.

Core contribution: finds the sparsest SAE-feature intervention to steer model behavior
using a Second-Order Cone Program (SOCP) formulation.

Formulation:
    minimize    1^T delta
    s.t.  w_j^T D^T delta >= gamma_j(h),  j = 1, ..., k   (concept subspace constraints)
          ||delta @ D||_2 <= epsilon                        (SOC: coherence budget)
          delta >= 0                                        (non-negativity)

Where:
    delta in R^{d_sae}  -- sparse feature activation perturbation
    D in R^{d_sae x d_model} -- SAE decoder matrix
    W = {w_1, ..., w_k} -- concept subspace directions
    h in R^{d_model} -- current residual stream activation for the input
    gamma_j(h) -- per-direction RHS (gap to class-1 mean projection)
    epsilon -- coherence budget (max L2 norm of perturbation in residual space)
"""

import logging
from typing import Optional

import cvxpy as cp
import numpy as np
import torch

from .base import SteeringMethod

logger = logging.getLogger(__name__)


class ConvexOptimalSteering(SteeringMethod):
    """SOCP-based optimal sparse steering in SAE feature space.

    Finds the minimum-L1 (sparsest) feature perturbation that:
    1. Shifts the concept subspace projections past target margins
    2. Keeps the residual-stream perturbation within an L2 budget

    This is solved per-input, producing an input-dependent steering vector.
    """

    def __init__(self, epsilon: float = 5.0,
                 solver: str = "SCS", max_iters: int = 10000,
                 prefilter_topk: int = 2000):
        super().__init__("convex_optimal")
        self.epsilon = epsilon
        self.solver = solver
        self.max_iters = max_iters
        self.prefilter_topk = prefilter_topk

        # Stored after compute
        self._delta: Optional[np.ndarray] = None
        self._solve_status: Optional[str] = None
        self._active_features: Optional[np.ndarray] = None
        self._feature_mask: Optional[np.ndarray] = None

    def compute_steering_vector(
        self,
        h: torch.Tensor,
        D: torch.Tensor,
        concept_subspace,
        sae_features: torch.Tensor = None,
        target_class: int = 1,
        epsilon: float = None,
        **kwargs,
    ) -> torch.Tensor:
        """Solve the SOCP for a single input to find the optimal sparse steering.

        Args:
            h: Current residual stream activation, shape (d_model,).
            D: SAE decoder matrix, shape (d_sae, d_model).
            concept_subspace: ConceptSubspace instance providing k concept
                directions and constraint RHS computation.
            sae_features: SAE feature activations for pre-filtering, shape (d_sae,).
            target_class: Which class to steer toward (0 or 1).
            epsilon: Override coherence budget.

        Returns:
            Steering vector in residual stream space, shape (d_model,).
        """
        if epsilon is not None:
            self.epsilon = epsilon

        # Convert inputs to numpy
        h_np = h.detach().cpu().numpy().astype(np.float64) if isinstance(h, torch.Tensor) else np.asarray(h, dtype=np.float64)
        D_np = D.detach().cpu().numpy().astype(np.float64) if isinstance(D, torch.Tensor) else np.asarray(D, dtype=np.float64)

        d_sae, d_model = D_np.shape

        # --- Multi-direction pre-filtering ---
        subspace_relevance = concept_subspace.compute_prefilter_relevance(D_np)

        n_keep = min(self.prefilter_topk, d_sae)
        n_half = n_keep // 2

        if sae_features is not None:
            if isinstance(sae_features, torch.Tensor):
                feat_np = sae_features.detach().cpu().numpy()
            else:
                feat_np = np.asarray(sae_features)
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
        logger.info(f"ConvexOptimal: {n_active}/{d_sae} active features after pre-filtering")

        D_sub = D_np[active_idx]  # (n_active, d_model)

        # Build k linear constraints: w_j^T D_sub^T delta >= rhs_j
        rhs_vec = concept_subspace.compute_rhs(h_np)  # (k,)
        DW = D_sub @ concept_subspace.directions.T  # (n_active, k)

        logger.info(
            f"ConvexOptimal: k={concept_subspace.n_directions} constraints, "
            f"rhs range = [{rhs_vec.min():.4f}, {rhs_vec.max():.4f}]"
        )

        delta = cp.Variable(n_active, nonneg=True)
        objective = cp.Minimize(cp.sum(delta))

        constraints = [DW[:, j] @ delta >= rhs_vec[j]
                       for j in range(concept_subspace.n_directions)]
        # Coherence constraint
        constraints.append(cp.norm(D_sub.T @ delta, 2) <= self.epsilon)

        problem = cp.Problem(objective, constraints)

        # Solve
        solver_kwargs = {"max_iters": self.max_iters, "verbose": False}
        try:
            problem.solve(solver=self.solver, **solver_kwargs)
        except cp.SolverError:
            logger.warning(f"ConvexOptimal: {self.solver} failed, trying ECOS fallback")
            try:
                problem.solve(solver="ECOS", verbose=False)
            except cp.SolverError:
                logger.error("ConvexOptimal: all solvers failed")
                self._solve_status = "failed"
                self._steering_vector = torch.zeros(d_model)
                return self._steering_vector

        self._solve_status = problem.status
        logger.info(f"ConvexOptimal: solver status = {problem.status}, optimal value = {problem.value}")

        if problem.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(f"ConvexOptimal: non-optimal status: {problem.status}")
            self._steering_vector = torch.zeros(d_model)
            self._delta = np.zeros(d_sae)
            return self._steering_vector

        # Extract solution
        delta_val = np.maximum(delta.value, 0)  # clip numerical noise

        # Map back to full feature space
        full_delta = np.zeros(d_sae)
        full_delta[active_idx] = delta_val
        self._delta = full_delta
        self._active_features = active_idx[delta_val > 1e-6]

        l0 = int((full_delta > 1e-6).sum())
        l1 = float(full_delta.sum())
        logger.info(f"ConvexOptimal: L0 = {l0}, L1 = {l1:.4f}")

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
        """Solve SOCP for a batch of inputs.

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

        for i in range(batch_size):
            h_i = activations[i]
            sae_feat_i = sae_features_batch[i] if sae_features_batch is not None else None
            vec = self.compute_steering_vector(
                h=h_i,
                D=D,
                concept_subspace=concept_subspace,
                sae_features=sae_feat_i,
                target_class=target_class,
                **kwargs,
            )
            results.append(vec)

        return results

    @property
    def delta(self) -> np.ndarray:
        """The sparse feature perturbation vector."""
        if self._delta is None:
            raise RuntimeError("SOCP not yet solved.")
        return self._delta

    @property
    def solve_status(self) -> str:
        return self._solve_status or "not_solved"

    @property
    def active_features(self) -> np.ndarray:
        """Indices of features with nonzero perturbation."""
        if self._active_features is None:
            raise RuntimeError("SOCP not yet solved.")
        return self._active_features

    def summary(self) -> dict:
        info = super().summary()
        info["solve_status"] = self.solve_status
        info["epsilon"] = self.epsilon
        if self._delta is not None:
            info["l0"] = int((self._delta > 1e-6).sum())
            info["l1"] = float(self._delta.sum())
            info["n_active_features"] = len(self._active_features) if self._active_features is not None else 0
        return info
