"""Convex Optimal Sparse Steering via SOCP.

Core contribution: finds the sparsest SAE-feature intervention to steer model behavior
using a Second-Order Cone Program (SOCP) formulation.

Formulation:
    minimize    1^T delta
    s.t.  (D @ w)^T delta >= tau' - w^T h - b    (linear: probe constraint)
          ||delta @ D||_2 <= epsilon               (SOC: coherence budget)
          delta >= 0                                (non-negativity)

Where:
    delta ∈ R^{d_sae}  — sparse feature activation perturbation
    D ∈ R^{d_sae x d_model} — SAE decoder matrix
    w ∈ R^{d_model} — probe weight vector
    b ∈ R — probe bias
    h ∈ R^{d_model} — current residual stream activation for the input
    tau' — desired probe margin (target class score)
    epsilon — coherence budget (max L2 norm of perturbation in residual space)
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
    1. Shifts the probe score past the target margin
    2. Keeps the residual-stream perturbation within an L2 budget

    This is solved per-input, producing an input-dependent steering vector.
    """

    def __init__(self, epsilon: float = 5.0, tau: float = 0.5,
                 solver: str = "SCS", max_iters: int = 10000,
                 prefilter_threshold: float = 0.01):
        super().__init__("convex_optimal")
        self.epsilon = epsilon
        self.tau = tau
        self.solver = solver
        self.max_iters = max_iters
        self.prefilter_threshold = prefilter_threshold

        # Stored after compute
        self._delta: Optional[np.ndarray] = None
        self._solve_status: Optional[str] = None
        self._active_features: Optional[np.ndarray] = None
        self._feature_mask: Optional[np.ndarray] = None

    def compute_steering_vector(
        self,
        h: torch.Tensor = None,
        probe_w: np.ndarray = None,
        probe_b: float = None,
        D: torch.Tensor = None,
        sae_features: torch.Tensor = None,
        target_class: int = 1,
        epsilon: float = None,
        tau: float = None,
        **kwargs,
    ) -> torch.Tensor:
        """Solve the SOCP for a single input to find the optimal sparse steering.

        Args:
            h: Current residual stream activation, shape (d_model,).
            probe_w: Probe weight vector, shape (d_model,).
            probe_b: Probe bias scalar.
            D: SAE decoder matrix, shape (d_sae, d_model).
            sae_features: SAE feature activations for pre-filtering, shape (d_sae,).
                If provided, features with activation below prefilter_threshold are excluded.
            target_class: Which class to steer toward (0 or 1). If 0, we negate the
                probe direction so the formulation stays the same.
            epsilon: Override coherence budget.
            tau: Override probe margin.

        Returns:
            Steering vector in residual stream space, shape (d_model,).
        """
        if epsilon is not None:
            self.epsilon = epsilon
        if tau is not None:
            self.tau = tau

        # Convert inputs to numpy
        h_np = h.detach().cpu().numpy().astype(np.float64) if isinstance(h, torch.Tensor) else np.asarray(h, dtype=np.float64)
        w_np = np.asarray(probe_w, dtype=np.float64)
        b_val = float(probe_b)
        D_np = D.detach().cpu().numpy().astype(np.float64) if isinstance(D, torch.Tensor) else np.asarray(D, dtype=np.float64)

        # If steering toward class 0, negate probe direction
        if target_class == 0:
            w_np = -w_np
            b_val = -b_val
            self.tau = -self.tau if self.tau < 0 else self.tau

        d_sae, d_model = D_np.shape

        # Pre-filter features to keep CVXPY fast
        if sae_features is not None:
            if isinstance(sae_features, torch.Tensor):
                feat_np = sae_features.detach().cpu().numpy()
            else:
                feat_np = np.asarray(sae_features)
            # Keep features with activation above threshold
            mask = np.abs(feat_np) > self.prefilter_threshold
            # Always include at least top 500 features by activation magnitude
            if mask.sum() < 500:
                top_indices = np.argsort(np.abs(feat_np))[::-1][:500]
                mask[top_indices] = True
            self._feature_mask = mask
        else:
            mask = np.ones(d_sae, dtype=bool)
            self._feature_mask = mask

        active_idx = np.where(mask)[0]
        n_active = len(active_idx)
        logger.info(f"ConvexOptimal: {n_active}/{d_sae} active features after pre-filtering")

        # Subselect decoder rows
        D_sub = D_np[active_idx]  # (n_active, d_model)

        # Precompute probe-projected decoder: (D @ w) for each active feature
        Dw = D_sub @ w_np  # (n_active,)

        # Current probe score: w^T h + b
        current_score = w_np @ h_np + b_val
        logger.info(f"ConvexOptimal: current probe score = {current_score:.4f}, target margin = {self.tau:.4f}")

        # RHS of probe constraint
        rhs = self.tau - current_score  # need Dw^T delta >= rhs

        # Define CVXPY problem
        delta = cp.Variable(n_active, nonneg=True)

        # Objective: minimize L1 (since delta >= 0, ||delta||_1 = 1^T delta)
        objective = cp.Minimize(cp.sum(delta))

        constraints = [
            # Probe constraint: projected perturbation must exceed margin
            Dw @ delta >= rhs,
            # Coherence constraint: L2 norm of perturbation in residual space
            cp.norm(D_sub.T @ delta, 2) <= self.epsilon,
        ]

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
        probe_w: np.ndarray,
        probe_b: float,
        D: torch.Tensor,
        sae_features_batch: torch.Tensor = None,
        target_class: int = 1,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Solve SOCP for a batch of inputs.

        Args:
            activations: Batch of activations, shape (batch, d_model).
            probe_w: Probe weight vector, shape (d_model,).
            probe_b: Probe bias scalar.
            D: SAE decoder matrix, shape (d_sae, d_model).
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
                probe_w=probe_w,
                probe_b=probe_b,
                D=D,
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
        info["tau"] = self.tau
        if self._delta is not None:
            info["l0"] = int((self._delta > 1e-6).sum())
            info["l1"] = float(self._delta.sum())
            info["n_active_features"] = len(self._active_features) if self._active_features is not None else 0
        return info
