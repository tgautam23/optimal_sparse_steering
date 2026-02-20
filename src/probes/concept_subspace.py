"""Concept Subspace via supervised PCA on class-conditional activations.

Replaces the single linear probe direction (w, b) with a subspace of
k orthogonal directions derived from PCA on centered activations,
selected by class separation.  This is more robust than a trained
classifier: it is based on data moments (no optimisation), captures
richer concept structure, and generalises better to out-of-distribution
inputs.

Construction from labeled activations {(h_i, y_i)}:
    1. Class means: mu_0, mu_1, difference mu_diff = mu_1 - mu_0
    2. PCA on centered activations -> eigenvectors v_1, ..., v_d
    3. Class separation per PC: s_j = v_j^T mu_diff
    4. Select top-k PCs by |s_j| (high-variance AND class-separating)
    5. Sign-correct: w_j = sign(s_j) * v_j
"""

import numpy as np


class ConceptSubspace:
    """Concept subspace for robust multi-direction steering constraints.

    Provides the same ``fit(X, y)`` interface as :class:`LinearProbe`
    but exposes *k* orthogonal concept directions instead of a single
    weight vector.

    Args:
        n_components: Number of PCA components to compute (intermediate).
        n_select: Number of final directions to keep (top by class
            separation).
    """

    def __init__(
        self,
        n_components: int = 50,
        n_select: int = 5,
    ) -> None:
        self.n_components = n_components
        self.n_select = n_select
        self._fitted = False

        # Populated by fit()
        self._directions: np.ndarray | None = None   # (k, d_model)
        self._class_separations: np.ndarray | None = None  # (k,)
        self._midpoints: np.ndarray | None = None     # (k,)
        self._mu_diff: np.ndarray | None = None        # (d_model,)
        self._mu0: np.ndarray | None = None
        self._mu1: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConceptSubspace":
        """Compute concept subspace from labeled activation data.

        Args:
            X: Activation matrix of shape ``(n, d_model)``.
            y: Binary label vector of shape ``(n,)`` with values in {0, 1}.

        Returns:
            ``self``, to allow method chaining.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        mask0 = y == 0
        mask1 = y == 1

        self._mu0 = X[mask0].mean(axis=0)
        self._mu1 = X[mask1].mean(axis=0)
        self._mu_diff = self._mu1 - self._mu0

        # ---- PCA via truncated SVD on centered data ----
        mean_all = X.mean(axis=0)
        X_centered = X - mean_all

        n_samples, d_model = X_centered.shape
        n_components = min(self.n_components, n_samples, d_model)

        # Economy SVD: X_centered = U S V^T, columns of V are eigenvectors
        # of X^T X (i.e. PCA directions)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Vt shape: (min(n, d), d_model) -- rows are eigenvectors
        V = Vt[:n_components]  # (n_components, d_model)

        # ---- Class separation per PC ----
        separations = V @ self._mu_diff  # s_j = v_j^T mu_diff, shape (n_components,)

        # ---- Select top-k by |s_j| ----
        n_select = min(self.n_select, n_components)
        top_idx = np.argsort(np.abs(separations))[::-1][:n_select]

        selected_V = V[top_idx]           # (k, d_model)
        selected_s = separations[top_idx]  # (k,)

        # ---- Sign-correct: w_j = sign(s_j) * v_j ----
        signs = np.sign(selected_s)
        signs[signs == 0] = 1.0  # avoid zero sign
        self._directions = signs[:, None] * selected_V  # (k, d_model)
        self._class_separations = signs * selected_s     # now all positive

        # Midpoints: w_j^T (mu_0 + mu_1) / 2
        midpoint_vec = (self._mu0 + self._mu1) / 2  # (d_model,)
        self._midpoints = self._directions @ midpoint_vec  # (k,)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "ConceptSubspace has not been fitted yet. Call fit() first."
            )

    @property
    def directions(self) -> np.ndarray:
        """Concept directions of shape ``(k, d_model)``, unit-norm rows."""
        self._check_fitted()
        return self._directions

    @property
    def class_separations(self) -> np.ndarray:
        """Class separation scores ``s_j`` of shape ``(k,)``."""
        self._check_fitted()
        return self._class_separations

    @property
    def midpoints(self) -> np.ndarray:
        """Decision midpoints ``w_j^T (mu_0+mu_1)/2`` of shape ``(k,)``."""
        self._check_fitted()
        return self._midpoints

    @property
    def mu_diff(self) -> np.ndarray:
        """Class mean difference ``mu_1 - mu_0`` of shape ``(d_model,)``."""
        self._check_fitted()
        return self._mu_diff

    @property
    def n_directions(self) -> int:
        """Number of selected concept directions ``k``."""
        self._check_fitted()
        return self._directions.shape[0]

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    def compute_rhs(self, h: np.ndarray) -> np.ndarray:
        """Compute the RHS vector for multi-constraint QP/SOCP.

        For each direction j the constraint requires:
            ``w_j^T (h + D^T delta) >= w_j^T mu_1``

        Rearranging gives ``(D w_j)^T delta >= w_j^T mu_1 - w_j^T h``,
        so ``gamma_j(h) = w_j^T mu_1 - w_j^T h``.

        Args:
            h: Current activation, shape ``(d_model,)``.

        Returns:
            RHS array of shape ``(k,)``.
        """
        self._check_fitted()
        h = np.asarray(h, dtype=np.float64)
        proj_h = self._directions @ h          # (k,)
        target = self._directions @ self._mu1  # (k,)
        return target - proj_h

    def compute_thresholds(self) -> np.ndarray:
        """Compute target thresholds (without input subtraction).

        Returns:
            Thresholds of shape ``(k,)``: ``w_j^T mu_1`` for each
            concept direction.
        """
        self._check_fitted()
        return self._directions @ self._mu1

    def compute_prefilter_relevance(self, D: np.ndarray) -> np.ndarray:
        """Compute per-feature relevance for pre-filtering.

        ``relevance_i = max_j |d_i^T w_j| / ||d_i||``

        Args:
            D: SAE decoder matrix, shape ``(d_sae, d_model)``.

        Returns:
            Relevance scores of shape ``(d_sae,)``.
        """
        self._check_fitted()
        D = np.asarray(D, dtype=np.float64)
        D_norms = np.linalg.norm(D, axis=1)  # (d_sae,)
        D_norms = np.maximum(D_norms, 1e-8)

        # D @ W^T -> (d_sae, k)
        DW = D @ self._directions.T  # (d_sae, k)
        # max_j |d_i^T w_j| / ||d_i||
        relevance = np.max(np.abs(DW), axis=1) / D_norms  # (d_sae,)
        return relevance
