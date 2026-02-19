"""Linear probe wrapping sklearn LogisticRegression.

Trains a logistic regression classifier on residual-stream activations
to identify concept directions. The learned weight vector and bias feed
directly into the SOCP steering formulation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class LinearProbe:
    """Logistic-regression probe for binary classification on activations.

    Wraps :class:`sklearn.linear_model.LogisticRegression` and exposes
    convenience properties (``weight_vector``, ``bias``) that are consumed
    downstream by the SOCP optimisation step.

    Args:
        d_model: Dimensionality of the residual-stream activations.
        C: Inverse regularisation strength (passed to LogisticRegression).
        max_iter: Maximum iterations for the solver.
    """

    def __init__(
        self,
        d_model: int,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.d_model = d_model
        self._clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Core sklearn-delegating methods
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """Train the probe on activation vectors and binary labels.

        Args:
            X: Activation matrix of shape ``(n, d_model)``.
            y: Binary label vector of shape ``(n,)``.

        Returns:
            ``self``, to allow method chaining.
        """
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions.

        Args:
            X: Activation matrix of shape ``(n, d_model)``.

        Returns:
            Integer label array of shape ``(n,)``.
        """
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Activation matrix of shape ``(n, d_model)``.

        Returns:
            Probability array of shape ``(n, 2)`` where column 0 is P(y=0)
            and column 1 is P(y=1).
        """
        return self._clf.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on the given data.

        Args:
            X: Activation matrix of shape ``(n, d_model)``.
            y: True binary labels of shape ``(n,)``.

        Returns:
            Fraction of correctly classified samples.
        """
        return self._clf.score(X, y)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw logit scores (w^T x + b).

        Args:
            X: Activation matrix of shape ``(n, d_model)``.

        Returns:
            Array of shape ``(n,)`` with the signed distance to the
            decision boundary.
        """
        return self._clf.decision_function(X)

    # ------------------------------------------------------------------
    # Properties used by the SOCP formulation
    # ------------------------------------------------------------------

    @property
    def weight_vector(self) -> np.ndarray:
        """Learned weight vector of shape ``(d_model,)``.

        Raises:
            RuntimeError: If the probe has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Probe has not been fitted yet. Call fit() first."
            )
        return self._clf.coef_[0]

    @property
    def bias(self) -> float:
        """Learned bias (intercept) scalar.

        Raises:
            RuntimeError: If the probe has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Probe has not been fitted yet. Call fit() first."
            )
        return self._clf.intercept_[0]
