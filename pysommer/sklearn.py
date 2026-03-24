"""Scikit-learn-like estimator wrappers for pysommer."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .mmes import mmes

try:  # pragma: no cover - optional import
    from sklearn.exceptions import NotFittedError
except ImportError:  # pragma: no cover - sklearn may be absent
    class NotFittedError(RuntimeError):
        """Raised when estimator methods are used before fitting."""


def _as_2d(a: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Convert an input vector/matrix to a 2D float array."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D")
    return arr


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute an R-squared score without depending on sklearn."""
    yt = _as_2d(y_true, "y_true")
    yp = _as_2d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    ss_res = float(np.sum((yt - yp) ** 2))
    yt_mean = np.mean(yt, axis=0, keepdims=True)
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0
    return 1.0 - (ss_res / ss_tot)


class MMESRegressor:
    """Scikit-learn-like regressor wrapper around ``pysommer.mmes``.

    This first implementation targets matrix mode only (``Y, X, Z, K``).
    Formula-mode wrappers can be added later as separate estimators.
    """

    def __init__(
        self,
        Z: Sequence[np.ndarray] | None = None,
        K: Sequence[np.ndarray] | None = None,
        R: np.ndarray | None = None,
        iters: int = 50,
        tolpar: float = 1e-6,
        tolparinv: float = 1e-6,
        ai: bool = True,
        pev: bool = True,
        verbose: bool = False,
        stepweight: np.ndarray | None = None,
        emweight: np.ndarray | None = None,
        theta_init: np.ndarray | None = None,
        method: str = "newton_di_sp",
    ) -> None:
        self.Z = Z
        self.K = K
        self.R = R
        self.iters = iters
        self.tolpar = tolpar
        self.tolparinv = tolparinv
        self.ai = ai
        self.pev = pev
        self.verbose = verbose
        self.stepweight = stepweight
        self.emweight = emweight
        self.theta_init = theta_init
        self.method = method

        # Attributes populated during fit.
        self.result_: dict[str, Any] | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.u_: list[np.ndarray] | None = None
        self.fitted_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.converged_: bool | None = None
        self.n_features_in_: int | None = None
        self.n_samples_fit_: int | None = None
        self.X_fit_: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn compatibility."""
        _ = deep
        return {
            "Z": self.Z,
            "K": self.K,
            "R": self.R,
            "iters": self.iters,
            "tolpar": self.tolpar,
            "tolparinv": self.tolparinv,
            "ai": self.ai,
            "pev": self.pev,
            "verbose": self.verbose,
            "stepweight": self.stepweight,
            "emweight": self.emweight,
            "theta_init": self.theta_init,
            "method": self.method,
        }

    def set_params(self, **params: Any) -> "MMESRegressor":
        """Set constructor parameters for sklearn compatibility."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for MMESRegressor")
            setattr(self, key, value)
        return self

    def _validate_fit_inputs(
        self,
        X: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
        Z: Sequence[np.ndarray] | None,
        K: Sequence[np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]:
        """Validate and normalize fit inputs."""
        if self.method not in {"newton_di_sp", "ai_mme_sp"}:
            raise ValueError("method must be one of: 'newton_di_sp', 'ai_mme_sp'")
        if self.iters <= 0:
            raise ValueError("iters must be > 0")
        if self.tolpar <= 0 or self.tolparinv <= 0:
            raise ValueError("tolpar and tolparinv must be > 0")

        x_arr = _as_2d(X, "X")
        y_arr = _as_2d(y, "y")

        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
            raise ValueError("X and y must contain only finite values")

        z_terms = self.Z if Z is None else Z
        k_terms = self.K if K is None else K
        if z_terms is None or k_terms is None:
            raise ValueError("Z and K must be provided either at init or fit time")
        if len(z_terms) != len(k_terms):
            raise ValueError("Z and K must have the same number of terms")

        for i, (z_i, k_i) in enumerate(zip(z_terms, k_terms)):
            z_arr = np.asarray(z_i, dtype=float)
            k_arr = np.asarray(k_i, dtype=float)
            if z_arr.ndim != 2:
                raise ValueError(f"Z[{i}] must be 2D")
            if k_arr.ndim != 2:
                raise ValueError(f"K[{i}] must be 2D")
            if z_arr.shape[0] != x_arr.shape[0]:
                raise ValueError(f"Z[{i}] row count must match X/y rows")
            if k_arr.shape[0] != k_arr.shape[1]:
                raise ValueError(f"K[{i}] must be square")
            if z_arr.shape[1] != k_arr.shape[0]:
                raise ValueError(f"Z[{i}] columns must match K[{i}] size")

        return x_arr, y_arr, z_terms, k_terms

    def fit(
        self,
        X: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
        *,
        Z: Sequence[np.ndarray] | None = None,
        K: Sequence[np.ndarray] | None = None,
        R: np.ndarray | None = None,
    ) -> "MMESRegressor":
        """Fit the MMES model in matrix mode."""
        x_arr, y_arr, z_terms, k_terms = self._validate_fit_inputs(X, y, Z, K)
        r_eff = self.R if R is None else R

        out = mmes(
            Y=y_arr,
            X=x_arr,
            Z=z_terms,
            K=k_terms,
            R=r_eff,
            iters=self.iters,
            tolpar=self.tolpar,
            tolparinv=self.tolparinv,
            ai=self.ai,
            pev=self.pev,
            verbose=self.verbose,
            stepweight=self.stepweight,
            emweight=self.emweight,
            theta_init=self.theta_init,
            method=self.method,
        )

        self.result_ = out
        self.coef_ = np.asarray(out["beta"], dtype=float)
        self.intercept_ = np.asarray(self.coef_[0, :], dtype=float)
        self.theta_ = np.asarray(out["theta"], dtype=float)
        self.u_ = [np.asarray(u_i, dtype=float) for u_i in out["u"]]
        self.fitted_ = np.asarray(out["fitted"], dtype=float)
        self.residuals_ = np.asarray(out["residuals"], dtype=float)
        self.converged_ = bool(out["converged"])
        self.n_features_in_ = int(x_arr.shape[1])
        self.n_samples_fit_ = int(x_arr.shape[0])
        self.X_fit_ = x_arr.copy()
        return self

    def predict(self, X: np.ndarray | Sequence[float], include_random: bool = False) -> np.ndarray:
        """Predict responses for X.

        If ``include_random`` is True and ``X`` exactly matches training design,
        return stored fitted values (fixed + random). Otherwise return fixed-only
        predictions ``X @ beta``.
        """
        if self.coef_ is None or self.n_features_in_ is None:
            raise NotFittedError("This MMESRegressor instance is not fitted yet")

        x_arr = _as_2d(X, "X")
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than seen in fit")

        if (
            include_random
            and self.X_fit_ is not None
            and self.fitted_ is not None
            and x_arr.shape == self.X_fit_.shape
            and np.allclose(x_arr, self.X_fit_)
        ):
            return self.fitted_.copy()

        return x_arr @ self.coef_

    def score(self, X: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> float:
        """Return R-squared score on given data."""
        y_pred = self.predict(X)
        return _r2_score(np.asarray(y, dtype=float), y_pred)
