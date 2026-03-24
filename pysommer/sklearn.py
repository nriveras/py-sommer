"""Scikit-learn-like estimator wrappers for pysommer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from .formula import (
    VSMCall,
    build_prediction_random_terms,
    build_random_from_vsm,
    parse_fixed_formula,
)
from .mmes import mmes, mmes_formula, _normalize_random
from .predict import predict_mmes, summarize_predictions

try:  # pragma: no cover - optional import
    from sklearn.base import BaseEstimator
    from sklearn.exceptions import NotFittedError
except ImportError:  # pragma: no cover - sklearn may be absent
    class BaseEstimator:  # type: ignore[no-redef]
        """Fallback base class when sklearn is not installed."""

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


class MMESRegressor(BaseEstimator):
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
        self.pevs_: list[np.ndarray] | None = None
        self.fitted_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.converged_: bool | None = None
        self.n_features_in_: int | None = None
        self.n_samples_fit_: int | None = None
        self.X_fit_: np.ndarray | None = None
        self.Z_fit_: list[np.ndarray] | None = None

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
        self.pevs_ = [np.asarray(pev_i, dtype=float) for pev_i in out["pevs"]]
        self.fitted_ = np.asarray(out["fitted"], dtype=float)
        self.residuals_ = np.asarray(out["residuals"], dtype=float)
        self.converged_ = bool(out["converged"])
        self.n_features_in_ = int(x_arr.shape[1])
        self.n_samples_fit_ = int(x_arr.shape[0])
        self.X_fit_ = x_arr.copy()
        self.Z_fit_ = [np.asarray(z_i, dtype=float).copy() for z_i in z_terms]
        return self

    def predict(
        self,
        X: np.ndarray | Sequence[float],
        include_random: bool = False,
        Z: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Predict responses for X.

        If ``include_random`` is True and ``X`` exactly matches training design,
        return stored fitted values (fixed + random). For new data, provide
        aligned ``Z`` design matrices to reuse fitted random effects.
        """
        if self.coef_ is None or self.n_features_in_ is None:
            raise NotFittedError("This MMESRegressor instance is not fitted yet")

        x_arr = _as_2d(X, "X")
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than seen in fit")

        if (
            include_random
            and Z is None
            and self.X_fit_ is not None
            and self.fitted_ is not None
            and x_arr.shape == self.X_fit_.shape
            and np.allclose(x_arr, self.X_fit_)
        ):
            return self.fitted_.copy()

        if include_random and Z is not None:
            if self.result_ is None:
                raise NotFittedError("This MMESRegressor instance is not fitted yet")
            return predict_mmes(self.result_, x_arr, Z=Z, include_random=True)

        return x_arr @ self.coef_

    def predict_summary(
        self,
        X: np.ndarray | Sequence[float],
        include_random: bool = False,
        Z: Sequence[np.ndarray] | None = None,
        interval: float = 0.95,
    ) -> dict[str, np.ndarray | float]:
        """Return conditional predictions with approximate uncertainty bands."""
        if self.result_ is None or self.coef_ is None or self.n_features_in_ is None:
            raise NotFittedError("This MMESRegressor instance is not fitted yet")

        x_arr = _as_2d(X, "X")
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than seen in fit")

        z_terms = Z
        if include_random and z_terms is None:
            if (
                self.X_fit_ is not None
                and self.Z_fit_ is not None
                and x_arr.shape == self.X_fit_.shape
                and np.allclose(x_arr, self.X_fit_)
            ):
                z_terms = self.Z_fit_
            else:
                raise ValueError("Z must be provided when include_random=True for new samples")

        return summarize_predictions(
            fit=self.result_,
            X=x_arr,
            Z=z_terms,
            include_random=include_random,
            interval=interval,
        )

    def score(self, X: np.ndarray | Sequence[float], y: np.ndarray | Sequence[float]) -> float:
        """Return R-squared score on given data."""
        y_pred = self.predict(X)
        return _r2_score(np.asarray(y, dtype=float), y_pred)


class MMESFormulaRegressor(BaseEstimator):
    """Scikit-learn-like regressor using formula-interface for mixed model REML.

    This estimator uses string formulas for fixed/random effects specification,
    similar to R's model specification (e.g., "y ~ 1 + x") and pysommer's
    lightweight formula interface (vsm, ism, dsm, usm).

    Parameters
    ----------
    fixed : str
        Fixed effects formula string (e.g., "y ~ 1 + x1 + x2").
    random : VSMCall | Sequence[VSMCall]
        Random structure declarations from vsm/ism/dsm/usm.
    R : np.ndarray, optional
        Residual covariance matrix (default: identity).
    iters : int, default=50
        Maximum EM iterations.
    tolpar : float, default=1e-6
        Parameter convergence tolerance.
    tolparinv : float, default=1e-6
        Matrix inversion tolerance.
    ai : bool, default=True
        Use AI algorithm for variance component updates.
    pev : bool, default=True
        Compute prediction error variances.
    verbose : bool, default=False
        Print iteration details.
    method : str, default="newton_di_sp"
        Solver method ("newton_di_sp" or "ai_mme_sp").
    """

    def __init__(
        self,
        fixed: str,
        random: VSMCall | Sequence[VSMCall],
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
        self.fixed = fixed
        self.random = random
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
        self.pevs_: list[np.ndarray] | None = None
        self.fitted_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.converged_: bool | None = None
        self.n_features_in_: int | None = None
        self.n_samples_fit_: int | None = None
        self.fixed_names_: list[str] | None = None
        self.random_names_: list[str] | None = None
        self.data_fit_: Mapping[str, Any] | None = None
        self.response_name_: str | None = None
        self.random_prediction_metadata_: list[dict[str, Any]] | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn compatibility."""
        _ = deep
        return {
            "fixed": self.fixed,
            "random": self.random,
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

    def set_params(self, **params: Any) -> "MMESFormulaRegressor":
        """Set constructor parameters for sklearn compatibility."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for MMESFormulaRegressor")
            setattr(self, key, value)
        return self

    def fit(self, data: Mapping[str, Any]) -> "MMESFormulaRegressor":
        """Fit the formula-based mixed model.

        Parameters
        ----------
        data : Mapping[str, Any] (e.g., dict or DataFrame)
            Data dictionary/frame with columns referenced in fixed/random formulas.

        Returns
        -------
        self : MMESFormulaRegressor
        """
        if self.method not in {"newton_di_sp", "ai_mme_sp"}:
            raise ValueError("method must be one of: 'newton_di_sp', 'ai_mme_sp'")
        if self.iters <= 0:
            raise ValueError("iters must be > 0")
        if self.tolpar <= 0 or self.tolparinv <= 0:
            raise ValueError("tolpar and tolparinv must be > 0")

        # Call formula-based solver.
        out = mmes_formula(
            fixed=self.fixed,
            random=self.random,
            data=data,
            R=self.R,
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
        self.pevs_ = [np.asarray(pev_i, dtype=float) for pev_i in out["pevs"]]
        self.fitted_ = np.asarray(out["fitted"], dtype=float)
        self.residuals_ = np.asarray(out["residuals"], dtype=float)
        self.converged_ = bool(out["converged"])
        self.n_features_in_ = len(out.get("fixed_names", []))
        self.n_samples_fit_ = int(out["fitted"].shape[0])
        self.fixed_names_ = list(out.get("fixed_names", []))
        self.random_names_ = list(out.get("random_names", []))
        self.data_fit_ = data
        random_terms = _normalize_random(self.random)
        _, _, _, random_metadata = build_random_from_vsm(
            random=random_terms,
            data=data,
            return_metadata=True,
        )
        self.random_prediction_metadata_ = list(random_metadata)
        # Extract response name from fixed formula (before the ~).
        if "~" in self.fixed:
            response_part = self.fixed.split("~", 1)[0].strip()
            # Handle multivariate responses separated by +
            self.response_name_ = response_part.split("+")[0].strip() if "+" not in response_part else None
        return self

    def predict(self, data: Mapping[str, Any], include_random: bool = False) -> np.ndarray:
        """Predict responses for new data.

        Parameters
        ----------
        data : Mapping[str, Any]
            Data dictionary/frame with columns referenced in fixed/random formulas.
        include_random : bool, default=False
            If True and data matches training data exactly, return fitted values
            (fixed + random). Otherwise return fixed-only predictions.

        Returns
        -------
        predictions : np.ndarray
            Array of predictions with shape (n_samples, n_responses).
        """
        if self.coef_ is None or self.n_features_in_ is None:
            raise NotFittedError("This MMESFormulaRegressor instance is not fitted yet")

        _, x_new, _ = parse_fixed_formula(fixed=self.fixed, data=data)
        x_new = np.asarray(x_new, dtype=float)

        if x_new.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Data has {x_new.shape[1]} features but expected {self.n_features_in_}"
            )

        if include_random:
            if self.result_ is None or self.random_prediction_metadata_ is None:
                raise NotFittedError("This MMESFormulaRegressor instance is not fitted yet")
            z_terms, _ = build_prediction_random_terms(self.random_prediction_metadata_, data)
            return predict_mmes(self.result_, x_new, Z=z_terms, include_random=True)

        return x_new @ self.coef_

    def predict_summary(
        self,
        data: Mapping[str, Any],
        include_random: bool = False,
        interval: float = 0.95,
    ) -> dict[str, Any]:
        """Return predictions plus approximate uncertainty summaries."""
        if self.result_ is None or self.coef_ is None or self.n_features_in_ is None:
            raise NotFittedError("This MMESFormulaRegressor instance is not fitted yet")

        _, x_new, _ = parse_fixed_formula(fixed=self.fixed, data=data)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Data has {x_new.shape[1]} features but expected {self.n_features_in_}"
            )

        z_terms = None
        random_status: list[dict[str, Any]] = []
        if include_random:
            if self.random_prediction_metadata_ is None:
                raise NotFittedError("This MMESFormulaRegressor instance is not fitted yet")
            z_terms, random_status = build_prediction_random_terms(
                self.random_prediction_metadata_,
                data,
            )

        summary = summarize_predictions(
            fit=self.result_,
            X=x_new,
            Z=z_terms,
            include_random=include_random,
            interval=interval,
        )
        summary["random_effect_status"] = random_status
        return summary

    def score(
        self,
        data: Mapping[str, Any],
        response_name: str | None = None,
    ) -> float:
        """Return R-squared score on given data.

        Parameters
        ----------
        data : Mapping[str, Any]
            Data dictionary/frame containing response and predictor columns.
        response_name : str, optional
            Name of response column. If None, extracted from fixed formula.

        Returns
        -------
        score : float
            R-squared score.
        """
        if self.coef_ is None or self.n_features_in_ is None:
            raise NotFittedError("This MMESFormulaRegressor instance is not fitted yet")

        # Extract response column.
        from .formula import _fetch_col

        resp_name = response_name or self.response_name_
        if resp_name is None:
            raise ValueError(
                "Could not determine response name. Provide response_name explicitly."
            )

        y = _fetch_col(data, resp_name).astype(float)
        y_pred = self.predict(data)
        return _r2_score(_as_2d(y, "y"), y_pred)
