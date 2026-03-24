"""Prediction helpers for fitted pysommer mixed models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    """Normalize a vector or matrix to a 2D float array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D")
    return arr


def _rowwise_quadratic(z_term: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Return per-row variances for Z C Z' with optional trait stacking."""
    if cov.ndim == 2:
        return np.sum(z_term * (z_term @ cov), axis=1, keepdims=True)
    if cov.ndim == 3:
        pieces = [
            np.sum(z_term * (z_term @ cov[:, :, trait]), axis=1)
            for trait in range(cov.shape[2])
        ]
        return np.column_stack(pieces)
    raise ValueError("Each PEV matrix must be 2D or 3D")


def _residual_variance(theta: np.ndarray, n_rows: int) -> np.ndarray:
    """Broadcast the residual variance component across prediction rows."""
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim == 1:
        return np.full((n_rows, 1), float(theta_arr[-1]), dtype=float)
    if theta_arr.ndim == 2:
        return np.tile(theta_arr[-1, :].reshape(1, -1), (n_rows, 1))
    raise ValueError("theta must be 1D or 2D")


def predict_mmes(
    fit: Mapping[str, Any],
    X: np.ndarray | Sequence[float],
    Z: Sequence[np.ndarray] | None = None,
    *,
    include_random: bool = False,
) -> np.ndarray:
    """Predict responses from a fitted ``mmes`` result.

    Parameters
    ----------
    fit : Mapping[str, Any]
        Result dictionary returned by ``pysommer.mmes`` or ``pysommer.mmes_formula``.
    X : array-like
        Fixed-effects design matrix for the prediction rows.
    Z : sequence of arrays, optional
        Random-effects design matrices aligned to the fitted random-effect levels.
    include_random : bool, default=False
        If True, add ``Z_i @ u_i`` contributions using the supplied ``Z`` terms.
    """
    beta = _as_2d_array(fit["beta"], "fit['beta']")
    x_arr = _as_2d_array(X, "X")
    if x_arr.shape[1] != beta.shape[0]:
        raise ValueError("X has a different number of columns than fit['beta']")

    pred = x_arr @ beta
    if not include_random:
        return pred

    if Z is None:
        raise ValueError("Z must be provided when include_random=True")

    u_terms = fit.get("u")
    if u_terms is None:
        raise ValueError("fit must include 'u' when include_random=True")
    if len(Z) != len(u_terms):
        raise ValueError("Z must have the same number of terms as fit['u']")

    for idx, (z_term, u_term) in enumerate(zip(Z, u_terms)):
        z_arr = np.asarray(z_term, dtype=float)
        u_arr = _as_2d_array(u_term, f"fit['u'][{idx}]")
        if z_arr.ndim != 2:
            raise ValueError(f"Z[{idx}] must be 2D")
        if z_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"Z[{idx}] row count must match X")
        if z_arr.shape[1] != u_arr.shape[0]:
            raise ValueError(f"Z[{idx}] columns must match fit['u'][{idx}] rows")
        pred = pred + (z_arr @ u_arr)

    return pred


def summarize_predictions(
    fit: Mapping[str, Any],
    X: np.ndarray | Sequence[float],
    Z: Sequence[np.ndarray] | None = None,
    *,
    include_random: bool = False,
    interval: float = 0.95,
) -> dict[str, np.ndarray | float]:
    """Summarize conditional predictions and approximate uncertainty.

    The returned uncertainty uses the residual variance plus random-effect PEV
    contributions from the supplied design matrices. Fixed-effect coefficient
    uncertainty is not currently included.
    """
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must be between 0 and 1")

    pred = predict_mmes(fit=fit, X=X, Z=Z, include_random=include_random)
    variance = _residual_variance(theta=np.asarray(fit["theta"], dtype=float), n_rows=pred.shape[0])
    random_variance = np.zeros_like(variance)

    if include_random:
        if Z is None:
            raise ValueError("Z must be provided when include_random=True")
        pevs = fit.get("pevs")
        if pevs is None:
            raise ValueError("fit must include 'pevs' to summarize random-effect uncertainty")
        if len(Z) != len(pevs):
            raise ValueError("Z must have the same number of terms as fit['pevs']")

        for idx, (z_term, pev_term) in enumerate(zip(Z, pevs)):
            z_arr = np.asarray(z_term, dtype=float)
            pev_arr = np.asarray(pev_term, dtype=float)
            if z_arr.ndim != 2:
                raise ValueError(f"Z[{idx}] must be 2D")
            if z_arr.shape[0] != pred.shape[0]:
                raise ValueError(f"Z[{idx}] row count must match X")
            random_variance = random_variance + _rowwise_quadratic(z_arr, pev_arr)

    total_variance = np.maximum(variance + random_variance, 0.0)
    prediction_sd = np.sqrt(total_variance)
    z_score = float(NormalDist().inv_cdf(0.5 + interval / 2.0))

    return {
        "predictions": pred,
        "prediction_variance": total_variance,
        "prediction_sd": prediction_sd,
        "interval_lower": pred - z_score * prediction_sd,
        "interval_upper": pred + z_score * prediction_sd,
        "residual_variance": variance,
        "random_effect_variance": random_variance,
        "interval": float(interval),
    }