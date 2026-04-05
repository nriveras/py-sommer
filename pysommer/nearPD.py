"""Nearest positive-definite projection — C++ core wrapper."""

from __future__ import annotations

import numpy as np

from ._cpp._sommer_core import near_pd_cpp as _near_pd_cpp  # type: ignore[import-not-found]


def nearPD(
    x0: np.ndarray,
    maxit: int = 100,
    eig_tol: float = 1e-6,
    conv_tol: float = 1e-7,
) -> np.ndarray:
    x = np.asarray(x0, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x0 must be a square matrix")
    return np.asarray(_near_pd_cpp(x, maxit, eig_tol, conv_tol))


def near_pd(
    x0: np.ndarray,
    maxit: int = 100,
    eig_tol: float = 1e-6,
    conv_tol: float = 1e-7,
) -> np.ndarray:
    """Alias using snake_case naming."""
    return nearPD(x0=x0, maxit=maxit, eig_tol=eig_tol, conv_tol=conv_tol)
