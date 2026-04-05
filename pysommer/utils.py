"""Core utility helpers — thin wrappers around sommer C++ core."""

from __future__ import annotations

import numpy as np

from ._cpp._sommer_core import (  # type: ignore[import-not-found]
    is_diagonal_mat as _is_diagonal_mat,
    is_identity_mat as _is_identity_mat,
    make_full as _make_full,
    mat_to_vec_cpp as _mat_to_vec_cpp,
    scale_cpp as _scale_cpp,
    seq_cpp as _seq_cpp,
    var_cols as _var_cols,
    vec_to_mat_cpp as _vec_to_mat_cpp,
)


def seq_cpp(a: int, b: int) -> np.ndarray:
    """Inclusive integer sequence [a, b]."""
    return np.asarray(_seq_cpp(a, b), dtype=float)


def var_cols(x: np.ndarray) -> np.ndarray:
    """Sample variance (ddof=1) of each column."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    return np.asarray(_var_cols(x))


def scale_cpp(x: np.ndarray) -> np.ndarray:
    """Center and scale columns using sample standard deviation."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    return np.asarray(_scale_cpp(x))


def make_full(x: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return a full-rank basis for the column-space of x using SVD."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    return np.asarray(_make_full(x))


def is_identity_mat(x: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True when x is a square identity matrix within tolerance."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return False
    return bool(_is_identity_mat(x))


def is_diagonal_mat(x: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True when x is a square diagonal matrix within tolerance."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return False
    return bool(_is_diagonal_mat(x))


def mat_to_vec_cpp(x: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Extract upper-triangle entries where x2(i, j) > 0 using C++ loop order."""
    x = np.asarray(x, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    return np.asarray(_mat_to_vec_cpp(x, x2))


def vec_to_mat_cpp(x: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Pack a vector into upper-triangle entries where x2(i, j) > 0."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float)
    return np.asarray(_vec_to_mat_cpp(x, x2))
