"""Core utility helpers translated from sommer C++ helpers."""

from __future__ import annotations

import numpy as np


def seq_cpp(a: int, b: int) -> np.ndarray:
    """Inclusive integer sequence [a, b]."""
    if b < a:
        return np.array([], dtype=int)
    return np.arange(a, b + 1, dtype=int)


def var_cols(x: np.ndarray) -> np.ndarray:
    """Sample variance (ddof=1) of each column."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    return np.var(x, axis=0, ddof=1)


def scale_cpp(x: np.ndarray) -> np.ndarray:
    """Center and scale columns using sample standard deviation."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    means = np.mean(x, axis=0, keepdims=True)
    sds = np.sqrt(var_cols(x))
    sds[sds == 0.0] = 1.0
    return (x - means) / sds


def make_full(x: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return a full-rank basis for the column-space of x using SVD."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s > tol))
    if rank == x.shape[1]:
        return x.copy()
    # Equivalent to taking first rank left singular vectors as a basis.
    return u[:, :rank]


def is_identity_mat(x: np.ndarray, atol: float = 1e-12) -> bool:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return False
    return np.allclose(x, np.eye(x.shape[0]), atol=atol, rtol=0.0)


def is_diagonal_mat(x: np.ndarray, atol: float = 1e-12) -> bool:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return False
    off_diag = x - np.diag(np.diag(x))
    return np.allclose(off_diag, 0.0, atol=atol, rtol=0.0)


def mat_to_vec_cpp(x: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Extract upper-triangle entries where x2(i, j) > 0 using C++ loop order."""
    x = np.asarray(x, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x.shape != x2.shape or x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x and x2 must be same-size square matrices")

    out = []
    ncol = x.shape[1]
    for i in range(ncol):
        for j in range(ncol):
            if i <= j and x2[i, j] > 0:
                out.append(x[i, j])
    return np.asarray(out, dtype=float)


def vec_to_mat_cpp(x: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Pack a vector into upper-triangle entries where x2(i, j) > 0."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float)
    if x2.ndim != 2 or x2.shape[0] != x2.shape[1]:
        raise ValueError("x2 must be square")

    ncol = x2.shape[1]
    out = np.zeros((ncol, ncol), dtype=float)
    counter = 0
    for j in range(ncol):
        for i in range(ncol):
            if i <= j and x2[i, j] > 0:
                if counter >= x.size:
                    raise ValueError("x is too short for x2 constraints")
                out[i, j] = x[counter]
                counter += 1
    if counter != x.size:
        raise ValueError("x has extra elements not consumed by x2 constraints")
    return out
