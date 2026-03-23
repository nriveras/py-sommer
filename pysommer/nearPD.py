"""Nearest positive-definite projection (Higham-style, matching sommer nearPDcpp)."""

from __future__ import annotations

import numpy as np


def nearPD(
    x0: np.ndarray,
    maxit: int = 100,
    eig_tol: float = 1e-6,
    conv_tol: float = 1e-7,
) -> np.ndarray:
    x = np.asarray(x0, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x0 must be a square matrix")

    x = (x + x.T) / 2.0
    n = x.shape[0]
    d_s = np.zeros((n, n), dtype=float)

    for _ in range(maxit):
        y = x
        r = y - d_s
        eigvals, eigvecs = np.linalg.eigh((r + r.T) / 2.0)
        thresh = eig_tol * eigvals[0]
        keep = eigvals > thresh
        if not np.any(keep):
            break
        q = eigvecs[:, keep]
        dp = eigvals[keep]
        x = (q * dp) @ q.T
        d_s = x - r

        denom = np.linalg.norm(y, ord=np.inf)
        if denom == 0:
            if np.linalg.norm(y - x, ord=np.inf) <= conv_tol:
                break
        else:
            conv = np.linalg.norm(y - x, ord=np.inf) / denom
            if conv <= conv_tol:
                break

    return (x + x.T) / 2.0


def near_pd(
    x0: np.ndarray,
    maxit: int = 100,
    eig_tol: float = 1e-6,
    conv_tol: float = 1e-7,
) -> np.ndarray:
    """Alias using snake_case naming."""
    return nearPD(x0=x0, maxit=maxit, eig_tol=eig_tol, conv_tol=conv_tol)
