"""Covariance structure constructors."""

from __future__ import annotations

import numpy as np


def AR1(n: int, rho: float = 0.25) -> np.ndarray:
    """AR(1) correlation matrix."""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def CS(n: int, rho: float = 0.25) -> np.ndarray:
    """Compound-symmetry correlation matrix."""
    out = np.full((n, n), float(rho), dtype=float)
    np.fill_diagonal(out, 1.0)
    return out


def ARMA(n: int, rho: float = 0.25, lam: float = 0.25) -> np.ndarray:
    """ARMA-like matrix matching sommer::ARMA constructor."""
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    m = d.copy().astype(float)
    m[np.tril_indices(n, k=-1)] -= 1.0
    m[np.triu_indices(n, k=1)] -= 1.0
    mm = rho ** m
    nn = np.full((n, n), float(lam), dtype=float)
    np.fill_diagonal(nn, 1.0)
    return mm * nn
