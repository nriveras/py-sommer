"""Relationship matrix constructors translated from sommer core C++ routines."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _impute_marker_means(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D marker matrix")
    out = x.copy()
    cols = np.where(np.isnan(out).any(axis=0))[0]
    for c in cols:
        mu = np.nanmean(out[:, c])
        out[np.isnan(out[:, c]), c] = mu
    return out


def _filter_markers(x: np.ndarray, min_maf: float) -> np.ndarray:
    pfreq = np.mean(x + 1.0, axis=0) / 2.0
    maf = np.minimum(pfreq, 1.0 - pfreq)
    idx_maf = maf > min_maf
    x2 = x[:, idx_maf]
    if x2.size == 0:
        raise ValueError("No markers remaining after min_maf filter")
    xvar = np.var(x2, axis=0, ddof=0)
    idx_var = xvar > 0.0
    x3 = x2[:, idx_var]
    if x3.size == 0:
        raise ValueError("No polymorphic markers remaining")
    return x3


def A_mat(X: np.ndarray, min_maf: float = 0.0, return_imputed: bool = False):
    """Additive genomic relationship matrix (vanRaden)."""
    X = _impute_marker_means(np.asarray(X, dtype=float))
    Xf = _filter_markers(X, min_maf=min_maf)

    n, p = Xf.shape
    ms012 = np.mean(Xf + 1.0, axis=0)
    freq = ms012 / 2.0
    v = 2.0 * np.mean(freq * (1.0 - freq))
    if v <= 0:
        raise ValueError("Invalid marker variance scaling for A matrix")

    freqmat = np.ones((n, 1)) @ freq[None, :]
    w = (Xf + 1.0) - (2.0 * freqmat)
    k = w @ w.T
    A = k / (v * p)

    if return_imputed:
        return {"X": X, "A": A}
    return A


def D_mat(
    X: np.ndarray,
    nishio: bool = True,
    min_maf: float = 0.0,
    return_imputed: bool = False,
):
    """Dominance genomic relationship matrix (Nishio/Satoh or Su)."""
    X = _impute_marker_means(np.asarray(X, dtype=float))
    Xf = _filter_markers(X, min_maf=min_maf)

    n = Xf.shape[0]
    xd = 1.0 - np.abs(Xf)

    if nishio:
        m = xd - np.mean(xd, axis=0, keepdims=True)
        p = np.mean(Xf + 1.0, axis=0) / 2.0
        var_hw = np.sum((2.0 * p * (1.0 - p)) ** 2)
    else:
        p = np.sum(Xf + 1.0, axis=0) / (2.0 * n)
        q = 1.0 - p
        p2q = 2.0 * (p * q)
        var_hw = np.sum(p2q * (1.0 - p2q))
        m = xd - p2q[None, :]

    if var_hw <= 0:
        raise ValueError("Invalid marker variance scaling for D matrix")

    k = m @ m.T
    D = k / var_hw

    if return_imputed:
        return {"X": X, "D": D}
    return D


def E_mat(
    X: np.ndarray,
    nishio: bool = True,
    interaction: str = "A#A",
    min_maf: float = 0.02,
) -> np.ndarray:
    """Epistatic matrix via Hadamard product of A and/or D."""
    interaction = interaction.upper()
    if interaction == "A#A":
        A = A_mat(X, min_maf=min_maf)
        return A * A
    if interaction == "A#D":
        A = A_mat(X, min_maf=min_maf)
        D = D_mat(X, nishio=nishio, min_maf=min_maf)
        return A * D
    if interaction == "D#D":
        D = D_mat(X, nishio=nishio, min_maf=min_maf)
        return D * D
    raise ValueError("interaction must be one of: 'A#A', 'A#D', 'D#D'")


def _safe_inv_sympd(a: np.ndarray, tolparinv: float) -> np.ndarray:
    a = (a + a.T) / 2.0
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.inv(a + np.eye(a.shape[0]) * tolparinv)


def H_mat(
    A: np.ndarray,
    G: np.ndarray,
    tau: float = 1.0,
    omega: float = 1.0,
    tolparinv: float = 1e-6,
    genotyped_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Single-step H matrix using A and G22.

    If genotyped_index is not provided, G is assumed to correspond to the last m
    individuals in A, where m = G.shape[0].
    """
    A = np.asarray(A, dtype=float)
    G = np.asarray(G, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be square")
    if G.shape[0] > A.shape[0]:
        raise ValueError("G cannot have more rows than A")

    n = A.shape[0]
    m = G.shape[0]

    if genotyped_index is None:
        mask_g = np.zeros(n, dtype=bool)
        mask_g[n - m :] = True
    else:
        mask_g = np.asarray(genotyped_index, dtype=bool)
        if mask_g.shape[0] != n:
            raise ValueError("genotyped_index must have length nrow(A)")
        if np.sum(mask_g) != m:
            raise ValueError("sum(genotyped_index) must equal nrow(G)")

    mask_ng = ~mask_g
    idx1 = np.where(mask_ng)[0]
    idx2 = np.where(mask_g)[0]

    A12 = A[np.ix_(idx1, idx2)]
    A21 = A[np.ix_(idx2, idx1)]
    A22 = A[np.ix_(idx2, idx2)]

    A22inv = _safe_inv_sympd(A22, tolparinv=tolparinv)
    G22inv = _safe_inv_sympd(G, tolparinv=tolparinv)

    H22p = (tau * G22inv) + ((1.0 - omega) * A22inv)
    H22inv = _safe_inv_sympd(H22p, tolparinv=tolparinv)

    H11 = A12 @ A22inv @ (H22inv - A22) @ A22inv @ A21
    H12 = A12 @ A22inv @ (H22inv - A22)
    H21 = (H22inv - A22) @ A22inv @ A21
    H22 = H22inv - A22

    addon = np.block([[H11, H12], [H21, H22]])
    # Rebuild A in partitioned order [non-genotyped, genotyped].
    Ap = A[np.ix_(np.r_[idx1, idx2], np.r_[idx1, idx2])]
    Hp = Ap + addon

    # Return to original order.
    order = np.r_[idx1, idx2]
    inv_order = np.argsort(order)
    return Hp[np.ix_(inv_order, inv_order)]
