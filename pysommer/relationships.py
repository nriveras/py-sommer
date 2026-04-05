"""Relationship matrix constructors — C++ core wrappers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._cpp._sommer_core import (  # type: ignore[import-not-found]
    amat as _amat,
    dmat as _dmat,
    emat as _emat,
    hmat as _hmat,
)


def A_mat(X: np.ndarray, min_maf: float = 0.0, return_imputed: bool = False):
    """Additive genomic relationship matrix (vanRaden)."""
    X = np.asarray(X, dtype=float)
    A = np.asarray(_amat(X, True, min_maf))
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
    X = np.asarray(X, dtype=float)
    D = np.asarray(_dmat(X, nishio, min_maf))
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
    X = np.asarray(X, dtype=float)
    interaction = interaction.upper()
    if interaction == "A#A":
        A = np.asarray(_amat(X, True, min_maf))
        return np.asarray(_emat(A, A))
    if interaction == "A#D":
        A = np.asarray(_amat(X, True, min_maf))
        D = np.asarray(_dmat(X, nishio, min_maf))
        return np.asarray(_emat(A, D))
    if interaction == "D#D":
        D = np.asarray(_dmat(X, nishio, min_maf))
        return np.asarray(_emat(D, D))
    raise ValueError("interaction must be one of: 'A#A', 'A#D', 'D#D'")


def H_mat(
    A: np.ndarray,
    G: np.ndarray,
    tau: float = 1.0,
    omega: float = 1.0,
    tolparinv: float = 1e-6,
    genotyped_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Single-step H matrix using A and G22."""
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
        # Default: last m individuals are genotyped → non-genotyped get index=1
        index = np.ones(n, dtype=float)
        index[n - m:] = 0.0
    else:
        mask_g = np.asarray(genotyped_index, dtype=bool)
        if mask_g.shape[0] != n:
            raise ValueError("genotyped_index must have length nrow(A)")
        if np.sum(mask_g) != m:
            raise ValueError("sum(genotyped_index) must equal nrow(G)")
        # C++ convention: index=TRUE means non-genotyped, FALSE means genotyped
        index = (~mask_g).astype(float)

    return np.asarray(_hmat(A, G, index, tolparinv, tau, omega))
