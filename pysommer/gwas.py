"""GWAS helper wrappers around C++ core: scorecalc and gwasForLoop."""

from __future__ import annotations

import numpy as np

from ._cpp._sommer_core import (  # type: ignore[import-not-found]
    gwas_for_loop as _gwas_for_loop,
    scorecalc as _scorecalc,
)


def scorecalc(
    Mimv: np.ndarray,
    Ymv: np.ndarray,
    Zmv: np.ndarray,
    Xmv: np.ndarray,
    Vinv: np.ndarray,
    nt: int,
    min_maf: float = 0.0,
    tolparinv: float = 1e-6,
) -> np.ndarray:
    """Compute GWAS scores, effects, and SEs for marker design columns.

    Returns an array of shape (n_markers, nt, 3) where slices are:
    - 0: score proxy x = v2 / (v2 + F)
    - 1: marker effect estimates
    - 2: marker effect standard errors
    """
    return np.asarray(_scorecalc(
        np.asarray(Mimv, dtype=float),
        np.asarray(Ymv, dtype=float),
        np.asarray(Zmv, dtype=float),
        np.asarray(Xmv, dtype=float),
        np.asarray(Vinv, dtype=float),
        nt, min_maf,
    ))


def gwasForLoop(
    M: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    X: np.ndarray,
    Vinv: np.ndarray,
    min_maf: float = 0.0,
    display_progress: bool = False,
) -> np.ndarray:
    """Loop over markers and compute scorecalc outputs.

    Returns shape (n_markers, n_traits, 3).
    """
    return np.asarray(_gwas_for_loop(
        np.asarray(M, dtype=float),
        np.asarray(Y, dtype=float),
        np.asarray(Z, dtype=float),
        np.asarray(X, dtype=float),
        np.asarray(Vinv, dtype=float),
        min_maf, display_progress,
    ))
