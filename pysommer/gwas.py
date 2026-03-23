"""GWAS helper translations: scorecalc and gwasForLoop."""

from __future__ import annotations

import numpy as np

from .solver import _safe_inv


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
    """Compute GWAS scores, effects, and SEs for a marker block.

    Returns an array of shape (n_markers, nt, 3) where slices are:
    - 0: score proxy x = v2 / (v2 + v1 * F)
    - 1: marker effect estimates
    - 2: marker effect standard errors
    """
    Mimv = np.asarray(Mimv, dtype=float)
    Ymv = np.asarray(Ymv, dtype=float)
    Zmv = np.asarray(Zmv, dtype=float)
    Xmv = np.asarray(Xmv, dtype=float)
    Vinv = np.asarray(Vinv, dtype=float)

    n = Ymv.shape[0]
    z_m = Zmv @ Mimv
    xz_m = np.hstack([Xmv, z_m])
    p = xz_m.shape[1]

    v1 = 1.0
    v2 = max(float(n - p), 1.0)

    w = xz_m.T @ (Vinv @ xz_m)
    w_inv = _safe_inv(w, tolparinv=tolparinv)

    # Minor allele frequency check (same coding assumption as sommer C++ routine).
    pf = np.mean(Mimv + 1.0, axis=0) / 2.0
    maf = np.min(np.vstack([pf, 1.0 - pf]), axis=0)
    if np.min(maf) <= min_maf:
        return np.zeros((Mimv.shape[1], nt, 3), dtype=float)

    b = w_inv @ (xz_m.T @ (Vinv @ Ymv))
    e = Ymv - (xz_m @ b)

    m_var = float(np.asarray((e.T @ (Vinv @ e)) / v2).squeeze())
    b_var = w_inv * m_var

    marker_idx = np.arange(Xmv.shape[1], b.shape[0])
    b_marker = b[marker_idx, :].reshape(-1)
    b_marker_var = b_var[np.ix_(marker_idx, marker_idx)]
    se_marker = np.sqrt(np.maximum(np.diag(b_marker_var), 0.0))

    f_stat = np.square(b_marker / np.maximum(se_marker, 1e-12))
    x_score = v2 / (v2 + v1 * f_stat)

    out = np.zeros((Mimv.shape[1], nt, 3), dtype=float)
    out[:, :, 0] = x_score.reshape(Mimv.shape[1], nt)
    out[:, :, 1] = b_marker.reshape(Mimv.shape[1], nt)
    out[:, :, 2] = se_marker.reshape(Mimv.shape[1], nt)
    return out


def gwasForLoop(
    M: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    X: np.ndarray,
    Vinv: np.ndarray,
    min_maf: float = 0.0,
) -> np.ndarray:
    """Loop over markers and compute scorecalc outputs.

    Returns shape (n_markers, n_traits, 3).
    """
    M = np.asarray(M, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    X = np.asarray(X, dtype=float)
    Vinv = np.asarray(Vinv, dtype=float)

    if Y.ndim == 1:
        Y = Y[:, None]

    nt = Y.shape[1]
    d_nt = np.eye(nt, dtype=float)
    y_mv = Y.T.reshape(-1, 1, order="F")
    z_mv = np.kron(d_nt, Z)
    x_mv = np.kron(d_nt, X)

    n_markers = M.shape[1]
    out = np.zeros((n_markers, nt, 3), dtype=float)
    for i in range(n_markers):
        mi = M[:, i : i + 1]
        mimv = np.kron(d_nt, mi)
        sc = scorecalc(
            Mimv=mimv,
            Ymv=y_mv,
            Zmv=z_mv,
            Xmv=x_mv,
            Vinv=Vinv,
            nt=nt,
            min_maf=min_maf,
        )
        out[i, :, :] = sc[0, :, :]

    return out
