"""GWAS helper translations: scorecalc and gwasForLoop."""

from __future__ import annotations

import numpy as np

from .solver import _safe_inv


def _n_marker_blocks(mimv: np.ndarray, nt: int) -> int:
    if nt <= 1:
        return mimv.shape[1]
    if mimv.shape[1] % nt != 0:
        raise ValueError("For nt > 1, Mimv columns must be divisible by nt")
    return mimv.shape[1] // nt


def _marker_maf(mimv: np.ndarray, nt: int) -> np.ndarray:
    """Compute marker-level MAF, collapsing trait-expanded columns when needed."""
    pf_cols = np.mean(mimv + 1.0, axis=0) / 2.0
    n_markers = _n_marker_blocks(mimv, nt)
    if nt > 1:
        pf = pf_cols.reshape(n_markers, nt).mean(axis=1)
    else:
        pf = pf_cols
    return np.minimum(pf, 1.0 - pf)


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
    Mimv = np.asarray(Mimv, dtype=float)
    Ymv = np.asarray(Ymv, dtype=float)
    Zmv = np.asarray(Zmv, dtype=float)
    Xmv = np.asarray(Xmv, dtype=float)
    Vinv = np.asarray(Vinv, dtype=float)

    if Ymv.ndim == 1:
        Ymv = Ymv[:, None]

    n_markers = _n_marker_blocks(Mimv, nt)
    out = np.zeros((n_markers, nt, 3), dtype=float)

    maf = _marker_maf(Mimv, nt)
    if np.min(maf) <= min_maf:
        return out

    n = Ymv.shape[0]
    z_m = Zmv @ Mimv
    xz_m = np.hstack([Xmv, z_m])
    p = xz_m.shape[1]

    v2 = max(float(n - p), 1.0)

    w = xz_m.T @ (Vinv @ xz_m)
    try:
        w_inv = _safe_inv(w, tolparinv=tolparinv)
    except np.linalg.LinAlgError:
        return out

    b = w_inv @ (xz_m.T @ (Vinv @ Ymv))
    e = Ymv - (xz_m @ b)

    m_var = float(np.asarray((e.T @ (Vinv @ e)) / v2).squeeze())
    b_var = w_inv * m_var

    marker_idx = np.arange(Xmv.shape[1], b.shape[0])
    b_marker = b[marker_idx, :].reshape(-1)
    b_marker_var = b_var[np.ix_(marker_idx, marker_idx)]
    se_marker = np.sqrt(np.maximum(np.diag(b_marker_var), 0.0))

    if b_marker.size % nt != 0:
        raise ValueError("Marker effect vector length is incompatible with nt")
    n_markers_fit = b_marker.size // nt

    f_stat = np.square(b_marker / np.maximum(se_marker, 1e-12))
    x_score = v2 / (v2 + f_stat)

    out = np.zeros((n_markers_fit, nt, 3), dtype=float)
    out[:, :, 0] = x_score.reshape(n_markers_fit, nt)
    out[:, :, 1] = b_marker.reshape(n_markers_fit, nt)
    out[:, :, 2] = se_marker.reshape(n_markers_fit, nt)
    return out


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
    del display_progress  # Progress bar support is not implemented in Python.

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
    z_mv = np.kron(Z, d_nt)
    x_mv = np.kron(X, d_nt)

    n_markers = M.shape[1]
    out = np.zeros((n_markers, nt, 3), dtype=float)

    for i in range(n_markers):
        mi = M[:, i : i + 1]
        mimv = np.kron(mi, d_nt) if nt > 1 else mi
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
