"""Matrix-based REML solvers — thin wrappers around sommer C++ core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ._cpp._sommer_core import (  # type: ignore[import-not-found]
    mnr as _mnr,
)


@dataclass
class SolverResult:
    """Container for mixed-model solver outputs and convergence metadata."""

    theta: np.ndarray
    beta: np.ndarray
    u: List[np.ndarray]
    log_likelihood: float
    converged: bool
    iterations: int
    fitted: np.ndarray
    residuals: np.ndarray
    pevs: List[np.ndarray]


def _as_col(y: np.ndarray) -> np.ndarray:
    """Return y as an n x 1 column vector for univariate solvers."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2 and y.shape[1] == 1:
        return y
    raise ValueError("Only univariate y is currently supported (shape n or n x 1)")


def _safe_inv(a: np.ndarray, tolparinv: float) -> np.ndarray:
    """Invert a symmetric matrix, adding ridge regularization if needed."""
    a = (a + a.T) / 2.0
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.inv(a + np.eye(a.shape[0]) * tolparinv)


def _default_weights(iters: int, stepweight, emweight):
    """Return per-iteration step and EM-mixing weights as 1-D float arrays."""
    if stepweight is None:
        sw = np.full(iters, 0.9, dtype=float)
        if iters > 0:
            sw[0] = 0.5
        if iters > 1:
            sw[1] = 0.7
    else:
        sw = np.asarray(stepweight, dtype=float).ravel()

    if emweight is None:
        ew = np.zeros(iters, dtype=float)
    else:
        ew = np.asarray(emweight, dtype=float).ravel()

    return sw, ew


def _parse_mnr_result(
    res: dict,
    y: np.ndarray,
    x: np.ndarray,
    z_list: list[np.ndarray],
) -> SolverResult:
    """Convert the py::dict returned by C++ MNR/newton_di_sp into a SolverResult."""
    beta = np.asarray(res["b"])
    convergence = bool(res["convergence"])

    # Monitor: rows = variance parameters, columns = iterations
    monitor = np.asarray(res["monitor"])
    if monitor.ndim == 2:
        theta = monitor[:, -1]
        iterations = int(monitor.shape[1])
    else:
        theta = monitor
        iterations = 1

    # logLik
    llik = np.asarray(res["llik"])
    if llik.ndim >= 2:
        loglik = float(llik[0, -1])
    elif llik.ndim == 1:
        loglik = float(llik[-1])
    else:
        loglik = float(llik)

    # u partitions and PEVs from uList/uPevList
    u_list_raw = res.get("uList", [])
    u_pev_raw = res.get("uPevList", [])

    u_list: list[np.ndarray] = []
    pevs: list[np.ndarray] = []

    if u_list_raw is not None and len(u_list_raw) > 0:
        for item in u_list_raw:
            u_list.append(np.asarray(item))
    else:
        u_all = np.asarray(res.get("u", np.zeros((0, 1))))
        u_list = [u_all]

    if u_pev_raw is not None and len(u_pev_raw) > 0:
        for item in u_pev_raw:
            pevs.append(np.asarray(item))
    else:
        pevs = [np.full((u.shape[0], u.shape[0]), np.nan) for u in u_list]

    # C++ returns fitted = X*beta only; add random part for full fitted
    fitted = x @ beta
    for zi, ui in zip(z_list, u_list):
        fitted = fitted + zi @ ui
    residuals = y - fitted

    return SolverResult(
        theta=theta,
        beta=beta,
        u=u_list,
        log_likelihood=loglik,
        converged=convergence,
        iterations=iterations,
        fitted=fitted,
        residuals=residuals,
        pevs=pevs,
    )


def _call_mnr(
    y: np.ndarray,
    x: np.ndarray,
    z_list: list[np.ndarray],
    k_list: list[np.ndarray],
    r: np.ndarray,
    iters: int,
    tolpar: float,
    tolparinv: float,
    ai: bool,
    pev: bool,
    verbose: bool,
    stepweight: np.ndarray | None,
    emweight: np.ndarray | None,
) -> dict:
    """Prepare inputs and call the C++ MNR solver."""
    n = y.shape[0]
    nt = 1  # univariate

    sw, ew = _default_weights(iters, stepweight, emweight)

    # MNR expects:
    #   Y: (n, nt) dense matrix
    #   X: list of (n, p) dense matrices (one per trait)
    #   Gx: list of (nt, nt) scaling matrices (one per random term)
    #   Z: list of (n, q) dense matrices (one per random term)
    #   K: list of (q, q) dense matrices (one per random term)
    #   R: list of (n, n) sparse residual matrices
    #   Ge: list of (nt, nt) initial variance matrices (one per random+residual term)
    #   GeI: list of (nt, nt) constraint matrices (one per random+residual term)
    #   W: (n, n) weight matrix
    n_random = len(z_list)
    n_re = n_random + 1  # + residual

    vary = float(np.var(y, ddof=1))
    if vary <= 0:
        vary = 1.0
    init_var = vary / max(n_re, 1)

    X_list = [x]
    Gx = [np.eye(nt) for _ in range(n_random)]
    R_list = [r]  # single residual term

    # Initial variance component estimates (scaled by vary)
    Ge = [np.full((nt, nt), init_var) for _ in range(n_re)]
    # Constraint indicators (1 = estimate this parameter)
    GeI = [np.ones((nt, nt)) for _ in range(n_re)]

    W = np.eye(n)

    return _mnr(
        y, X_list, Gx, z_list, k_list, R_list, Ge, GeI,
        W, False,
        iters, tolpar, tolparinv,
        ai, pev, verbose, False,
        sw, ew,
    )


def newton_di_sp(
    Y: np.ndarray,
    X: np.ndarray,
    Z: Sequence[np.ndarray],
    K: Sequence[np.ndarray],
    R: np.ndarray | None = None,
    theta_init: np.ndarray | None = None,
    iters: int = 50,
    tolpar: float = 1e-6,
    tolparinv: float = 1e-6,
    ai: bool = True,
    pev: bool = True,
    verbose: bool = False,
    stepweight: np.ndarray | None = None,
    emweight: np.ndarray | None = None,
) -> SolverResult:
    """Direct-inversion REML for univariate Gaussian mixed models.

    Model: y = Xb + sum_i Z_i u_i + e
    with u_i ~ N(0, theta_i K_i), e ~ N(0, theta_e R).
    """
    y = _as_col(Y)
    x = np.asarray(X, dtype=float)
    n = y.shape[0]

    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError("X must be 2D with same number of rows as Y")
    if len(Z) != len(K):
        raise ValueError("Z and K must have the same number of terms")

    z_list = [np.asarray(zi, dtype=float) for zi in Z]
    k_list = [np.asarray(ki, dtype=float) for ki in K]

    if R is None:
        r = np.eye(n)
    else:
        r = np.asarray(R, dtype=float)

    res = _call_mnr(y, x, z_list, k_list, r,
                    iters, tolpar, tolparinv, ai, pev, verbose,
                    stepweight, emweight)

    return _parse_mnr_result(res, y, x, z_list)


def ai_mme_sp(
    Y: np.ndarray,
    X: np.ndarray,
    Z: Sequence[np.ndarray],
    K: Sequence[np.ndarray],
    R: np.ndarray | None = None,
    theta_init: np.ndarray | None = None,
    iters: int = 50,
    tolpar: float = 1e-6,
    tolparinv: float = 1e-6,
    pev: bool = True,
    verbose: bool = False,
    stepweight: np.ndarray | None = None,
    emweight: np.ndarray | None = None,
) -> SolverResult:
    """Henderson-style MME REML updates (AI/EM hybrid).

    Note: This now uses the same C++ newton_di_sp core via MNR, which
    supports both AI and EM weighting schemes. The separate Henderson
    MME pathway may be exposed in a future release.
    """
    y = _as_col(Y)
    x = np.asarray(X, dtype=float)
    n = y.shape[0]

    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError("X must be 2D with same number of rows as Y")
    if len(Z) != len(K):
        raise ValueError("Z and K must have the same number of terms")

    z_list = [np.asarray(zi, dtype=float) for zi in Z]
    k_list = [np.asarray(ki, dtype=float) for ki in K]

    if R is None:
        r = np.eye(n)
    else:
        r = np.asarray(R, dtype=float)

    res = _call_mnr(y, x, z_list, k_list, r,
                    iters, tolpar, tolparinv, True, pev, verbose,
                    stepweight, emweight)

    return _parse_mnr_result(res, y, x, z_list)
