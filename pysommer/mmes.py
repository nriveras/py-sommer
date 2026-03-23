"""Simplified matrix-based mmes wrapper (Phase 1)."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .solver import newton_di_sp


def mmes(
    Y: np.ndarray,
    X: np.ndarray,
    Z: Sequence[np.ndarray],
    K: Sequence[np.ndarray],
    R: np.ndarray | None = None,
    iters: int = 50,
    tolpar: float = 1e-6,
    tolparinv: float = 1e-6,
    ai: bool = True,
    pev: bool = True,
    verbose: bool = False,
    stepweight: np.ndarray | None = None,
    emweight: np.ndarray | None = None,
    theta_init: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Fit a univariate mixed model using explicit matrices.

    This intentionally avoids R-style formula parsing and focuses on the
    computational core.
    """
    fit = newton_di_sp(
        Y=Y,
        X=X,
        Z=Z,
        K=K,
        R=R,
        theta_init=theta_init,
        iters=iters,
        tolpar=tolpar,
        tolparinv=tolparinv,
        ai=ai,
        pev=pev,
        verbose=verbose,
        stepweight=stepweight,
        emweight=emweight,
    )

    return {
        "beta": fit.beta,
        "u": fit.u,
        "theta": fit.theta,
        "logLik": fit.log_likelihood,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "fitted": fit.fitted,
        "residuals": fit.residuals,
        "pevs": fit.pevs,
    }
