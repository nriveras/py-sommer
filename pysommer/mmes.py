"""Matrix-based and formula-like mmes wrappers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .formula import VSMCall, build_random_from_vsm, parse_fixed_formula
from .solver import ai_mme_sp, newton_di_sp


def _run_mmes_matrix_core(
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
    method: str = "newton_di_sp",
) -> Dict[str, Any]:
    """Shared computational backend for matrix-based mmes calls."""
    if method == "newton_di_sp":
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
    elif method == "ai_mme_sp":
        fit = ai_mme_sp(
            Y=Y,
            X=X,
            Z=Z,
            K=K,
            R=R,
            theta_init=theta_init,
            iters=iters,
            tolpar=tolpar,
            tolparinv=tolparinv,
            pev=pev,
            verbose=verbose,
            stepweight=stepweight,
            emweight=emweight,
        )
    else:
        raise ValueError("method must be one of: 'newton_di_sp', 'ai_mme_sp'")

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


def _run_mmes_matrix_dispatch(
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
    method: str = "newton_di_sp",
) -> Dict[str, Any]:
    """Dispatch matrix mode to univariate core or independent-trait multivariate path."""
    y_arr = np.asarray(Y, dtype=float)

    # Univariate route.
    if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
        return _run_mmes_matrix_core(
            Y=y_arr,
            X=X,
            Z=Z,
            K=K,
            R=R,
            iters=iters,
            tolpar=tolpar,
            tolparinv=tolparinv,
            ai=ai,
            pev=pev,
            verbose=verbose,
            stepweight=stepweight,
            emweight=emweight,
            theta_init=theta_init,
            method=method,
        )

    # Independent-trait multivariate route: solve one trait at a time with shared model terms.
    if y_arr.ndim != 2:
        raise ValueError("Y must be 1D or 2D array")

    n_traits = y_arr.shape[1]
    trait_fits: list[dict[str, Any]] = []
    for j in range(n_traits):
        yj = y_arr[:, j : j + 1]
        fit_j = _run_mmes_matrix_core(
            Y=yj,
            X=X,
            Z=Z,
            K=K,
            R=R,
            iters=iters,
            tolpar=tolpar,
            tolparinv=tolparinv,
            ai=ai,
            pev=pev,
            verbose=verbose,
            stepweight=stepweight,
            emweight=emweight,
            theta_init=theta_init,
            method=method,
        )
        trait_fits.append(fit_j)

    beta = np.hstack([f["beta"] for f in trait_fits])
    theta = np.column_stack([np.asarray(f["theta"]).reshape(-1) for f in trait_fits])
    fitted = np.hstack([f["fitted"] for f in trait_fits])
    residuals = np.hstack([f["residuals"] for f in trait_fits])
    u = [np.hstack([f["u"][i] for f in trait_fits]) for i in range(len(trait_fits[0]["u"]))]

    # Preserve per-trait likelihood/iterations while exposing an aggregate summary.
    loglik = np.asarray([f["logLik"] for f in trait_fits], dtype=float)
    iterations = np.asarray([f["iterations"] for f in trait_fits], dtype=int)
    converged = bool(all(bool(f["converged"]) for f in trait_fits))

    pevs = []
    for i in range(len(trait_fits[0]["pevs"])):
        pevs_i = [np.asarray(f["pevs"][i]) for f in trait_fits]
        pevs.append(np.stack(pevs_i, axis=2))

    return {
        "beta": beta,
        "u": u,
        "theta": theta,
        "logLik": loglik,
        "converged": converged,
        "iterations": iterations,
        "fitted": fitted,
        "residuals": residuals,
        "pevs": pevs,
        "trait_fits": trait_fits,
        "multivariate_mode": "independent",
    }


def _normalize_random(random: Sequence[VSMCall] | VSMCall | None) -> list[VSMCall]:
    if random is None:
        return []
    if isinstance(random, VSMCall):
        return [random]
    return list(random)


def mmes(
    Y: np.ndarray | None = None,
    X: np.ndarray | None = None,
    Z: Sequence[np.ndarray] | None = None,
    K: Sequence[np.ndarray] | None = None,
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
    method: str = "newton_di_sp",
    fixed: str | None = None,
    random: Sequence[VSMCall] | VSMCall | None = None,
    data: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fit a mixed model using matrix inputs or a lightweight formula interface.

    Matrix mode:
    - Provide `Y`, `X`, `Z`, `K`.

    Formula-like mode (R-sommer style wrapper):
    - Provide `fixed`, `random`, `data` where `random` is built with `vsm(...)`.
    """
    formula_mode = fixed is not None or data is not None or random is not None

    if formula_mode:
        if fixed is None or data is None:
            raise ValueError("Formula-like mode requires 'fixed' and 'data'")
        return mmes_formula(
            fixed=fixed,
            random=_normalize_random(random),
            data=data,
            R=R,
            iters=iters,
            tolpar=tolpar,
            tolparinv=tolparinv,
            ai=ai,
            pev=pev,
            verbose=verbose,
            stepweight=stepweight,
            emweight=emweight,
            theta_init=theta_init,
            method=method,
        )

    if Y is None or X is None or Z is None or K is None:
        raise ValueError("Matrix mode requires Y, X, Z, and K")

    return _run_mmes_matrix_dispatch(
        Y=Y,
        X=X,
        Z=Z,
        K=K,
        R=R,
        iters=iters,
        tolpar=tolpar,
        tolparinv=tolparinv,
        ai=ai,
        pev=pev,
        verbose=verbose,
        stepweight=stepweight,
        emweight=emweight,
        theta_init=theta_init,
        method=method,
    )


def mmes_formula(
    fixed: str,
    random: Sequence[VSMCall] | VSMCall,
    data: Mapping[str, Any],
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
    method: str = "newton_di_sp",
) -> Dict[str, Any]:
    """Fit mmes from a lightweight formula-like interface.

    Supported random declarations:
    - vsm(ism(group), Gu=K)
    - vsm(dsm(env), ism(group), Gu=K)
    """
    random_terms = _normalize_random(random)
    y, x, fixed_names = parse_fixed_formula(fixed=fixed, data=data)
    z_terms, k_terms, random_names = build_random_from_vsm(random=random_terms, data=data)

    out = _run_mmes_matrix_dispatch(
        Y=y,
        X=x,
        Z=z_terms,
        K=k_terms,
        R=R,
        iters=iters,
        tolpar=tolpar,
        tolparinv=tolparinv,
        ai=ai,
        pev=pev,
        verbose=verbose,
        stepweight=stepweight,
        emweight=emweight,
        theta_init=theta_init,
        method=method,
    )
    out["fixed_names"] = fixed_names
    out["random_names"] = random_names
    return out
