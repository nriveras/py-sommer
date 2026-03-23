"""Matrix-based REML solvers inspired by sommer core routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class SolverResult:
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
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2 and y.shape[1] == 1:
        return y
    raise ValueError("Only univariate y is currently supported (shape n or n x 1)")


def _safe_inv(a: np.ndarray, tolparinv: float) -> np.ndarray:
    a = (a + a.T) / 2.0
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.inv(a + np.eye(a.shape[0]) * tolparinv)


def _build_v(theta: np.ndarray, dv: Sequence[np.ndarray]) -> np.ndarray:
    v = np.zeros_like(dv[0])
    for t, dvi in zip(theta, dv):
        v += t * dvi
    return (v + v.T) / 2.0


def _initialize_theta(
    y: np.ndarray,
    m: int,
    theta_init: np.ndarray | None,
    tolpar: float,
) -> np.ndarray:
    if theta_init is None:
        vy = float(np.var(y, ddof=1))
        theta = np.full(m, vy / max(m, 1), dtype=float)
        theta[theta <= tolpar] = max(tolpar * 10.0, 1e-4)
        return theta

    theta = np.asarray(theta_init, dtype=float).reshape(-1)
    if theta.size != m:
        raise ValueError("theta_init must have len(Z)+1 elements")
    return np.maximum(theta, tolpar)


def _initialize_weights(
    iters: int,
    stepweight: np.ndarray | None,
    emweight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if stepweight is None:
        step = np.full(iters, 0.9, dtype=float)
        if iters > 0:
            step[0] = 0.5
        if iters > 1:
            step[1] = 0.7
    else:
        step = np.asarray(stepweight, dtype=float).reshape(-1)
        if step.size < iters:
            raise ValueError("stepweight must have at least 'iters' elements")

    if emweight is None:
        em = np.zeros(iters, dtype=float)
    else:
        em = np.asarray(emweight, dtype=float).reshape(-1)
        if em.size < iters:
            raise ValueError("emweight must have at least 'iters' elements")

    return step, em


def _partition_indices(block_sizes: Sequence[int]) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    for sz in block_sizes:
        end = start + int(sz)
        parts.append((start, end))
        start = end
    return parts


def _loglik_reml(y: np.ndarray, x: np.ndarray, v: np.ndarray, tolparinv: float) -> float:
    n = y.shape[0]
    p = x.shape[1]

    vi = _safe_inv(v, tolparinv=tolparinv)
    xvix = x.T @ vi @ x
    xvix_i = _safe_inv(xvix, tolparinv=tolparinv)

    beta = xvix_i @ (x.T @ vi @ y)
    e = y - x @ beta

    sign_v, logdet_v = np.linalg.slogdet(v)
    sign_x, logdet_x = np.linalg.slogdet(xvix)
    if sign_v <= 0 or sign_x <= 0:
        return float("-inf")

    quad = float(np.sum(e.T @ vi @ e))
    return -0.5 * (logdet_v + logdet_x + quad + (n - p) * np.log(2.0 * np.pi))


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
    for zi in z_list:
        if zi.shape[0] != n:
            raise ValueError("Each Z_i must have same number of rows as Y")

    if R is None:
        r = np.eye(n)
    else:
        r = np.asarray(R, dtype=float)
        if r.shape != (n, n):
            raise ValueError("R must be n x n")

    m = len(z_list) + 1
    theta = _initialize_theta(y=y, m=m, theta_init=theta_init, tolpar=tolpar)
    stepweight, emweight = _initialize_weights(
        iters=iters,
        stepweight=stepweight,
        emweight=emweight,
    )

    dv = [zi @ ki @ zi.T for zi, ki in zip(z_list, k_list)] + [r]

    converged = False
    loglik = float("-inf")
    beta = np.zeros((x.shape[1], 1), dtype=float)
    vi = np.eye(n)
    pmat = np.eye(n)

    for it in range(iters):
        v = _build_v(theta, dv)
        vi = _safe_inv(v, tolparinv=tolparinv)

        xvix = x.T @ vi @ x
        xvix_i = _safe_inv(xvix, tolparinv=tolparinv)
        vi_x = vi @ x
        pmat = vi - vi_x @ xvix_i @ vi_x.T

        beta = xvix_i @ (x.T @ vi @ y)
        py = pmat @ y

        score = np.zeros(m, dtype=float)
        info = np.zeros((m, m), dtype=float)

        p_dv = [pmat @ dvi for dvi in dv]
        for i in range(m):
            quad_i = float(np.sum(y.T @ p_dv[i] @ py))
            score[i] = -0.5 * np.trace(p_dv[i]) + 0.5 * quad_i

        for i in range(m):
            for j in range(i, m):
                if ai:
                    vij = 0.5 * float(np.sum(y.T @ p_dv[i] @ p_dv[j] @ py))
                else:
                    vij = 0.5 * np.trace(p_dv[i] @ p_dv[j])
                info[i, j] = vij
                info[j, i] = vij

        info += np.eye(m) * tolparinv
        try:
            ai_update = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            ai_update = np.linalg.pinv(info) @ score

        em_update = score / np.maximum(np.diag(info), tolparinv)
        update = (1.0 - emweight[it]) * ai_update + emweight[it] * em_update

        theta_new = theta + stepweight[it] * update
        theta_new = np.maximum(theta_new, tolpar)

        diff = np.linalg.norm(theta_new - theta)
        theta = theta_new

        loglik = _loglik_reml(y, x, v, tolparinv=tolparinv)
        if verbose:
            print(
                f"iter={it + 1} logLik={loglik:.6f} diff={diff:.3e} "
                f"theta={np.array2string(theta, precision=6)}"
            )

        if diff < tolpar:
            converged = True
            break

    fitted_fixed = x @ beta
    resid = y - fitted_fixed

    u_hat = []
    pevs = []
    for i, (zi, ki) in enumerate(zip(z_list, k_list)):
        gi = theta[i] * ki
        ui = gi @ zi.T @ vi @ resid
        u_hat.append(ui)

        if pev:
            c_uu = gi - gi @ zi.T @ pmat @ zi @ gi
        else:
            c_uu = np.full((gi.shape[0], gi.shape[0]), np.nan)
        pevs.append(c_uu)

    fitted = fitted_fixed.copy()
    for zi, ui in zip(z_list, u_hat):
        fitted += zi @ ui
    resid = y - fitted

    return SolverResult(
        theta=theta,
        beta=beta,
        u=u_hat,
        log_likelihood=loglik,
        converged=converged,
        iterations=it + 1,
        fitted=fitted,
        residuals=resid,
        pevs=pevs,
    )


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
    """Henderson-style MME REML updates (simplified ai_mme_sp translation).

    This solver focuses on univariate models and a dense-matrix implementation.
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
    for zi in z_list:
        if zi.shape[0] != n:
            raise ValueError("Each Z_i must have same number of rows as Y")

    if R is None:
        r0 = np.eye(n)
    else:
        r0 = np.asarray(R, dtype=float)
        if r0.shape != (n, n):
            raise ValueError("R must be n x n")

    m = len(z_list) + 1
    theta = _initialize_theta(y=y, m=m, theta_init=theta_init, tolpar=tolpar)
    stepweight, emweight = _initialize_weights(
        iters=iters,
        stepweight=stepweight,
        emweight=emweight,
    )

    # Used for comparable logLik and residual updates.
    dv = [zi @ ki @ zi.T for zi, ki in zip(z_list, k_list)] + [r0]

    q_sizes = [zi.shape[1] for zi in z_list]
    u_parts = _partition_indices(q_sizes)
    q_total = int(np.sum(q_sizes))
    z_all = np.hstack(z_list) if len(z_list) > 0 else np.zeros((n, 0), dtype=float)

    converged = False
    loglik = float("-inf")
    beta = np.zeros((x.shape[1], 1), dtype=float)
    u_hat_stack = np.zeros((q_total, 1), dtype=float)
    c_uu = np.zeros((q_total, q_total), dtype=float)

    r0_inv = _safe_inv(r0, tolparinv=tolparinv)

    for it in range(iters):
        # Build scaled inverse residual covariance.
        r_inv = r0_inv / max(theta[-1], tolpar)

        # Build block-diagonal G^{-1}.
        if q_total > 0:
            g_inv = np.zeros((q_total, q_total), dtype=float)
            for (start, end), ki, th in zip(u_parts, k_list, theta[:-1]):
                ki_inv = _safe_inv(ki, tolparinv=tolparinv)
                g_inv[start:end, start:end] = ki_inv / max(th, tolpar)
        else:
            g_inv = np.zeros((0, 0), dtype=float)

        x_r = x.T @ r_inv
        c11 = x_r @ x
        rhs1 = x_r @ y

        if q_total > 0:
            c12 = x_r @ z_all
            c21 = c12.T
            c22 = z_all.T @ r_inv @ z_all + g_inv
            rhs2 = z_all.T @ r_inv @ y

            c = np.block([[c11, c12], [c21, c22]])
            rhs = np.vstack([rhs1, rhs2])
        else:
            c = c11
            rhs = rhs1

        c = (c + c.T) / 2.0
        c_inv = _safe_inv(c, tolparinv=tolparinv)
        bu = c_inv @ rhs

        beta = bu[: x.shape[1], :]
        if q_total > 0:
            u_hat_stack = bu[x.shape[1] :, :]
            c_uu = c_inv[x.shape[1] :, x.shape[1] :]
        else:
            u_hat_stack = np.zeros((0, 1), dtype=float)
            c_uu = np.zeros((0, 0), dtype=float)

        fitted = x @ beta
        if q_total > 0:
            fitted = fitted + z_all @ u_hat_stack
        resid = y - fitted

        theta_new = theta.copy()

        # Random-effect variance component updates.
        for idx, ((start, end), ki) in enumerate(zip(u_parts, k_list)):
            ui = u_hat_stack[start:end, :]
            cuu_i = c_uu[start:end, start:end]
            ki_inv = _safe_inv(ki, tolparinv=tolparinv)

            em_est = float(np.asarray((ui.T @ ki_inv @ ui) / max(ki.shape[0], 1)).squeeze())
            ai_est = float(
                np.asarray((ui.T @ ki_inv @ ui + np.trace(ki_inv @ cuu_i)) / max(ki.shape[0], 1)).squeeze()
            )

            proposed = (1.0 - emweight[it]) * ai_est + emweight[it] * em_est
            theta_new[idx] = max(
                (1.0 - stepweight[it]) * theta[idx] + stepweight[it] * proposed,
                tolpar,
            )

        # Residual variance update.
        e_quad = float(np.asarray(resid.T @ r0_inv @ resid).squeeze())
        sigma_e = e_quad / max(n - x.shape[1], 1)
        theta_new[-1] = max(
            (1.0 - stepweight[it]) * theta[-1] + stepweight[it] * sigma_e,
            tolpar,
        )

        diff = np.linalg.norm(theta_new - theta)
        theta = theta_new

        v = _build_v(theta, dv)
        loglik = _loglik_reml(y, x, v, tolparinv=tolparinv)
        if verbose:
            print(
                f"iter={it + 1} logLik={loglik:.6f} diff={diff:.3e} "
                f"theta={np.array2string(theta, precision=6)}"
            )

        if diff < tolpar:
            converged = True
            break

    u_hat: list[np.ndarray] = []
    pevs: list[np.ndarray] = []
    for start, end in u_parts:
        u_hat.append(u_hat_stack[start:end, :])
        if pev:
            pevs.append(c_uu[start:end, start:end])
        else:
            n_u = end - start
            pevs.append(np.full((n_u, n_u), np.nan))

    fitted = x @ beta
    for zi, ui in zip(z_list, u_hat):
        fitted += zi @ ui
    resid = y - fitted

    return SolverResult(
        theta=theta,
        beta=beta,
        u=u_hat,
        log_likelihood=loglik,
        converged=converged,
        iterations=it + 1,
        fitted=fitted,
        residuals=resid,
        pevs=pevs,
    )
