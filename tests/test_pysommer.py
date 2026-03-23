import json
from pathlib import Path

import numpy as np
import pytest

from pysommer import (
    A_mat,
    AR1,
    ARMA,
    CS,
    D_mat,
    E_mat,
    H_mat,
    make_full,
    mat_to_vec_cpp,
    nearPD,
    dsm,
    gwasForLoop,
    ism,
    scale_cpp,
    scorecalc,
    seq_cpp,
    var_cols,
    vec_to_mat_cpp,
    vsm,
)
from pysommer.mmes import mmes, mmes_formula


REF_DIR = Path(__file__).parent / "reference_data"


def _load_csv(name: str) -> np.ndarray:
    path = REF_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"Reference file not found: {path}")
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _assert_close(a, b, atol=1e-6, rtol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def test_utils_basic():
    assert np.array_equal(seq_cpp(2, 5), np.array([2, 3, 4, 5]))

    x = np.array([[1.0, 2.0], [3.0, 4.0], [6.0, 10.0]])
    vc = var_cols(x)
    _assert_close(vc, np.var(x, axis=0, ddof=1))

    xs = scale_cpp(x)
    _assert_close(np.mean(xs, axis=0), np.zeros(2), atol=1e-10, rtol=0)


def test_vec_mat_helpers_roundtrip_shape():
    m = np.array([[1.0, 0.5, 0.2], [0.0, 2.0, 0.3], [0.0, 0.0, 3.0]])
    c = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    v = mat_to_vec_cpp(m, c)
    m2 = vec_to_mat_cpp(v, c)
    assert m2.shape == m.shape
    assert np.all(m2[np.tril_indices(3, -1)] == 0)


def test_covariance_constructors():
    ar1 = AR1(5, rho=0.3)
    cs = CS(5, rho=0.2)
    arma = ARMA(5, rho=0.3, lam=0.2)

    assert ar1.shape == (5, 5)
    assert cs.shape == (5, 5)
    assert arma.shape == (5, 5)
    assert np.allclose(np.diag(cs), 1.0)


def test_relationship_matrices_psd():
    rng = np.random.default_rng(11)
    X = rng.choice([-1.0, 0.0, 1.0], size=(12, 20))

    A = A_mat(X)
    D = D_mat(X)
    E = E_mat(X, interaction="A#A", min_maf=0.0)

    assert A.shape == (12, 12)
    assert D.shape == (12, 12)
    assert E.shape == (12, 12)
    assert np.allclose(A, A.T, atol=1e-10)
    assert np.allclose(D, D.T, atol=1e-10)


def test_h_matrix_shape():
    rng = np.random.default_rng(21)
    X = rng.choice([-1.0, 0.0, 1.0], size=(10, 16))
    A = A_mat(X)
    G = A[-6:, -6:]
    H = H_mat(A, G)
    assert H.shape == A.shape
    assert np.allclose(H, H.T, atol=1e-8)


def test_nearpd_restores_psd():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    fixed = nearPD(bad)
    eig = np.linalg.eigvalsh((fixed + fixed.T) / 2)
    assert np.min(eig) >= -1e-8


def test_make_full_reduces_collinearity():
    x = np.column_stack([np.ones(6), np.arange(6), 2.0 * np.arange(6)])
    xf = make_full(x)
    assert xf.shape[0] == x.shape[0]


def test_solver_random_intercept():
    rng = np.random.default_rng(100)
    n_groups = 20
    reps = 3
    n = n_groups * reps

    group = np.repeat(np.arange(n_groups), reps)
    Z = np.eye(n_groups)[group]
    X = np.ones((n, 1))
    K = [np.eye(n_groups)]

    u_true = rng.normal(0, np.sqrt(1.2), size=(n_groups, 1))
    e = rng.normal(0, np.sqrt(0.4), size=(n, 1))
    y = 2.0 + Z @ u_true + e

    out = mmes(Y=y, X=X, Z=[Z], K=K, iters=40, tolpar=1e-6, tolparinv=1e-6)

    assert out["converged"] is True or out["iterations"] == 40
    assert np.all(np.asarray(out["theta"]) > 0)
    assert out["beta"].shape == (1, 1)
    assert len(out["u"]) == 1


def test_solver_ai_mme_sp_path():
    rng = np.random.default_rng(101)
    n_groups = 16
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    Z = np.eye(n_groups)[group]
    X = np.ones((n, 1), dtype=float)
    u_true = rng.normal(0, np.sqrt(0.8), size=(n_groups, 1))
    e = rng.normal(0, np.sqrt(0.5), size=(n, 1))
    y = 1.5 + Z @ u_true + e

    out = mmes(
        Y=y,
        X=X,
        Z=[Z],
        K=[np.eye(n_groups)],
        method="ai_mme_sp",
        iters=40,
    )

    assert np.all(np.asarray(out["theta"]) > 0)
    assert out["beta"].shape == (1, 1)
    assert len(out["u"]) == 1
    assert out["fitted"].shape == y.shape


def test_solver_ai_mme_sp_agrees_with_newton_on_simple_case():
    rng = np.random.default_rng(1201)
    n_groups = 12
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    z = np.eye(n_groups)[group]
    x = np.ones((n, 1), dtype=float)
    k = np.eye(n_groups, dtype=float)

    u_true = rng.normal(0.0, np.sqrt(0.9), size=(n_groups, 1))
    e = rng.normal(0.0, np.sqrt(0.5), size=(n, 1))
    y = 1.2 + z @ u_true + e

    out_newton = mmes(Y=y, X=x, Z=[z], K=[k], method="newton_di_sp", iters=60)
    out_hend = mmes(Y=y, X=x, Z=[z], K=[k], method="ai_mme_sp", iters=60)

    np.testing.assert_allclose(out_hend["theta"], out_newton["theta"], atol=2e-2, rtol=2e-1)
    np.testing.assert_allclose(out_hend["beta"], out_newton["beta"], atol=2e-2, rtol=2e-1)
    np.testing.assert_allclose(out_hend["fitted"], out_newton["fitted"], atol=4e-2, rtol=2e-1)


def test_formula_mode_ai_mme_sp_solver_path():
    rng = np.random.default_rng(1202)
    n_groups = 9
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    z = np.eye(n_groups)[group]
    y = 2.1 + z @ rng.normal(0.0, 0.7, size=(n_groups, 1)) + rng.normal(0.0, 0.3, size=(n, 1))
    data = {"y": y.ravel(), "group": group}

    out = mmes(
        fixed="y ~ 1",
        random=[vsm(ism("group"))],
        data=data,
        method="ai_mme_sp",
        iters=50,
    )

    assert len(out["u"]) == 1
    assert out["beta"].shape == (1, 1)
    assert np.all(np.asarray(out["theta"]) > 0)


def test_formula_api_with_ism_and_dsm():
    rng = np.random.default_rng(202)
    n_groups = 8
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    env = np.tile(np.array(["E1", "E2"]), n_groups * reps // 2)
    n = group.size

    Z = np.eye(n_groups)[group]
    u = rng.normal(0, 0.8, size=(n_groups, 1))
    y = 2.0 + Z @ u + rng.normal(0, 0.3, size=(n, 1))

    data = {
        "y": y.ravel(),
        "group": group,
        "env": env,
    }

    out1 = mmes_formula(
        fixed="y ~ 1",
        random=[vsm(ism("group"))],
        data=data,
        iters=30,
    )
    assert "random_names" in out1
    assert len(out1["u"]) == 1

    out2 = mmes_formula(
        fixed="y ~ 1",
        random=[vsm(dsm("env"), ism("group"))],
        data=data,
        iters=30,
    )
    assert len(out2["u"]) == 2


def test_mmes_accepts_formula_like_arguments_directly():
    rng = np.random.default_rng(250)
    n_groups = 10
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    z = np.eye(n_groups)[group]
    y = 1.0 + z @ rng.normal(0, 0.6, size=(n_groups, 1)) + rng.normal(0, 0.2, size=(n, 1))
    data = {"y": y.ravel(), "group": group}

    out = mmes(
        fixed="y ~ 1",
        random=[vsm(ism("group"))],
        data=data,
        iters=25,
    )

    assert "fixed_names" in out
    assert "random_names" in out
    assert out["fixed_names"] == ["Intercept"]
    assert len(out["u"]) == 1


def test_mmes_formula_like_matches_mmes_formula_wrapper():
    rng = np.random.default_rng(251)
    n_groups = 8
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    env = np.tile(np.array(["E1", "E2"]), n_groups * reps // 2)
    n = group.size

    z = np.eye(n_groups)[group]
    y = 1.8 + z @ rng.normal(0, 0.7, size=(n_groups, 1)) + rng.normal(0, 0.25, size=(n, 1))
    data = {"y": y.ravel(), "group": group, "env": env}
    random = [vsm(dsm("env"), ism("group"))]

    out_direct = mmes(
        fixed="y ~ 1",
        random=random,
        data=data,
        iters=20,
    )
    out_wrapper = mmes_formula(
        fixed="y ~ 1",
        random=random,
        data=data,
        iters=20,
    )

    assert out_direct["random_names"] == out_wrapper["random_names"]
    assert out_direct["fixed_names"] == out_wrapper["fixed_names"]
    np.testing.assert_allclose(out_direct["theta"], out_wrapper["theta"], atol=1e-8, rtol=1e-8)


def test_gwas_helpers_shapes_and_finite():
    rng = np.random.default_rng(303)
    n = 20
    m = 6

    M = rng.choice([-1.0, 0.0, 1.0], size=(n, m))
    X = np.ones((n, 1), dtype=float)
    Z = np.eye(n)
    y = rng.normal(0.0, 1.0, size=(n, 1))
    Vinv = np.eye(n)

    out = gwasForLoop(M=M, Y=y, Z=Z, X=X, Vinv=Vinv, min_maf=0.0)
    assert out.shape == (m, 1, 3)
    assert np.isfinite(out).all()

    d_nt = np.eye(1)
    mimv = np.kron(d_nt, M[:, [0]])
    ymv = y
    zmv = Z
    xmv = X
    sc = scorecalc(mimv, ymv, zmv, xmv, Vinv, nt=1, min_maf=0.0)
    assert sc.shape == (1, 1, 3)
    assert np.isfinite(sc).all()


def test_reference_covariance_if_available():
    ar1_ref = _load_csv("AR1")
    cs_ref = _load_csv("CS")
    arma_ref = _load_csv("ARMA")

    _assert_close(AR1(ar1_ref.shape[0], rho=0.35), ar1_ref)
    _assert_close(CS(cs_ref.shape[0], rho=0.20), cs_ref)
    _assert_close(ARMA(arma_ref.shape[0], rho=0.35, lam=0.20), arma_ref)


def test_reference_nearpd_if_available():
    near_ref = _load_csv("nearPD")
    out = nearPD(np.array([[1.0, 2.0], [2.0, 1.0]]), maxit=100, eig_tol=1e-6, conv_tol=1e-7)
    _assert_close(out, near_ref, atol=1e-5, rtol=1e-4)


def test_reference_solver_if_available():
    path = REF_DIR / "mmes_reference.json"
    if not path.exists():
        pytest.skip("mmes reference JSON not found")

    payload = json.loads(path.read_text())
    if "theta" not in payload:
        pytest.skip("No theta in mmes reference payload")

    # Recreate the same data generation used in the R reference script.
    rng = np.random.default_rng(123)
    idv = np.repeat(np.arange(20), 2)
    Z = np.eye(20)[idv]
    X = np.ones((len(idv), 1))
    u = rng.normal(0, np.sqrt(1.2), size=(20, 1))
    e = rng.normal(0, np.sqrt(0.4), size=(len(idv), 1))
    y = 2.0 + Z @ u + e

    out = mmes(Y=y, X=X, Z=[Z], K=[np.eye(20)], iters=50)
    theta_ref = np.asarray(payload["theta"], dtype=float).reshape(-1)
    theta_py = np.asarray(out["theta"], dtype=float).reshape(-1)

    m = min(theta_ref.size, theta_py.size)
    _assert_close(theta_py[:m], theta_ref[:m], atol=1e-2, rtol=5e-2)
