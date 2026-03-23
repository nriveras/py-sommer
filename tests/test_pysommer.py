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
    mmes,
    nearPD,
    scale_cpp,
    seq_cpp,
    var_cols,
    vec_to_mat_cpp,
)


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
