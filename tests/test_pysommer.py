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
    usm,
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


def test_matrix_mode_multivariate_independent_traits():
    rng = np.random.default_rng(1301)
    n_groups = 10
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    z = np.eye(n_groups)[group]
    x = np.ones((n, 1), dtype=float)
    k = np.eye(n_groups, dtype=float)

    u1 = rng.normal(0.0, np.sqrt(0.7), size=(n_groups, 1))
    u2 = rng.normal(0.0, np.sqrt(1.1), size=(n_groups, 1))
    e1 = rng.normal(0.0, np.sqrt(0.3), size=(n, 1))
    e2 = rng.normal(0.0, np.sqrt(0.4), size=(n, 1))
    y = np.hstack([1.0 + z @ u1 + e1, 2.0 + z @ u2 + e2])

    out = mmes(Y=y, X=x, Z=[z], K=[k], iters=50)

    assert out["beta"].shape == (1, 2)
    assert out["theta"].shape == (2, 2)
    assert out["fitted"].shape == y.shape
    assert out["residuals"].shape == y.shape
    assert len(out["u"]) == 1
    assert out["u"][0].shape == (n_groups, 2)


def test_formula_mode_multivariate_independent_traits():
    rng = np.random.default_rng(1302)
    n_groups = 9
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    z = np.eye(n_groups)[group]

    y1 = 1.5 + z @ rng.normal(0.0, 0.8, size=(n_groups, 1)) + rng.normal(0.0, 0.3, size=(group.size, 1))
    y2 = 0.5 + z @ rng.normal(0.0, 0.6, size=(n_groups, 1)) + rng.normal(0.0, 0.35, size=(group.size, 1))
    data = {"y1": y1.ravel(), "y2": y2.ravel(), "group": group}

    out = mmes(
        fixed="y1 + y2 ~ 1",
        random=[vsm(ism("group"))],
        data=data,
        iters=40,
    )

    assert out["beta"].shape == (1, 2)
    assert out["fitted"].shape == (group.size, 2)
    assert out["fixed_names"] == ["Intercept"]
    assert len(out["u"]) == 1
    assert out["u"][0].shape == (n_groups, 2)


def test_formula_usm_with_custom_covariance_runs():
    rng = np.random.default_rng(1303)
    n_groups = 8
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    env = np.tile(np.array(["E1", "E2"]), group.size // 2)

    z = np.eye(n_groups)[group]
    y = 1.8 + z @ rng.normal(0.0, 0.7, size=(n_groups, 1)) + rng.normal(0.0, 0.25, size=(group.size, 1))
    data = {"y": y.ravel(), "group": group, "env": env}
    cu = AR1(2, rho=0.35)

    out = mmes_formula(
        fixed="y ~ 1",
        random=[vsm(usm("env"), ism("group"), Cu=cu)],
        data=data,
        iters=35,
    )

    assert len(out["u"]) == 1
    assert out["u"][0].shape == (n_groups * 2, 1)
    assert "usm(env)xism(group)" in out["random_names"]


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


def test_gwas_helpers_shapes_and_finite_univariate():
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

    sc = scorecalc(Mimv=M[:, [0]], Ymv=y, Zmv=Z, Xmv=X, Vinv=Vinv, nt=1, min_maf=0.0)
    assert sc.shape == (1, 1, 3)
    assert np.isfinite(sc).all()


def test_gwas_multivariate_matches_markerwise_scorecalc():
    rng = np.random.default_rng(304)
    n = 30
    n_levels = 10
    m = 4
    nt = 2

    group = np.repeat(np.arange(n_levels), n // n_levels)
    Z = np.eye(n_levels)[group]
    X = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])

    M = rng.choice([-1.0, 0.0, 1.0], size=(n_levels, m))
    Y = rng.normal(0.0, 1.0, size=(n, nt))
    Vinv = np.eye(n * nt)

    out = gwasForLoop(M=M, Y=Y, Z=Z, X=X, Vinv=Vinv, min_maf=0.0)
    assert out.shape == (m, nt, 3)
    assert np.isfinite(out).all()

    d_nt = np.eye(nt, dtype=float)
    y_mv = Y.T.reshape(-1, 1, order="F")
    z_mv = np.kron(Z, d_nt)
    x_mv = np.kron(X, d_nt)

    expected = np.zeros_like(out)
    for i in range(m):
        mi_mv = np.kron(M[:, [i]], d_nt)
        sc_i = scorecalc(
            Mimv=mi_mv,
            Ymv=y_mv,
            Zmv=z_mv,
            Xmv=x_mv,
            Vinv=Vinv,
            nt=nt,
            min_maf=0.0,
        )
        expected[i, :, :] = sc_i[0, :, :]

    np.testing.assert_allclose(out, expected, atol=1e-10, rtol=1e-10)


def test_gwas_min_maf_filters_low_frequency_markers():
    rng = np.random.default_rng(305)
    n = 16

    Z = np.eye(n)
    X = np.ones((n, 1), dtype=float)
    Y = rng.normal(0.0, 1.0, size=(n, 1))
    Vinv = np.eye(n)

    # Marker 0 is monomorphic at -1 coding -> MAF 0; others are polymorphic.
    M = np.column_stack(
        [
            -np.ones(n),
            np.tile(np.array([-1.0, 1.0]), n // 2),
            np.tile(np.array([-1.0, 0.0, 1.0, 0.0]), n // 4),
        ]
    )

    out = gwasForLoop(M=M, Y=Y, Z=Z, X=X, Vinv=Vinv, min_maf=0.05)

    assert np.allclose(out[0, :, :], 0.0)
    assert np.isfinite(out[1:, :, :]).all()
    assert not np.allclose(out[1, :, :], 0.0)


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


def test_edge_case_small_sample():
    """Test stability with minimal sample size (n=5)."""
    rng = np.random.default_rng(501)
    n = 5
    n_groups = 2
    group = np.array([0, 0, 0, 1, 1])
    Z = np.eye(n_groups)[group]
    X = np.ones((n, 1), dtype=float)
    y = rng.normal(0.0, 1.0, size=(n, 1))

    out = mmes(Y=y, X=X, Z=[Z], K=[np.eye(n_groups)], iters=20)
    assert out["converged"] or out["iterations"] == 20
    assert np.all(np.asarray(out["theta"]) > 0)


def test_edge_case_large_variance_ratio():
    """Test stability with extreme variance ratios."""
    rng = np.random.default_rng(502)
    n = 30
    n_groups = 10
    group = np.repeat(np.arange(n_groups), 3)
    Z = np.eye(n_groups)[group]
    X = np.ones((n, 1), dtype=float)

    # Very large random effect variance vs residual
    u_large = rng.normal(0.0, 100.0, size=(n_groups, 1))
    e_small = rng.normal(0.0, 0.01, size=(n, 1))
    y = 1.0 + Z @ u_large + e_small

    out = mmes(Y=y, X=X, Z=[Z], K=[np.eye(n_groups)], iters=40)
    assert np.all(np.isfinite(np.asarray(out["theta"])))


def test_edge_case_perfect_collinearity_in_random_effects():
    """Test behavior with perfectly replicated random effect levels."""
    rng = np.random.default_rng(503)
    n = 20
    group = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    Z = np.eye(2)[group]
    X = np.ones((n, 1), dtype=float)
    y = 2.0 + rng.normal(0.0, 0.3, size=(n, 1))

    out = mmes(Y=y, X=X, Z=[Z], K=[np.eye(2)], iters=30)
    assert np.isfinite(out["theta"]).all()
    assert np.isfinite(out["beta"]).all()


def test_edge_case_identity_relationship_matrix():
    """Test with identity relationship matrix (unrelated individuals)."""
    rng = np.random.default_rng(504)
    n = 25
    Z = np.eye(n)
    X = np.ones((n, 1), dtype=float)
    y = rng.normal(2.0, 1.0, size=(n, 1))

    out = mmes(Y=y, X=X, Z=[Z], K=[np.eye(n)], iters=20)
    assert np.isfinite(out["theta"]).all()


def test_edge_case_balanced_vs_unbalanced_design():
    """Compare solver behavior on balanced vs unbalanced designs."""
    rng = np.random.default_rng(505)
    n_groups = 5

    # Balanced: 4 reps per group
    group_bal = np.repeat(np.arange(n_groups), 4)
    Z_bal = np.eye(n_groups)[group_bal]
    y_bal = 1.5 + Z_bal @ rng.normal(0.0, 0.8, size=(n_groups, 1)) + rng.normal(0.0, 0.3, size=(group_bal.size, 1))

    # Unbalanced: varying reps per group
    group_unbal = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 4])
    Z_unbal = np.eye(n_groups)[group_unbal]
    y_unbal = 1.5 + Z_unbal @ rng.normal(0.0, 0.8, size=(n_groups, 1)) + rng.normal(0.0, 0.3, size=(group_unbal.size, 1))

    out_bal = mmes(Y=y_bal.ravel(), X=np.ones((group_bal.size, 1)), Z=[Z_bal], K=[np.eye(n_groups)], iters=30)
    out_unbal = mmes(Y=y_unbal.ravel(), X=np.ones((group_unbal.size, 1)), Z=[Z_unbal], K=[np.eye(n_groups)], iters=30)

    assert np.isfinite(out_bal["theta"]).all()
    assert np.isfinite(out_unbal["theta"]).all()


def test_solver_consistency_across_methods():
    """Regression test: both solvers should produce similar results on same data."""
    rng = np.random.default_rng(506)
    n_groups = 12
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    Z = np.eye(n_groups)[group]
    X = np.ones((n, 1), dtype=float)
    u_true = rng.normal(0.0, np.sqrt(0.7), size=(n_groups, 1))
    e = rng.normal(0.0, np.sqrt(0.3), size=(n, 1))
    y = 1.5 + Z @ u_true + e

    out_newton = mmes(Y=y, X=X, Z=[Z], K=[np.eye(n_groups)], method="newton_di_sp", iters=50)
    out_ai = mmes(Y=y, X=X, Z=[Z], K=[np.eye(n_groups)], method="ai_mme_sp", iters=50)

    np.testing.assert_allclose(out_newton["beta"], out_ai["beta"], atol=1e-1, rtol=1e-1)
    np.testing.assert_allclose(out_newton["theta"], out_ai["theta"], atol=1e-1, rtol=1e-1)


def test_matrix_vs_formula_interface_consistency():
    """Regression test: matrix and formula interfaces should give identical results."""
    rng = np.random.default_rng(507)
    n_groups = 8
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    Z = np.eye(n_groups)[group]
    y = 2.0 + Z @ rng.normal(0.0, 0.8, size=(n_groups, 1)) + rng.normal(0.0, 0.3, size=(group.size, 1))

    # Matrix interface
    X = np.ones((group.size, 1), dtype=float)
    out_matrix = mmes(Y=y.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], iters=35)

    # Formula interface
    data = {"y": y.ravel(), "group": group}
    out_formula = mmes(fixed="y ~ 1", random=[vsm(ism("group"))], data=data, iters=35)

    np.testing.assert_allclose(out_matrix["beta"], out_formula["beta"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(out_matrix["theta"], out_formula["theta"], atol=1e-8, rtol=1e-8)


def test_multivariate_independent_vs_sequential():
    """Regression test: multivariate fit should match sequential univariate fits."""
    rng = np.random.default_rng(508)
    n_groups = 6
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    Z = np.eye(n_groups)[group]
    X = np.ones((group.size, 1), dtype=float)

    y1 = 1.0 + Z @ rng.normal(0.0, 0.6, size=(n_groups, 1)) + rng.normal(0.0, 0.2, size=(group.size, 1))
    y2 = 2.0 + Z @ rng.normal(0.0, 0.8, size=(n_groups, 1)) + rng.normal(0.0, 0.25, size=(group.size, 1))

    # Multivariate fit
    Y_mv = np.hstack([y1, y2])
    out_mv = mmes(Y=Y_mv, X=X, Z=[Z], K=[np.eye(n_groups)], iters=35)

    # Sequential univariate fits
    out_y1 = mmes(Y=y1.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], iters=35)
    out_y2 = mmes(Y=y2.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], iters=35)

    np.testing.assert_allclose(out_mv["beta"][:, 0], out_y1["beta"].ravel(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(out_mv["beta"][:, 1], out_y2["beta"].ravel(), atol=1e-6, rtol=1e-6)


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


# ============================================================================
# Step 6: End-to-End Examples (Multiple Random Terms, Custom Matrices, Predictions)
# ============================================================================


def test_multiple_random_terms_two_random_effects():
    """Test model with multiple random effects terms (e.g., genetic + spatial)."""
    rng = np.random.default_rng(601)
    n_fam = 6
    n_spatial = 3
    reps = 3
    n_obs = n_fam * reps

    fam = np.repeat(np.arange(n_fam), reps)
    spatial = np.array([0, 1, 2] * n_fam)  # Cycle through 3 spatial locations

    Z_fam = np.eye(n_fam)[fam]
    Z_spatial = np.eye(n_spatial)[spatial]

    X = np.ones((n_obs, 1))
    K_fam = np.eye(n_fam)
    K_spatial = np.eye(n_spatial)

    u_fam = rng.normal(0.0, np.sqrt(0.5), size=(n_fam, 1))
    u_spatial = rng.normal(0.0, np.sqrt(0.3), size=(n_spatial, 1))
    e = rng.normal(0.0, np.sqrt(0.2), size=(n_obs, 1))
    y = 1.5 + Z_fam @ u_fam + Z_spatial @ u_spatial + e

    out = mmes(Y=y.ravel(), X=X, Z=[Z_fam, Z_spatial], K=[K_fam, K_spatial], iters=40)

    # Verify convergence and structure
    assert out["converged"] or (out["status"] == 0)
    # Theta contains variance for 2 random effects + 1 residual = 3 components
    assert out["theta"].size >= 2
    assert out["u"][0].shape == (n_fam, 1)
    assert out["u"][1].shape == (n_spatial, 1)
    assert np.isfinite(out["beta"]).all()
    assert np.isfinite(out["theta"]).all()


def test_multiple_random_terms_three_random_effects():
    """Test model with three random effects (genetic, maternal, spatial)."""
    rng = np.random.default_rng(602)
    n_ind = 8
    n_mat = 4
    n_loc = 2
    n_obs = n_ind * n_mat * n_loc

    # Create full factorial design for simplicity
    ind = np.repeat(np.arange(n_ind), n_mat * n_loc)
    mat = np.tile(np.repeat(np.arange(n_mat), n_loc), n_ind)
    loc = np.tile(np.arange(n_loc), n_ind * n_mat)

    Z_ind = np.eye(n_ind)[ind]
    Z_mat = np.eye(n_mat)[mat]
    Z_loc = np.eye(n_loc)[loc]

    X = np.ones((n_obs, 1))
    K_ind = np.eye(n_ind)
    K_mat = np.eye(n_mat)
    K_loc = np.eye(n_loc)

    u_ind = rng.normal(0.0, np.sqrt(0.4), size=(n_ind, 1))
    u_mat = rng.normal(0.0, np.sqrt(0.2), size=(n_mat, 1))
    u_loc = rng.normal(0.0, np.sqrt(0.15), size=(n_loc, 1))
    e = rng.normal(0.0, np.sqrt(0.25), size=(n_obs, 1))
    y = 2.0 + Z_ind @ u_ind + Z_mat @ u_mat + Z_loc @ u_loc + e

    out = mmes(
        Y=y.ravel(),
        X=X,
        Z=[Z_ind, Z_mat, Z_loc],
        K=[K_ind, K_mat, K_loc],
        iters=40,
    )

    assert out["converged"] or (out["status"] == 0)
    assert len(out["u"]) == 3
    assert out["u"][0].shape == (n_ind, 1)
    assert out["u"][1].shape == (n_mat, 1)
    assert out["u"][2].shape == (n_loc, 1)


def test_custom_relationship_matrix_identity():
    """Test with an explicit identity relationship matrix (should match default)."""
    rng = np.random.default_rng(603)
    n_groups = 10
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n_obs = group.size

    Z = np.eye(n_groups)[group]
    X = np.ones((n_obs, 1))

    y = 1.5 + Z @ rng.normal(0.0, np.sqrt(0.6), size=(n_groups, 1)) + rng.normal(
        0.0, np.sqrt(0.4), size=(n_obs, 1)
    )

    # With identity matrix (default)
    out_identity = mmes(Y=y.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], iters=35)

    # With explicit identity
    K_explicit = np.eye(n_groups, dtype=float)
    out_explicit = mmes(Y=y.ravel(), X=X, Z=[Z], K=[K_explicit], iters=35)

    # Both should give same results
    np.testing.assert_allclose(out_identity["beta"], out_explicit["beta"], rtol=1e-6)
    np.testing.assert_allclose(out_identity["theta"], out_explicit["theta"], rtol=1e-6)


def test_custom_relationship_matrix_compound_symmetry():
    """Test with compound symmetry (CS) relationship matrix."""
    rng = np.random.default_rng(604)
    n_groups = 8
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    n_obs = group.size

    Z = np.eye(n_groups)[group]
    X = np.ones((n_obs, 1))

    # Compound symmetry relationship matrix: correlation rho between all pairs
    rho = 0.3
    K_cs = CS(n_groups, rho=rho)

    y = 1.2 + Z @ rng.normal(0.0, np.sqrt(0.5), size=(n_groups, 1)) + rng.normal(
        0.0, np.sqrt(0.3), size=(n_obs, 1)
    )

    out_cs = mmes(Y=y.ravel(), X=X, Z=[Z], K=[K_cs], iters=35)

    # Should converge without issues
    assert out_cs["converged"] or (out_cs["status"] == 0)
    assert np.isfinite(out_cs["beta"]).all()
    assert np.isfinite(out_cs["theta"]).all()


def test_custom_relationship_matrix_ar1():
    """Test with AR1 (autoregressive-1) relationship matrix."""
    rng = np.random.default_rng(605)
    n_times = 12
    n_reps = 3
    n_obs = n_times * n_reps

    time = np.repeat(np.arange(n_times), n_reps)
    Z = np.eye(n_times)[time]
    X = np.ones((n_obs, 1))

    # AR1 relationship matrix with correlation phi
    phi = 0.8
    K_ar1 = AR1(n_times, rho=phi)

    y = (
        2.0
        + Z @ rng.normal(0.0, np.sqrt(0.7), size=(n_times, 1))
        + rng.normal(0.0, np.sqrt(0.4), size=(n_obs, 1))
    )

    out_ar1 = mmes(Y=y.ravel(), X=X, Z=[Z], K=[K_ar1], iters=40)

    assert out_ar1["converged"] or (out_ar1["status"] == 0)
    assert np.isfinite(out_ar1["beta"]).all()
    assert np.isfinite(out_ar1["theta"]).all()


def test_prediction_fitted_values_manual():
    """Test manual prediction using fitted = X @ beta + Z @ u."""
    rng = np.random.default_rng(606)
    n_groups = 12
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n_obs = group.size

    Z = np.eye(n_groups)[group]
    X = np.ones((n_obs, 1))

    y = 1.5 + Z @ rng.normal(0.0, np.sqrt(0.6), size=(n_groups, 1)) + rng.normal(
        0.0, np.sqrt(0.3), size=(n_obs, 1)
    )

    out = mmes(Y=y.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], iters=35)

    # Manual prediction
    beta = out["beta"].ravel()
    u = out["u"][0].ravel()
    yhat_manual = X.ravel() * beta[0] + (Z @ u.reshape(-1, 1)).ravel()

    # Compare with model's fitted values
    yhat_model = out["fitted"].ravel()

    np.testing.assert_allclose(yhat_manual, yhat_model, rtol=1e-8)


def test_prediction_residuals_manual():
    """Test that residuals = y - fitted."""
    rng = np.random.default_rng(607)
    n_groups = 10
    reps = 2
    group = np.repeat(np.arange(n_groups), reps)
    n_obs = group.size

    Z = np.eye(n_groups)[group]
    X = np.ones((n_obs, 1))
    y = 2.0 + Z @ rng.normal(0.0, np.sqrt(0.5), size=(n_groups, 1)) + rng.normal(
        0.0, np.sqrt(0.4), size=(n_obs, 1)
    )

    out = mmes(Y=y.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], iters=30)

    # Manual residuals
    resid_manual = y.ravel() - out["fitted"].ravel()

    # Compare with model's residuals
    resid_model = out["residuals"].ravel()

    np.testing.assert_allclose(resid_manual, resid_model, rtol=1e-8)


def test_prediction_for_new_levels():
    """Test prediction for new group levels using random effects structure."""
    rng = np.random.default_rng(608)
    n_groups_train = 8
    reps = 4
    group_train = np.repeat(np.arange(n_groups_train), reps)
    n_train = group_train.size

    Z_train = np.eye(n_groups_train)[group_train]
    X_train = np.ones((n_train, 1))

    y_train = (
        1.5
        + Z_train @ rng.normal(0.0, np.sqrt(0.6), size=(n_groups_train, 1))
        + rng.normal(0.0, np.sqrt(0.3), size=(n_train, 1))
    )

    out = mmes(
        Y=y_train.ravel(), X=X_train, Z=[Z_train], K=[np.eye(n_groups_train)], iters=35
    )

    # For new groups not in training set, predict using fixed effects only
    n_new_groups = 3
    n_new = n_new_groups * reps
    X_new = np.ones((n_new, 1))

    # Prediction for new groups: use fixed effects (intercept)
    yhat_new = (X_new @ out["beta"]).ravel()

    # Should be close to the fixed effect value
    beta_intercept = out["beta"][0, 0]
    np.testing.assert_allclose(yhat_new, np.ones(n_new) * beta_intercept, rtol=1e-10)


def test_prediction_with_multiple_random_terms():
    """Test prediction in model with multiple random effects."""
    rng = np.random.default_rng(609)
    n_fam = 6
    n_spatial = 3
    reps = 3
    n_obs = n_fam * reps

    fam = np.repeat(np.arange(n_fam), reps)
    spatial = np.array([0, 1, 2] * n_fam)  # Cycle through 3 spatial locations

    Z_fam = np.eye(n_fam)[fam]
    Z_spatial = np.eye(n_spatial)[spatial]
    X = np.ones((n_obs, 1))

    u_fam = rng.normal(0.0, np.sqrt(0.5), size=(n_fam, 1))
    u_spatial = rng.normal(0.0, np.sqrt(0.3), size=(n_spatial, 1))
    e = rng.normal(0.0, np.sqrt(0.2), size=(n_obs, 1))
    y = 1.5 + Z_fam @ u_fam + Z_spatial @ u_spatial + e

    out = mmes(Y=y.ravel(), X=X, Z=[Z_fam, Z_spatial], K=[np.eye(n_fam), np.eye(n_spatial)], iters=40)

    # Manual prediction with multiple terms
    beta = out["beta"].ravel()
    u_fam_est = out["u"][0].ravel()
    u_spatial_est = out["u"][1].ravel()

    yhat_manual = (
        X.ravel() * beta[0]
        + (Z_fam @ u_fam_est.reshape(-1, 1)).ravel()
        + (Z_spatial @ u_spatial_est.reshape(-1, 1)).ravel()
    )

    yhat_model = out["fitted"].ravel()

    np.testing.assert_allclose(yhat_manual, yhat_model, rtol=1e-8)
