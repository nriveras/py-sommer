import numpy as np
import pytest

from pysommer.mmes import mmes
from pysommer.sklearn import MMESRegressor


def _make_random_intercept_data(seed: int = 1901):
    rng = np.random.default_rng(seed)
    n_groups = 12
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    X = np.ones((n, 1), dtype=float)
    Z = np.eye(n_groups)[group]
    K = np.eye(n_groups, dtype=float)

    u = rng.normal(0.0, np.sqrt(0.7), size=(n_groups, 1))
    e = rng.normal(0.0, np.sqrt(0.3), size=(n, 1))
    y = 1.7 + Z @ u + e
    return X, y, Z, K


def test_mmes_regressor_fit_predict_score():
    X, y, Z, K = _make_random_intercept_data()
    est = MMESRegressor(Z=[Z], K=[K], iters=40)
    est.fit(X, y)

    y_pred = est.predict(X)
    assert y_pred.shape == y.shape

    score = est.score(X, y)
    assert np.isfinite(score)


def test_mmes_regressor_fitted_attributes():
    X, y, Z, K = _make_random_intercept_data()
    est = MMESRegressor(Z=[Z], K=[K], iters=35)
    est.fit(X, y)

    assert hasattr(est, "n_features_in_")
    assert est.n_features_in_ == X.shape[1]
    assert est.coef_.shape == (1, 1)
    assert est.theta_.shape[0] >= 2
    assert len(est.u_) == 1
    assert est.fitted_.shape == y.shape
    assert est.residuals_.shape == y.shape


def test_mmes_regressor_fit_requires_zk():
    X, y, Z, K = _make_random_intercept_data()

    est = MMESRegressor(iters=20)
    with pytest.raises(ValueError, match="Z and K"):
        est.fit(X, y)

    est.fit(X, y, Z=[Z], K=[K])
    assert est.coef_.shape == (1, 1)


def test_mmes_regressor_get_set_params_roundtrip():
    X, y, Z, K = _make_random_intercept_data()
    est = MMESRegressor(Z=[Z], K=[K], iters=25, method="newton_di_sp")

    params = est.get_params()
    assert params["iters"] == 25
    assert params["method"] == "newton_di_sp"

    est.set_params(iters=35, method="ai_mme_sp")
    est.fit(X, y)
    assert est.get_params()["iters"] == 35
    assert est.get_params()["method"] == "ai_mme_sp"


def test_mmes_regressor_predict_before_fit_raises():
    X, _, _, _ = _make_random_intercept_data()
    est = MMESRegressor()
    with pytest.raises(Exception):
        est.predict(X)


def test_mmes_regressor_with_sklearn_clone_if_available():
    sklearn = pytest.importorskip("sklearn.base")
    X, y, Z, K = _make_random_intercept_data()

    est = MMESRegressor(Z=[Z], K=[K], iters=30)
    est2 = sklearn.clone(est)

    assert isinstance(est2, MMESRegressor)
    est2.fit(X, y)
    assert est2.coef_.shape == (1, 1)


def test_mmes_regressor_parity_with_functional_mmes():
    X, y, Z, K = _make_random_intercept_data(seed=1902)

    out_fun = mmes(Y=y, X=X, Z=[Z], K=[K], method="newton_di_sp", iters=50)
    est = MMESRegressor(Z=[Z], K=[K], method="newton_di_sp", iters=50)
    est.fit(X, y)

    np.testing.assert_allclose(out_fun["beta"], est.coef_, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.asarray(out_fun["theta"]).reshape(-1), np.asarray(est.theta_).reshape(-1), rtol=1e-8, atol=1e-8)

    y_fit_est = est.predict(X, include_random=True)
    np.testing.assert_allclose(out_fun["fitted"], y_fit_est, rtol=1e-8, atol=1e-8)


def test_mmes_regressor_multivariate_traits():
    rng = np.random.default_rng(1903)
    n_groups = 8
    reps = 3
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    X = np.ones((n, 1), dtype=float)
    Z = np.eye(n_groups)[group]
    K = np.eye(n_groups, dtype=float)

    y1 = 1.0 + Z @ rng.normal(0.0, 0.5, size=(n_groups, 1)) + rng.normal(0.0, 0.2, size=(n, 1))
    y2 = 2.0 + Z @ rng.normal(0.0, 0.8, size=(n_groups, 1)) + rng.normal(0.0, 0.25, size=(n, 1))
    Y = np.hstack([y1, y2])

    est = MMESRegressor(Z=[Z], K=[K], iters=40)
    est.fit(X, Y)

    assert est.coef_.shape == (1, 2)
    assert est.theta_.shape == (2, 2)
    assert est.fitted_.shape == (n, 2)
    assert est.u_[0].shape == (n_groups, 2)


def test_mmes_regressor_input_validation():
    X, y, Z, K = _make_random_intercept_data(seed=1904)
    est = MMESRegressor(Z=[Z], K=[K], iters=30)

    with pytest.raises(ValueError, match="same number of rows"):
        est.fit(X[:-1], y)

    y_bad = y.copy()
    y_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        est.fit(X, y_bad)

    with pytest.raises(ValueError, match="method"):
        MMESRegressor(Z=[Z], K=[K], method="bad_method").fit(X, y)


def test_mmes_regressor_predict_new_samples_fixed_only():
    X, y, Z, K = _make_random_intercept_data(seed=1905)
    est = MMESRegressor(Z=[Z], K=[K], iters=30)
    est.fit(X, y)

    X_new = np.ones((5, X.shape[1]), dtype=float)
    y_new = est.predict(X_new)

    assert y_new.shape == (5, y.shape[1])
    np.testing.assert_allclose(y_new.ravel(), np.ones(5) * est.coef_[0, 0], rtol=1e-8, atol=1e-8)
