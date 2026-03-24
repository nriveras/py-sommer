import numpy as np
import pytest

from pysommer.mmes import mmes
from pysommer.sklearn import MMESRegressor, MMESFormulaRegressor
from pysommer.formula import ism, vsm


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


def test_mmes_regressor_pipeline_with_function_transformer_if_available():
    pipeline_mod = pytest.importorskip("sklearn.pipeline")
    preprocessing_mod = pytest.importorskip("sklearn.preprocessing")

    rng = np.random.default_rng(1952)
    n_groups = 10
    reps = 4
    group = np.repeat(np.arange(n_groups), reps)
    n = group.size

    x = rng.normal(0.0, 1.0, size=n)
    x_raw = x.reshape(-1, 1)
    z = np.eye(n_groups)[group]
    k = np.eye(n_groups, dtype=float)
    y = 1.2 + 0.6 * x[:, None] + z @ rng.normal(0.0, np.sqrt(0.5), size=(n_groups, 1)) + rng.normal(0.0, np.sqrt(0.25), size=(n, 1))

    def add_intercept(x_in):
        return np.column_stack([np.ones(x_in.shape[0]), x_in])

    pipe = pipeline_mod.Pipeline(
        [
            ("intercept", preprocessing_mod.FunctionTransformer(add_intercept)),
            ("mmes", MMESRegressor(Z=[z], K=[k], iters=25)),
        ]
    )

    pipe.fit(x_raw, y)
    y_pred = pipe.predict(x_raw)
    assert y_pred.shape == y.shape
    score = pipe.score(x_raw, y)
    assert np.isfinite(score)


def test_mmes_regressor_manual_kfold_clone_if_available():
    sklearn_base = pytest.importorskip("sklearn.base")
    model_selection_mod = pytest.importorskip("sklearn.model_selection")

    X, y, Z, K = _make_random_intercept_data(seed=1953)

    base = MMESRegressor(iters=20)
    kf = model_selection_mod.KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        z_train = Z[train_idx, :]

        est = sklearn_base.clone(base)
        est.fit(x_train, y_train, Z=[z_train], K=[K])
        scores.append(est.score(x_test, y_test))

    assert len(scores) == 3
    assert all(np.isfinite(s) for s in scores)


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


def test_mmes_regressor_predict_new_samples_with_random_design():
    X, y, Z, K = _make_random_intercept_data(seed=1906)
    est = MMESRegressor(Z=[Z], K=[K], iters=30)
    est.fit(X, y)

    X_new = np.ones((4, 1), dtype=float)
    Z_new = [np.eye(K.shape[0])[np.array([0, 3, 5, 9])]]

    y_new = est.predict(X_new, include_random=True, Z=Z_new)
    expected = X_new @ est.coef_ + Z_new[0] @ est.u_[0]

    np.testing.assert_allclose(y_new, expected, rtol=1e-8, atol=1e-8)


def test_mmes_regressor_predict_summary_for_new_samples():
    X, y, Z, K = _make_random_intercept_data(seed=1907)
    est = MMESRegressor(Z=[Z], K=[K], iters=30)
    est.fit(X, y)

    X_new = np.ones((3, 1), dtype=float)
    Z_new = [np.eye(K.shape[0])[np.array([1, 4, 8])]]

    summary = est.predict_summary(X_new, include_random=True, Z=Z_new)

    assert summary["predictions"].shape == (3, 1)
    assert summary["prediction_sd"].shape == (3, 1)
    assert np.all(summary["prediction_variance"] >= 0.0)


# ============================================================================
# Tests for MMESFormulaRegressor (formula-mode interface)
# ============================================================================


def _make_formula_data(seed: int = 2501):
    """Generate synthetic data for formula-mode tests."""
    rng = np.random.default_rng(seed)
    n_groups = 10
    reps = 4
    group_vec = np.repeat(np.arange(n_groups), reps)
    n = group_vec.size

    x_val = rng.normal(0.5, 0.8, size=n)
    u = rng.normal(0.0, np.sqrt(0.6), size=(n_groups, 1))
    e = rng.normal(0.0, np.sqrt(0.4), size=(n, 1))
    y_val = 2.0 + 0.8 * x_val[:, None] + u[group_vec] + e

    data = {
        "y": y_val.ravel(),
        "x": x_val,
        "group": group_vec,
    }
    return data


def test_mmes_formula_regressor_fit_predict():
    """Test basic formula estimator fit/predict workflow."""
    data = _make_formula_data()
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        iters=40,
    )
    est.fit(data)

    # Check fitted attributes
    assert est.coef_ is not None
    assert est.coef_.shape[1] == 1  # Single response
    assert est.n_features_in_ == 2  # Intercept and x
    assert est.fixed_names_ == ["Intercept", "x"]
    assert len(est.random_names_) == 1

    # Predict
    y_pred = est.predict(data)
    assert y_pred.shape == (len(data["y"]), 1)
    assert np.isfinite(y_pred).all()


def test_mmes_formula_regressor_score():
    """Test R-squared scoring for formula estimator."""
    data = _make_formula_data(seed=2502)
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        iters=35,
    )
    est.fit(data)

    score = est.score(data)
    assert 0.0 <= score <= 1.0
    assert np.isfinite(score)


def test_mmes_formula_regressor_get_set_params():
    """Test parameter protocol for formula estimator."""
    random_term = vsm(ism("group"))
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=random_term,
        iters=50,
        method="ai_mme_sp",
    )

    params = est.get_params()
    assert params["fixed"] == "y ~ 1 + x"
    assert params["iters"] == 50
    assert params["method"] == "ai_mme_sp"

    est.set_params(iters=25)
    assert est.iters == 25
    params_updated = est.get_params()
    assert params_updated["iters"] == 25


def test_mmes_formula_regressor_not_fitted_error():
    """Test that predict raises NotFittedError before fit."""
    from pysommer.sklearn import NotFittedError

    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
    )
    data = _make_formula_data()

    with pytest.raises(NotFittedError):
        est.predict(data)

    with pytest.raises(NotFittedError):
        est.score(data)


def test_mmes_formula_regressor_parity_with_functional():
    """Verify formula estimator produces same results as functional mmes_formula."""
    from pysommer.mmes import mmes_formula

    data = _make_formula_data(seed=2503)

    # Functional API
    result_func = mmes_formula(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        data=data,
        iters=40,
    )

    # Estimator API
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        iters=40,
    )
    est.fit(data)

    # Compare results
    np.testing.assert_allclose(est.coef_, result_func["beta"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(est.theta_, result_func["theta"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(est.fitted_, result_func["fitted"], rtol=1e-6, atol=1e-6)
    assert est.converged_ == result_func["converged"]


def test_mmes_formula_regressor_multivariate():
    """Test formula estimator with multiple response variables."""
    rng = np.random.default_rng(2504)
    n_groups = 8
    reps = 5
    group_vec = np.repeat(np.arange(n_groups), reps)
    n = group_vec.size

    x_val = rng.normal(0.0, 1.0, size=n)
    u = rng.normal(0.0, np.sqrt(0.5), size=(n_groups, 1))
    e1 = rng.normal(0.0, np.sqrt(0.3), size=(n, 1))
    e2 = rng.normal(0.0, np.sqrt(0.3), size=(n, 1))

    y1 = 1.5 + 0.5 * x_val[:, None] + u[group_vec] + e1
    y2 = 2.0 - 0.3 * x_val[:, None] + 0.8 * u[group_vec] + e2

    data = {
        "y1": y1.ravel(),
        "y2": y2.ravel(),
        "x": x_val,
        "group": group_vec,
    }

    est = MMESFormulaRegressor(
        fixed="y1 + y2 ~ 1 + x",
        random=vsm(ism("group")),
        iters=35,
    )
    est.fit(data)

    assert est.coef_.shape == (2, 2)  # 2 responses, 2 fixed effects each
    y_pred = est.predict(data)
    assert y_pred.shape == (n, 2)


def test_mmes_formula_regressor_input_validation():
    """Test input validation for formula estimator."""
    data = _make_formula_data()
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
    )

    # Missing column
    bad_data = {"y": data["y"], "x": data["x"]}  # Missing 'group'
    with pytest.raises(KeyError, match="group"):
        est.fit(bad_data)

    # Bad tolerance
    est_bad = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        tolpar=-1.0,
    )
    with pytest.raises(ValueError, match="tolpar"):
        est_bad.fit(data)


def test_mmes_formula_regressor_with_sklearn_clone_if_available():
    """Test sklearn clone compatibility if sklearn is available."""
    try:
        from sklearn.base import clone
    except ImportError:
        pytest.skip("sklearn not installed")

    data = _make_formula_data(seed=2505)
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        iters=30,
    )
    est.fit(data)

    # Clone should work with get_params/set_params
    est_cloned = clone(est)
    assert est_cloned.fixed == est.fixed
    assert est_cloned.iters == est.iters
    assert est_cloned.coef_ is None  # Not fitted


def test_mmes_formula_regressor_predict_include_random():
    """Test include_random flag in formula estimator predict."""
    data = _make_formula_data(seed=2506)
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        iters=30,
    )
    est.fit(data)

    # Predict with exact training data - with/without random
    y_fixed_only = est.predict(data, include_random=False)
    y_with_random = est.predict(data, include_random=True)

    # With exact training data match, include_random=True should return stored fitted values
    assert np.allclose(y_with_random, est.fitted_)

    # Fixed-only should be different from stored fitted (unless by chance)
    assert not np.allclose(y_fixed_only, est.fitted_)


def test_mmes_formula_regressor_predict_new_data_seen_and_unseen_levels():
    data = _make_formula_data(seed=2507)
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        iters=35,
    )
    est.fit(data)

    new_data = {
        "y": np.zeros(3, dtype=float),
        "x": np.array([data["x"][0], data["x"][1], 0.25], dtype=float),
        "group": np.array([0, 3, 999]),
    }

    pred = est.predict(new_data, include_random=True)
    fixed_only = est.predict(new_data, include_random=False)

    expected_seen = fixed_only[:2] + est.u_[0][[0, 3]]
    np.testing.assert_allclose(pred[:2], expected_seen, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(pred[2:], fixed_only[2:], rtol=1e-8, atol=1e-8)


def test_mmes_formula_regressor_predict_summary_reports_zeroed_rows():
    data = _make_formula_data(seed=2508)
    est = MMESFormulaRegressor(
        fixed="y ~ 1 + x",
        random=vsm(ism("group")),
        iters=35,
    )
    est.fit(data)

    new_data = {
        "y": np.zeros(4, dtype=float),
        "x": np.array([0.0, 0.5, -0.3, 1.2], dtype=float),
        "group": np.array([0, 5, 999, 1000]),
    }

    summary = est.predict_summary(new_data, include_random=True)

    assert summary["predictions"].shape == (4, 1)
    assert summary["prediction_sd"].shape == (4, 1)
    assert len(summary["random_effect_status"]) == 1
    assert summary["random_effect_status"][0]["matched_rows"] == 2
    assert summary["random_effect_status"][0]["zeroed_rows"] == 2
