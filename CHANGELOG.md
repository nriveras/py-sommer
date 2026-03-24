# Changelog

All notable changes to this project will be documented in this file.

## 0.2.0 - Unreleased

- Added `sommer_version` metadata to `tests/reference_data/mmes_reference.json` export via `tests/test_sommer_reference.R` for explicit upstream-version traceability.
- Added public prediction helpers `predict_mmes` and `summarize_predictions` for matrix-mode fits, including conditional prediction intervals based on residual variance and random-effect PEV contributions.
- Extended `MMESRegressor.predict()` to accept aligned `Z` terms for out-of-sample random-effect prediction and added `predict_summary()`.
- Extended `MMESFormulaRegressor` so known factor levels reuse fitted random effects on new data while unseen levels fall back to fixed-only prediction, and added `predict_summary()` with per-term matched/zeroed row counts.
- Added regression tests covering public prediction helpers, matrix estimator out-of-sample random-effect prediction, formula-mode seen/unseen level handling, and uncertainty summaries.
- Added a scikit-learn-like matrix-mode estimator interface via `MMESRegressor` with `fit`, `predict`, `score`, `get_params`, and `set_params`.
- Added a formula-mode estimator interface via `MMESFormulaRegressor` supporting R-like formula syntax (`"y ~ 1 + x"`) with `vsm`/`ism`/`dsm`/`usm` random effect declarations.
- Added formula-mode regression tests (9 tests) covering fit/predict/score, parameter protocol, parity with functional API, clone compatibility, multivariate responses, input validation, and `include_random` prediction behavior.
- Added notebook examples in `notebooks/start_implementation.ipynb` comparing formula estimator vs functional API, demonstrating sklearn.clone compatibility, and R package comparison framework.
- Both estimator classes achieve numerical parity with underlying functional APIs (`mmes` and `mmes_formula`).
- Added documentation and notebook examples for `MMESRegressor` and `MMESFormulaRegressor` usage.
- Added richer sklearn interoperability examples for pipeline composition and manual KFold/clone cross-validation patterns with mixed-model matrix handling notes.
- Consolidated completed roadmap milestones from README into changelog-oriented release history.

## 0.1.0 - 2026-03-24

- First public Python package release of pysommer.
- Ported core computational routines from sommer for relationship matrices, covariance structures, near positive-definite projection, REML solvers, and mixed model fitting.
- Added matrix-based and formula-like entry points through `mmes`.
- Added tests against bundled reference data and project metadata for packaging and publication.
