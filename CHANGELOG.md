# Changelog

All notable changes to this project will be documented in this file.

## 0.2.0 - Unreleased

- Added a scikit-learn-like matrix-mode estimator interface via `MMESRegressor` with `fit`, `predict`, `score`, `get_params`, and `set_params`.
- Added a formula-mode estimator interface via `MMESFormulaRegressor` supporting R-like formula syntax (`"y ~ 1 + x"`) with `vsm`/`ism`/`dsm`/`usm` random effect declarations.
- Added formula-mode regression tests (8 tests) covering fit/predict/score, parameter protocol, parity with functional API, clone compatibility, multivariate responses, and input validation.
- Added notebook examples (cells 11-14) comparing formula estimator vs functional API, demonstrating sklearn.clone compatibility, and R package comparison framework.
- Both estimator classes achieve numerical parity with underlying functional APIs (`mmes` and `mmes_formula`).
- Added documentation and notebook examples for `MMESRegressor` and `MMESFormulaRegressor` usage.
- Consolidated completed roadmap milestones from README into changelog-oriented release history.

## 0.1.0 - 2026-03-24

- First public Python package release of pysommer.
- Ported core computational routines from sommer for relationship matrices, covariance structures, near positive-definite projection, REML solvers, and mixed model fitting.
- Added matrix-based and formula-like entry points through `mmes`.
- Added tests against bundled reference data and project metadata for packaging and publication.
