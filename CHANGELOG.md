# Changelog

All notable changes to this project will be documented in this file.

## 0.2.0 - Unreleased

- Added a scikit-learn-like matrix-mode estimator interface via `MMESRegressor` with `fit`, `predict`, `score`, `get_params`, and `set_params`.
- Added estimator regression tests covering API behavior, parity with `mmes`, multivariate handling, and validation errors.
- Added documentation and notebook examples for `MMESRegressor` usage.
- Consolidated completed roadmap milestones from README into changelog-oriented release history.

## 0.1.0 - 2026-03-24

- First public Python package release of pysommer.
- Ported core computational routines from sommer for relationship matrices, covariance structures, near positive-definite projection, REML solvers, and mixed model fitting.
- Added matrix-based and formula-like entry points through `mmes`.
- Added tests against bundled reference data and project metadata for packaging and publication.
