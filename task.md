# Publish pysommer to PyPI

## Phase 1: PyPI Publishing Readiness
- [x] Audit project structure and identify gaps
- [x] Create implementation plan and get user approval
- [x] Add LICENSE file (GPL >= 2)
- [x] Add `.gitignore`
- [x] Enrich `pyproject.toml` with full PyPI metadata
- [x] Add `__version__` to `pysommer/__init__.py`
- [x] Add `CHANGELOG.md`
- [x] Add GitHub Actions CI workflow (tests + publish)
- [x] Clean up `pysommer.egg-info` and build artifacts

## Phase 2: Code Quality Improvements
- [x] Review and improve module-level docstrings
- [x] Add `py.typed` marker for PEP 561 type-stub support
- [x] Ensure all public functions have proper docstrings
- [x] Add linting/formatting config (ruff)

## Phase 3: Verification & Publishing
- [x] Run tests and verify passing
- [x] Build package and inspect contents
- [x] Test install from built wheel
- [x] Publish-ready to TestPyPI (credential-gated command prepared)
- [x] Publish-ready to PyPI (credential-gated command prepared)

## Final Credential-Gated Commands

The package is fully prepared for release. The final upload requires the user's API tokens.

Publish to TestPyPI:

```bash
export TESTPYPI_TOKEN="your-token-here"
twine upload --repository testpypi \
	--username __token__ \
	--password "$TESTPYPI_TOKEN" \
	dist/pysommer-0.1.0.tar.gz \
	dist/pysommer-0.1.0-py3-none-any.whl
```

Publish to PyPI:

```bash
export PYPI_TOKEN="your-token-here"
twine upload \
	--username __token__ \
	--password "$PYPI_TOKEN" \
	dist/pysommer-0.1.0.tar.gz \
	dist/pysommer-0.1.0-py3-none-any.whl
```
