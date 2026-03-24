# Publish pysommer to PyPI

## Phase 1: PyPI Publishing Readiness
- [x] Audit project structure and identify gaps
- [x] Create implementation plan and get user approval
- [x] Add LICENSE file (GPL >= 2)
- [x] Add `.gitignore`
- [x] Enrich [pyproject.toml](file:///Users/nico/Desktop/Projects/py-sommer/pyproject.toml) with full PyPI metadata
- [x] Add `__version__` to [__init__.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/__init__.py)
- [x] Add `CHANGELOG.md`
- [x] Add GitHub Actions CI workflow (tests + publish)
- [x] Clean up [pysommer.egg-info](file:///Users/nico/Desktop/Projects/py-sommer/pysommer.egg-info) and build artifacts

## Phase 2: Code Quality Improvements
- [ ] Review and improve module-level docstrings
- [x] Add `py.typed` marker for PEP 561 type-stub support
- [ ] Ensure all public functions have proper docstrings
- [x] Add linting/formatting config (ruff)

## Phase 3: Verification & Publishing
- [x] Run tests and verify passing
- [x] Build package and inspect contents
- [ ] Test install from built wheel
- [ ] Publish to TestPyPI first
- [ ] Publish to PyPI
