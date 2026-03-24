# Publishing pysommer to PyPI

This document provides step-by-step instructions to publish pysommer to TestPyPI and PyPI.

## Prerequisites

Ensure you have:
- PyPI API token (starts with `pypi-`)
- TestPyPI API token (starts with `pypi-`)
- All tests passing: `uv run pytest -q` (58 passing)
- Clean build: `uv build` creates `dist/pysommer-0.1.0-py3-none-any.whl` and `dist/pysommer-0.1.0.tar.gz`
- Metadata validation: `uv run --with twine twine check dist/*` (both PASSED)

## Current Status (Completed)

✅ **Phase 1: PyPI Publishing Readiness**
- Project structure audited and gaps identified
- Implementation plan created and approved
- LICENSE added (GPL >= 2)
- `.gitignore` configured
- `pyproject.toml` enriched with PyPI metadata
- `__version__` added to `__init__.py`
- `CHANGELOG.md` created and updated
- GitHub Actions CI workflow added
- Build artifacts cleaned

✅ **Phase 2: Code Quality**
- Module-level docstrings reviewed and improved
  - Enhanced all public function docstrings with Parameters, Returns, Examples
  - Improved coverage across covariance.py, relationships.py, utils.py, gwas.py, formula.py, mmes.py
- PEP 561 type-stub marker (`py.typed`) added
- Linting/formatting config added (ruff)

✅ **Phase 3: Verification**
- Tests passing: 58 tests, 1 skipped
- Package builds successfully
- Wheel installation verified in clean venv
- Distributions validated with twine

## Publishing Steps

### Option 1: Using Environment Variables (Recommended)

```bash
cd /Users/nico/Desktop/Projects/py-sommer

# Set tokens
export TESTPYPI_TOKEN="your-testpypi-token-here"
export PYPI_TOKEN="your-pypi-token-here"

# Publish to TestPyPI
python -m pip install twine
twine upload --repository testpypi \
  --username __token__ \
  --password "$TESTPYPI_TOKEN" \
  dist/pysommer-0.1.0.tar.gz \
  dist/pysommer-0.1.0-py3-none-any.whl

# Verify TestPyPI upload
pip install --index-url https://test.pypi.org/simple/ pysommer

# Publish to PyPI (production)
twine upload \
  --username __token__ \
  --password "$PYPI_TOKEN" \
  dist/pysommer-0.1.0.tar.gz \
  dist/pysommer-0.1.0-py3-none-any.whl
```

### Option 2: Using .pypirc Configuration

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE
```

Then publish:

```bash
cd /Users/nico/Desktop/Projects/py-sommer

# Publish to TestPyPI
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*
```

### Option 3: Using GitHub Actions (CI/CD)

The project has a `.github/workflows/publish.yml` workflow configured to:
- Run tests on all pushes
- Publish to PyPI on tagged releases (tags matching `v*`)

To trigger:
```bash
git tag v0.1.0
git push origin v0.1.0
```

## Post-Publication Verification

After publishing to PyPI:

```bash
# Install from PyPI
pip install pysommer

# Verify installation
python -c "import pysommer; print(f'pysommer {pysommer.__version__} installed successfully')"

# Quick functionality test
python -c "
from pysommer import A_mat, mmes
import numpy as np

# Test A_mat
X = np.random.randn(5, 10)
A = A_mat(X)
print(f'✓ A_mat works: shape {A.shape}')

# Test formula estimator
from pysommer import MMESFormulaRegressor, vsm, ism
data = {
    'y': np.random.randn(15),
    'g': np.repeat(np.arange(5), 3)
}
est = MMESFormulaRegressor(fixed='y ~ 1', random=vsm(ism('g')), iters=10)
est.fit(data)
pred = est.predict(data)
print(f'✓ MMESFormulaRegressor works: predictions shape {pred.shape}')
print('✓ All functionality verified')
"
```

## Troubleshooting

### "Invalid token" error
- Verify token is correct and not expired
- GitHub: go to https://pypi.org/manage/account/tokens/
- TestPyPI: go to https://test.pypi.org/manage/account/tokens/

### "File already exists" error
- If a version is already published, increment version in `pyproject.toml`
- Rebuild: `uv build`

### "Metadata validation failed"
- Run: `uv run --with twine twine check dist/*`
- Fix any issues and rebuild

## Rollback

If issues arise after publishing:
1. Upload a patch version (0.1.1) with fixes
2. Update `CHANGELOG.md` documenting the fix
3. Tag and redeploy

## Success Criteria

✅ Package appears on https://pypi.org/project/pysommer/
✅ Installation works: `pip install pysommer`
✅ All imports function correctly
✅ GitHub Actions publish workflow runs successfully on tagged commits
