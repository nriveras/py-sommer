# Publish pysommer to PyPI

The package is functionally complete (9 modules, 39+ tests passing, comprehensive README), but lacks the metadata, licensing, and infrastructure needed for PyPI publication.

**PyPI** (Python Package Index, https://pypi.org) is the standard repository for Python packages — it's what `pip install` uses by default. It is the most respected and universally used registry for Python libraries.

## User Review Required

> [!IMPORTANT]
> **Author information**: The plan uses placeholder author name/email. Please provide your preferred name and email for the package metadata.

> [!IMPORTANT]
> **License choice**: The upstream R sommer package uses GPL (≥ 2). Since pysommer is a derived work, it must use a GPL-compatible license. The plan uses **GPL-3.0-or-later**. Confirm if this is acceptable.

> [!IMPORTANT]
> **PyPI account**: You will need a PyPI account (https://pypi.org/account/register/) and a TestPyPI account (https://test.pypi.org/account/register/) to publish. The CI workflow uses trusted publishing (no API tokens needed if configured on PyPI).

> [!WARNING]
> **Package name [pysommer](file:///Users/nico/Desktop/Projects/py-sommer/pysommer)**: Before we proceed, verify that the name is available on PyPI. I checked that [pysommer](file:///Users/nico/Desktop/Projects/py-sommer/pysommer) does not appear to be taken, but you should confirm at https://pypi.org/project/pysommer/.

## Proposed Changes

### Repository Hygiene

#### [NEW] [LICENSE](file:///Users/nico/Desktop/Projects/py-sommer/LICENSE)

Full GPL-3.0-or-later license text. Required by PyPI and by upstream GPL obligations.

#### [NEW] [.gitignore](file:///Users/nico/Desktop/Projects/py-sommer/.gitignore)

Standard Python `.gitignore` covering `__pycache__`, `.venv`, `*.egg-info`, `dist/`, [build/](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/solver.py#45-51), [.DS_Store](file:///Users/nico/Desktop/Projects/py-sommer/.DS_Store), etc.

#### [NEW] [CHANGELOG.md](file:///Users/nico/Desktop/Projects/py-sommer/CHANGELOG.md)

Initial changelog entry for v0.1.0 documenting the current feature set. Good practice for any published package.

---

### Package Metadata

#### [MODIFY] [pyproject.toml](file:///Users/nico/Desktop/Projects/py-sommer/pyproject.toml)

Enrich with all fields needed for a quality PyPI listing:
- `license` — GPL-3.0-or-later (SPDX identifier)
- `authors` — name and email
- `keywords` — mixed models, REML, quantitative genetics, genomics, breeding
- `classifiers` — Development Status, License, Python versions, Topic
- `[project.urls]` — Homepage, Repository, Bug Tracker, Documentation
- `[tool.ruff]` — linting and formatting configuration
- `[tool.pytest.ini_options]` — test discovery defaults

#### [MODIFY] [__init__.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/__init__.py)

Add `__version__ = "0.1.0"` for programmatic version access (`pysommer.__version__`).

---

### Code Quality

#### [NEW] [pysommer/py.typed](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/py.typed)

Empty marker file (PEP 561) signaling that pysommer ships inline type annotations. This allows tools like `mypy` and `pyright` to use the type information from pysommer.

---

### CI/CD

#### [NEW] [.github/workflows/ci.yml](file:///Users/nico/Desktop/Projects/py-sommer/.github/workflows/ci.yml)

GitHub Actions workflow with two jobs:
1. **test** — runs `uv sync && uv run pytest` on Python 3.9, 3.11, 3.12 across ubuntu-latest
2. **publish** — builds and uploads to PyPI using trusted publishing, triggered only on GitHub Releases

---

### Cleanup

- Delete [pysommer.egg-info/](file:///Users/nico/Desktop/Projects/py-sommer/pysommer.egg-info) directory (build artifact that shouldn't be committed)
- The `.gitignore` will prevent it from being re-committed

## Verification Plan

### Automated Tests

Run the existing test suite to confirm nothing is broken:

```bash
cd /Users/nico/Desktop/Projects/py-sommer
uv run pytest tests/test_pysommer.py -q
```

### Build Verification

Build the package and inspect its contents:

```bash
cd /Users/nico/Desktop/Projects/py-sommer
uv run python -m build
# Inspect the built wheel
unzip -l dist/pysommer-0.1.0-py3-none-any.whl
# Verify metadata
uv run python -c "import pysommer; print(pysommer.__version__)"
```

### Install Verification

Install from the built wheel into a fresh venv to confirm it works:

```bash
cd /tmp
python3 -m venv test-pysommer-install
source test-pysommer-install/bin/activate
pip install /Users/nico/Desktop/Projects/py-sommer/dist/pysommer-0.1.0-py3-none-any.whl
python -c "from pysommer import mmes; print('import ok')"
deactivate
rm -rf test-pysommer-install
```

### Manual Verification

After implementation, ask the user to:
1. Review the PyPI metadata preview by running `uv run python -m build` and inspecting `dist/`
2. Create accounts on PyPI and TestPyPI if not already done
3. Upload to TestPyPI first: `uv run twine upload --repository testpypi dist/*`
4. Test install from TestPyPI: `pip install --index-url https://test.pypi.org/simple/ pysommer`
5. When satisfied, upload to production PyPI or push a GitHub Release to trigger the CI workflow
