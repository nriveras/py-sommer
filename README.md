# py-sommer: Python Translation of sommer Core

This repository contains a Python translation of core computational components from the R package sommer.

## Repository Layout

The repository root is focused on the Python translation project (`pysommer`).

The original/forked upstream R package source has been moved to:

- `upstream_r_sommer/`

This keeps Python packaging and development workflows clean while preserving upstream reference code.

Current scope focuses on explicit matrix-based mixed model computation in Python (NumPy/SciPy), including:
1. Utility matrix helpers
2. Relationship matrices (A, D, E, H)
3. Covariance constructors (AR1, CS, ARMA)
4. nearPD projection
5. A first-pass direct-inversion REML solver and simplified matrix-based mmes wrapper

## Credits and Attribution

This project is derived from and inspired by the original sommer package and its scientific and software contributions.

Primary original author of sommer:
1. Giovanny Covarrubias-Pazaran (author/creator of the R package sommer)

Original software and research sources that this translation builds on:
1. sommer R package repository: https://github.com/covaruber/sommer
2. Covarrubias-Pazaran, G. (2016). Genome assisted prediction of quantitative traits using the R package sommer. PLoS ONE. https://doi.org/10.1371/journal.pone.0156744
3. Maier et al. (2015). Joint analysis of psychiatric disorders increases accuracy of risk prediction for schizophrenia, bipolar disorder, and major depressive disorder. American Journal of Human Genetics. https://doi.org/10.1016/j.ajhg.2014.12.006
4. Jensen et al. (1997). (Methodological mixed model reference used by sommer)

Important attribution note:
1. This is an independent Python re-implementation of selected computational routines.
2. It is not the official sommer package and does not replace the original R implementation.
3. For full sommer features (formula interface, full API, and latest methods), use the upstream R package.

## License Lineage

The upstream sommer package is distributed under GPL (>= 2). Please review licensing obligations before redistribution or integration of derived work.

## Python Installation (uv)

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sync dependencies from this repository (recommended)

```bash
git clone https://github.com/nriveras/py-sommer
cd py-sommer
uv sync
```

### Include development dependencies

```bash
uv sync --dev
```

### Verify installation

```bash
uv run python -c "import pysommer; print('ok')"
```

### Run tests

```bash
uv run pytest tests/test_pysommer.py -q
```

## Python Model Setup Example

`pysommer.mmes` supports both explicit matrix input and a lightweight formula-like interface.

```python
import numpy as np
from pysommer import mmes

# n observations, one fixed intercept term.
n_groups = 20
reps = 3
group = np.repeat(np.arange(n_groups), reps)
n = group.size

X = np.ones((n, 1), dtype=float)         # Fixed effects design matrix
Z_id = np.eye(n_groups)[group]            # Random effect design matrix for group IDs
K_id = np.eye(n_groups, dtype=float)      # Covariance/relationship matrix for random term

# Example response.
rng = np.random.default_rng(123)
u = rng.normal(0, np.sqrt(1.2), size=(n_groups, 1))
e = rng.normal(0, np.sqrt(0.4), size=(n, 1))
y = 2.0 + Z_id @ u + e

fit = mmes(
	Y=y,
	X=X,
	Z=[Z_id],
	K=[K_id],
	iters=50,
)

print("converged:", fit["converged"])
print("theta:", fit["theta"])
print("beta:", fit["beta"].ravel())

# Predicted values (fixed + random)
yhat = X @ fit["beta"] + Z_id @ fit["u"][0]
```

## Example: Formula-like API (`vsm` / `ism` / `dsm`)

This is the high-level interface added in Next step 1.

```python
import numpy as np
from pysommer import mmes, vsm, ism, dsm

rng = np.random.default_rng(202)
n_groups = 8
reps = 4
group = np.repeat(np.arange(n_groups), reps)
env = np.tile(np.array(["E1", "E2"]), n_groups * reps // 2)

Z = np.eye(n_groups)[group]
y = 2.0 + Z @ rng.normal(0, 0.8, size=(n_groups, 1)) + rng.normal(0, 0.3, size=(group.size, 1))

data = {
	"y": y.ravel(),
	"group": group,
	"env": env,
}

# Random intercept by group
fit_ism = mmes(
	fixed="y ~ 1",
	random=[vsm(ism("group"))],
	data=data,
	iters=40,
)

# Environment-specific group effects via diagonal-by-level structure
fit_dsm = mmes(
	fixed="y ~ 1",
	random=[vsm(dsm("env"), ism("group"))],
	data=data,
	iters=40,
)

print("ISM random terms:", fit_ism["random_names"])
print("DSM random terms:", fit_dsm["random_names"])
```

## Example: Henderson-style AI solver (`ai_mme_sp`)

This is the second REML path implemented in Next step 2. Use `method="ai_mme_sp"`.

```python
import numpy as np
from pysommer import mmes

rng = np.random.default_rng(101)
n_groups = 16
reps = 3
group = np.repeat(np.arange(n_groups), reps)
n = group.size

Z = np.eye(n_groups)[group]
X = np.ones((n, 1), dtype=float)
y = 1.5 + Z @ rng.normal(0, np.sqrt(0.8), size=(n_groups, 1)) + rng.normal(0, np.sqrt(0.5), size=(n, 1))

fit_henderson = mmes(
	Y=y,
	X=X,
	Z=[Z],
	K=[np.eye(n_groups)],
	method="ai_mme_sp",
	iters=50,
)

print("converged:", fit_henderson["converged"])
print("theta:", np.asarray(fit_henderson["theta"]).round(6))
print("beta:", np.asarray(fit_henderson["beta"]).ravel().round(6))
```

## Next steps

The following major pieces are still pending:

1. [x] Add a high-level API closer to R `sommer` style (`mmes`/`vsm` formula-like interface) instead of matrix-only inputs.
2. [x] Implement the second REML solver path (`ai_mme_sp` / Henderson-based AI) in Python.
3. [ ] Extend solver coverage for broader multivariate and advanced covariance structures beyond the current first-pass univariate core.
4. [ ] Add GWAS helper translations (`scorecalc`, `gwasForLoop`) and tests.
5. [ ] Expand cross-language validation with more real datasets and edge-case regression tests.
6. [ ] Improve user-facing docs with more end-to-end examples (multiple random terms, custom relationship matrices, prediction workflows).

## Use In Jupyter Notebook

### 1. Sync project dependencies

```bash
uv sync --dev
```

### 2. Install a Jupyter kernel backed by this project environment

```bash
uv run jupyter kernelspec remove py-sommer -f || true
uv run python -m ipykernel install --user --name py-sommer --display-name "Python (py-sommer)"
```

### 3. Start Jupyter from the project environment

```bash
uv run jupyter lab
```

### 4. Select the kernel in the notebook UI

In VS Code or Jupyter, choose the kernel named **Python (py-sommer)** for notebook execution.

### Generate R reference outputs (optional cross-check)

```bash
Rscript tests/test_sommer_reference.R
```
