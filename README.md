# py-sommer: Python Translation of sommer Core

This repository contains a Python translation of core computational components from the R package sommer.

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

`pysommer.mmes` currently works with explicit matrices (not formula strings).

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
