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
uv run pytest tests/ -q
```

Run only sklearn-like interface tests (including formula-mode estimator tests):

```bash
uv run pytest tests/test_sklearn_interface.py -q
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

## Example: Multivariate solver coverage (independent-trait mode)

Step 3 adds broader multivariate coverage by allowing `Y` to have multiple columns.
The current implementation fits each trait with shared model terms and returns
stacked outputs (`beta`, `theta`, `u`, `fitted`, `residuals`).

```python
import numpy as np
from pysommer import mmes

rng = np.random.default_rng(1301)
n_groups = 10
reps = 3
group = np.repeat(np.arange(n_groups), reps)
n = group.size

Z = np.eye(n_groups)[group]
X = np.ones((n, 1), dtype=float)
K = np.eye(n_groups, dtype=float)

y1 = 1.0 + Z @ rng.normal(0.0, np.sqrt(0.7), size=(n_groups, 1)) + rng.normal(0.0, np.sqrt(0.3), size=(n, 1))
y2 = 2.0 + Z @ rng.normal(0.0, np.sqrt(1.1), size=(n_groups, 1)) + rng.normal(0.0, np.sqrt(0.4), size=(n, 1))
Y = np.hstack([y1, y2])

fit_mv = mmes(Y=Y, X=X, Z=[Z], K=[K], iters=50)

print("beta shape:", fit_mv["beta"].shape)      # (p, n_traits)
print("theta shape:", fit_mv["theta"].shape)    # (n_vc, n_traits)
print("u[0] shape:", fit_mv["u"][0].shape)      # (n_levels, n_traits)
```

## Example: Advanced covariance structure in formula mode (`usm` + `Cu`)

Step 3 also adds an advanced covariance declaration in formula mode:
`vsm(usm(env), ism(group), Cu=...)`.

`Cu` is a known covariance matrix across levels of the `usm(...)` factor and
is combined with `Gu` (or identity if omitted) through a Kronecker product.

```python
import numpy as np
from pysommer import AR1, mmes, vsm, usm, ism

rng = np.random.default_rng(1303)
n_groups = 8
reps = 4
group = np.repeat(np.arange(n_groups), reps)
env = np.tile(np.array(["E1", "E2"]), group.size // 2)

Z = np.eye(n_groups)[group]
y = 1.8 + Z @ rng.normal(0.0, 0.7, size=(n_groups, 1)) + rng.normal(0.0, 0.25, size=(group.size, 1))

data = {
	"y": y.ravel(),
	"group": group,
	"env": env,
}

fit_usm = mmes(
	fixed="y ~ 1",
	random=[vsm(usm("env"), ism("group"), Cu=AR1(2, rho=0.35))],
	data=data,
	iters=35,
)

print("random terms:", fit_usm["random_names"])
print("u shape:", fit_usm["u"][0].shape)
```

## Example: GWAS helpers (`scorecalc`, `gwasForLoop`)

Step 4 adds direct Python translations of the sommer GWAS helper routines.

```python
import numpy as np
from pysommer import gwasForLoop, scorecalc

rng = np.random.default_rng(1701)
n = 24
n_levels = 8
n_traits = 2
n_markers = 5

group = np.repeat(np.arange(n_levels), n // n_levels)
Z = np.eye(n_levels)[group]
X = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
M = rng.choice([-1.0, 0.0, 1.0], size=(n_levels, n_markers))
Y = rng.normal(0.0, 1.0, size=(n, n_traits))
Vinv = np.eye(n * n_traits)

# Marker-by-marker GWAS helper (markers x traits x [score, effect, se])
out = gwasForLoop(M=M, Y=Y, Z=Z, X=X, Vinv=Vinv, min_maf=0.0)
print("gwas output shape:", out.shape)

# Direct scorecalc call for marker 1 in multivariate form
D = np.eye(n_traits)
Ymv = Y.T.reshape(-1, 1, order="F")
Zmv = np.kron(Z, D)
Xmv = np.kron(X, D)
Mimv = np.kron(M[:, [0]], D)
score_marker1 = scorecalc(Mimv=Mimv, Ymv=Ymv, Zmv=Zmv, Xmv=Xmv, Vinv=Vinv, nt=n_traits)
print("scorecalc marker shape:", score_marker1.shape)
```

For a complete Python-vs-R sommer parity check of these helper outputs, see the GWAS comparison section in [notebooks/compare_r_python_predictions.ipynb](notebooks/compare_r_python_predictions.ipynb).

## Example: Edge-case and Solver Regression Tests

Step 5 adds comprehensive regression tests for numerical stability and cross-language validation.

```python
import numpy as np
from pysommer import mmes

# Edge case 1: Small sample size (n=5)
rng = np.random.default_rng(501)
group_small = np.array([0, 0, 1, 1, 1])
Z_small = np.eye(2)[group_small]
y_small = 1.0 + Z_small @ rng.normal(0.0, np.sqrt(0.5), size=(2, 1)) + rng.normal(0.0, np.sqrt(0.2), size=(5, 1))

out_small = mmes(
    Y=y_small.ravel(),
    X=np.ones((5, 1)),
    Z=[Z_small],
    K=[np.eye(2)],
    iters=25
)
print("Converged on small sample:", out_small["status"] == 0)

# Edge case 2: Extreme variance ratio (σ²_u >> σ²_e)
n_groups = 8
group = np.repeat(np.arange(n_groups), 5)
Z = np.eye(n_groups)[group]
u_var = rng.normal(0.0, np.sqrt(100.0), size=(n_groups, 1))  # Very large
e_var = rng.normal(0.0, np.sqrt(0.01), size=(group.size, 1))  # Very small
y_extreme = 2.0 + Z @ u_var + e_var

out_extreme = mmes(Y=y_extreme.ravel(), X=np.ones((group.size, 1)), Z=[Z], K=[np.eye(n_groups)], iters=40)
print("Handled extreme variance ratio:", np.isfinite(out_extreme["theta"]).all())

# Solver consistency: Newton vs AI/EM
y = 1.5 + Z @ rng.normal(0.0, np.sqrt(0.7), size=(n_groups, 1)) + rng.normal(0.0, np.sqrt(0.3), size=(group.size, 1))
X = np.ones((group.size, 1))

newton_fit = mmes(Y=y.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], method="newton_di_sp", iters=30)
ai_fit = mmes(Y=y.ravel(), X=X, Z=[Z], K=[np.eye(n_groups)], method="ai_mme_sp", iters=30)

max_diff = np.max(np.abs(newton_fit["theta"] - ai_fit["theta"]))
print(f"Solver methods agree: {max_diff < 0.1} (diff={max_diff:.6f})")
```

All regression tests, edge-case coverage, and cross-language validation examples are demonstrated in [notebooks/compare_r_python_predictions.ipynb](notebooks/compare_r_python_predictions.ipynb). The test suite includes 30 comprehensive tests validating:
- Numerical stability on extreme inputs (small samples, extreme variance ratios, collinearity)
- Solver consistency across methods (Newton vs AI/EM)
- Interface consistency (matrix mode vs formula mode)
- Multivariate independence validation (multivariate fits vs sequential univariate fits)

## Example: End-to-End Workflows (Multiple Random Terms, Custom Matrices, Predictions)

Step 6 introduces complete practical workflows combining multiple random effects, custom relationship matrices, and prediction pipelines.

### Multiple Random Effects Example

Fit a model with three independent random effects (genetic family + spatial block + environment):

```python
import numpy as np
from pysommer import mmes

rng = np.random.default_rng(6001)

# Setup: factorial design with families, blocks, environments
n_fam, n_block, n_env = 8, 3, 2
reps = 2
n_obs = n_fam * n_block * n_env * reps

fam = np.repeat(np.arange(n_fam), n_block * n_env * reps)
block = np.tile(np.repeat(np.arange(n_block), n_env * reps), n_fam)
env = np.tile(np.repeat(np.arange(n_env), reps), n_fam * n_block)

# Design matrices for each random effect
Z_fam = np.eye(n_fam)[fam]
Z_block = np.eye(n_block)[block]
Z_env = np.eye(n_env)[env]

# Generate response with three random components
X = np.ones((n_obs, 1))
u_fam = rng.normal(0.0, np.sqrt(0.8), size=(n_fam, 1))
u_block = rng.normal(0.0, np.sqrt(0.3), size=(n_block, 1))
u_env = rng.normal(0.0, np.sqrt(0.2), size=(n_env, 1))
e = rng.normal(0.0, np.sqrt(0.15), size=(n_obs, 1))

y = 1.8 + Z_fam @ u_fam + Z_block @ u_block + Z_env @ u_env + e

# Fit with multiple random effects
fit = mmes(
	Y=y.ravel(),
	X=X,
	Z=[Z_fam, Z_block, Z_env],
	K=[np.eye(n_fam), np.eye(n_block), np.eye(n_env)],
	iters=40
)

print(f"Fixed effect: {fit['beta'][0,0]:.4f}")
print(f"Variance components: {fit['theta'].flatten()}")
print(f"Number of random effect types: {len(fit['u'])}")
```

### Custom Relationship Matrices Example

Use AR1 (autoregressive-1, for temporal autocorrelation) and CS (compound symmetry, for shared environmental correlation):

```python
import numpy as np
from pysommer import mmes, AR1, CS

rng = np.random.default_rng(6002)

# Data with temporal and spatial structure
n_time, n_loc, reps = 8, 4, 3
n_obs = n_time * n_loc * reps

time_idx = np.repeat(np.arange(n_time), n_loc * reps)
loc_idx = np.tile(np.repeat(np.arange(n_loc), reps), n_time)

Z_time = np.eye(n_time)[time_idx]
Z_loc = np.eye(n_loc)[loc_idx]
X = np.ones((n_obs, 1))

# AR1 relationship (phi=0.7 autocorrelation for time series)
K_time = AR1(n_time, rho=0.7)

# Compound Symmetry (rho=0.4 for location clustering)
K_loc = CS(n_loc, rho=0.4)

# Generate response
u_time = rng.normal(0.0, np.sqrt(0.6), size=(n_time, 1))
u_loc = rng.normal(0.0, np.sqrt(0.4), size=(n_loc, 1))
e = rng.normal(0.0, np.sqrt(0.2), size=(n_obs, 1))
y = 2.5 + Z_time @ u_time + Z_loc @ u_loc + e

# Fit with custom relationship structures
fit = mmes(
	Y=y.ravel(),
	X=X,
	Z=[Z_time, Z_loc],
	K=[K_time, K_loc],
	iters=40
)

print(f"Time effect variance (AR1): {fit['theta'].flatten()[0]:.4f}")
print(f"Location effect variance (CS): {fit['theta'].flatten()[1]:.4f}")
```

### Prediction Workflow Example

Complete workflow: fitted values, out-of-sample random-effect predictions for known levels, uncertainty summaries, and cross-validation:

```python
import numpy as np
from pysommer import mmes, predict_mmes, summarize_predictions

rng = np.random.default_rng(6003)

# Training data
n_train_fam = 10
n_train_reps = 4
n_train = n_train_fam * n_train_reps

fam_train = np.repeat(np.arange(n_train_fam), n_train_reps)
Z_train = np.eye(n_train_fam)[fam_train]
X_train = np.ones((n_train, 1))

u_true = rng.normal(0.0, np.sqrt(0.5), size=(n_train_fam, 1))
e_train = rng.normal(0.0, np.sqrt(0.3), size=(n_train, 1))
y_train = 2.0 + Z_train @ u_true + e_train

# Fit model
fit = mmes(Y=y_train.ravel(), X=X_train, Z=[Z_train], K=[np.eye(n_train_fam)], iters=35)

# ---- Prediction Stream 1: Training data fitted values and residuals ----
beta = fit["beta"][0, 0]
yhat_train = predict_mmes(fit, X_train, Z=[Z_train], include_random=True).ravel()
resid_train = y_train.ravel() - yhat_train

print(f"Training RMSE: {np.sqrt(np.mean(resid_train**2)):.4f}")
print(f"Fixed effect estimate: {beta:.4f}")

# ---- Prediction Stream 2: New rows with seen and unseen families ----
# Reuse fitted BLUPs for seen families by passing aligned Z rows.
# Leave rows for unseen families all-zero so they fall back to fixed effects.
X_new = np.ones((4, 1))
Z_new = np.zeros((4, n_train_fam))
Z_new[0, 1] = 1.0   # Seen family 1
Z_new[1, 7] = 1.0   # Seen family 7
# Rows 2 and 3 are unseen families, so their random-effect contribution stays 0.

yhat_new = predict_mmes(fit, X_new, Z=[Z_new], include_random=True)
summary = summarize_predictions(fit, X_new, Z=[Z_new], include_random=True)

print("Predictions:", yhat_new.ravel().round(4))
print("Prediction SD:", summary["prediction_sd"].ravel().round(4))
print("95% interval first row:", summary["interval_lower"][0, 0].round(4), summary["interval_upper"][0, 0].round(4))

# ---- Prediction Stream 3: Cross-validation concept ----
# Fit on subset, predict on holdout
split_idx = 8
y_cv_train = y_train[:split_idx * n_train_reps]
Z_cv_train = Z_train[:split_idx * n_train_reps, :split_idx]
X_cv_train = X_train[:split_idx * n_train_reps]

fit_cv = mmes(
	Y=y_cv_train.ravel(),
	X=X_cv_train,
	Z=[Z_cv_train],
	K=[np.eye(split_idx)],
	iters=35
)

y_cv_test = y_train[split_idx * n_train_reps:]
yhat_cv = np.ones(len(y_cv_test)) * fit_cv["beta"][0, 0]
cv_rmse = np.sqrt(np.mean((y_cv_test.ravel() - yhat_cv)**2))

print(f"Cross-validation RMSE (held-out families): {cv_rmse:.4f}")
```

`summarize_predictions(...)` returns conditional uncertainty bands based on the fitted residual variance plus random-effect PEV contributions. Fixed-effect coefficient uncertainty is not yet included.

### Genomic Selection Example

Combine genomic relationship matrix with spatial structure for breeding program decisions:

```python
import numpy as np
from pysommer import mmes, ARMA

rng = np.random.default_rng(6004)

# Simulated genomic + spatial model
n_lines, n_plots, n_env = 12, 4, 2
reps = 1
n_obs = n_lines * n_plots * n_env * reps

lines = np.repeat(np.arange(n_lines), n_plots * n_env * reps)
plots = np.tile(np.repeat(np.arange(n_plots), n_env * reps), n_lines)
envs = np.tile(np.repeat(np.arange(n_env), reps), n_lines * n_plots)

Z_lines = np.eye(n_lines)[lines]
Z_plots = np.eye(n_plots)[plots]
Z_env = np.eye(n_env)[envs]
X = np.ones((n_obs, 1))

# Simulate genomic relationship matrix (kinship from marker data)
G = np.eye(n_lines) + 0.1 * rng.normal(0, 0.1, size=(n_lines, n_lines))
G = (G + G.T) / 2  # Symmetrize

# Spatial ARMA structure for plots
K_plots = ARMA(n_plots, p=1, q=0, rho_p=0.6)

# Generate phenotypes
u_genomic = rng.normal(0.0, np.sqrt(0.7), size=(n_lines, 1))
u_spatial = rng.normal(0.0, np.sqrt(0.3), size=(n_plots, 1))
u_env = rng.normal(0.0, np.sqrt(0.15), size=(n_env, 1))
e = rng.normal(0.0, np.sqrt(0.25), size=(n_obs, 1))

y = 3.2 + Z_lines @ u_genomic + Z_plots @ u_spatial + Z_env @ u_env + e

# Fit with genomic kinship matrix
fit = mmes(
	Y=y.ravel(),
	X=X,
	Z=[Z_lines, Z_plots, Z_env],
	K=[G, K_plots, np.eye(n_env)],
	iters=40
)

# Genomic breeding values (GEBVs) for selection
gebvs = fit["u"][0].ravel()
top_lines = np.argsort(gebvs)[-3:][::-1]

print(f"Top 3 lines by genomic merit:")
for idx in top_lines:
	print(f"  Line {idx}: GEBV = {gebvs[idx]:.4f}")
```

For complete examples with R sommer comparison and detailed output, see [notebooks/compare_r_python_predictions.ipynb](notebooks/compare_r_python_predictions.ipynb).

## Example: Scikit-Learn-like Interface (`MMESRegressor`)

`MMESRegressor` provides a familiar estimator API with `fit`, `predict`, `predict_summary`, `score`, `get_params`, and `set_params` while reusing the same solver backend as `mmes`.

```python
import numpy as np
from pysommer import MMESRegressor

rng = np.random.default_rng(7001)
n_groups = 10
reps = 4
group = np.repeat(np.arange(n_groups), reps)
n = group.size

X = np.ones((n, 1), dtype=float)
Z = np.eye(n_groups)[group]
K = np.eye(n_groups, dtype=float)

y = 2.0 + Z @ rng.normal(0.0, np.sqrt(0.7), size=(n_groups, 1)) + rng.normal(0.0, np.sqrt(0.3), size=(n, 1))

est = MMESRegressor(Z=[Z], K=[K], iters=40, method="newton_di_sp")
est.fit(X, y)

# Fixed-only predictions for arbitrary new design rows.
yhat_fixed = est.predict(X)

# Training-aligned fitted values (fixed + random) when X matches fit-time X.
yhat_fitted = est.predict(X, include_random=True)

# New rows can reuse fitted random effects when you pass aligned Z rows.
X_new = np.ones((3, 1), dtype=float)
Z_new = [np.eye(n_groups)[np.array([0, 4, 9])]]
yhat_known = est.predict(X_new, include_random=True, Z=Z_new)
summary = est.predict_summary(X_new, include_random=True, Z=Z_new)

print("coef:", est.coef_.ravel())
print("theta:", est.theta_)
print("score:", est.score(X, y))
print("prediction sd:", summary["prediction_sd"].ravel())

# Parameter API compatible with sklearn cloning / search tools.
params = est.get_params()
est.set_params(iters=50)
```

## Example: Formula-Mode Scikit-Learn-like Interface (`MMESFormulaRegressor`)

`MMESFormulaRegressor` exposes sklearn-style methods while using formula mode
(`fixed` / `random` / `data`) under the hood. For new data, known random-effect
levels reuse fitted BLUPs and unseen levels automatically fall back to the
fixed-effect component.

```python
import numpy as np
from pysommer import MMESFormulaRegressor, vsm, ism

rng = np.random.default_rng(7002)
n_groups = 8
reps = 5
group = np.repeat(np.arange(n_groups), reps)
n = group.size

x = rng.normal(0.0, 1.0, size=n)
u = rng.normal(0.0, np.sqrt(0.6), size=(n_groups, 1))
e = rng.normal(0.0, np.sqrt(0.35), size=(n, 1))
y = 1.7 + 0.9 * x[:, None] + u[group] + e

data = {
	"y": y.ravel(),
	"x": x,
	"group": group,
}

est = MMESFormulaRegressor(
	fixed="y ~ 1 + x",
	random=vsm(ism("group")),
	iters=40,
	method="newton_di_sp",
)
est.fit(data)

# Fixed-only predictions from formula-derived fixed design.
yhat_fixed = est.predict(data)

# Include random effects for exact training data.
yhat_fitted = est.predict(data, include_random=True)

new_data = {
	"y": np.zeros(3),
	"x": np.array([x[0], x[1], 0.25]),
	"group": np.array([0, 3, 999]),
}

# group 0 and 3 reuse fitted random effects; group 999 is unseen and gets
# fixed-only prediction.
yhat_new = est.predict(new_data, include_random=True)
summary = est.predict_summary(new_data, include_random=True)

print("fixed names:", est.fixed_names_)
print("random names:", est.random_names_)
print("coef:", est.coef_.ravel())
print("theta:", est.theta_)
print("score:", est.score(data))
print("new predictions:", yhat_new.ravel())
print("random effect status:", summary["random_effect_status"])

# sklearn-compatible parameter API.
params = est.get_params()
est.set_params(iters=50)
```

Optional sklearn interoperability dependency:

```bash
uv sync --extra sklearn
```

## Example: Richer sklearn Interoperability Patterns (Pipeline + Cross-Validation)

`MMESRegressor` works with sklearn pipelines and cloning utilities.

### Pipeline pattern with `FunctionTransformer`

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pysommer import MMESRegressor

rng = np.random.default_rng(7003)
n_groups = 10
reps = 4
group = np.repeat(np.arange(n_groups), reps)
n = group.size

x = rng.normal(0.0, 1.0, size=n)
X_raw = x.reshape(-1, 1)
Z = np.eye(n_groups)[group]
K = np.eye(n_groups, dtype=float)

y = 1.4 + 0.7 * x[:, None] + Z @ rng.normal(0.0, np.sqrt(0.6), size=(n_groups, 1)) + rng.normal(0.0, np.sqrt(0.3), size=(n, 1))

def add_intercept(x_in):
	return np.column_stack([np.ones(x_in.shape[0]), x_in])

pipe = Pipeline([
	("intercept", FunctionTransformer(add_intercept)),
	("mmes", MMESRegressor(Z=[Z], K=[K], iters=35)),
])

pipe.fit(X_raw, y)
print("pipeline score:", pipe.score(X_raw, y))
```

### Cross-validation pattern with `KFold` + `clone`

```python
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from pysommer import MMESRegressor

rng = np.random.default_rng(7004)
n_groups = 12
reps = 5
group = np.repeat(np.arange(n_groups), reps)
n = group.size

X = np.ones((n, 1), dtype=float)
Z = np.eye(n_groups)[group]
K = np.eye(n_groups, dtype=float)
y = 2.0 + Z @ rng.normal(0.0, np.sqrt(0.7), size=(n_groups, 1)) + rng.normal(0.0, np.sqrt(0.3), size=(n, 1))

base = MMESRegressor(iters=30)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in kf.split(X):
	X_train, X_test = X[train_idx], X[test_idx]
	y_train, y_test = y[train_idx], y[test_idx]
	Z_train = Z[train_idx, :]

	est = clone(base)
	est.fit(X_train, y_train, Z=[Z_train], K=[K])
	scores.append(est.score(X_test, y_test))

print("cv scores:", scores)
print("mean cv score:", float(np.mean(scores)))
```

`cross_val_score` is not shown for this mixed-model workflow because each fold
needs explicit per-fold handling of `Z` rows while maintaining compatible `K`
dimensions.

## Next steps

1. [X] Add a formula-mode estimator interface (for `fixed` / `random` / `data`) with sklearn-style methods.
2. [X] Add richer sklearn interoperability examples (pipeline and cross-validation patterns) after formula-mode estimator support lands.
3. [X] Expand prediction helpers for out-of-sample random effect handling and uncertainty summaries.
4. [ ] Publish the next release to TestPyPI, validate install/docs rendering, then publish to PyPI.

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

The generated file `tests/reference_data/mmes_reference.json` includes
`sommer_version`, which records the exact installed R `sommer` version used
to produce the reference outputs.

The random-intercept reference model is loaded from
`tests/reference_data/mmes_input.csv`, so the same observations can be reused
in the Python-vs-R comparison section of
`notebooks/start_implementation.ipynb`.
