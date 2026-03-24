# pysommer

Python implementation of the core mixed-model computational routines from the R
[sommer](https://github.com/covaruber/sommer) package by Giovanny
Covarrubias-Pazaran.

`pysommer` provides REML-based mixed linear model fitting via `mmes` — a Python
analogue of `sommer::mmes` — together with formula-style random-effect
declarations, scikit-learn-compatible estimators, prediction helpers,
relationship-matrix constructors, covariance structures, and GWAS scoring
utilities. All linear algebra is built on NumPy and SciPy; no R dependency is
required.

## Features

- **Two solver backends** — Newton-Raphson (`newton_di_sp`) and
  Average-Information REML (`ai_mme_sp`)
- **Matrix API** — pass `X`, `Z`, `K` arrays directly to `mmes`
- **Formula API** — R-like syntax (`"y ~ 1 + x"`) with `vsm` / `ism` / `dsm` /
  `usm` random-effect declarations
- **Multivariate responses** — fit multiple traits simultaneously by passing a
  multi-column `Y`
- **scikit-learn estimators** — `MMESRegressor` and `MMESFormulaRegressor`
  support `fit`, `predict`, `score`, `clone`, pipeline composition, and
  `KFold`-based cross-validation
- **Prediction helpers** — `predict_mmes` and `summarize_predictions` produce
  fitted values, residuals, and conditional prediction intervals; known
  random-effect levels reuse fitted BLUPs while unseen levels fall back to
  fixed-only predictions
- **Covariance structures** — `AR1`, `CS`, `ARMA` for use as `K` matrices
- **Relationship matrices** — `A_mat`, `D_mat`, `E_mat`, `H_mat`
  (additive/dominance/epistatic/H genomic)
- **Near-PD projection** — `nearPD` / `near_pd` for covariance regularisation
- **GWAS helpers** — `scorecalc` and `gwasForLoop` for marker-based association
  scoring

---

## Installation

```bash
pip install pysommer
```

### From source with uv (recommended)

```bash
git clone https://github.com/nriveras/pysommer
cd pysommer
uv sync
```

Include development tools (pytest, JupyterLab, scikit-learn):

```bash
uv sync --dev
```

Include the optional scikit-learn extras only:

```bash
uv sync --extra sklearn
```

Verify:

```bash
uv run python -c "import pysommer; print(pysommer.__version__)"
```

Run the test suite:

```bash
uv run pytest tests/ -q
```

Run only sklearn-like interface tests (including formula-mode estimator tests):

```bash
uv run pytest tests/test_sklearn_interface.py -q
```

---

## Quick Start

```python
import numpy as np
from pysommer import mmes

rng = np.random.default_rng(42)
n_groups, reps = 12, 4
group = np.repeat(np.arange(n_groups), reps)
n = group.size

X = np.ones((n, 1))                    # fixed-effects design matrix (intercept)
Z = np.eye(n_groups)[group]            # incidence matrix for group random effect
K = np.eye(n_groups)                   # identity relationship matrix

y = 2.0 + Z @ rng.normal(0, 0.9, (n_groups, 1)) + rng.normal(0, 0.5, (n, 1))

fit = mmes(Y=y, X=X, Z=[Z], K=[K], iters=50)

print("converged:", fit["converged"])
print("beta (intercept):", fit["beta"].ravel())
print("theta (var components):", fit["theta"].ravel())

# Fitted values: fixed + random
yhat = X @ fit["beta"] + Z @ fit["u"][0]
```

---

## API Reference

### Matrix API (`mmes`)

Pass `X`, `Z`, `K` arrays directly. Multiple random terms are supported.

```python
from pysommer import mmes

fit = mmes(
    Y=y,
    X=X,
    Z=[Z_fam, Z_block, Z_env],
    K=[np.eye(n_fam), np.eye(n_block), np.eye(n_env)],
    method="newton_di_sp",   # or "ai_mme_sp"
    iters=50,
)
```

Key result keys: `beta`, `theta`, `u`, `fitted`, `residuals`, `converged`,
`iterations`.

### Formula API (`mmes` with `vsm` / `ism` / `dsm` / `usm`)

Use a Wilkinson-Rogers formula string and a dict of arrays for data:

```python
from pysommer import mmes, vsm, ism, dsm, usm

data = {"y": y.ravel(), "group": group, "env": env}

# Random intercept by group
fit = mmes(fixed="y ~ 1", random=[vsm(ism("group"))], data=data, iters=40)

# Unstructured environment x group interaction
fit = mmes(fixed="y ~ 1", random=[vsm(usm("env"), ism("group"))], data=data, iters=40)

# Diagonal-by-environment group effects
fit = mmes(fixed="y ~ 1", random=[vsm(dsm("env"), ism("group"))], data=data, iters=40)
```

### Scikit-learn Estimators

#### `MMESRegressor` — matrix-mode estimator

```python
from pysommer import MMESRegressor

est = MMESRegressor(Z=[Z], K=[K], iters=40)
est.fit(X, y)

yhat        = est.predict(X)
yhat_blup   = est.predict(X, include_random=True)
r2          = est.score(X, y)
summary     = est.predict_summary(X, include_random=True)

print(est.coef_)   # fixed-effect coefficients
print(est.theta_)  # variance components
```

Out-of-sample prediction with known random-effect levels:

```python
X_new = np.ones((3, 1))
Z_new = [np.eye(n_groups)[np.array([0, 4, 9])]]
yhat_new = est.predict(X_new, include_random=True, Z=Z_new)
```

#### `MMESFormulaRegressor` — formula-mode estimator

Known factor levels reuse fitted BLUPs. Unseen levels automatically fall back
to fixed-only predictions.

```python
from pysommer import MMESFormulaRegressor, vsm, ism

est = MMESFormulaRegressor(
    fixed="y ~ 1 + x",
    random=vsm(ism("group")),
    iters=40,
)
est.fit(data)

new_data = {"y": np.zeros(3), "x": np.array([0.1, 0.5, -0.3]), "group": np.array([0, 3, 999])}
yhat = est.predict(new_data, include_random=True)   # group 999 -> fixed only
summary = est.predict_summary(new_data, include_random=True)

print(est.fixed_names_)              # ["Intercept", "x"]
print(est.random_names_)             # ["group"]
print(summary["random_effect_status"])  # per-term matched / zeroed row counts
```

Both estimators are compatible with `sklearn.base.clone`, `Pipeline`, and manual
`KFold` cross-validation. See the
[sklearn interop section](#richer-sklearn-interoperability-patterns-pipeline--cross-validation) below.

### Prediction helpers (matrix API)

```python
from pysommer import predict_mmes, summarize_predictions

yhat    = predict_mmes(fit, X_new, Z=[Z_new], include_random=True)
summary = summarize_predictions(fit, X_new, Z=[Z_new], include_random=True, interval=0.95)

print(summary["prediction_sd"])
print(summary["interval_lower"])
print(summary["interval_upper"])
```

`summarize_predictions` returns conditional uncertainty bands based on the
fitted residual variance plus random-effect PEV contributions. Fixed-effect
coefficient uncertainty is not yet included.

### Covariance structures

```python
from pysommer import AR1, CS, ARMA

K_time  = AR1(n_time, rho=0.7)                    # first-order autoregressive
K_space = CS(n_loc, rho=0.4)                       # compound symmetry
K_plot  = ARMA(n_plots, p=1, q=0, rho_p=0.6)      # ARMA(p,q)
```

Pass any of these as `K` entries in `mmes` or the sklearn estimators.

### Relationship matrices

```python
from pysommer import A_mat, D_mat, E_mat, H_mat

A = A_mat(markers)              # additive genomic relationship (VanRaden)
D = D_mat(markers)              # dominance relationship
E = E_mat(markers)              # epistatic relationship
H = H_mat(A, A22, A_inv)        # H matrix (pedigree + genomic)
```

### GWAS helpers

```python
from pysommer import gwasForLoop, scorecalc

# Marker-by-marker GWAS: returns (n_markers, n_traits, 3) array [score, effect, se]
out = gwasForLoop(M=M, Y=Y, Z=Z, X=X, Vinv=Vinv, min_maf=0.05)

# Single-marker multivariate score
score = scorecalc(Mimv=Mimv, Ymv=Ymv, Zmv=Zmv, Xmv=Xmv, Vinv=Vinv, nt=n_traits)
```

---

## Usage Examples

### Multiple random effects

```python
import numpy as np
from pysommer import mmes

rng = np.random.default_rng(6001)
n_fam, n_block, n_env = 8, 3, 2
reps = 2
n_obs = n_fam * n_block * n_env * reps

fam   = np.repeat(np.arange(n_fam), n_block * n_env * reps)
block = np.tile(np.repeat(np.arange(n_block), n_env * reps), n_fam)
env   = np.tile(np.repeat(np.arange(n_env), reps), n_fam * n_block)

Z_fam, Z_block, Z_env = np.eye(n_fam)[fam], np.eye(n_block)[block], np.eye(n_env)[env]
X = np.ones((n_obs, 1))

y = (1.8
     + Z_fam   @ rng.normal(0, np.sqrt(0.8), (n_fam, 1))
     + Z_block @ rng.normal(0, np.sqrt(0.3), (n_block, 1))
     + Z_env   @ rng.normal(0, np.sqrt(0.2), (n_env, 1))
     + rng.normal(0, np.sqrt(0.15), (n_obs, 1)))

fit = mmes(
    Y=y.ravel(), X=X,
    Z=[Z_fam, Z_block, Z_env],
    K=[np.eye(n_fam), np.eye(n_block), np.eye(n_env)],
    iters=40,
)

print("variance components:", fit["theta"].flatten())
print("random effect types:", len(fit["u"]))
```

### Custom relationship matrices (AR1, CS)

```python
import numpy as np
from pysommer import mmes, AR1, CS

rng = np.random.default_rng(6002)
n_time, n_loc, reps = 8, 4, 3
n_obs = n_time * n_loc * reps

time_idx = np.repeat(np.arange(n_time), n_loc * reps)
loc_idx  = np.tile(np.repeat(np.arange(n_loc), reps), n_time)
Z_time, Z_loc = np.eye(n_time)[time_idx], np.eye(n_loc)[loc_idx]
X = np.ones((n_obs, 1))

K_time = AR1(n_time, rho=0.7)   # first-order autoregressive (temporal)
K_loc  = CS(n_loc, rho=0.4)     # compound symmetry (location clustering)

y = (2.5
     + Z_time @ rng.normal(0, np.sqrt(0.6), (n_time, 1))
     + Z_loc  @ rng.normal(0, np.sqrt(0.4), (n_loc, 1))
     + rng.normal(0, np.sqrt(0.2), (n_obs, 1)))

fit = mmes(Y=y.ravel(), X=X, Z=[Z_time, Z_loc], K=[K_time, K_loc], iters=40)
print("time variance (AR1):", fit["theta"].flatten()[0])
print("location variance (CS):", fit["theta"].flatten()[1])
```

### Advanced covariance in formula mode (`usm` + `Cu`)

`Cu` is a known covariance matrix across levels of the `usm(...)` factor,
combined with `Gu` (or identity if omitted) through a Kronecker product.

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

data = {"y": y.ravel(), "group": group, "env": env}

fit_usm = mmes(
    fixed="y ~ 1",
    random=[vsm(usm("env"), ism("group"), Cu=AR1(2, rho=0.35))],
    data=data,
    iters=35,
)

print("random terms:", fit_usm["random_names"])
print("u shape:", fit_usm["u"][0].shape)
```

### Multivariate response

```python
import numpy as np
from pysommer import mmes

rng = np.random.default_rng(1301)
n_groups, reps = 10, 3
group = np.repeat(np.arange(n_groups), reps)
n = group.size
Z, X, K = np.eye(n_groups)[group], np.ones((n, 1)), np.eye(n_groups)

y1 = 1.0 + Z @ rng.normal(0, np.sqrt(0.7), (n_groups, 1)) + rng.normal(0, np.sqrt(0.3), (n, 1))
y2 = 2.0 + Z @ rng.normal(0, np.sqrt(1.1), (n_groups, 1)) + rng.normal(0, np.sqrt(0.4), (n, 1))
Y  = np.hstack([y1, y2])

fit_mv = mmes(Y=Y, X=X, Z=[Z], K=[K], iters=50)
print("beta shape:", fit_mv["beta"].shape)    # (p, n_traits)
print("theta shape:", fit_mv["theta"].shape)  # (n_vc, n_traits)
print("u[0] shape:", fit_mv["u"][0].shape)    # (n_levels, n_traits)
```

### Prediction workflow (seen and unseen levels, uncertainty)

```python
import numpy as np
from pysommer import mmes, predict_mmes, summarize_predictions

rng = np.random.default_rng(6003)
n_fam, n_reps = 10, 4
fam = np.repeat(np.arange(n_fam), n_reps)
Z_train = np.eye(n_fam)[fam]
X_train = np.ones((n_fam * n_reps, 1))

y_train = (2.0
           + Z_train @ rng.normal(0, np.sqrt(0.5), (n_fam, 1))
           + rng.normal(0, np.sqrt(0.3), (n_fam * n_reps, 1)))

fit = mmes(Y=y_train.ravel(), X=X_train, Z=[Z_train], K=[np.eye(n_fam)], iters=35)

yhat_train = predict_mmes(fit, X_train, Z=[Z_train], include_random=True).ravel()
print(f"Training RMSE: {float(np.sqrt(np.mean((y_train.ravel() - yhat_train)**2))):.4f}")

# New rows: seen families reuse fitted BLUPs; all-zero rows fall back to fixed effects
X_new = np.ones((4, 1))
Z_new = np.zeros((4, n_fam))
Z_new[0, 1] = 1.0   # family 1 (seen)
Z_new[1, 7] = 1.0   # family 7 (seen)
# rows 2 and 3 -> unseen families, random contribution = 0

yhat_new = predict_mmes(fit, X_new, Z=[Z_new], include_random=True)
summary   = summarize_predictions(fit, X_new, Z=[Z_new], include_random=True)

print("Predictions:", yhat_new.ravel().round(4))
print("Prediction SD:", summary["prediction_sd"].ravel().round(4))
```

### Genomic selection (GEBVs)

```python
import numpy as np
from pysommer import mmes, ARMA

rng = np.random.default_rng(6004)
n_lines, n_plots, n_env = 12, 4, 2
n_obs = n_lines * n_plots * n_env

lines = np.repeat(np.arange(n_lines), n_plots * n_env)
plots = np.tile(np.repeat(np.arange(n_plots), n_env), n_lines)
envs  = np.tile(np.arange(n_env), n_lines * n_plots)

Z_lines = np.eye(n_lines)[lines]
Z_plots = np.eye(n_plots)[plots]
Z_env   = np.eye(n_env)[envs]
X = np.ones((n_obs, 1))

# Genomic relationship matrix from markers
G = np.eye(n_lines) + 0.1 * rng.normal(0, 0.1, (n_lines, n_lines))
G = (G + G.T) / 2

K_plots = ARMA(n_plots, p=1, q=0, rho_p=0.6)

y = (3.2
     + Z_lines @ rng.normal(0, np.sqrt(0.7), (n_lines, 1))
     + Z_plots @ rng.normal(0, np.sqrt(0.3), (n_plots, 1))
     + Z_env   @ rng.normal(0, np.sqrt(0.15), (n_env, 1))
     + rng.normal(0, np.sqrt(0.25), (n_obs, 1)))

fit = mmes(
    Y=y.ravel(), X=X,
    Z=[Z_lines, Z_plots, Z_env],
    K=[G, K_plots, np.eye(n_env)],
    iters=40,
)

gebvs     = fit["u"][0].ravel()
top_lines = np.argsort(gebvs)[-3:][::-1]
print("Top 3 lines by GEBV:")
for i in top_lines:
    print(f"  Line {i}: {gebvs[i]:.4f}")
```

### GWAS scoring

```python
import numpy as np
from pysommer import gwasForLoop, scorecalc

rng = np.random.default_rng(1701)
n, n_levels, n_traits, n_markers = 24, 8, 2, 5

group = np.repeat(np.arange(n_levels), n // n_levels)
Z    = np.eye(n_levels)[group]
X    = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
M    = rng.choice([-1.0, 0.0, 1.0], size=(n_levels, n_markers))
Y    = rng.normal(0.0, 1.0, size=(n, n_traits))
Vinv = np.eye(n * n_traits)

# markers x traits x [score, effect, se]
out = gwasForLoop(M=M, Y=Y, Z=Z, X=X, Vinv=Vinv, min_maf=0.0)
print("gwas output shape:", out.shape)
```

For a Python-vs-R sommer parity check of GWAS outputs, see
[notebooks/compare_r_python_predictions.ipynb](notebooks/compare_r_python_predictions.ipynb).

### sklearn-like interface (`MMESRegressor`)

`MMESRegressor` provides a familiar estimator API while reusing the same
solver backend as `mmes`.

```python
import numpy as np
from pysommer import MMESRegressor

rng = np.random.default_rng(7001)
n_groups, reps = 10, 4
group = np.repeat(np.arange(n_groups), reps)
n = group.size

X = np.ones((n, 1), dtype=float)
Z = np.eye(n_groups)[group]
K = np.eye(n_groups, dtype=float)
y = 2.0 + Z @ rng.normal(0, np.sqrt(0.7), (n_groups, 1)) + rng.normal(0, np.sqrt(0.3), (n, 1))

est = MMESRegressor(Z=[Z], K=[K], iters=40, method="newton_di_sp")
est.fit(X, y)

yhat_fixed  = est.predict(X)
yhat_fitted = est.predict(X, include_random=True)

# Out-of-sample prediction reusing fitted random effects for known levels
X_new   = np.ones((3, 1), dtype=float)
Z_new   = [np.eye(n_groups)[np.array([0, 4, 9])]]
summary = est.predict_summary(X_new, include_random=True, Z=Z_new)

print("coef:", est.coef_.ravel())
print("theta:", est.theta_)
print("score:", est.score(X, y))
print("prediction sd:", summary["prediction_sd"].ravel())

# Parameter API compatible with sklearn cloning / search tools
params = est.get_params()
est.set_params(iters=50)
```

### Formula-mode sklearn interface (`MMESFormulaRegressor`)

```python
import numpy as np
from pysommer import MMESFormulaRegressor, vsm, ism

rng = np.random.default_rng(7002)
n_groups, reps = 8, 5
group = np.repeat(np.arange(n_groups), reps)
n = group.size

x = rng.normal(0.0, 1.0, size=n)
u = rng.normal(0.0, np.sqrt(0.6), size=(n_groups, 1))
e = rng.normal(0.0, np.sqrt(0.35), size=(n, 1))
y = 1.7 + 0.9 * x[:, None] + u[group] + e

data = {"y": y.ravel(), "x": x, "group": group}

est = MMESFormulaRegressor(
    fixed="y ~ 1 + x",
    random=vsm(ism("group")),
    iters=40,
    method="newton_di_sp",
)
est.fit(data)

yhat_fixed  = est.predict(data)
yhat_fitted = est.predict(data, include_random=True)

new_data = {
    "y": np.zeros(3),
    "x": np.array([x[0], x[1], 0.25]),
    "group": np.array([0, 3, 999]),  # group 999 is unseen -> fixed only
}
yhat_new = est.predict(new_data, include_random=True)
summary  = est.predict_summary(new_data, include_random=True)

print("fixed names:", est.fixed_names_)
print("random names:", est.random_names_)
print("new predictions:", yhat_new.ravel())
print("random effect status:", summary["random_effect_status"])
```

### Richer sklearn Interoperability Patterns (Pipeline + Cross-Validation)

#### Pipeline with `FunctionTransformer`

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pysommer import MMESRegressor

rng = np.random.default_rng(7003)
n_groups, reps = 10, 4
group = np.repeat(np.arange(n_groups), reps)
n = group.size
x = rng.normal(0.0, 1.0, size=n)
Z, K = np.eye(n_groups)[group], np.eye(n_groups)
y = 1.4 + 0.7 * x[:, None] + Z @ rng.normal(0, np.sqrt(0.6), (n_groups, 1)) + rng.normal(0, np.sqrt(0.3), (n, 1))

def add_intercept(x_in):
    return np.column_stack([np.ones(x_in.shape[0]), x_in])

pipe = Pipeline([
    ("intercept", FunctionTransformer(add_intercept)),
    ("mmes",      MMESRegressor(Z=[Z], K=[K], iters=35)),
])
pipe.fit(x.reshape(-1, 1), y)
print("pipeline score:", pipe.score(x.reshape(-1, 1), y))
```

#### Manual K-fold cross-validation

Each fold requires explicit row-subsetting of `Z`. `cross_val_score` is not
used because `Z` must be subset together with `X` per fold.

```python
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from pysommer import MMESRegressor

rng = np.random.default_rng(7004)
n_groups, reps = 12, 5
group = np.repeat(np.arange(n_groups), reps)
n = group.size
X = np.ones((n, 1), dtype=float)
Z, K = np.eye(n_groups)[group], np.eye(n_groups)
y = 2.0 + Z @ rng.normal(0, np.sqrt(0.7), (n_groups, 1)) + rng.normal(0, np.sqrt(0.3), (n, 1))

base = MMESRegressor(iters=30)
kf   = KFold(n_splits=3, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in kf.split(X):
    est = clone(base)
    est.fit(X[train_idx], y[train_idx], Z=[Z[train_idx, :]], K=[K])
    scores.append(est.score(X[test_idx], y[test_idx]))

print("CV scores:", np.round(scores, 4))
print("Mean CV score:", float(np.mean(scores)))
```

Optional scikit-learn dependency:

```bash
uv sync --extra sklearn
```

---

## Jupyter Notebook Setup

```bash
# 1. Sync project dependencies
uv sync --dev

# 2. Register a kernel
uv run jupyter kernelspec remove pysommer -f || true
uv run python -m ipykernel install --user --name pysommer --display-name "Python (pysommer)"

# 3. Start JupyterLab
uv run jupyter lab
```

Select the **Python (pysommer)** kernel in the notebook UI.

- Starter notebook: [notebooks/start_implementation.ipynb](notebooks/start_implementation.ipynb)
- R-vs-Python comparison: [notebooks/compare_r_python_predictions.ipynb](notebooks/compare_r_python_predictions.ipynb)

---

## Testing Against R sommer (Optional)

Generate R reference outputs (requires R + the `sommer` package):

```bash
Rscript tests/test_sommer_reference.R
```

This writes `tests/reference_data/mmes_reference.json` (includes `sommer_version`)
and `tests/reference_data/mmes_input.csv`. Both files are used by the
Python-vs-R comparison sections in the notebooks and the pytest suite.

---

## Attribution

This project is a Python re-implementation of selected computational routines
from the R `sommer` package. It is not the official package and does not replace
the original R implementation. For full sommer features (formula interface, full
API, and latest methods), use the upstream R package.

- Primary author of R sommer: Giovanny Covarrubias-Pazaran
- R sommer repository: <https://github.com/covaruber/sommer>
- Covarrubias-Pazaran, G. (2016). Genome assisted prediction of quantitative
  traits using the R package sommer. *PLoS ONE*.
  <https://doi.org/10.1371/journal.pone.0156744>
- Maier et al. (2015). Joint analysis of psychiatric disorders increases accuracy
  of risk prediction. *American Journal of Human Genetics*.
  <https://doi.org/10.1016/j.ajhg.2014.12.006>
- Jensen et al. (1997). Methodological mixed-model reference used by sommer.

## License

The upstream R sommer package is distributed under GPL (>= 2). Review licensing
obligations before redistribution or integration of derived work. See
[LICENSE](LICENSE).

---

## Repository Layout

```
pysommer/              Python package source
tests/                 pytest suite + R reference scripts + reference_data/
notebooks/             Jupyter notebooks (starter + R comparison)
upstream_r_sommer/     Unmodified upstream R sommer source (reference only)
```
