# Translate sommer R Package Core to Python

Translate the core computational functions of the [sommer](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#337-364) R package (mixed model REML solver) to a pure-Python package using **numpy** and **scipy** only. The C++ code (RcppArmadillo) will be re-implemented in Python with numpy linear algebra.

## User Review Required

> [!IMPORTANT]
> **Scope decision**: The full [sommer](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#337-364) package has a complex formula-parsing API (`mmes()`, `vsm()`, etc.) that is deeply tied to R's formula system. I propose translating only the **computational core** in Phase 1, keeping the Python API as explicit matrix-based calls (no formula parsing). A higher-level API can be added later.

> [!WARNING]
> **GWAS functions**: [scorecalc](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#203-205) and [gwasForLoop](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#220-222) are specialized GWAS tools. Should I include these in the first pass, or defer them? They add ~200 lines of C++ but are secondary to the core mixed model solver.

> [!IMPORTANT]
> **Solver choice**: The package has two REML solvers:
> 1. [newton_di_sp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#521-1283) — Direct-Inversion Newton-Raphson/AI (for p > n problems)
> 2. [ai_mme_sp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#1366-2162) — Henderson-based AI (for n > p problems)
>
> I propose starting with [newton_di_sp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#521-1283) only, as it's the default and most commonly used. We can add [ai_mme_sp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#1366-2162) later.

---

## Proposed Changes

### R Test Suite

#### [NEW] [test_sommer_reference.R](file:///Users/nico/Desktop/Projects/py-sommer/tests/test_sommer_reference.R)

Comprehensive R test script that:
- Tests each core function with known inputs
- Saves numerical outputs to CSV/JSON files in `tests/reference_data/`
- Covers: `A.mat`, `D.mat`, `E.mat`, `H.mat`, `AR1`, `CS`, `ARMA`, [nearPDcpp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#291-293), [scaleCpp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#154-168), [makeFull](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#94-96), and a simple `mmes()` univariate model
- Uses small, reproducible datasets (synthetic marker matrices, the built-in datasets)

---

### Python Package (`pysommer/`)

#### [NEW] [__init__.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/__init__.py)

Package entry point, exports all public functions.

#### [NEW] [utils.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/utils.py)

Python equivalents of C++ utility functions:

| C++ function | Python implementation |
|---|---|
| [seqCpp(a, b)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#36-48) | `np.arange(a, b+1)` |
| [varCols(x)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#130-153) | `np.var(x, axis=0, ddof=1)` |
| [scaleCpp(x)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#154-168) | Center + scale by column std |
| [makeFull(X)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#94-96) | SVD-based full-rank column extraction |
| [isIdentity_mat(x)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#105-107) | `np.allclose(x, np.eye(n))` |
| [isDiagonal_mat(x)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#218-229) | Check off-diagonals are zero |
| [mat_to_vecCpp(x, x2)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#49-73) | Extract upper-tri elements where constraint > 0 |
| [vec_to_matCpp(x, x2)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#48-50) | Reconstruct matrix from vector + constraint |

#### [NEW] [relationships.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/relationships.py)

Relationship matrix functions:

| R/C++ function | Description |
|---|---|
| `A_mat(X)` | Additive (VanRaden) genomic relationship matrix |
| `D_mat(X)` | Dominance relationship matrix (Nishio or Su) |
| `E_mat(X)` | Epistatic relationship matrix (Hadamard product) |
| `H_mat(A, G)` | Combined pedigree-genomic H matrix |

#### [NEW] [covariance.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/covariance.py)

Covariance structure constructors:

| R function | Python |
|---|---|
| `AR1(n, rho)` | AR(1) correlation matrix |
| `CS(n, rho)` | Compound symmetry matrix |
| `ARMA(n, rho, lambda)` | ARMA correlation matrix |

#### [NEW] [nearPD.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/nearPD.py)

[nearPD(X, maxit, eig_tol, conv_tol)](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#291-293) — Nearest positive definite matrix algorithm, translated from the C++ [nearPDcpp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/RcppExports.cpp#291-293).

#### [NEW] [solver.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/solver.py)

The REML solver, translated from [newton_di_sp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp#521-1283) in [MNR.cpp](file:///Users/nico/Desktop/Projects/py-sommer/upstream_r_sommer/src/MNR.cpp). This is the largest and most complex piece (~600 lines of C++ → ~400 lines of Python). Implements:
- V matrix construction from variance components
- Projection matrix P = Vi - Vi X (X'Vi X)^-1 X' Vi  
- Score (first derivatives) and Information matrix (second derivatives)
- NR / AI iteration with parameter restraint
- BLUP and PEV calculation on last iteration

#### [NEW] [mmes.py](file:///Users/nico/Desktop/Projects/py-sommer/pysommer/mmes.py)

A simplified `mmes()` function that takes explicit matrices (Y, X, Z, K) rather than formulas. This wraps the solver with:
- Variance component initialization
- Step weight and EM weight scheduling
- Result packaging (beta, u, theta, etc.)

---

### Python Test Suite

#### [NEW] [test_pysommer.py](file:///Users/nico/Desktop/Projects/py-sommer/tests/test_pysommer.py)

pytest-based test suite that:
- Tests each function independently with the same inputs used in R
- Loads reference outputs from `tests/reference_data/`
- Compares Python vs R results within `atol=1e-6, rtol=1e-4`
- Tests solver convergence on a simple model

---

## Verification Plan

### Automated Tests

1. **R reference generation** (run once to generate reference data):
   ```bash
   cd /Users/nico/Desktop/Projects/py-sommer
   Rscript tests/test_sommer_reference.R
   ```
   This creates `tests/reference_data/*.csv` with known-good outputs.

2. **Python test suite**:
   ```bash
   cd /Users/nico/Desktop/Projects/py-sommer
   python -m pytest tests/test_pysommer.py -v
   ```
   Validates all Python implementations against R reference values.

3. **Cross-validation checks**:
   - Relationship matrices: element-wise comparison within tolerance
   - Covariance structures: exact match (pure arithmetic)
   - Solver: variance component estimates within 1% of R values
   - BLUPs: correlation > 0.999 with R BLUPs

### Manual Verification
- User can run both R and Python on the same dataset and compare results side-by-side
- A Jupyter notebook could be created later to demonstrate equivalence
