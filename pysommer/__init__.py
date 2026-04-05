"""pysommer: Python translation of sommer computational core (Phase 1)."""

__version__ = "0.3.0"

from .covariance import AR1, ARMA, CS
from .formula import dsm, ism, usm, vsm
from .gwas import gwasForLoop, scorecalc
from .mmes import mmes, mmes_formula
from .nearPD import nearPD, near_pd
from .predict import predict_mmes, summarize_predictions
from .relationships import A_mat, D_mat, E_mat, H_mat
from .sklearn import MMESFormulaRegressor, MMESRegressor
from .solver import SolverResult, ai_mme_sp, newton_di_sp
from .utils import (
    is_diagonal_mat,
    is_identity_mat,
    make_full,
    mat_to_vec_cpp,
    scale_cpp,
    seq_cpp,
    var_cols,
    vec_to_mat_cpp,
)

__all__ = [
    "__version__",
    "A_mat",
    "AR1",
    "ARMA",
    "CS",
    "D_mat",
    "E_mat",
    "H_mat",
    "SolverResult",
    "ai_mme_sp",
    "dsm",
    "gwasForLoop",
    "is_diagonal_mat",
    "MMESFormulaRegressor",
    "MMESRegressor",
    "is_identity_mat",
    "ism",
    "make_full",
    "mat_to_vec_cpp",
    "mmes",
    "mmes_formula",
    "nearPD",
    "near_pd",
    "newton_di_sp",
    "scale_cpp",
    "scorecalc",
    "seq_cpp",
    "predict_mmes",
    "summarize_predictions",
    "usm",
    "var_cols",
    "vec_to_mat_cpp",
    "vsm",
]
