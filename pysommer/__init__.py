"""pysommer: Python translation of sommer computational core (Phase 1)."""

from .covariance import AR1, ARMA, CS
from .mmes import mmes
from .nearPD import nearPD, near_pd
from .relationships import A_mat, D_mat, E_mat, H_mat
from .solver import SolverResult, newton_di_sp
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
    "A_mat",
    "AR1",
    "ARMA",
    "CS",
    "D_mat",
    "E_mat",
    "H_mat",
    "SolverResult",
    "is_diagonal_mat",
    "is_identity_mat",
    "make_full",
    "mat_to_vec_cpp",
    "mmes",
    "nearPD",
    "near_pd",
    "newton_di_sp",
    "scale_cpp",
    "seq_cpp",
    "var_cols",
    "vec_to_mat_cpp",
]
