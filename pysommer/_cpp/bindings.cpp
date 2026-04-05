// bindings.cpp — pybind11 module definition for _sommer_core
// Provides NumPy ↔ Armadillo conversions and exposes all C++ functions.

#define ARMA_DONT_PRINT_ERRORS
#define ARMA_64BIT_WORD 1
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ============================================================================
// Forward declarations of functions from sommer_core.cpp
// ============================================================================
arma::vec seqCpp(const int & a, const int & b);
arma::vec mat_to_vecCpp(const arma::mat & x, const arma::mat & x2);
arma::mat vec_to_matCpp(const arma::vec & x, const arma::mat & x2);
arma::cube vec_to_cubeCpp(const arma::vec & x, const std::vector<arma::mat> & g);
arma::vec varCols(const arma::mat & x);
arma::mat scaleCpp(const arma::mat & x);
arma::mat makeFull(const arma::mat & X);
bool isIdentity_mat(const arma::mat x);
bool isIdentity_spmat(const arma::sp_mat x);
bool isDiagonal_mat(const arma::mat x);
bool isDiagonal_spmat(const arma::sp_mat x);
arma::mat amat(const arma::mat & Xo, const bool & vanraden, double minMAF);
arma::mat dmat(const arma::mat & Xo, const bool & nishio, double minMAF);
arma::mat emat(const arma::mat & X1, const arma::mat & X2);
arma::mat hmat(const arma::mat & A, const arma::mat & G22, const arma::vec & index, double tolparinv, double tau, double omega);
arma::cube scorecalc(const arma::mat & Mimv, const arma::mat & Ymv, const arma::mat & Zmv, const arma::mat & Xmv, const arma::mat & Vinv, int nt, double minMAF);
arma::cube gwasForLoop(const arma::mat & M, const arma::mat & Y, const arma::mat & Z, const arma::mat & X, const arma::mat & Vinv, double minMAF, bool display_progress);
arma::mat nearPDcpp(const arma::mat X0, const int & maxit, const double & eig_tol, const double & conv_tol);
arma::vec mat_to_vecCpp2(const arma::mat & x, const arma::mat & x2);

py::dict newton_di_sp_cpp(
    const arma::sp_mat & Y, const std::vector<arma::sp_mat> & X,
    const std::vector<arma::mat> & Gx, const std::vector<arma::sp_mat> & Z,
    const std::vector<arma::sp_mat> & K, const std::vector<arma::sp_mat> & R,
    const std::vector<arma::mat> & Ge, const std::vector<arma::mat> & GeI,
    const arma::sp_mat & W, const bool & isInvW,
    int iters, double tolpar, double tolparinv,
    const bool & ai, const bool & pev,
    const bool & verbose, const bool & retscaled,
    const arma::vec & stepweight, const arma::vec & emweight,
    const std::vector<arma::mat> & thetaConstOri, const arma::vec & thetaIndex);

py::dict ai_mme_sp_cpp(
    const arma::sp_mat & X, const std::vector<arma::sp_mat> & ZI,
    const arma::vec & Zind, const std::vector<arma::sp_mat> & AiI,
    const arma::sp_mat & y0, const std::vector<arma::sp_mat> & SI,
    const std::vector<arma::mat> & partitionsS,
    const arma::sp_mat & H, const bool & useH,
    int nIters, double tolParConvLL, double tolParConvNorm,
    double tolParInv, const std::vector<arma::mat> & thetaI,
    const std::vector<arma::mat> & thetaCI, const arma::mat & thetaF,
    const arma::vec & addScaleParam, const arma::vec & weightEmInf,
    const arma::vec & weightInf, const bool & verbose);

py::dict MNR_cpp(
    const arma::mat & Y, const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & Gx, const std::vector<arma::mat> & Z,
    const std::vector<arma::mat> & K, const std::vector<arma::sp_mat> & R,
    const std::vector<arma::mat> & Ge, const std::vector<arma::mat> & GeI,
    const arma::mat & W, const bool & isInvW,
    int iters, double tolpar, double tolparinv,
    const bool & ai, const bool & pev,
    const bool & verbose, const bool & retscaled,
    const arma::vec & stepweight, const arma::vec & emweight);


// ============================================================================
// NumPy ↔ Armadillo conversion helpers
// ============================================================================

// Convert a NumPy 2D array to arma::mat (column-major copy)
arma::mat numpy_to_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim == 1) {
        return arma::mat(static_cast<double*>(buf.ptr), buf.shape[0], 1, true);
    }
    if (buf.ndim != 2) throw std::runtime_error("Expected 1D or 2D array for mat conversion");
    // NumPy is row-major, Armadillo is column-major → copy and transpose
    arma::mat m(static_cast<double*>(buf.ptr), buf.shape[1], buf.shape[0], true);
    return m.t();
}

// Convert arma::mat to NumPy 2D array
py::array_t<double> mat_to_numpy(const arma::mat & m) {
    // Armadillo is col-major, NumPy expects row-major → transpose first
    arma::mat mt = m.t();
    auto result = py::array_t<double>({(ssize_t)m.n_rows, (ssize_t)m.n_cols});
    auto buf = result.request();
    std::memcpy(buf.ptr, mt.memptr(), mt.n_elem * sizeof(double));
    return result;
}

// Convert a NumPy 1D array to arma::vec
arma::vec numpy_to_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) throw std::runtime_error("Expected 1D array for vec conversion");
    return arma::vec(static_cast<double*>(buf.ptr), buf.shape[0], true);
}

// Convert arma::vec to NumPy 1D array
py::array_t<double> vec_to_numpy(const arma::vec & v) {
    auto result = py::array_t<double>({(ssize_t)v.n_elem});
    auto buf = result.request();
    std::memcpy(buf.ptr, v.memptr(), v.n_elem * sizeof(double));
    return result;
}

// Convert arma::cube to NumPy 3D array (slices × rows × cols)
py::array_t<double> cube_to_numpy(const arma::cube & c) {
    auto result = py::array_t<double>({(ssize_t)c.n_rows, (ssize_t)c.n_cols, (ssize_t)c.n_slices});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t s = 0; s < c.n_slices; ++s) {
        for (size_t r = 0; r < c.n_rows; ++r) {
            for (size_t col = 0; col < c.n_cols; ++col) {
                ptr[r * c.n_cols * c.n_slices + col * c.n_slices + s] = c(r, col, s);
            }
        }
    }
    return result;
}

// Convert NumPy 2D array to arma::sp_mat (dense → sparse)
arma::sp_mat numpy_to_spmat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    return arma::sp_mat(numpy_to_mat(arr));
}

// Convert scipy CSC sparse matrix components to arma::sp_mat
arma::sp_mat csc_to_spmat(py::array_t<double> data,
                           py::array_t<int> indices,
                           py::array_t<int> indptr,
                           int nrow, int ncol) {
    auto d_buf = data.request();
    auto i_buf = indices.request();
    auto p_buf = indptr.request();

    arma::urowvec arm_i(i_buf.shape[0]);
    arma::urowvec arm_p(p_buf.shape[0]);
    arma::vec arm_x(d_buf.shape[0]);

    int* i_ptr = static_cast<int*>(i_buf.ptr);
    int* p_ptr = static_cast<int*>(p_buf.ptr);
    double* d_ptr = static_cast<double*>(d_buf.ptr);

    for (ssize_t k = 0; k < i_buf.shape[0]; ++k) arm_i(k) = i_ptr[k];
    for (ssize_t k = 0; k < p_buf.shape[0]; ++k) arm_p(k) = p_ptr[k];
    for (ssize_t k = 0; k < d_buf.shape[0]; ++k) arm_x(k) = d_ptr[k];

    return arma::sp_mat(arm_i, arm_p, arm_x, nrow, ncol);
}


// ============================================================================
// PYBIND11 MODULE
// ============================================================================

PYBIND11_MODULE(_sommer_core, m) {
    m.doc() = "C++ core of the sommer mixed model solver (adapted from R sommer MNR.cpp)";

    // --- Utility functions ---
    m.def("seq_cpp", [](int a, int b) -> py::array_t<double> {
        return vec_to_numpy(seqCpp(a, b));
    }, py::arg("a"), py::arg("b"), "Inclusive integer sequence [a, b]");

    m.def("mat_to_vec_cpp", [](py::array_t<double> x, py::array_t<double> x2) -> py::array_t<double> {
        return vec_to_numpy(mat_to_vecCpp(numpy_to_mat(x), numpy_to_mat(x2)));
    }, py::arg("x"), py::arg("x2"));

    m.def("vec_to_mat_cpp", [](py::array_t<double> x, py::array_t<double> x2) -> py::array_t<double> {
        return mat_to_numpy(vec_to_matCpp(numpy_to_vec(x), numpy_to_mat(x2)));
    }, py::arg("x"), py::arg("x2"));

    m.def("vec_to_cube_cpp", [](py::array_t<double> x, py::list g) -> py::array_t<double> {
        arma::vec xv = numpy_to_vec(x);
        std::vector<arma::mat> gv;
        for (auto item : g) gv.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        return cube_to_numpy(vec_to_cubeCpp(xv, gv));
    }, py::arg("x"), py::arg("g"));

    m.def("var_cols", [](py::array_t<double> x) -> py::array_t<double> {
        return vec_to_numpy(varCols(numpy_to_mat(x)));
    }, py::arg("x"), "Column-wise variance (Welford's algorithm)");

    m.def("scale_cpp", [](py::array_t<double> x) -> py::array_t<double> {
        return mat_to_numpy(scaleCpp(numpy_to_mat(x)));
    }, py::arg("x"), "Center and scale columns");

    m.def("make_full", [](py::array_t<double> x) -> py::array_t<double> {
        return mat_to_numpy(makeFull(numpy_to_mat(x)));
    }, py::arg("x"), "SVD-based full-rank column basis");

    m.def("is_identity_mat", [](py::array_t<double> x) -> bool {
        return isIdentity_mat(numpy_to_mat(x));
    }, py::arg("x"));

    m.def("is_diagonal_mat", [](py::array_t<double> x) -> bool {
        return isDiagonal_mat(numpy_to_mat(x));
    }, py::arg("x"));

    // --- Relationship matrices ---
    m.def("amat", [](py::array_t<double> Xo, bool vanraden, double minMAF) -> py::array_t<double> {
        return mat_to_numpy(amat(numpy_to_mat(Xo), vanraden, minMAF));
    }, py::arg("Xo"), py::arg("vanraden") = true, py::arg("minMAF") = 0.0,
    "Genomic relationship matrix (additive)");

    m.def("dmat", [](py::array_t<double> Xo, bool nishio, double minMAF) -> py::array_t<double> {
        return mat_to_numpy(dmat(numpy_to_mat(Xo), nishio, minMAF));
    }, py::arg("Xo"), py::arg("nishio") = true, py::arg("minMAF") = 0.0,
    "Dominance relationship matrix");

    m.def("emat", [](py::array_t<double> X1, py::array_t<double> X2) -> py::array_t<double> {
        return mat_to_numpy(emat(numpy_to_mat(X1), numpy_to_mat(X2)));
    }, py::arg("X1"), py::arg("X2"), "Hadamard (element-wise) product");

    m.def("hmat", [](py::array_t<double> A, py::array_t<double> G22,
                      py::array_t<double> index, double tolparinv,
                      double tau, double omega) -> py::array_t<double> {
        return mat_to_numpy(hmat(numpy_to_mat(A), numpy_to_mat(G22),
                                 numpy_to_vec(index), tolparinv, tau, omega));
    }, py::arg("A"), py::arg("G22"), py::arg("index"),
       py::arg("tolparinv") = 1e-6, py::arg("tau") = 1.0, py::arg("omega") = 1.0,
    "Single-step H matrix");

    // --- GWAS ---
    m.def("scorecalc", [](py::array_t<double> Mimv, py::array_t<double> Ymv,
                           py::array_t<double> Zmv, py::array_t<double> Xmv,
                           py::array_t<double> Vinv, int nt, double minMAF) -> py::array_t<double> {
        return cube_to_numpy(scorecalc(numpy_to_mat(Mimv), numpy_to_mat(Ymv),
                                        numpy_to_mat(Zmv), numpy_to_mat(Xmv),
                                        numpy_to_mat(Vinv), nt, minMAF));
    }, py::arg("Mimv"), py::arg("Ymv"), py::arg("Zmv"), py::arg("Xmv"),
       py::arg("Vinv"), py::arg("nt"), py::arg("minMAF") = 0.0);

    m.def("gwas_for_loop", [](py::array_t<double> M, py::array_t<double> Y,
                               py::array_t<double> Z, py::array_t<double> X,
                               py::array_t<double> Vinv, double minMAF,
                               bool display_progress) -> py::array_t<double> {
        return cube_to_numpy(gwasForLoop(numpy_to_mat(M), numpy_to_mat(Y),
                                          numpy_to_mat(Z), numpy_to_mat(X),
                                          numpy_to_mat(Vinv), minMAF, display_progress));
    }, py::arg("M"), py::arg("Y"), py::arg("Z"), py::arg("X"),
       py::arg("Vinv"), py::arg("minMAF") = 0.0, py::arg("display_progress") = false);

    // --- nearPD ---
    m.def("near_pd_cpp", [](py::array_t<double> X0, int maxit, double eig_tol, double conv_tol) -> py::array_t<double> {
        return mat_to_numpy(nearPDcpp(numpy_to_mat(X0), maxit, eig_tol, conv_tol));
    }, py::arg("X0"), py::arg("maxit") = 100, py::arg("eig_tol") = 1e-6, py::arg("conv_tol") = 1e-7,
    "Nearest positive-definite projection");

    m.def("mat_to_vec_cpp2", [](py::array_t<double> x, py::array_t<double> x2) -> py::array_t<double> {
        return vec_to_numpy(mat_to_vecCpp2(numpy_to_mat(x), numpy_to_mat(x2)));
    }, py::arg("x"), py::arg("x2"));

    // --- Solvers ---
    m.def("newton_di_sp", [](
        py::array_t<double> Y, py::list X_list, py::list Gx_list,
        py::list Z_list, py::list K_list, py::list R_list,
        py::list Ge_list, py::list GeI_list,
        py::array_t<double> W, bool isInvW,
        int iters, double tolpar, double tolparinv,
        bool ai_flag, bool pev, bool verbose, bool retscaled,
        py::array_t<double> stepweight, py::array_t<double> emweight_arr,
        py::list thetaConstOri_list, py::array_t<double> thetaIndex_arr
    ) -> py::dict {
        // Convert inputs
        arma::sp_mat Y_sp = numpy_to_spmat(Y);
        std::vector<arma::sp_mat> X_v, Z_v, K_v, R_v;
        std::vector<arma::mat> Gx_v, Ge_v, GeI_v, tCO_v;
        for (auto item : X_list) X_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : Gx_list) Gx_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : Z_list) Z_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : K_list) K_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : R_list) R_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : Ge_list) Ge_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : GeI_list) GeI_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : thetaConstOri_list) tCO_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        arma::sp_mat W_sp = numpy_to_spmat(W);
        arma::vec sw = numpy_to_vec(stepweight);
        arma::vec ew = numpy_to_vec(emweight_arr);
        arma::vec ti = numpy_to_vec(thetaIndex_arr);

        return newton_di_sp_cpp(Y_sp, X_v, Gx_v, Z_v, K_v, R_v, Ge_v, GeI_v,
                                W_sp, isInvW, iters, tolpar, tolparinv,
                                ai_flag, pev, verbose, retscaled, sw, ew, tCO_v, ti);
    }, "Direct inversion REML solver (newton_di_sp)");

    m.def("ai_mme_sp", [](
        py::array_t<double> X, py::list ZI_list, py::array_t<double> Zind_arr,
        py::list AiI_list, py::array_t<double> y0,
        py::list SI_list, py::list partitionsS_list,
        py::array_t<double> H, bool useH,
        int nIters, double tolParConvLL, double tolParConvNorm,
        double tolParInv, py::list thetaI_list, py::list thetaCI_list,
        py::array_t<double> thetaF, py::array_t<double> addScaleParam,
        py::array_t<double> weightEmInf, py::array_t<double> weightInf,
        bool verbose
    ) -> py::dict {
        arma::sp_mat X_sp = numpy_to_spmat(X);
        std::vector<arma::sp_mat> ZI_v, AiI_v, SI_v;
        std::vector<arma::mat> pS_v, tI_v, tCI_v;
        for (auto item : ZI_list) ZI_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : AiI_list) AiI_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : SI_list) SI_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : partitionsS_list) pS_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : thetaI_list) tI_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : thetaCI_list) tCI_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        arma::sp_mat H_sp = numpy_to_spmat(H);
        arma::sp_mat y0_sp = numpy_to_spmat(y0);

        return ai_mme_sp_cpp(X_sp, ZI_v, numpy_to_vec(Zind_arr), AiI_v, y0_sp,
                              SI_v, pS_v, H_sp, useH, nIters, tolParConvLL,
                              tolParConvNorm, tolParInv, tI_v, tCI_v,
                              numpy_to_mat(thetaF), numpy_to_vec(addScaleParam),
                              numpy_to_vec(weightEmInf), numpy_to_vec(weightInf), verbose);
    }, "Henderson MME REML with AI/EM hybrid (ai_mme_sp)");

    m.def("mnr", [](
        py::array_t<double> Y, py::list X_list, py::list Gx_list,
        py::list Z_list, py::list K_list, py::list R_list,
        py::list Ge_list, py::list GeI_list,
        py::array_t<double> W, bool isInvW,
        int iters, double tolpar, double tolparinv,
        bool ai_flag, bool pev, bool verbose, bool retscaled,
        py::array_t<double> stepweight, py::array_t<double> emweight_arr
    ) -> py::dict {
        arma::mat Y_m = numpy_to_mat(Y);
        std::vector<arma::mat> X_v, Gx_v, Z_v, K_v, Ge_v, GeI_v;
        std::vector<arma::sp_mat> R_v;
        for (auto item : X_list) X_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : Gx_list) Gx_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : Z_list) Z_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : K_list) K_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : R_list) R_v.push_back(numpy_to_spmat(item.cast<py::array_t<double>>()));
        for (auto item : Ge_list) Ge_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        for (auto item : GeI_list) GeI_v.push_back(numpy_to_mat(item.cast<py::array_t<double>>()));
        arma::mat W_m = numpy_to_mat(W);

        return MNR_cpp(Y_m, X_v, Gx_v, Z_v, K_v, R_v, Ge_v, GeI_v,
                       W_m, isInvW, iters, tolpar, tolparinv,
                       ai_flag, pev, verbose, retscaled,
                       numpy_to_vec(stepweight), numpy_to_vec(emweight_arr));
    }, "Dense Newton-Raphson REML solver (MNR)");

    // --- CSC sparse helper ---
    m.def("csc_to_spmat", &csc_to_spmat,
          py::arg("data"), py::arg("indices"), py::arg("indptr"),
          py::arg("nrow"), py::arg("ncol"),
          "Convert scipy CSC sparse matrix components to internal sparse format");
}
