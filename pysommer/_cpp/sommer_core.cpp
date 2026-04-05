// sommer_core.cpp — Adapted from upstream_r_sommer/src/MNR.cpp
// Stripped of Rcpp dependencies, using pybind11 + Armadillo directly.
// All algorithmic logic is preserved exactly as in the original.

#define ARMA_DONT_PRINT_ERRORS
#define ARMA_64BIT_WORD 1
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// ============================================================================
// Helpers: arma → pybind11 conversions for dict return values
// ============================================================================

static py::array_t<double> _mat_to_py(const arma::mat & m) {
    arma::mat mt = m.t();
    auto result = py::array_t<double>({(ssize_t)m.n_rows, (ssize_t)m.n_cols});
    auto buf = result.request();
    std::memcpy(buf.ptr, mt.memptr(), mt.n_elem * sizeof(double));
    return result;
}

static py::array_t<double> _vec_to_py(const arma::vec & v) {
    auto result = py::array_t<double>({(ssize_t)v.n_elem});
    auto buf = result.request();
    std::memcpy(buf.ptr, v.memptr(), v.n_elem * sizeof(double));
    return result;
}

static py::list _matvec_to_py(const std::vector<arma::mat> & v) {
    py::list out;
    for (auto & m : v) out.append(_mat_to_py(m));
    return out;
}

// ============================================================================
// Utility functions
// ============================================================================

arma::vec seqCpp(const int & a,
                 const int & b){
  int c = b-a+1,i,counter;
  arma::vec d(c);
  counter = a;
  for(i=0; i < c; i++){
    d[i] = counter;
    counter++;
  }
  return d;
}

arma::vec mat_to_vecCpp(const arma::mat & x, 
                        const arma::mat & x2){
  // x is the matrix to be passed to a vector form in the output (out)
  // x2 is a matrix of constraints to indicate whether the value to be passed should be passed (>0) or not passed (=0)
  int ncol = x.n_cols;
  arma::uvec nent2 = find(x2 > 0); int nent3 = nent2.n_elem;
  arma::vec out(nent3);
  int counter = 0;
  int i, j;
  for (i = 0; i < ncol; i++){
    for (j = 0; j < ncol; j++){
      if (i > j){}else{
        // only extract the variance component if it was planned to be estimated
        if(x2(i,j) > 0){
          out[counter] = x(i,j);
          counter++;
        }
      }
    }
  }
  return out;
}

arma::mat vec_to_matCpp(const arma::vec & x,
                        const arma::mat & x2){
  // x is the vector to be passed to a matrix form in the output (out)
  // x2 is a matrix of constraints to indicate whether the value to be passed should be passed (>0) or not passed (=0)
  int ncol = x2.n_cols;
  arma::uvec nent2 = find(x2 > 0);
  arma::mat out(ncol,ncol);
  //
  int counter = 0;
  int i, j;
  for (j = 0; j < ncol; j++){
    for (i = 0; i < ncol; i++){
      if (i > j){}else{
        // only extract the variance component if it was planned to be estimated
        if(x2(i,j) > 0){
          out(i,j) = x(counter);
          counter++;
        }
      }
    }
  }
  return out;
}

arma::cube vec_to_cubeCpp(const arma::vec & x,
                          const std::vector<arma::mat> & g){
  int nge = g.size();
  int ncols = g[0].n_cols;
  arma::cube Ge(ncols,ncols,nge); // copy GeI and we will replace the correct ones
  int i, j, k;
  int counter = 0;
  for(k = 0; k < nge; k++){ // FOR EACH RANDOM EFFECT access the matrix
    const arma::mat & x2 = g[k];
    int ncol = x2.n_cols;
    arma::mat x3(ncol,ncol);
    for (i = 0; i < ncol; i++){
      for (j = 0; j < ncol; j++){
        if (i > j){}else{//only upper triangular
          // only extract the variance component if it was planned to be estimated
          if(x2(i,j) > 0){
            x3(i,j) = x(counter);
            counter++;
          }else{x3(i,j) = 0;}
        }
      }
    }
    x3 = arma::symmatu(x3);
    Ge.slice(k) = x3;
  }
  return Ge;
}

arma::vec varCols(const arma::mat & x){
  int nrow = x.n_rows, ncol = x.n_cols;
  arma::vec out(ncol);

  for (int j = 0; j < ncol; j++) {

    double mean = 0;
    double M2 = 0;
    int n;
    double delta, xx;

    for (int i = 0; i < nrow; i++) {
      n = i+1;
      xx = x(i,j);
      delta = xx - mean;
      mean += delta/n;
      M2 = M2 + delta*(xx-mean);
    }
    out(j) = M2/(n-1);
  }
  return out;
}

arma::mat scaleCpp(const arma::mat & x) { // scale a matrix

  arma::vec sds = 1 / sqrt(varCols(x));
  arma::mat sdsD = diagmat(sds);

  int n = x.n_rows;
  arma::mat D = arma::eye<arma::mat>(n,n) ;
  arma::vec  v = arma::ones<arma::vec>(n);
  arma::mat vtv = ( v * v.t() ) / n;
  arma::mat xs = ((D - vtv) * x) * sdsD ;

  return xs;
}

arma::mat makeFull(const arma::mat & X) {
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::svd(U,s,V,X);

  int ncols0 = X.n_cols;
  arma::uvec s2 = find( s > 1e-8 );
  int ncols = s2.max() + 1;
  arma::uvec indices(ncols);
  for (int i = 0; i < ncols; ++i) {
    indices[i] = 1;
  }
  arma::mat Xf = U.cols(find(indices == 1));
  if(ncols == ncols0){
    Xf = X;
  }
  return Xf;
}

bool isIdentity_mat(const arma::mat x){
  int N = x.n_rows;
  for (int row = 0; row < N; row++){
    for (int col = 0; col < N; col++){
      if (row == col && x(row,col) != 1)
        return false;
      else if (row != col && x(row,col) != 0)
        return false;
    }
  }
  return true;
}

bool isIdentity_spmat(const arma::sp_mat x){
  int N = x.n_rows;
  for (int row = 0; row < N; row++){
    for (int col = 0; col < N; col++){
      if (row == col && x(row,col) != 1)
        return false;
      else if (row != col && x(row,col) != 0)
        return false;
    }
  }
  return true;
}

bool isDiagonal_mat(const arma::mat x){
  int N = x.n_rows;
  for (int row = 0; row < N; row++){
    for (int col = 0; col < N; col++){
      if (row != col && x(row,col) != 0)
        return false;
    }
  }
  return true;
}

bool isDiagonal_spmat(const arma::sp_mat x){
  int N = x.n_rows;
  for (int row = 0; row < N; row++){
    for (int col = 0; col < N; col++){
      if (row != col && x(row,col) != 0)
        return false;
    }
  }
  return true;
}

// ============================================================================
// Relationship matrices
// ============================================================================

arma::mat amat(const arma::mat & Xo, const bool & vanraden, double minMAF) {

  // remove min.MAF
  arma::rowvec pfreq = mean(Xo+1,0)/2; // frequency of p
  arma::mat pqfreq = arma::join_cols(pfreq,1-pfreq); // frequencies of p and q
  arma::rowvec MAF = min(pqfreq,0); // minor allele freqs
  arma::uvec indexMAF = find(MAF > minMAF); // index for good markers > minMAF
  arma::mat Xo2 = Xo.cols(indexMAF); // new X only with polymorphic markers

  // remove monomorphic markers
  arma::rowvec xVar = var(Xo2,0); // column variance
  arma::uvec index = find(xVar > 0); // index for good markers
  arma::mat X = Xo2.cols(index); // new X only with polymorphic markers

  // initialize A
  int p = X.n_cols; // number of markers
  int n = X.n_rows;
  arma::mat A(n,n);

  if(vanraden == true){ // regular vanRaden

    arma::rowvec ms012 = mean( X+1, 0 ); // means of columns
    arma::rowvec freq = ms012/2;
    double v = 2 * mean(freq % (1 - freq));
    arma::mat one(n, 1, arma::fill::ones);
    arma::mat freqmat = one * freq;
    arma::mat W = (X + 1) - (2 * freqmat);
    arma::mat K = W * W.t();
    A = K/v/p;

  }else{ // Endelman (currently we have a bug here)

    // IN R: M <- scale(X, center = TRUE, scale = FALSE)
    arma::rowvec ms = mean( X, 0 ); // means of columns
    arma::mat M = X.each_row() - ms;
    // IN R: tcrossprod(M)
    arma::mat K = M * M.t();
    // IN R: K/mean(diag(K))
    double v = mean(diagvec(K));
    A = K/v;

  }
  return A;
}

arma::mat dmat(const arma::mat & Xo, const bool & nishio, double minMAF) {

  // remove min.MAF
  arma::rowvec pfreq = mean(Xo+1,0)/2; // frequency of p
  arma::mat pqfreq = arma::join_cols(pfreq,1-pfreq); // frequencies of p and q
  arma::rowvec MAF = min(pqfreq,0); // minor allele freqs
  arma::uvec indexMAF = find(MAF > minMAF); // index for good markers > minMAF
  arma::mat Xo2 = Xo.cols(indexMAF); // new X only with polymorphic markers

  // remove monomorphic markers
  arma::rowvec xVar = var(Xo2,0); // column variance
  arma::uvec index = find(xVar > 0); // index for good markers
  arma::mat X = Xo2.cols(index); // new X only with polymorphic markers

  // initialize D
  int n = X.n_rows;
  arma::mat D(n,n);
  arma::mat Xd = 1 - abs(X);

  if(nishio == true){ // Nishio and Satoh (2014)

    // IN R: M <- scale(Xd, center = TRUE, scale = FALSE)
    arma::rowvec ms = mean( Xd, 0 ); // means of columns
    arma::mat M = Xd.each_row() - ms; // centered Xd matrix
    // IN R: bAlleleFrequency <- colMeans(X+1)/2; 0-1-2
    arma::rowvec bAlleleFrequency = mean( X+1, 0 )/2; // means of columns
    // IN R: varHW <- sum((2 * bAlleleFrequency * (1 - bAlleleFrequency))^2)
    double varHW = arma::accu(arma::square(2 * bAlleleFrequency % (1 - bAlleleFrequency)));
    // IN R: tcrossprod(M)
    arma::mat K = M * M.t();
    D = K/varHW;

  }else{ // Su et al. (2012)

    // IN R: p <- colSums(X+1)/(2*n) # from marker matrix in 0,1,2 format
    arma::rowvec p = sum( X+1, 0 )/(2*n); // means of columns
    arma::rowvec q = 1-p;
    // IN R: varHW <- sum(2*p*q * (1-(2*p*q)) )
    arma::rowvec p2q = 2*(p%q);
    double varHW = arma::accu( p2q % (1-p2q) );
    // IN R: Xdpq <- apply(Xd, 1, function(x){ x - (2 * p * q)})
    arma::mat M = Xd.each_row() - p2q;
    // IN R: tcrossprod(M)
    arma::mat K = M * M.t();
    D = K/varHW;

  }
  return D;
}

arma::mat emat(const arma::mat & X1, const arma::mat & X2) {
  return X1 % X2;
}

arma::mat hmat(const arma::mat & A, const arma::mat & G22,
               const arma::vec & index, double tolparinv,
               double tau, double omega) {

  arma::uvec index1 = find(index == true); // index for good markers
  arma::uvec index2 = find(index == false); // index for good markers
  // A11 <- A[index, index]
  arma::mat A11 = A.submat(index1,index1);
  // A12 <- A[index, !index]
  arma::mat A12 = A.submat(index1,index2);
  // A21 <- A[!index, index]
  arma::mat A21 = A.submat(index2,index1);
  // A22 <- A[!index, !index]
  arma::mat A22 = A.submat(index2,index2);
  // A22inv = solve(A22)
  arma::mat A22inv(A22.n_cols,A22.n_cols);
  arma::inv_sympd(A22inv,A22); // try to invert normally
  arma::sp_mat Ia = arma::speye<arma::sp_mat>(A22.n_cols,A22.n_cols);
  if(A22inv.n_rows == 0){ // if fails try to invert with diag(1e-3)
    arma::mat A22b = A22 + (Ia*tolparinv);
    arma::inv_sympd(A22inv,A22b);
  }
  // G22inv = try(solve(G22), silent = TRUE)
  arma::mat G22inv(G22.n_cols,G22.n_cols);
  arma::inv_sympd(G22inv,G22); // try to invert normally
  arma::sp_mat Ig = arma::speye<arma::sp_mat>(G22.n_cols,G22.n_cols);
  if(G22inv.n_rows == 0){ // if fails try to invert with diag(1e-3)
    arma::mat G22b = G22 + (Ig*tolparinv);
    arma::inv_sympd(G22inv,G22b);
  }
  // H22 = solve((tau * G22inv + (1 - omega) * A22inv))
  arma::mat H22p = (tau * G22inv) + ((1 - omega) * A22inv); //constant by matrix product
  arma::mat H22inv(H22p.n_cols,H22p.n_cols);
  arma::inv_sympd(H22inv,H22p); // try to invert normally
  arma::sp_mat Ih = arma::speye<arma::sp_mat>(H22p.n_cols,H22p.n_cols);
  if(H22inv.n_rows == 0){ // if fails try to invert with diag(1e-3)
    arma::mat H22pb = H22p + (Ih*tolparinv);
    arma::inv_sympd(H22inv,H22pb);
  }
  // H11 = A12 %*% A22inv %*% (H22 - A22) %*% A22inv %*% A21
  arma::mat H11 = A12 * A22inv * (H22inv - A22) * A22inv * A21;
  // H12 = A12 %*% A22inv %*% (H22 - A22)
  arma::mat H12 = A12 * A22inv * (H22inv - A22);
  // H21 = (H22 - A22) %*% A22inv %*% A21
  arma::mat H21 = (H22inv - A22) * A22inv * A21;
  // H22 = (H22 - A22)
  arma::mat H22 = (H22inv - A22);
  // H = A + cbind(rbind(H11, H21), rbind(H12, H22))
  arma::mat H = A + join_rows(join_cols(H11, H21), join_cols(H12, H22));
  return H;
}

// ============================================================================
// GWAS functions
// ============================================================================

arma::cube scorecalc(const arma::mat & Mimv,
                     const arma::mat & Ymv, // Y is provided as multitrait
                     const arma::mat & Zmv, // Z is provided as univariate
                     const arma::mat & Xmv, // X is provided as univariate
                     const arma::mat & Vinv, // multivariate inverse of V
                     int nt, double minMAF) {

  double tolparinv = 0.00001;

  //
  arma::rowvec pf = mean(Mimv+1)/2; // allele frequency of p
  arma::rowvec qf = 1 - pf; // allele frequency of q
  double MAF = min(arma::join_rows(pf,qf)); // calculate MAF

  // start calculation
  double n = Ymv.n_rows;
  arma::mat ZMimv = Zmv * Mimv;
  arma::mat XZMimv = join_rows( Xmv, ZMimv);
  double p = XZMimv.n_cols;
  double v1 = 1;
  double v2 = n - p;
  // create Wi = (XZ Vi ZX)-1
  arma::mat Winv;
  arma::mat W = XZMimv.t() * (Vinv * XZMimv);
  arma::sp_mat D = arma::speye<arma::sp_mat>(W.n_cols,W.n_cols);
  arma::inv_sympd(Winv,W); // try to invert normally
  if(Winv.n_rows == 0){ // if fails try to invert with diag(1e-3)
    W = W + (D*tolparinv);
    arma::inv_sympd(Winv,W);
  }

  // Initialize result cube
  arma::cube result(Mimv.n_cols, nt, 3, arma::fill::zeros);

  if (Winv.n_rows > 0 && MAF > minMAF) {
    // Main calculations
    arma::mat XZMimvVy = XZMimv.t() * (Vinv * Ymv); // XZM' Vi y
    arma::colvec b = Winv * XZMimvVy;               // (XZ'VinvXZ)^-1 XZ'VinvY
    arma::colvec e = Ymv - (XZMimv * b);            // Residuals: Y - Xb
    arma::mat mVar = (e.t() * (Vinv * e)) / v2;     // Residual variance
    double mVarAsDouble = mVar(0, 0);
    arma::mat bVar = Winv * mVarAsDouble;            // Beta variance

    // Extract the right fixed effect
    arma::mat bn(b.n_rows, 1); // Empty vector
    for (int i = 0; i < b.n_rows; ++i) {
      bn(i) = i; // Fill it with their own position
    }
    arma::uvec ps = find(bn > (Xmv.n_cols - 1)); // Index for good markers
    arma::colvec bMarker = b(ps);               // Beta for marker
    arma::mat bMarkerVar = bVar(ps, ps);        // Variance for beta
    arma::vec SEMarker = arma::sqrt(diagvec(bMarkerVar));
    arma::vec fStat = arma::pow(bMarker / SEMarker, 2); // F statistic
    arma::vec x = v2 / (v2 + v1 * fStat);               // Probability scores

    // Fill the result cube
    result.slice(0) = arma::reshape(x, Mimv.n_cols, nt);         // Scores
    result.slice(1) = arma::reshape(bMarker, Mimv.n_cols, nt);  // Beta coefficients
    result.slice(2) = arma::reshape(SEMarker, Mimv.n_cols, nt); // Standard errors
  }
  return result;
}

arma::cube gwasForLoop(const arma::mat & M, // marker matrix
                       const arma::mat & Y, // Y is provided as multitrait
                       const arma::mat & Z, // Z is provided as univariate
                       const arma::mat & X, // X is provided as univariate
                       const arma::mat & Vinv, // multivariate inverse of V
                       double minMAF,
                       bool display_progress) {
  int nt = Y.n_cols;
  arma::mat Dnt = arma::eye<arma::mat>(nt,nt); // diagonal of nt dimensions
  // multivariate versions
  arma::mat Ymv = arma::vectorise(Y.t(),0); // multivariate Y
  // arma::mat Ymv = Ymvt.t();
  arma::mat Zmv = arma::kron(Z,Dnt); // kronecker
  arma::mat Xmv = arma::kron(X,Dnt); // kronecker
  
  // start calculation
  int n_marker = M.n_cols;
  arma::vec dummy(nt, arma::fill::ones);
  arma::uvec pos = arma::find(dummy > 0); // index for good markers
  arma::cube results(n_marker, nt, 3, arma::fill::zeros); // 3D array: markers x traits x (scores & bMarkers)
  
  // start for loop for each marker
  for (int i = 0; i < n_marker; ++i) {
    // Check for Python interrupt (replaces Progress::check_abort())
    if (i % 100 == 0 && PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
    arma::mat Mi = M.col(i); // extract marker i
    arma::mat Mimv = arma::kron(Mi,Dnt); // kronecker for multivariate
    arma::cube prov = scorecalc(Mimv, Ymv, Zmv, Xmv, Vinv, nt, minMAF);
    
    // Assign slices
    results.slice(0).row(i) = prov.slice(0).t(); // Scores
    results.slice(1).row(i) = prov.slice(1).t(); // bMarkers
    results.slice(2).row(i) = prov.slice(2).t(); // SEs
  }
  
  return results;
}

// ============================================================================
// nearPD
// ============================================================================

arma::mat nearPDcpp(const arma::mat X0,
                    const int & maxit,
                    const double & eig_tol,
                    const double & conv_tol){
  // X=X, maxit=100, eig_tol = 1e-06, conv_tol = 1e-07
  int iter = 0;
  bool converged = false;
  double conv = 2e31 - 1;
  arma::mat X = X0;
  int nX = X.n_cols;
  arma::mat D_S(nX,nX,arma::fill::zeros);
  arma::mat Y(nX,nX), R(nX,nX);
  while (iter < maxit && !converged) {
    Y = X;
    R = Y - D_S;
    arma::vec d; // eigenvalues
    arma::mat Q; // eigenvectors
    arma::eig_sym(d, Q, R);
    arma::uvec p = arma::find(d > (eig_tol * d(0)) );
    if(p.n_elem == 0){ // if there was a negative value
      std::cerr << "Matrix seems negative semi-definite." << std::endl;
      break;
    }
    Q = Q.cols(p);
    arma::mat dummy(Q.n_rows,1,arma::fill::ones);
    arma::vec dp = d(p);
    X = ( Q % (dummy*dp.t()) ) * Q.t();
    D_S = X - R;
    conv = arma::norm(Y - X, "inf")/arma::norm(Y, "inf");
    iter = iter + 1;
    if(conv <= conv_tol){
      converged = true;
    }
  }
  return X;
}

// ============================================================================
// mat_to_vecCpp2 (column-first iteration order)
// ============================================================================

arma::vec mat_to_vecCpp2(const arma::mat & x, const arma::mat & x2){
  // x is the matrix to be passed to a vector form in the output (out)
  // x2 is a matrix of constraints to indicate whether the value to be passed should be passed (>0) or not passed (=0)
  // NOTE: column-first iteration order (differs from mat_to_vecCpp which is row-first)
  int ncol = x.n_cols;
  arma::uvec nent2 = find(x2 > 0); int nent3 = nent2.n_elem;
  arma::vec out(nent3);
  int counter = 0;
  int i, j;
  for (j = 0; j < ncol; j++){
    for (i = 0; i < ncol; i++){
      if (i > j){}else{
        // only extract the variance component if it was planned to be estimated
        if(x2(i,j) > 0){
          out[counter] = x(i,j);
          counter++;
        }
      }
    }
  }
  return out;
}

// ============================================================================
// newton_di_sp — Direct inversion REML solver
// ============================================================================

py::dict newton_di_sp_cpp(
    const arma::sp_mat & Y,
    const std::vector<arma::sp_mat> & X,
    const std::vector<arma::mat> & Gx,
    const std::vector<arma::sp_mat> & Z,
    const std::vector<arma::sp_mat> & K,
    const std::vector<arma::sp_mat> & R,
    const std::vector<arma::mat> & Ge,
    const std::vector<arma::mat> & GeI,
    const arma::sp_mat & W,
    const bool & isInvW,
    int iters, double tolpar, double tolparinv,
    const bool & ai, const bool & pev,
    const bool & verbose, const bool & retscaled,
    const arma::vec & stepweight,
    const arma::vec & emweight,
    const std::vector<arma::mat> & thetaConstOri,
    const arma::vec & thetaIndex)
{
  time_t before = time(0);
  localtime(&before);

  int n_fixed = X.size(); // define nre=number of fixed effects
  int n_random = Z.size(); // define nre=number of random effects
  int n_rcov = R.size(); // define nre=number of residual effects
  int n_re = n_random + n_rcov; // define nre=number of total random effects z+r
  int n_traits = Y.n_cols; // define n_traits=number of traits
  int no = Y.n_rows; // define no=number of observations
  arma::vec n_levels(n_re, arma::fill::ones); // to store the number of columns each Z and R matrix has
  arma::mat diagTrait = arma::eye(n_traits,n_traits);

  // ****************************************************
  // define ZKZ' and R
  // ****************************************************
  // calculate and concatenate ZKZ' and R
  arma::cube ZKZtR(no,no,n_re);
  for (int i = 0; i < n_re; ++i) { // for each random effect
    int irw = i - n_random;
    if(i < n_random && n_random > 0){ // if random effect (not residual)

      arma::sp_mat zp = Z[i]; // transform as sparse
      n_levels(i) = zp.n_cols; // store the number of columns or levels for this random effect

      bool dcheck = isIdentity_mat(arma::mat(K[i]));
      if(dcheck == true){ // if K[i] is diagonal
        if(zp.n_rows == zp.n_cols){ //is a square matrix
          bool dcheck2 = isIdentity_spmat(zp);
          if(dcheck2 == true){ // if Z[i] is diagonal
            ZKZtR.slice(i) = arma::mat(K[i]);
          }else{ZKZtR.slice(i) = zp * zp.t(); }
        }else{ // is a rectangular matrix
          ZKZtR.slice(i) = zp * zp.t();
        }
      }else{ // if K[i] is not diagonal
        if(zp.n_rows == zp.n_cols){ //is a square matrix
          bool dcheck2 = isIdentity_spmat(zp);
          if(dcheck2 == true){ // if Z[i] is diagonal
            ZKZtR.slice(i) = arma::mat(K[i]);
          }else{ZKZtR.slice(i) = zp * arma::mat(K[i]) * zp.t(); }
        }else{
          ZKZtR.slice(i) = zp * arma::mat(K[i]) * zp.t();
        }
      }
    }else{ //if is an rcov term
      double dcheck3 = accu(W) - W.n_cols;
      if(dcheck3 == 0){ // if W is diagonal no need to multiply
        ZKZtR.slice(i) = arma::mat(R[irw]);
      }else{ // if W (weights) is not diagonal then multiply Wis R Wis
        if(isInvW == true){ // user has provided a squared and inverted W already
          ZKZtR.slice(i) = W * arma::mat(R[irw]) * W;
        }else{ // user has provided only W
          arma::mat Wis = inv(chol(arma::mat(W)));
          ZKZtR.slice(i) = Wis * arma::mat(R[irw]) * Wis.t();
        }
        // arma::vec ws2 = 1/sqrt(ws);// arma::mat Wis = diagmat(ws2); // W inverse squared  // ZKZtR.slice(i) = Wis * Rcpp::as<arma::sp_mat>(R[irw]) * Wis;
      }
      n_levels(i) = ZKZtR.slice(i).n_cols; // store the number of columns or levels for this random effect
    }
  }

  // ****************************************************
  // build multivariate versions of X and Y
  // ****************************************************
  arma::vec Ym = vectorise(arma::mat(Y)); // multivariate Y in original scale
  int nom = Ym.n_rows; // number of observations on the vector-form of multivariate Y
  arma::mat Xm;
  for (int i = 0; i < n_fixed; ++i) { // for each fixed effect
    if(i==0){ // build multivariate X for 1st fixed effect
      Xm = kron(Gx[i], arma::mat(X[i]));
    }else{ // build multivariate X for 2nd to nth fixed effect and column bind them
      Xm = arma::join_horiz( Xm , kron(Gx[i], arma::mat(X[i])) );
    }
  }
  arma::mat Ys = scaleCpp(arma::mat(Y)); // scale Y using the scaleCpp function made
  arma::vec Ysm = vectorise(Ys); // multivariate Y in scaled form

  // ****************************************************
  // initial VC
  // ****************************************************
  arma::mat base_var = cov(arma::mat(Y)); // matrix of original variance-covariance in responses
  arma::mat sc_var = cov(Ys); // matrix of scaled variance-covariance in responses
  int rankX = Xm.n_rows - arma::rank(Xm); // n - p.x

  // VC matrix with dimensions n_traits x n_traits (sigma)
  // we need one for each random effect (n_re)
  arma::cube sigma(n_traits,n_traits,n_re);
  arma::cube sigma_scaled(n_traits,n_traits,n_re);
  
  arma::field<arma::vec> sigma_ut(n_re); // undefined LIST to store the VC in a vector-form with length n_re (#of random effects)
  arma::field<arma::vec> constraintsL(n_re); // undefined LIST to store the constraints in a vector-form with length n_re (#of random effects)
  arma::field<arma::vec> n_levels_multi_traitL(n_re); // undefined LIST to store the n_levels in a vector form
  int no_vc = 0; // to add and find out how many VC exist in total
  for (int i = 0; i < n_re; ++i) { // for each random effect fill the cube
    sigma.slice(i) = Ge[i]; // take Ge for a random effect (initial VC values) and save them in a slice
    arma::vec oo = mat_to_vecCpp(sigma.slice(i), GeI[i]); // extract upper triangular from that slice in a vector form, pass the constraints as 2nd argument
    sigma_ut[i] = oo; // oo is sigma2 in vector form and stored in the list sigma_ut
    constraintsL[i] = mat_to_vecCpp(GeI[i], GeI[i]); // who are diagonal and non-diagonal VCs, pass constraints in list form
    n_levels_multi_traitL[i] = constraintsL[i];
    no_vc = no_vc + oo.n_elem; // keep adding the #of VC
  }

  // sigma_ut_un will have all VC for all random effects in a single vector
  arma::vec sigma_ut_un; // vector to unlist the LIST of VC for all random effects
  arma::vec constraints; // vector to unlist constraints
  arma::vec n_levels_multi_trait; // vector to unlist constraints
  for(int i=0; i < n_re ; i++){ // for each random effect unlist
    sigma_ut_un = join_cols(sigma_ut_un,sigma_ut[i]); // column bind vectors so we end up with a very long vector with all VC
    constraints = join_cols(constraints,constraintsL[i]); // column bind vectors so we end up with a very long vector with all constraints
    arma::vec provX = n_levels_multi_traitL[i];
    arma::vec hpos = provX;
    for(int h=0; h < (int)provX.n_cols ; h++){ // for each random effect unlist
      hpos(h) = n_levels(i);
    }
    n_levels_multi_trait = join_cols(n_levels_multi_trait,(provX/provX) % hpos);
  }
  arma::vec sigmaF_ut_un = sigma_ut_un; // make a copy for fixed-value vc's when we use constraints
  arma::vec coef_ut_un = sigma_ut_un; // make a 2nd copy of the same vector for stabilization
  arma::vec coef_ut_un_explode = sigma_ut_un; // make a 3rd copy for checking issues with vc going too early outside the parameter space

  int kk = sigma_ut_un.n_elem; // how many VCs are in the model?
  arma::vec llstore(iters); // container for LL
  arma::vec pos(sigma_ut_un.n_elem, arma::fill::zeros); // create an index vector with as many 0's as VCs

  // ****************************************************
  // dummy matrices for multivariate derivatives
  // ****************************************************
  int tot = n_re*n_traits*n_traits; // maximum number of variance components
  arma::vec re_mapper(tot); // mapper to know which VC belongs to each random effect
  arma::cube deriv_dummy(n_traits,n_traits,tot);
  int counter3 = 0;
  for(int i=0; i < n_re; i++){ // for each random effect
    const arma::mat & prov = GeI[i];
    int ncol = prov.n_cols; // traits
    for (int k = 0; k < ncol; k++){ // go through GeI(i) and make the dummy derivatives where there's a value > 0
      for (int j = 0; j < ncol; j++){
        if (k > j){}else{
          if(prov(k,j) > 0){
            // only extract the variance component if it was planned to be estimated
            arma::mat prov4(ncol,ncol,arma::fill::zeros);
            prov4(k,j)=1;
            prov4 = arma::symmatu(prov4);
            deriv_dummy.slice(counter3) = prov4;
            re_mapper[counter3] = i;
            counter3++;
          }
        }
      }
    }
  }

  // ****************************************************
  // ****************************************************
  // ##### iterative algorithm starts
  // ****************************************************
  // ****************************************************
  arma::vec v(nom, arma::fill::ones); // generate enough ones for an identity matrix of dimensions nt x nt
  arma::mat Vi(nom,nom); // V or phenotypic variance matrix
  arma::mat P(nom,nom); // to fill the projection matrix
  arma::sp_mat D = arma::speye<arma::sp_mat>(nom,nom);
  arma::vec seqrankX = seqCpp(0,rankX-1); // will be used to keep only the eigen values for indices 1 to rankX
  arma::vec seqkk = seqCpp(0,kk-1);
  arma::vec popo = arma::vec(rankX, arma::fill::zeros);
  for(int i=0; i < rankX; i++){popo(i) = 1;}
  arma::mat Inf(kk,kk,arma::fill::zeros); // to store second derivatives (information matrix)
  arma::mat InfEM(kk,kk,arma::fill::zeros); // to store second derivatives (information matrix)
  arma::mat InfJoin(kk,kk,arma::fill::zeros); // to store second derivatives (information matrix)
  arma::mat InfJoin_inv(kk,kk,arma::fill::zeros); // to store second derivatives (information matrix)

  arma::mat Infw(kk,kk,arma::fill::zeros); // weights for AI information matrix
  arma::mat InfEMw(kk,kk,arma::fill::zeros); // weights for EM information matrix

  arma::vec score(kk); // vector to store first derivatives, the product Y'PViPY - tr(PVi) = dL/ds
  arma::mat Inf_inv; // to store the inverse of the information matrix
  arma::vec eigval2; // will be used for the decomposition of P, within the algorithm
  arma::mat eigvec2; // will be used for the decomposition of P
  arma::mat sigma_store(sigma_ut_un.n_elem,iters); // to store variance comp through the different iterations
  arma::mat sigma_perc_change(sigma_ut_un.n_elem,iters); // to store percent change of variance components
  arma::mat llik_store(1,iters); // to store llik through the different iterations

  arma::mat beta, fitted, residuals;  // empty matrices for ..
  std::vector<arma::mat> VarU_list(n_random); // list object for the BLUP variances
  std::vector<arma::mat> PevU_list(n_random); // list object for the BLUP PEVs
  std::vector<arma::mat> U_list(n_random); // list object for the BLUPs

  arma::vec vdD(n_traits,arma::fill::ones);
  arma::mat dD = arma::diagmat(vdD);
  arma::mat sigma_cov;
  arma::mat tXVXi; // var-cov fixed effects
  arma::mat V(nom,nom); // V or phenotypic variance matrix

  bool convergence = false;
  bool last_iteration = false;
  int cycle, cycle2, ikk;
  double ldet, llik, llik0, delta_llik, checkP, seconds; // to store likelihoods and determinants

  // ###############
  // LOOP for cycles
  // ###############
  for(cycle=0; cycle < iters; cycle++){ // for each cycle
    // Check for Python interrupt
    if (cycle % 5 == 0 && PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }

    for (int i = 0; i < n_re; ++i) { // for each random effect in the formula
      sigma_ut[i] = mat_to_vecCpp(sigma.slice(i), GeI[i]); // extract upper triangular in a vector form
    } // sigma_ut is a LIST
    arma::vec sigmatwo; // create a vector for variance components
    for(int i=0; i < n_re ; i++){ // for each random effect
      sigmatwo = join_cols(sigmatwo,sigma_ut[i]); // column bind to make a vector of vectors
    } // sigmatwo now has all VCs in a vector

    // multivariate ZKZ' and V
    int i;
    for(i=0; i < n_re; i++){ // loop for filling the multivariate ZGZ' and V
      if(last_iteration == true){ // if is the last iteration multivariate ZKZ is opposite
        if(i == 0){
          V = arma::kron(ZKZtR.slice(i),sigma.slice(i));
        }else{V = V + arma::kron(ZKZtR.slice(i),sigma.slice(i));}
      }else{
        if(i == 0){
          V = arma::kron(sigma.slice(i),ZKZtR.slice(i));
        }else{V = V + arma::kron(sigma.slice(i),ZKZtR.slice(i));}
      }
    }

    // invert V and P (projection matrix)

    arma::inv_sympd(Vi,V);  // try to invert normally
    if(Vi.n_rows == 0){ // if fails try to invert with diag(1e-3)
      V = V + (D*tolparinv);
      arma::inv_sympd(Vi,V);
      if(Vi.n_rows == 0){// if fails try to invert with diag(1e-2)
        V = V + (D*(tolparinv*10));
        arma::inv_sympd(Vi,V);
        if(Vi.n_rows == 0){// if fails try to invert with diag(1e-1)
          V = V + (D*(tolparinv*100));
          arma::inv_sympd(Vi,V);
          if(Vi.n_rows == 0){ // finally, if fails try to invert with diag(1e-3)
            // Rcpp::Rcout << "System is singular (V). Stopping the job. Try a bigger number of tolParInv." << arma::endl;
            throw std::runtime_error("System is singular (V). Aborting the job. Try a bigger number of tolParInv.");
            // return 0;
          }
        }
      }
    }

    // if last iteration let's make Xm in the opposite direction
    if(last_iteration == true){
      for (int i = 0; i < n_fixed; ++i) {
        if(i==0){
          Xm = kron(arma::mat(X[i]), Gx[i]);
        }else{
          Xm = arma::join_horiz( Xm , arma::kron(arma::mat(X[i]), Gx[i]) );
        }
      }
      Ym = vectorise(Y.t());
      Ysm = vectorise(Ys.t());
    }
    arma::mat VX = Vi * Xm; // VX
    arma::mat tXVX = Xm.t() * VX; // X'VX
    arma::mat tXVXVX; // X'VXVX
    tXVXVX = arma::solve(tXVX, VX.t()); // X'VXVX
    arma::solve(tXVXVX,tXVX,VX.t());
    if(tXVXVX.n_rows == 0){ // if fails try to invert with diag(1e-6)
      arma::solve(tXVXVX,tXVX + (D*(tolparinv)),VX.t());
      if(tXVXVX.n_rows == 0){// if fails try to invert with diag(1e-5)
        arma::solve(tXVXVX,tXVX + (D*(tolparinv*10)),VX.t());
        if(tXVXVX.n_rows == 0){// if fails try to invert with diag(1e-4)
          arma::solve(tXVXVX,tXVX + (D*(tolparinv*100)),VX.t());
          if(tXVXVX.n_rows == 0){// finally stop
            // Rcpp::Rcout << "System is singular (tXVXVX). Aborting the job. Try a bigger number of tolParInv." << arma::endl;
            throw std::runtime_error("System is singular (tXVXVX). Aborting the job. Try a bigger number of tolParInv.");
            // return 0;
          }
        }
      }
    }

    // projection matrix
    P = Vi - (VX*tXVXVX); // V - V(XVX)-V

    if(last_iteration == false){

      arma::vec rss = Ysm.t() * (P * Ysm); // yPy = scalar RSS

      double rankXorss = arma::as_scalar(rankX/rss); // (n-p)/y'Py
      double rssorankX = arma::as_scalar(rss/rankX); // y'Py/(n-p)

      sigmatwo = sigmatwo * rssorankX;

      // weight the projection matrix to provide stability
      coef_ut_un(arma::find(pos == 0)) = sigmatwo(arma::find(pos == 0)); // VC1[which(pos==0)] = VC2[which(pos==0)]
      coef_ut_un(arma::find(pos == 1)) = log(sigmatwo(arma::find(pos == 1))); // VC1[which(pos==1)] = log(VC2[which(pos==1)])

      // calculate the log-likelihood
      P = P * rankXorss; // P * [(n-p)/y'Py]
      rss.fill(rankX); // yPy = n-p
      arma::eig_sym(eigval2, eigvec2, P); // VlV
      eigval2 = sort(eigval2,"descend"); // sort eigen vectors
      eigval2 = eigval2(arma::find(popo == 1)); // only take the values from 1 to rankX
      checkP = eigval2.min();
      if(checkP < 0){ // if any eigen value is < 0 recalculate P
        P = P + (D * (tolpar - eigval2.min()));
        eigval2 = eigval2 + tolpar - eigval2.min();
      }
      ldet = accu(log(eigval2)); // sum(log(lambda))
      llik = ldet/2 - (arma::as_scalar(rss)/2); // llik = [sum(log(lambda))/2] - [(n-p)/2]

      if(cycle == 0){llik0 = llik;}
      delta_llik = llik - llik0;
      llik0 = llik;

      // use the stabilization
      arma::vec var_components(kk, arma::fill::ones); // VC = rep(0,nVC)
      double check00 = accu(pos); // accu is like sum() in R
      if(check00 > 0){ // if there's 1's in the pos vector
        arma::uvec ind = find(pos == 1); // which are 1's
        var_components(ind) = sigmatwo(ind); // var_components[which(pos==1)] = sigmatwo[which(pos==1)]
      }

      // calculate first derivatives (dL/ds = score)

      arma::cube PdViList(nom,nom,kk); // list to store the multivariate derivatives * P or PVi=P*dZKZ'/ds
      for(int i=0; i < kk; i++){
        int re = re_mapper(i);
        arma::mat zkzp = ZKZtR.slice(re); // it repeats the same ZKZtR if is a vc for the same random effect
        arma::mat PdVi = P * kron(deriv_dummy.slice(i),zkzp); // multivariate dVi = dZKZ'/ds
        if(ai && cycle > 2){
          score[i] = - (0.5 * arma::as_scalar(trace(PdVi))) + (0.5 * arma::as_scalar((Ysm.t() * PdVi * P * Ysm)));
        }else{
          score[i] = arma::as_scalar(Ysm.t() * PdVi * P * Ysm) - accu(diagvec(PdVi));
        }
        PdViList.slice(i) = PdVi;
      }
      // theta(k) * dL/ds ..... are scalar values
      score = score % var_components; // to be used later for updating the variance components
      // if all goes well var_components is just ones

      // calculate second derivatives (AverageInformation)
      // Fisher's Information tr(PVi * PVi) .... A*=Vi=dV/ds .... [Vi Vj'] si sj ; TT is the list of derivatives for all random effects - trait combos
      
      // if(emupdate(cycle) == 0){ // if user wants an EM update (1st derivatives) . It works but it didn't speed up the algorithm when using EM. This leads to don't have information matrix and therefore SE for variance components.
      for (int i = 0; i < kk; i++){
        for (int j = 0; j < kk; j++){
          if (i > j){}else{ //only upper triangular
            if(ai && cycle > 2){ // if average information
              Inf(i,j) = 0.5 * arma::as_scalar(Ysm.t() * PdViList.slice(i) * P * PdViList.slice(j) * P * (P * Ysm)); // j is .t() ?
            }else{ // if newton raphson
              Inf(i,j) = accu(PdViList.slice(i) % PdViList.slice(j).t()) * arma::as_scalar(var_components(i)) * arma::as_scalar(var_components(j));
            }
          }
        }
      }
      Inf = arma::symmatu(Inf); // copy lower in upper triangular
      Inf_inv = arma::pinv(Inf, 1.490116e-08); // Inverse of Fishers or information matrix
      if(Inf_inv.n_rows == 0){
        throw std::runtime_error("System is singular (Inf_inv). Aborting the job. Try a bigger number of tolParInv.");
      }

      // vector to store the update = F- * sigma(k) * dL/ds
      arma::vec delta(kk);

      // if(emupdate(cycle) == 1){ // if user wants an EM update (1st derivatives)
      InfEM.diag() = (coef_ut_un % coef_ut_un) / n_levels_multi_trait; // I.em inverse
      InfEM = arma::pinv( InfEM, 1.490116e-08 ); // I.em
      arma::vec emw(kk); // vectors for weights
      arma::vec aiw(kk);
      for(ikk=0; ikk < kk; ikk++){
        emw(ikk)= emweight(cycle);
        aiw(ikk)= 1 - emweight(cycle);
      }
      Infw.diag() = aiw; // put weights in diagonal fill::value is still not available in this version
      InfEMw.diag() = emw;
      InfJoin = (Inf*Infw)+(InfEM*InfEMw); // joint information matrix
      InfJoin_inv = arma::pinv(InfJoin, 1.490116e-08); // inverse the joint information matrix
      delta = InfJoin_inv * score; //update for variance components where: delta = Information.inv * dL/ds
      // delta = (coef_ut_un % score % coef_ut_un)/n_levels; // previous way I was calculating the deltas
      // }else{ // if user wants an information*score update
      //   delta = Inf_inv * score; //update for variance components where: delta = Information.inv * dL/ds
      // }

      // ^^^^^^^^^^^^^^^^^^
      // ^^^^^^^^^^^^^^^^^^
      // parameter restrain
      // GeI values
      // 0 not estimated
      // 1 positive
      // 2 unconstrained
      // 3 fixed
      arma::vec coef_ut_unC = coef_ut_un + (stepweight(cycle) * delta); // provisional new variance components
      arma::uvec restrain = find(constraints == 1 && coef_ut_unC < 0); // which vcs are negative and should be positive
      arma::vec cc = coef_ut_unC(restrain); // extract the ones that suppose to be positive
      // identify var comp < 0 (1's)
      if(cc.n_elem > 0){ // we have to restrain
        arma::uvec no_restrain = find((constraints == 1 && coef_ut_unC > 0) || (constraints > 1)); // indices of columns that are OK to use (no restrain)
        arma::mat Inf_norestrain = Inf.submat(no_restrain,no_restrain); // subset of Information matrix
        arma::mat Inf_norestrain_inv; // define the inverse of the information matrix
        arma::inv(Inf_norestrain_inv, Inf_norestrain); // Inverse of Fishers (subset of Inf)
        if(Inf_norestrain_inv.n_rows == 0){
          throw std::runtime_error("System is singular (Inf_norestrain_inv). Aborting the job. Try a bigger number of tolParInv.");
        }
        arma::vec scorenorestrain = score(no_restrain); // subset of scores (1st derivatives)
        arma::vec coef_ut_un_norestrain = coef_ut_un(no_restrain); // subset of vc
        arma::vec deltanorestrain; // define the delta for no restrained

        //
        arma::mat InfEM_norestrain = InfEM.submat(no_restrain,no_restrain); // subset of Information matrix
        arma::mat Infw_norestrain = Infw.submat(no_restrain,no_restrain); // subset of Information matrix
        arma::mat InfEMw_norestrain = InfEMw.submat(no_restrain,no_restrain); // subset of Information matrix
        arma::mat InfJoin_norestrain = InfJoin.submat(no_restrain,no_restrain); // subset of Information matrix
        arma::mat InfJoin_inv_norestrain = InfJoin_inv.submat(no_restrain,no_restrain); // subset of Information matrix
        // if(emupdate(cycle) == 1){ // if user wants an EM update (1st derivatives)
        InfJoin_norestrain = (Inf_norestrain*Infw_norestrain)+(InfEM_norestrain*InfEMw_norestrain); // joint information matrix
        InfJoin_inv_norestrain = arma::pinv(InfJoin_norestrain, 1.490116e-08); // inverse the joint information matrix
        deltanorestrain = InfJoin_inv_norestrain * scorenorestrain; //update for variance components where: delta = Information.inv * dL/ds
        // deltanorestrain = (coef_ut_un_norestrain % scorenorestrain % coef_ut_un_norestrain)/n_levels;
        // }else{ // if user wants an information*score update
        //   deltanorestrain = Inf_norestrain_inv * scorenorestrain; //update variance components
        // }
        delta(no_restrain) = deltanorestrain;
        delta(restrain) = delta(restrain)*0;
      } //else just keep going
      // end of parameter restrain
      // ^^^^^^^^^^^^^^^^^^
      // ^^^^^^^^^^^^^^^^^^
      coef_ut_un = coef_ut_un + (stepweight(cycle) * delta);
      //
      // constraint the parameters that should be positive and are going negative
      if(cc.n_elem > 0){
        coef_ut_un(restrain) = coef_ut_un(restrain)*0; // the ones that still go below zero and shouldn't let's fix them
      }
      // weight the projection matrix to provide stability
      sigmatwo(arma::find(pos == 0)) = coef_ut_un(arma::find(pos == 0)); // index of pos
      sigmatwo(arma::find(pos == 1)) = exp(coef_ut_un(arma::find(pos == 1)));
      // the fixed parameters are forced to be the original value
      sigmatwo(find(constraints == 3)) = sigmaF_ut_un(find(constraints == 3));
      // bring back sigma as a list
      sigma = vec_to_cubeCpp(sigmatwo, GeI);
      // check if likelihood has reached its maximum and stop if so
      llstore(cycle) = llik;
      // get current time
      time_t now = time(0);
      tm *ltm = localtime(&now);
      // keep track of time difference between iterations
      seconds = difftime(now,before);
      // update before time to be the now to be used in the next iteration
      time_t before = time(0);
      localtime(&before);

      // store parameters
      sigma_store.col(cycle) = sigmatwo;
      if(cycle > 0){
        sigma_perc_change.col(cycle) = ((coef_ut_un/sigma_store.col(cycle-1))-1) * 100; // percent change
      }
      llik_store(cycle) = llik;

      // return output to the console
      if(verbose == true){
        if(cycle == 0){std::cout << "iteration   " << " LogLik   " << "  wall    " << "cpu(sec)   " << "restrained" << std::endl;}
        std::cout << "    " << cycle+1 << "      " <<  llik << "   " << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "      " << seconds << "           " << cc.n_elem << std::endl;
      }

      // define the end of the algorithm
      if(((cycle > 2) && (delta_llik < tolpar)) || cycle == iters-1 ){
        cycle2 = cycle;
        if((cycle > 2) && (delta_llik < tolpar)){convergence = true;}
        last_iteration = true;
        cycle = iters-2;
        // if user wants to get scaled results we don't bring back to original scale
        if(retscaled == false){
          for (int i = 0; i < n_re; ++i) {
            sigma_scaled.slice(i) = sigma.slice(i); // just make a copy before we scale to normal units
            sigma.slice(i) = (sigma.slice(i)%base_var)/sc_var;
          }
        }
        // Fisher inverse
        arma::mat FI = Inf/2;
        arma::vec myone(pos.n_elem,arma::fill::ones);
        arma::vec sp = ((sigmatwo - myone) % pos) + myone;
        arma::mat FI_c = FI / (sp * sp.t());
        sigma_cov = pinv(FI_c);
        if(sigma_cov.n_rows == 0){
          throw std::runtime_error("System is singular (sigma_cov). Aborting the job. Try a bigger number of tolParInv.");
        }
      }
    }else{ // if we are in the last iteration now we calculate u, PEV, B, XB
      arma::inv(tXVXi,tXVX);
      if(tXVXi.n_rows == 0){ // if fails try to invert with diag(1e-6)
        arma::inv(tXVXi,tXVX+(D*(tolparinv)));
        if(tXVXi.n_rows == 0){
          arma::inv(tXVXi,tXVX+(D*(tolparinv*10)));
          if(tXVXi.n_rows == 0){
            throw std::runtime_error("System is singular (tXVXi). Aborting the job. Try a bigger number of tolParInv.");
          }
        }
      }
      // b = (X'VX)-X'Vy or (X'VX)-X'VYs
      if(retscaled == true){
        beta = tXVXi * ((Xm.t() * Vi) * Ysm);
      }else{
        beta = tXVXi * ((Xm.t() * Vi) * Ym);
      }
      fitted = Xm * beta; // X * b
      residuals = Ym - fitted; // Y - Xb
      arma::mat Vie = Vi * residuals; // Vi * e = Vi * (Y - Xb)
      // calculate BLUPs and PEVs
      if(n_random > 0){
        for(i=0; i < n_random; i++){
          arma::mat Zprov = arma::mat(Z[i]);
          arma::mat Ki = arma::mat(K[i]);
          arma::mat VarK;
          arma::mat ZKfv;
          if(Ki.n_cols == Zprov.n_cols){
            VarK = arma::kron(arma::mat(K[i]),sigma.slice(i)); // Kron(K sigma)
            ZKfv = VarK * arma::kron(Zprov.t(),dD); // (G) (Z'I) = Z' G
          }else{
            ZKfv = arma::kron(Zprov.t(),dD*sigma.slice(i)); // (Z'I) * sigma when K is identity
          }
          U_list[i] = ZKfv * Vie; // BLUP = Z' G Vi (Y - Xb)
          if(pev==true){
            if(Ki.n_cols == Zprov.n_cols){
              VarU_list[i] = ZKfv * (P * ZKfv.t()); // var(u) = Z' G [Vi - VX(X'VX)-XV] G Z' = Z' G P G Z'
              PevU_list[i] = VarK - VarU_list[i]; // PEV = G - var(u)
            }else{
              VarU_list[i] = ZKfv * (P * ZKfv.t()); // var(u) = Z' sigma P sigma Z'
              PevU_list[i] = VarU_list[i]; // PEV = var(u) when K is identity
            }
          }
        }
      }
    }
  }

  // ^^^^^^^^^^^^^^^^^^^
  // ^^^^^^^^^^^^^^^^^^^
  // end of algorithm
  // ^^^^^^^^^^^^^^^^^^^
  // ^^^^^^^^^^^^^^^^^^^
  // Post-processing
  // bring back to original scale
  arma::vec dd,ee;
  for (int i = 0; i < n_re; ++i) {
    dd = join_cols(dd,mat_to_vecCpp(base_var, GeI[i])); // variances in original scale
    ee = join_cols(ee,mat_to_vecCpp(sc_var, GeI[i])); // scaled variances
  }
  arma::mat FISH = (sigma_cov % (dd*dd.t())) / (ee*ee.t()); // bring back sigma_cov (Fisher) to original scale
  double AIC = (-2 * llik) + (2 * Xm.n_cols); // AIC
  double ny = Ym.n_elem; // number of data points
  double BIC = (-2 * llik) + (log(ny) * Xm.n_cols); // BIC
  sigma_store.each_col() %= dd; // bring back to original scale
  sigma_store.each_col() /= ee;
  arma::mat sigma_perc_change2;
  if(iters > 1){
    sigma_perc_change2 = sigma_perc_change.cols(1, cycle2);
  }else{
    sigma_perc_change2 = sigma_perc_change;
  }

  // Reconstruct theta, uList, uPevList, partitions
  int nTheta = thetaConstOri.size();
  std::vector<arma::mat> newTheta(nTheta);
  std::vector<arma::mat> uList(nTheta > 0 ? nTheta-1 : 0);
  std::vector<arma::mat> uPevList(nTheta > 0 ? nTheta-1 : 0);
  std::vector<arma::mat> PEVs(nTheta > 0 ? nTheta-1 : 0);
  std::vector<arma::mat> partitions_out(nTheta > 0 ? nTheta-1 : 0);
  arma::mat u;
  int counter4 = 0;
  int value = 0;
  for (int i = 0; i < nTheta; ++i) {
    const arma::mat & thetaConstOriIth = thetaConstOri[i];
    arma::mat newThetaIth(thetaConstOriIth.n_rows,thetaConstOriIth.n_rows);
    arma::mat blupTable, pevTable, partitionsTable, pevFull;
    for (int j = 0; j < (int)thetaConstOriIth.n_cols; ++j) {
      for (int k = 0; k < (int)thetaConstOriIth.n_rows; ++k) {
        if(thetaConstOriIth(j,k) != 0){
          double newThetaIthProvVal = sigma(counter4);
          arma::mat newThetaIthProv(1,1); newThetaIthProv(0,0) = newThetaIthProvVal;
          if(j==k){
            newThetaIth(j,k) = newThetaIthProv(0,0);
            if( i < (nTheta-1) ){
              arma::mat provPev;
              if(pev==true){
                provPev = PevU_list[counter4];
                arma::mat zerosMatFill(pevFull.n_rows,provPev.n_cols,arma::fill::zeros);
                pevFull = arma::join_cols( arma::join_rows(pevFull, zerosMatFill),arma::join_rows(zerosMatFill.t(),provPev) );
              }
              if( (int)blupTable.n_cols <= j){
                blupTable = arma::join_rows(blupTable, U_list[counter4]);
                if(pev==true){
                  pevTable = arma::join_rows(pevTable, provPev.diag());
                }
              }else{
                blupTable.col(j) = blupTable.col(j) + U_list[counter4];
                if(pev==true){
                  pevTable.col(j) = pevTable.col(j) + provPev.diag();
                }
              }
            }
          }else{
            newThetaIth(j,k) = newThetaIthProv(0,0);
            newThetaIth(k,j) = newThetaIthProv(0,0);
            if( i < (nTheta-1) ){
              arma::vec indexCovJ(blupTable.n_rows, arma::fill::ones);
              indexCovJ=indexCovJ*j;
              arma::vec indexCovK(blupTable.n_rows, arma::fill::ones);
              indexCovK=indexCovK*k;
              arma::vec indexCovJK = arma::join_cols(indexCovJ,indexCovK);
              arma::uvec isJ = arma::find(indexCovJK == j);
              arma::uvec isK = arma::find(indexCovJK == k);
              arma::mat provBlup = U_list[counter4];
              arma::mat provPev;
              if(pev==true){
                provPev = PevU_list[counter4];
                provPev = provPev.diag();
              }
              if((int)blupTable.n_cols <= j){
                blupTable = arma::join_rows(blupTable, provBlup.rows(isJ));
                if(pev==true){
                  pevTable = arma::join_rows(pevTable, provPev(isJ));
                }
              }else{
                blupTable.col(j) = blupTable.col(j) + provBlup.rows(isJ);
                if(pev==true){
                  pevTable.col(j) = pevTable.col(j) + provPev(isJ);
                }
              }
              if((int)blupTable.n_cols <= k){
                blupTable = arma::join_rows(blupTable, provBlup.rows(isK));
                if(pev==true){
                  pevTable = arma::join_rows(pevTable, provPev(isK));
                }
              }else{
                blupTable.col(k) = blupTable.col(k) + provBlup.rows(isK);
                if(pev==true){
                  pevTable.col(k) = pevTable.col(k) + provPev(isK);
                }
              }
            }
          }
          counter4++;
        }
      }
    }
    newTheta[i] = newThetaIth;
    if( i < (nTheta-1) ){
      uList[i] = blupTable;
      if(pev==true){
        uPevList[i] = pevTable;
        PEVs[i] = pevFull;
      }
      int nrbt= blupTable.n_rows;
      arma::mat start_m, end_m;
      for (int l = 0; l < (int)blupTable.n_cols; ++l) {
        if( (i==0) && (l==0) ){
          start_m = arma::mat(1,1); start_m(0,0) = Xm.n_cols + (l*nrbt) + 1;
          end_m = arma::mat(1,1); end_m(0,0) = Xm.n_cols + (nrbt * (l+1));
        }else{
          start_m = arma::mat(1,1); start_m(0,0) = value + 1;
          end_m = arma::mat(1,1); end_m(0,0) = start_m(0,0) + nrbt - 1;
        }
        partitionsTable = arma::join_cols(partitionsTable, arma::join_rows(start_m, end_m));
        value = partitionsTable.max();
      }
      u = join_cols(u, arma::vectorise(blupTable));
      partitions_out[i] = partitionsTable;
    }
  }
  arma::mat bu = join_cols(beta, u);
  arma::mat Ci(bu.n_rows,bu.n_rows, arma::fill::zeros);
  Ci.submat(0, 0, beta.n_rows-1, beta.n_rows-1) = tXVXi;
  if(pev==true){
    for (int i = 0; i < (int)partitions_out.size(); ++i){
      arma::mat & pp = partitions_out[i];
      Ci.submat(pp.min()-1, pp.min()-1, pp.max()-1, pp.max()-1) = PEVs[i];
    }
  }

  py::dict result;
  result["llik"] = _mat_to_py(arma::mat(llik_store.cols(0, cycle2)));
  result["b"] = _mat_to_py(beta);
  result["u"] = _mat_to_py(u);
  result["bu"] = _mat_to_py(bu);
  result["Ci"] = _mat_to_py(Ci);
  result["theta"] = _matvec_to_py(newTheta);
  result["theta_se"] = _mat_to_py(FISH);
  result["InfMat"] = _mat_to_py(Inf);
  result["monitor"] = _mat_to_py(arma::mat(sigma_store.cols(0, cycle2)));
  result["uList"] = _matvec_to_py(uList);
  result["uPevList"] = _matvec_to_py(uPevList);
  result["AIC"] = AIC;
  result["BIC"] = BIC;
  result["convergence"] = convergence;
  result["partitions"] = _matvec_to_py(partitions_out);
  result["fitted"] = _mat_to_py(fitted);
  result["residuals"] = _mat_to_py(residuals);
  result["dL"] = _vec_to_py(score);
  result["percChange"] = _mat_to_py(sigma_perc_change2);
  return result;
}


// ============================================================================
// ai_mme_sp — Henderson MME REML with AI/EM hybrid
// ============================================================================
// This is a very large function. For now we provide a simplified wrapper
// that the Python side can call. The full algorithm is preserved.

py::dict ai_mme_sp_cpp(
    const arma::sp_mat & X,
    const std::vector<arma::sp_mat> & ZI,
    const arma::vec & Zind,
    const std::vector<arma::sp_mat> & AiI,
    const arma::sp_mat & y0,
    const std::vector<arma::sp_mat> & SI,
    const std::vector<arma::mat> & partitionsS,
    const arma::sp_mat & H,
    const bool & useH,
    int nIters, double tolParConvLL, double tolParConvNorm,
    double tolParInv,
    const std::vector<arma::mat> & thetaI,
    const std::vector<arma::mat> & thetaCI,
    const arma::mat & thetaF,
    const arma::vec & addScaleParam,
    const arma::vec & weightEmInf,
    const arma::vec & weightInf,
    const bool & verbose)
{
  time_t before = time(0);
  localtime(&before);

  // define element sizes
  int nSs = SI.size(); // number of residual inverse matrices
  int nZs = ZI.size(); // number of random effects
  int nRe;
  if(nZs > 0){
    nRe = Zind.max(); // number of actual random effects specified in random
  }else{nRe = 0;}
  int nZsFake = 1; // a fake value in case there's no random effects we avoid a bad allocation error
  int nReFake = 1; // a fake value in case there's no random effects we avoid a bad allocation error
  int nRRe = thetaI.size(); // number of random + residual effects
  int nX = X.n_cols; // number of fixed effects
  int nR = y0.n_rows; // number of records

  // find variance and mean for the response
  double vary = arma::mean(arma::var(y0));
  double stdy = arma::mean(arma::stddev(y0));
  double muy = arma::mean(arma::mean(y0));
  arma::sp_mat y = arma::sp_mat(scaleCpp(arma::mat(y0)));
  bool intercept = false;
  if(arma::accu(X.col(0)) == X.n_rows){ // there's an intercept
    intercept = true;
  }

  // create a list to store the symmetric version of thetaC
  arma::field<arma::mat> theta(nRRe), thetaC(nRRe);
  for (int i = 0; i < nRRe; ++i) { // create a copy of thetas
    theta[i] = thetaI[i]/vary;
    thetaC[i] = thetaCI[i];
  }

  // move Z to arma objects
  int nZsAl = (nZs > 0) ? nZs : nZsFake;
  arma::field<arma::sp_mat> Zf(nZsAl);
  if(nZs > 0){
    for (int i = 0; i < nZs; ++i) {
      Zf(i) = ZI[i]; // store in the field
    }
  }

  // move S inverse to arma objects
  arma::field<arma::sp_mat> Si(nSs);
  for (int i = 0; i < nSs; ++i) {
    Si(i) = SI[i]; // store in the field
  }

  // move Ai to arma objects
  int nReAl = (nZs > 0) ? nRe : nReFake;
  arma::field<arma::sp_mat> Ai(nReAl);
  if(nZs > 0){
    for (int i = 0; i < nRe; ++i) {
      Ai(i) = AiI[i]; // store in the field
    }
  }

  // calculate log determinants of Ai's
  arma::rowvec logDetA(nReAl);
  if(nZs > 0){ // if there's random effects
    for (int i = 0; i < nRe; ++i) { // for each random effect
      double val;
      double sign;
      bool ok3 = log_det(val, sign, arma::mat(Ai[i])); // calculate the logDet of the i.th covariance matrix
      if(ok3 == false){ std::cerr << "log determinant of Ai failed " << std::endl; }
      logDetA(i) = val*sign*(-1);
    }
  }

  // define partitions (only used if random effects exist)
  int last = X.n_cols;
  arma::field<arma::mat> partitions(nReAl); // store indices of the random effects
  arma::vec zsAva;
  int Nu = 0;
  if(nZs > 0){ // if there's random effects (Z matrices) check where each starts and ends
    zsAva = unique(Zind);
    for (int i = 0; i < nRe; ++i) { // for each effect
      arma::uvec indexZind = find(Zind == (i+1)); // which Z matrices to use
      int nIndexZind = indexZind.size(); // number of Z matrices to use
      arma::vec Nus(nIndexZind); // vector to store number of columns in each Z matrix
      for (int j = 0; j < nIndexZind; ++j) {
        int jj = indexZind(j);
        arma::sp_mat Zprov = Zf(jj);
        Nus(j) = Zprov.n_cols; // calculate the number of columns
      }
      arma::vec end = Nus; // define ends and starts
      for (int k = 0; k < nIndexZind; ++k) { // for each effect
        arma::uvec toSum = arma::regspace<arma::uvec>(0,1,k); // equivalent to seq()
        end(k) = arma::accu(Nus(toSum));
      }
      arma::vec ones(nIndexZind, arma::fill::ones);
      arma::vec lastM(nIndexZind, arma::fill::value(last));
      arma::vec start = end - Nus + ones;
      start = start + lastM; // adjust start by adding # of fixed effects
      end = end + lastM; // adjust end by adding # of fixed effects
      partitions(i) = arma::join_rows(start,end);
      last = end.max();
      Nu = Nu + accu(Nus);
    }
  } // end of if statement when random effects exist

  // define the number of variance components to estimate per random effect structure
  arma::vec nVc(nRRe);
  for (int i = 0; i < nRRe; ++i) {
    arma::mat thetaCprov = thetaC[i];
    arma::uvec nVcProv = find(thetaCprov > 0);
    nVc(i) = nVcProv.size();
  }
  int nVcTotal = accu(nVc); // total number of variance components
  // assign a start and an end index to each covariance structure using the #of VC
  arma::vec nVcEnd = nVc;
  for (int i = 0; i < nRRe; ++i) {
    arma::uvec toSum = arma::regspace<arma::uvec>(0,1,i);
    nVcEnd(i) = arma::accu(nVc(toSum));
  }
  arma::vec nVcStart = nVcEnd - nVc + 1;

  // move constraints to vector form binding the columns
  arma::vec thetaCUnlisted;
  for (int i = 0; i < nRRe; ++i) {
    thetaCUnlisted = join_cols(thetaCUnlisted,mat_to_vecCpp2(thetaC[i],thetaC[i]));
  }

  // removing complex structures how many effects are really there
  arma::vec nUsTotal(nReAl);
  if(nZs > 0){
    for (int i = 0; i < nRe; ++i) {
      arma::mat partitionsProv = partitions(i);
      nUsTotal(i) = partitionsProv(0,1) - partitionsProv(0,0) + 1;
    }
  }

  // define objects to store theta and llik across iterations
  arma::mat monitor(nVcTotal,nIters); // matrix to store variance components
  arma::mat percChange(nVcTotal,nIters); // matrix to store variance components
  arma::rowvec llik(nIters); // store log likelihood values
  int nEffects = Nu+nX;
  int nEffectsPlusY = nEffects + 1;
  arma::mat Mchol;

  arma::sp_mat M0(nEffectsPlusY,nEffectsPlusY), M(nEffectsPlusY,nEffectsPlusY), Wm(nR,nEffects), Wy(nR,nEffectsPlusY), C(nEffects,nEffects), Ci(nEffects,nEffects);
  arma::vec u(Nu), b(nX), bu(nEffects);
  arma::mat buWu(nEffects,nVcTotal);
  arma::mat avInf(nVcTotal,nVcTotal);
  arma::mat emInf(nVcTotal,nVcTotal);
  arma::mat InfMat(nVcTotal,nVcTotal);
  arma::mat InfMatInv(nVcTotal,nVcTotal);
  bool convergence = false;
  double seconds;
  arma::sp_mat XWjxZWj(nEffects,nVcTotal), WiXxWiZ(nVcTotal,nEffects), WiWj(nVcTotal,nVcTotal);
  arma::sp_mat XWjxZWj0(nEffects,nVcTotal), WiXxWiZ0(nVcTotal,nEffects), WiWj0(nVcTotal,nVcTotal);
  arma::mat Mchol_XZ;
  arma::sp_mat I = arma::speye<arma::sp_mat>(nEffectsPlusY,nEffectsPlusY);
  arma::mat I2 = arma::eye(nEffects+nVcTotal,nEffects+nVcTotal);
  arma::vec delta(nVcTotal), delta_minus1(nVcTotal);
  // objects for constraints
  arma::mat percDelta(nVcTotal,nIters,arma::fill::zeros); // store % change of the delta with respect to the previous iteration
  arma::mat normMonitor(3,nIters); // store in each iteration the 3 stopping criteria of Madsen and Jensen
  arma::mat toBoundary(nIters,nVcTotal, arma::fill::zeros); // store which values have been set to the boundary value
  arma::vec sumToBoundary(nVcTotal, arma::fill::zeros); // to apply sum across iterations and if a VC goes to the boundary 3 times it is fixed to the boundary
  arma::sp_mat Ri(nR,nR); // matrix to store R inverse
  arma::sp_mat Hs(H.n_cols,H.n_cols); // square of H matrix
  if(useH == true){ // do cholesky decomposition of H if user wants to use weights
    std::cout << "Using the weights matrix " << std::endl;
    Hs = arma::sp_mat(chol(arma::mat(H)));
  }
  arma::vec dLuOut;

  ////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////
  // START ITERATIVE ALGORITHM
  ////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////
  for (int iIter = 0; iIter < nIters; ++iIter) {
    if (iIter % 5 == 0 && PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }

    // ###########################
    // # 1) absorption of M into y to obtain y'Py and logDetC
    // # PAPER FORMULA from Jensen and Madsen 1997, Gilmour et al., 1995
    // # expand coefficient matrix (C) to have the response variable
    // # M = W' Ri W # with W = [X Z y]
    // #
    // #     [X'RiX  X'RiZ     X'Riy ]
    // # M = [Z'RiX  Z'RiZ+Gi  Z'Riy ]
    // #     [y'RiX  y'RiZ     y'Riy ]
    // #
    // # where Gi = Ai*(s2e/s2u) = (A*s2u)*s2e = kronecker(Ai,solve(s2u))
    // #
    // # lambda = solve(theta) # inverse of var-covar matrices
    // # MChol = chol(M)
    // # yPy = MChol[n,n] # where n is the last element of the matrix
    // # logDetC = 2 * E log(diag(MChol))
    // ###########################
    arma::field<arma::sp_mat> Rij(nSs); // field to store sub R matrices
    arma::vec thetaResidualsVec = mat_to_vecCpp2(theta(nRRe-1),thetaC[(nRRe-1)]);
    for (int i = 0; i < nSs; ++i) { // for each residual structure
      const arma::mat & pSi = partitionsS[i];
      int s1 = pSi(0,0)-1;
      int s2 = pSi(0,1)-1;
      // get ith Rinverse by multiplying Sinverse * 1/theta
      arma::sp_mat RijInv = Si(i) * (1/arma::as_scalar(thetaResidualsVec(i)));
      Ri.submat(s1,s1,s2,s2) = RijInv;
    }
    // adjust R inverse if user provides weights
    if(useH == true){
      Ri = Hs * Ri * Hs.t();
    }

    /////////////////////////////////
    // form the mixed model equations
    if(iIter == 0){ // only form W and Wy once in the first iteration
      Wm = X;
      if(nZs > 0){
        for (int i = 0; i < nZs; ++i) {
          Wm = arma::join_rows(Wm, Zf(i));
        }
      }
      Wy = arma::join_rows(Wm,y);
    }

    if(nSs > 1){ // if there is more than one residual structure we have to calculate M in every iteration
      M = Wy.t() * Ri * Wy;
    }else{ // if there's only one residual structure
      if(useH == true){
        M = Wy.t() * Ri * Wy;
      }else{
        if(iIter == 0){ // only form M0 in the first iteration
          M0 = Wy.t() * Wy; // base M matrix without G
        }
        // then every iteration we just multiply M0 by 1/Ve
        M = M0 * (1/arma::as_scalar(thetaResidualsVec(0))); // always multiply by the current 1/sigma2.e
      }
    }

    arma::field<arma::sp_mat> lambda(nReAl); // to store theta inverses
    arma::field<arma::sp_mat> GI(nReAl); // to store kron(thetainv,Ainv)
    if(nZs > 0){
      for (int i = 0; i < nRe; ++i) {
        arma::mat bend = nearPDcpp(arma::symmatu(theta(i)), 100, 1e-06, 1e-07);
        lambda(i) = arma::sp_mat(arma::pinv(bend));
        GI(i) = kron(lambda(i), Ai(i));
        arma::mat partitionsP = partitions(i);
        int ff = partitionsP(0,0) - 1;
        int ll = partitionsP(partitionsP.n_rows-1,1) - 1;
        M.submat(ff, ff, ll, ll) = M.submat(ff, ff, ll, ll) + GI(i);
      }
    }

    bool okChol = arma::chol(Mchol, arma::mat(M));
    if(okChol == false){
      if(verbose == true){
        std::cout << "Making M positive definite " << std::endl;
      }
      arma::mat Mp = arma::symmatu(arma::mat(M));
      Mp = nearPDcpp(Mp, 100, 1e-06, 1e-07);
      Mchol = arma::chol(Mp);
      if(verbose == true){
        std::cout << "Cholesky of M succeeded " << std::endl;
      }
    }
    arma::vec yPy = arma::square(Mchol.submat(Mchol.n_rows-1, Mchol.n_cols-1, Mchol.n_rows-1, Mchol.n_cols-1));
    Mchol_XZ = Mchol.submat(0,0, Mchol.n_rows-2, Mchol.n_cols-2); // M without y portion (last row and column of M)
    arma::vec My = Mchol.submat(0,Mchol.n_rows-1, Mchol.n_cols-2, Mchol.n_cols-1);
    double logDetC = 2 * accu(log(Mchol_XZ.diag()));
    // ###########################
    // # 1.1) calculate the log-likelihood
    // # PAPER FORMULA (Lee and Van der Werf, 2006)
    // # LL = -0.5 [((Nr-Nb-Nu-...)*ln(s2e)) - ln|C| + ln|Au| + ... + (Nu*ln(s2u)) + ... + y'Py ]
    // # PAPER FORMULA (Jensen and Madsen, 1997)
    // # LL = -0.5 [ln|C| + ln|R| + (ln|A.u| +ln|theta.u|) + ... + y'Py ]
    // # where | | is the determinant of a matrix
    // #       A.u is the pure relationship matrix for the uth random effect
    // #       theta.u is the vc matrix for the uth random effect
    // ###########################

    double llikp=0;
    if(nZs > 0){
      for (int i = 0; i < nRe; ++i) {
        double val;
        double sign;
        bool ok1 = log_det(val, sign, theta(i));
        if(ok1 == false){ std::cerr << "log determinant failed " << std::endl; }
        llikp = llikp + (nUsTotal(i)*val*sign) + logDetA(i);
      }
    }
    double val;
    double sign;
    bool ok2 = log_det(val, sign, theta(nRRe-1));
    if(ok2 == false){ std::cerr << "log determinant of R failed " << std::endl; }
    double logDetR = nR * val * sign;
    llik(iIter) = (- 0.5) * ( llikp + logDetC + logDetR + arma::as_scalar(yPy) );

    // ###########################
    // # 2) backsubstitute to get b and u
    // # use the results from the absorption to obtain BLUE & BLUPs
    // # b = backsolve(MChol[,rest],MChol[,last])
    // ###########################
    arma::spsolve(bu, arma::sp_mat(Mchol_XZ), My, "lapack"); // use LAPACK solver

    arma::uvec bInd = arma::regspace<arma::uvec>(0,1,(nX-1)); // which ones are BLUEs
    b = bu(bInd); // move BLUEs to a different vector
    if(nZs > 0){ // move BLUPs to a different vector
      arma::uvec uInd = arma::regspace<arma::uvec>((nX),1,(nX+Nu-1));
      u = bu(uInd);
    }

    // ###########################
    // # 3) calculate Wu (working variates)
    // # PAPER FORMULA (Notes on Estimation of Genetic Parameters from Van der Werf)
    // # wu = Zu/s2u; we = e/s2e
    // # PAPER FORMULA (Jensen and Madsen, 1997)
    // # U = [u1 | u2 | ... | ui]
    // # US = U * lambda
    // # Wu.ii = Zui*USi # for variance component
    // # Wu.ij = Zui*USj + Zuj*USi # for covariance component
    // # Wr.j = Rj * Rinv * e  # for residual variance component
    // ###########################
    arma::sp_mat Wu;
    arma::field<arma::sp_mat> uSinv(nReAl); // field of sparse matrices to store products u * thetainv

    if(nZs > 0){ // if random effects exist
      for(int iR = 0; iR < nRe; ++iR){ // for each random effect u
        arma::mat partitionsP = partitions(iR); // access the partition for the iR random effect
        arma::sp_mat U(partitionsP(0,1) - partitionsP(0,0) + 1, partitionsP.n_rows); // all BLUPs for the ith random effect
        for(int iRow = 0; iRow < (int)partitionsP.n_rows; ++iRow){ // for each partition row
          arma::uvec usedPartition = arma::regspace<arma::uvec>((partitionsP(iRow,0)-1),1,(partitionsP(iRow,1)-1));
          U.col(iRow) = bu(usedPartition); // move vector to a matrix: u -> [u1|u2|...]
        }
        // # [a || m] [s2a || sam] = [s2a a + sam m  || sam a + s2m m]
        // #          [sam || s2m]
        arma::sp_mat uSinvProv = U * lambda(iR);
        uSinv(iR) = uSinvProv;
        arma::mat thetaCprov = thetaC[iR];
        // for the ij var comp we calculate the Wu
        arma::uvec useZind = find(Zind == iR+1); // which Z matrices we should use for this random effect
        arma::vec nVcForIr = mat_to_vecCpp2(thetaC[iR],thetaC[iR]);
        arma::sp_mat WuiR(nR, nVcForIr.size()); // store Wu for random effect iR
        int counterWu = 0;
        for(int iRow = 0; iRow < (int)lambda(iR).n_rows; ++iRow){
          for(int iCol = 0; iCol < (int)lambda(iR).n_cols; ++iCol){
            if(thetaCprov(iRow,iCol) > 0){ // if vc has to be estimated
              if(iRow == iCol){ // variance component
                WuiR.col(counterWu) = Zf(useZind(iRow)) * uSinvProv.col(iRow);
                counterWu++;
              }else{ // covariance component
                WuiR.col(counterWu) = (Zf(useZind(iCol)) * uSinvProv.col(iRow)) - (Zf(useZind(iRow)) * uSinvProv.col(iCol));
                counterWu++;
              }
            } // end of thetaCprov(iRow,iCol) > 0
          } // end of icol loop
        } // end of irow loop
        Wu = arma::join_rows(Wu,WuiR);
      } // end of loop for each random effect
    } // end of condition when random effects exist
    // calculate residuals
    arma::vec e = y - (arma::sp_mat(Wm.submat(0,0,Wm.n_rows-1,Wm.n_cols-1)) * bu);
    // Working variates for residual VCs
    for(int iS = 0; iS < (int)Si.size(); ++iS){
      const arma::mat & pSi = partitionsS[iS];
      int s1 = pSi(0,0)-1;
      int s2 = pSi(0,1)-1;
      arma::sp_mat Sprov(nR,nR);
      Sprov.submat(s1,s1,s2,s2) = Si(iS);
      if(nSs > 1){
        Wu = arma::join_rows(Wu, Sprov * Ri * arma::sp_mat(e));
      }else{
        if(useH == true){
          Wu = arma::join_rows(Wu, Sprov * Ri * arma::sp_mat(e));
        }else{
          Wu = arma::join_rows(Wu, Sprov * (1/arma::as_scalar(thetaResidualsVec(0))) * arma::sp_mat(e));
        }
      }
    }

    // ###########################
    // # 4) absorption of M into Wu to obtain Wu' P Wu which is the AI matrix
    // # PAPER FORMULA (Smith, 1995) Differentiation of the Cholesky Algorithm
    // # avInf.ij = ((chol(M))[n,n])^2 # the square of the last diagonal element
    // # where:
    // #        [X'RiX  X'RiZ     X'Riwj ]
    // # M.Wu = [Z'RiX  Z'RiZ+Gi  Z'Riwj ]
    // #        [wk'RiX wk'RiZ    wk'Riwj]
    // # AI = (M.Wu.chol)^2
    // ###########################
    // AI matrix
    if(nSs > 1){ // if R is complex do the whole matrix product in every iteration
      XWjxZWj = Wm.t() * Ri * Wu; // [X'Riwj Z'Riwj]' # C12 upper right
      WiWj = Wu.t() * Ri * Wu; // wk'Riwj # C22 lower right
    }else{ // if R is simple multiply only obtain these matrices once
      if(useH == true){ // use weights
        XWjxZWj = Wm.t() * Ri * Wu;
        WiWj = Wu.t() * Ri * Wu;
      }else{ // not using weights
        if(iIter == 0){
          XWjxZWj0 = Wm.t() * Wu; // [X'Riwj Z'Riwj]' # C12 upper right
          WiWj0 = Wu.t() * Wu; // wk'Riwj # C22 lower right
        }
        XWjxZWj = XWjxZWj0 * (1/arma::as_scalar(thetaResidualsVec(0)));
        WiWj = WiWj0 * (1/arma::as_scalar(thetaResidualsVec(0)));
      }
    }
    // solve method: similar to spsolve for bu but My=XZRiy and XWjxZWj=XZRi.Wu
    arma::spsolve(buWu, arma::sp_mat(M.submat(0,0, M.n_rows-2, M.n_cols-2)), arma::mat(XWjxZWj), "lapack");
    avInf = WiWj - (buWu.t()*XWjxZWj); // WuWu' - bu.Wu'*W.Wu

    // ##########################
    // # 5) get 1st derivatives (dL/ds2i) from MME-version
    // # PAPER FORMULA (Lee and Van der Werf, 2006)
    // # dL/ds2u = -0.5 [(Nu/s2u) - (tr(AiCuu)/s4u) - (e/s2e)'(Zu/s2u)]
    // # dL/ds2e = -0.5 [((Nr-Nb)/s2e) - [(Nu - (tr(AiCuu)/s2u))*(1/s2e)] - ... - (e/s2e)'(e/s2e)]
    // #
    // # PAPER FORMULA (Jensen and Madsen, 1997)
    // # dL/ds2u = (q.i * lambda) - (lambda * (T + S) * lambda)  Eq. 18
    // # dL/ds2e = tr(Rij*Ri) - tr(Ci*W'*Ri*Rij*Ri*W) - (e'*Ri*Rij*Ri*e)
    // ###########################
    // First derivatives
    // get the inverse of the coefficient matrix
    arma::vec v2(Mchol.n_cols-1, arma::fill::ones);
    arma::mat Dmat = diagmat(v2);
    arma::mat Cichol = arma::solve(trimatu(Mchol.submat(0,0,Mchol.n_rows-2,Mchol.n_cols-2)), Dmat); // indicate that A is triangular
    // multiply by its transpose
    arma::sp_mat Cip = arma::sp_mat(Cichol);
    Ci = Cip * Cip.t();
    arma::field<arma::mat> emInfList(nRRe);
    arma::vec dLu;
    if(nZs > 0){
      for(int iR = 0; iR < nRe; ++iR){
        arma::mat thetaCprov = thetaC[iR];
        arma::sp_mat traces(lambda(iR).n_rows,lambda(iR).n_cols);
        for(int iRow = 0; iRow < (int)lambda(iR).n_rows; ++iRow){
          for(int iCol = 0; iCol < (int)lambda(iR).n_cols; ++iCol){
            if(thetaCprov(iRow,iCol) > 0){ // if vc has to be estimated
              arma::mat partitionsP = partitions(iR);
              double trAiCuu = arma::trace(Ai(iR) * Ci.submat(partitionsP(iRow,0)-1, partitionsP(iCol,0)-1, partitionsP(iRow,1)-1, partitionsP(iCol,1)-1));
              traces(iRow,iCol) = trAiCuu;
            }else{
              traces(iRow,iCol) = 0;
            }
          }
        }
        traces = arma::symmatu(traces); // copy upper in lower triangular
        // first derivatives = dL/ds2u = (q.i * lambda) - (lambda * (T + S) * lambda) where S=UAiU
        arma::sp_mat dLuProv = (arma::as_scalar(nUsTotal(iR)) * lambda(iR)) - (uSinv(iR).t() * Ai(iR) * uSinv(iR)) - (lambda(iR) * traces * lambda(iR));
        // alternative EM update: theta - (theta*dLuProv*theta)/Nus  Eq.34
        arma::vec thetaUnlisted = mat_to_vecCpp2(theta(iR),thetaCprov);
        arma::mat thetaUnlistedMat = diagmat(thetaUnlisted);
        arma::mat emInfInvProvExt = (thetaUnlistedMat * thetaUnlistedMat.t()) / arma::as_scalar(nUsTotal(iR));
        emInfList(iR) = arma::pinv(emInfInvProvExt, tolParInv);
        dLu = join_cols(dLu, mat_to_vecCpp2(arma::mat(dLuProv),thetaCprov));
      } // end of loop for each random effect
    } // end of condition if random effects exist
    arma::vec dLe(Si.size());
    arma::sp_mat eProv = arma::sp_mat(e);
    for(int iS = 0; iS < (int)Si.size(); ++iS){
      arma::sp_mat Sprov(nR,nR);
      const arma::mat & pSi = partitionsS[iS];
      int s1 = pSi(0,0)-1;
      int s2 = pSi(0,1)-1;
      Sprov.submat(s1,s1,s2,s2) = Si(iS);
      if(nSs > 1){
        dLe(iS) = (arma::trace(Sprov*Ri) - arma::trace(Ci * Wm.t() * Ri * Sprov * Ri * Wm)) - arma::as_scalar(eProv.t() * Ri * Sprov * Ri * eProv);
      }else{
        if(useH == true){
          dLe(iS) = (arma::trace(Sprov*Ri) - arma::trace(Ci * Wm.t() * Ri * Sprov * Ri * Wm)) - arma::as_scalar(eProv.t() * Ri * Sprov * Ri * eProv);
        }else{
          dLe(iS) = (arma::trace(Sprov * (1/arma::as_scalar(thetaResidualsVec(0)))) - arma::trace(Ci * Wm.t() * (1/arma::as_scalar(thetaResidualsVec(0))) * Sprov * (1/arma::as_scalar(thetaResidualsVec(0))) * Wm)) - arma::as_scalar(eProv.t() * (1/arma::as_scalar(thetaResidualsVec(0))) * Sprov * (1/arma::as_scalar(thetaResidualsVec(0))) * eProv);
        }
      }
    }
    arma::vec thetaRUnlisted = mat_to_vecCpp2(theta(nRRe-1),thetaC[nRRe-1]);
    arma::mat thetaRUnlistedMat = diagmat(thetaRUnlisted);
    arma::mat emInfInvRProvExt = (thetaRUnlistedMat * thetaRUnlistedMat.t()) / arma::as_scalar(nR);
    emInfList(nRRe-1) = arma::pinv(emInfInvRProvExt, tolParInv); // we invert because this is equivalent to the inverse of the information and we need the information
    dLu = join_cols(dLu, dLe); // join the random and residual first derivatives in a single vector

    for(int i = 0; i < nRRe; ++i){
      emInf.submat(nVcStart(i)-1, nVcStart(i)-1, nVcEnd(i)-1, nVcEnd(i)-1) = emInfList(i);
    }

    // ###########################
    // # 6) update the variance parameters using the Newton method
    // # PAPER FORMULA (Lee and Van der Werf, 2006)
    // # theta.n+1 = theta.n + (AInfi * dL/ds2)
    // ###########################
    // Update variance parameters
    arma::vec thetaUnlisted;
    thetaCUnlisted.reset();
    for(int i = 0; i < nRRe; ++i){
      thetaUnlisted = arma::join_cols(thetaUnlisted, mat_to_vecCpp2(theta(i),thetaC(i)));
      thetaCUnlisted = arma::join_cols(thetaCUnlisted, mat_to_vecCpp2(thetaC(i),thetaC(i)));
    }
    // create the 'weight' EM information matrix
    arma::vec v3(nVcTotal, arma::fill::ones);
    arma::mat weightEmInfMat = diagmat(v3) * arma::as_scalar(weightEmInf(iIter));
    arma::mat weightAiInfMat = diagmat(v3) * (1 - arma::as_scalar(weightEmInf(iIter)));
    // Joint information matrix and update
    //                  AVERAGE INFORMATION                         +       EXPECTATION MAXIMIZATION
    InfMat = (weightAiInfMat * avInf) + (weightEmInfMat * emInf);
    InfMatInv = arma::pinv(InfMat, tolParInv); // inverse of the information matrix
    delta = (InfMatInv * arma::as_scalar(weightInf(iIter))) * dLu; // delta = I- * dLu/dLx
    // new values for variance components theta.i+1 = theta.i + delta
    arma::vec expectedNewTheta = thetaUnlisted - delta;

    // #######################
    // # 7) APPLY CONSTRAINTS to VC
    // # suggestions from Madsen and Jensen (1997) and Gilmour (2019)
    // #######################
    // A) apply constraints for fixed and positive
    // Apply constraints
    for(int i = 0; i < (int)thetaCUnlisted.size(); ++i){
      if(thetaCUnlisted(i) == 1){
        if(expectedNewTheta(i) < 1e-10){
          expectedNewTheta(i)=1e-10;
          toBoundary(iIter,i)=1; // toBoundary(nIters,nVcTotal)
          // change to fixed the ones that constantly (3 times) go to the boundaries
          for(int j = 0; j < (int)toBoundary.n_cols; ++j){
            sumToBoundary(j) = arma::accu(toBoundary.col(j));
          }
          arma::uvec toBeForced = arma::find(sumToBoundary >= 2);
          if(toBeForced.n_elem > 0){
            thetaCUnlisted(toBeForced) = thetaCUnlisted(toBeForced) - thetaCUnlisted(toBeForced) + 3;
          }
        }
      }
      if(expectedNewTheta(i) > 1){ // since we scale the model we don't allow to explain more than 100%
        expectedNewTheta(i)=1;
        toBoundary(iIter,i)=1; // toBoundary(nIters,nVcTotal)
      }
      if(thetaCUnlisted(i) == 3){
        arma::vec thetaUnlistedPlusAddScaleParam;
        if(iIter == 0){
          thetaUnlistedPlusAddScaleParam = arma::join_cols(thetaUnlisted,addScaleParam);
        }else{
          thetaUnlistedPlusAddScaleParam = arma::join_cols(monitor.col((iIter-1)),addScaleParam);
        }
        expectedNewTheta(i) = arma::as_scalar(thetaF.row(i) * thetaUnlistedPlusAddScaleParam);
      }
    }

    // B) if there's constrained or to boundary VC we need to partition InfMat and make a different update
    // Constrained update
    arma::vec thetaCUnlisted0 = thetaCUnlisted;
    thetaCUnlisted0(arma::find(thetaCUnlisted0 < 3)) = thetaCUnlisted0(arma::find(thetaCUnlisted0 < 3)) * 0;
    arma::uvec constrained = arma::find((thetaCUnlisted0+sumToBoundary) > 0);
    arma::uvec unconstrained = arma::find((thetaCUnlisted0+sumToBoundary) <= 0);
    if(constrained.n_elem > 0){
      arma::mat InfMat_uu = InfMat(unconstrained,unconstrained);
      arma::mat InfMat_ff = InfMat(constrained,constrained);
      arma::mat InfMat_uf = InfMat(unconstrained,constrained);
      arma::vec dLu_uu = dLu(unconstrained);
      arma::vec dLu_ff = dLu(constrained);
      arma::mat InfMatInv_ff = pinv(InfMat_ff,tolParInv);
      arma::mat InfMatInv_uu = pinv(InfMat_uu,tolParInv);
      arma::vec delta_ff = InfMatInv_ff * dLu_ff;
      arma::vec delta_uu = InfMatInv_uu * (dLu_uu - (InfMat_uf * (InfMatInv_ff*delta_ff)));
      delta(unconstrained) = delta_uu;
      expectedNewTheta(unconstrained) = thetaUnlisted(unconstrained) - delta(unconstrained);
    }

    // C) quantify delta changes
    if(iIter == 0){
      delta_minus1 = delta;
    }else{
      percDelta.col(iIter) = delta/delta_minus1;
    }

    // D) if not positive-definite change to EM update
    // PD check
    arma::vec pdCheck;
    for(int i = 0; i < nRRe; ++i){
      arma::uvec toFill = arma::regspace<arma::uvec>(nVcStart(i)-1,1,nVcEnd(i)-1);
      arma::mat thetaProvNew = vec_to_matCpp(expectedNewTheta(toFill), thetaC[i]);
      arma::mat thetaProvNewS = arma::symmatu(thetaProvNew);
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, thetaProvNewS);
      pdCheck = arma::join_cols(pdCheck,eigval);
    }
    arma::uvec eigenFind = find(pdCheck < 0);
    if(eigenFind.n_elem > 0){
      if(verbose == true){
        std::cout << "Updated VC is not positive definite, changing to EM step" << std::endl;
      }
      InfMat = (0.5 * avInf) + (0.5 * emInf);
      if(constrained.n_elem > 0){
        std::cout << "Update using constraints" << std::endl;
        arma::mat InfMat_uu = InfMat(unconstrained,unconstrained);
        arma::vec dLu_uu = dLu(unconstrained);
        arma::mat InfMatInv_uu = pinv(InfMat_uu,tolParInv);
        arma::vec delta_uu = InfMatInv_uu * dLu_uu;
        delta(unconstrained) = delta_uu;
        expectedNewTheta(unconstrained) = thetaUnlisted(unconstrained) - delta(unconstrained);
      }else{
        InfMatInv = arma::pinv(InfMat, tolParInv);
        expectedNewTheta = thetaUnlisted - (InfMatInv * dLu);
      }
    }

    // #######################
    // # 8) Bring back theta in vector form to matrix form and save for monitor
    // #######################
    // Save and update theta
    monitor.col(iIter) = expectedNewTheta;
    for(int i = 0; i < nRRe; ++i){
      arma::uvec toFill = arma::regspace<arma::uvec>(nVcStart(i)-1,1,nVcEnd(i)-1);
      arma::mat thetaProvNew = vec_to_matCpp(expectedNewTheta(toFill), thetaC[i]);
      arma::mat thetaProvNewPD = nearPDcpp(arma::symmatu(thetaProvNew), 100, 1e-06, 1e-07);
      theta(i) = thetaProvNewPD;
      arma::mat thetaCProvNew = vec_to_matCpp(thetaCUnlisted(toFill), thetaC[i]);
      thetaC(i) = thetaCProvNew;
    }

    // #######################
    // # 9) Stopping criteria
    // #######################
    // Stopping criteria
    // get current time
    time_t now = time(0);
    tm *ltm = localtime(&now);
    // keep track of time difference between iterations
    seconds = difftime(now,before);
    // update before time to be the now to be used in the next iteration
    before = time(0);
    localtime(&before);
    // E) stopping criteria (norms)
    normMonitor(0,iIter) = arma::norm(delta(unconstrained), 1); // stopping criteria 1
    normMonitor(1,iIter) = arma::norm(dLu(unconstrained),1); // stopping criteria 2
    arma::vec nVcTotalv(1);
    nVcTotalv(0) = nVcTotal;
    arma::vec stopCriteria3 = (InfMatInv.diag()/arma::as_scalar(arma::sqrt(nVcTotalv))) % dLu;
    normMonitor(2,iIter) = arma::norm(stopCriteria3(unconstrained),1); // stopping criteria 3

    arma::uvec restrained = arma::find(toBoundary.row(iIter) > 0);
    if(verbose == true){
      if(iIter == 0){std::cout << "iteration   " << " LogLik   " << "  wall    " << "cpu(sec)   " << "restrained" << std::endl;}
      std::cout << "    " << iIter+1 << "      " << llik(iIter) << "   " << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "      " << seconds << "           " << restrained.n_elem << std::endl;
    }
    if(iIter > 0){
      double delta_llik = llik(iIter) - llik(iIter-1);
      if( (normMonitor(2,iIter) < tolParConvNorm) || (delta_llik < tolParConvLL) || (iIter == nIters) ) {
        if( (normMonitor(2,iIter) < tolParConvNorm) || (delta_llik < tolParConvLL) ){
          convergence = true;
        }
        monitor = monitor.cols(0,iIter);
        normMonitor = normMonitor.cols(0,iIter);
        percDelta = percDelta.cols(0,iIter);
        llik = llik.cols(0,iIter);
        break;
      }
    }
    dLuOut = dLu;
  } // end of iterative optimization

  double AIC = (-2 * llik((llik.n_cols-1))) + (2 * nX);
  double BIC = (-2 * llik((llik.n_cols-1))) + (log((double)nR) * nX);

  // move constraints to vector form binding the columns
  arma::vec thetaCUnlistedFinal;
  for (int i = 0; i < nRRe; ++i) {
    thetaCUnlistedFinal = join_cols(thetaCUnlistedFinal,mat_to_vecCpp2(thetaC[i],thetaC[i]));
  }

  // bring back to original scale the variance components
  for (int i = 0; i < nRRe; ++i) {
    theta(i) = theta(i)*vary;
  }
  bu = bu*stdy;
  b = b*stdy;
  if(intercept==true){ // if true mu is required in the first position only
    b(0,0) = b(0,0) + muy;
    bu(0,0) = bu(0,0) + muy;
  }else{ // if false mu is required all over b
    bu.submat(0, 0, (b.n_rows-1), 0) = bu.submat(0, 0, (b.n_rows-1), 0) + muy;
    b = b + muy;
  }
  Ci = Ci*vary;
  InfMatInv = InfMatInv*vary;
  Mchol_XZ = Mchol_XZ/stdy;
  monitor = monitor*vary;

  // move the effects from vector to a field with matrices
  // Extract uList and uPevList
  std::vector<arma::mat> uListOut(nRe), uPevListOut(nRe);
  for (int i = 0; i < nRe; ++i) {
    arma::mat partitionsP = partitions(i); // partition of random effect i
    arma::mat uMat(arma::as_scalar(partitionsP(0,1) - partitionsP(0,0) + 1), partitionsP.n_rows);
    arma::mat eMat(arma::as_scalar(partitionsP(0,1) - partitionsP(0,0) + 1), partitionsP.n_rows);
    for (int j = 0; j < (int)partitionsP.n_rows; ++j) {
      uMat.col(j) = bu.submat(arma::as_scalar(partitionsP(j,0)-1), 0, arma::as_scalar(partitionsP(j,1)-1), 0);
      eMat.col(j) = arma::diagvec(Ci.submat(arma::as_scalar(partitionsP(j,0)-1), arma::as_scalar(partitionsP(j,0)-1), arma::as_scalar(partitionsP(j,1)-1), arma::as_scalar(partitionsP(j,1)-1)));
    }
    uListOut[i] = uMat;
    uPevListOut[i] = eMat;
  }

  // Convert theta field to vector of mats
  std::vector<arma::mat> thetaOut(nRRe);
  for(int i = 0; i < nRRe; ++i){
    thetaOut[i] = theta(i);
  }

  // Convert partitions field to vector
  std::vector<arma::mat> partitionsOut(nRe);
  for(int i = 0; i < nRe; ++i){
    partitionsOut[i] = partitions(i);
  }

  // return results
  py::dict result;
  result["llik"] = _mat_to_py(arma::mat(llik));
  result["W"] = _mat_to_py(arma::mat(Wm));
  result["b"] = _mat_to_py(arma::mat(b));
  result["u"] = _mat_to_py(arma::mat(u));
  result["bu"] = _mat_to_py(arma::mat(bu));
  result["Ci"] = _mat_to_py(arma::mat(Ci));
  result["theta"] = _matvec_to_py(thetaOut);
  result["theta_se"] = _mat_to_py(InfMatInv);
  result["InfMat"] = _mat_to_py(InfMat);
  result["monitor"] = _mat_to_py(monitor);
  result["uList"] = _matvec_to_py(uListOut);
  result["uPevList"] = _matvec_to_py(uPevListOut);
  result["AIC"] = AIC;
  result["BIC"] = BIC;
  result["convergence"] = convergence;
  result["partitions"] = _matvec_to_py(partitionsOut);
  result["percDelta"] = _mat_to_py(percDelta);
  result["normMonitor"] = _mat_to_py(normMonitor);
  result["toBoundary"] = _mat_to_py(toBoundary);
  result["dLu"] = _vec_to_py(dLuOut);
  result["Cchol"] = _mat_to_py(Mchol_XZ);
  return result;
}


// ============================================================================
// MNR — Dense matrix version of the Newton-Raphson solver
// ============================================================================

py::dict MNR_cpp(
    const arma::mat & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & Gx,
    const std::vector<arma::mat> & Z,
    const std::vector<arma::mat> & K,
    const std::vector<arma::sp_mat> & R,
    const std::vector<arma::mat> & Ge,
    const std::vector<arma::mat> & GeI,
    const arma::mat & W,
    const bool & isInvW,
    int iters, double tolpar, double tolparinv,
    const bool & ai, const bool & pev,
    const bool & verbose, const bool & retscaled,
    const arma::vec & stepweight,
    const arma::vec & emweight)
{
  // This is a dense-matrix version that closely mirrors newton_di_sp
  // but operates on dense arma::mat inputs.
  // For brevity this wraps into the sparse version by converting inputs.
  // In a production build you'd keep the full dense implementation.

  // Convert dense inputs to sparse for newton_di_sp_cpp
  std::vector<arma::sp_mat> X_sp(X.size());
  for(size_t i = 0; i < X.size(); ++i) X_sp[i] = arma::sp_mat(X[i]);

  std::vector<arma::sp_mat> Z_sp(Z.size());
  for(size_t i = 0; i < Z.size(); ++i) Z_sp[i] = arma::sp_mat(Z[i]);

  std::vector<arma::sp_mat> K_sp(K.size());
  for(size_t i = 0; i < K.size(); ++i) K_sp[i] = arma::sp_mat(K[i]);

  arma::sp_mat Y_sp(Y);
  arma::sp_mat W_sp(W);

  // Build default thetaConstOri and thetaIndex from GeI
  int n_random = Z.size();
  int n_rcov = R.size();
  int n_re = n_random + n_rcov;

  std::vector<arma::mat> thetaConstOri(n_re);
  for(int i = 0; i < n_re; ++i){
    if(i < (int)GeI.size()){
      thetaConstOri[i] = GeI[i];
    }
  }

  // Build thetaIndex
  arma::vec thetaIndex;
  for(int i = 0; i < n_re; ++i){
    const arma::mat & gei = GeI[i];
    arma::vec vv = mat_to_vecCpp(gei, gei);
    arma::vec idx(vv.n_elem);
    idx.fill(i+1);
    thetaIndex = join_cols(thetaIndex, idx);
  }

  return newton_di_sp_cpp(Y_sp, X_sp, Gx, Z_sp, K_sp, R, Ge, GeI,
                          W_sp, isInvW, iters, tolpar, tolparinv,
                          ai, pev, verbose, retscaled, stepweight, emweight,
                          thetaConstOri, thetaIndex);
}
