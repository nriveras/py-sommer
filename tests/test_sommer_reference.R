#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(sommer)
  library(jsonlite)
})

set.seed(123)
out_dir <- file.path("tests", "reference_data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_mat <- function(name, x) {
  write.csv(as.matrix(x), file = file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
}

# Marker matrix in {-1, 0, 1} with a few missing values.
n <- 10
p <- 14
X <- matrix(sample(c(-1, 0, 1), n * p, replace = TRUE), nrow = n, ncol = p)
X[sample(length(X), 5)] <- NA

Ares <- A.mat(X, min.MAF = 0, return.imputed = TRUE)
Dres <- D.mat(X, nishio = TRUE, min.MAF = 0, return.imputed = TRUE)
Eres <- E.mat(Ares$X, type = "A#A", min.MAF = 0)

A <- Ares$A
D <- Dres$D
E <- Eres

# H matrix: use the last 6 individuals as genotyped.
A_ped <- A
G22 <- A_ped[(n - 5):n, (n - 5):n]
H <- H.mat(A = A_ped, G = G22, tau = 1, omega = 1, tolparinv = 1e-6)

# Covariance structure constructors.
ar1 <- AR1(1:6, rho = 0.35)
cs <- CS(1:6, rho = 0.20)
arma_m <- ARMA(1:6, rho = 0.35, lambda = 0.20)

# Utility helpers from C++ exports.
x_small <- matrix(c(1, 2, 3, 4, 6, 8, 1, 1, 1), nrow = 3, byrow = TRUE)
scaled <- sommer:::scaleCpp(x_small)
full_rank <- sommer:::makeFull(cbind(1, 1:6, 2 * (1:6), rnorm(6)))

# nearPD projection.
bad_pd <- matrix(c(1, 2, 2, 1), 2, 2)
npd <- sommer:::nearPDcpp(bad_pd, 100, 1e-6, 1e-7)

write_mat("A_mat", A)
write_mat("D_mat", D)
write_mat("E_mat", E)
write_mat("H_mat", H)
write_mat("AR1", ar1)
write_mat("CS", cs)
write_mat("ARMA", arma_m)
write_mat("scaleCpp", scaled)
write_mat("makeFull", full_rank)
write_mat("nearPD", npd)

# Simple random-intercept model for reference variance components.
id <- as.factor(rep(1:20, each = 2))
u <- rnorm(20, 0, sqrt(1.2))
e <- rnorm(length(id), 0, sqrt(0.4))
y <- 2 + u[as.integer(id)] + e
dat <- data.frame(y = y, id = id)

solver_out <- list()
fit <- try(
  mmes(
    y ~ 1,
    random = ~id,
    rcov = ~units,
    data = dat,
    nIters = 30,
    tolParConvNorm = 1e-6,
    tolParConvLL = 1e-6,
    verbose = FALSE,
    dateWarning = FALSE
  ),
  silent = TRUE
)

if (!inherits(fit, "try-error")) {
  # sommer object structure may vary by version; keep a robust subset.
  solver_out$converged <- isTRUE(fit$convergence)
  if (!is.null(fit$sigma)) {
    solver_out$theta <- as.numeric(fit$sigma)
  } else if (!is.null(fit$sigma_scaled)) {
    solver_out$theta <- as.numeric(fit$sigma_scaled)
  }
  if (!is.null(fit$Beta)) {
    solver_out$beta <- as.numeric(fit$Beta)
  }
}

write_json(solver_out, path = file.path(out_dir, "mmes_reference.json"), auto_unbox = TRUE, pretty = TRUE)

cat("Reference data written to:", out_dir, "\n")
