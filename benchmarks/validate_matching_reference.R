# Cross-language matching parity against the CRAN Matching source mirror.
#
# Usage:
#   Rscript benchmarks/validate_matching_reference.R PATH_TO_MATCHING_CHECKOUT
#
# The checkout must be https://github.com/cran/Matching at tag 4.10-15. The
# harness invokes the package's pure-R Rmatch reference path because that path
# exposes the same fixed-score Abadie-Imbens equations without requiring a
# platform-specific compiled package installation.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("supply exactly one path to the Matching source checkout")
}

matching_root <- normalizePath(args[[1]], mustWork = TRUE)
reference_commit <- system2(
  "git", c("-C", matching_root, "rev-parse", "HEAD"), stdout = TRUE
)
expected_commit <- "1208eaa7bfa888b1fc903481dddfb8c0dffa40d5"
if (length(reference_commit) != 1 || reference_commit[[1]] != expected_commit) {
  stop(paste0("Matching checkout must be at recorded commit ", expected_commit))
}

suppressPackageStartupMessages(library(MASS))
source(file.path(matching_root, "R", "Matching.R"))

outcome <- matrix(c(0, 2, 5, 3, 8, 12), ncol = 1)
treatment <- matrix(c(0, 0, 0, 1, 1, 1), ncol = 1)
logit_score <- matrix(c(0, 4, 10, 1, 6, 9), ncol = 1)
unit_weights <- matrix(1, nrow = 6, ncol = 1)

fit_reference <- function(all_code) {
  Rmatch(
    Y = outcome,
    Tr = treatment,
    X = logit_score,
    Z = logit_score,
    V = unit_weights,
    All = all_code,
    M = 1,
    BiasAdj = 0,
    Weight = 1,
    Weight.matrix = 1,
    Var.calc = 1,
    weight = unit_weights,
    SAMPLE = 0,
    ccc = sqrt(.Machine$double.eps),
    cdd = .Machine$double.eps,
    ecaliper = NULL,
    restrict = NULL
  )
}

description <- read.dcf(file.path(matching_root, "DESCRIPTION"))
cat("reference_commit=", reference_commit[[1]], "\n", sep = "")
cat("matching_version=", description[1, "Version"], "\n", sep = "")
all_codes <- c(att = 0, ate = 1, atc = 2)
for (name in names(all_codes)) {
  result <- fit_reference(all_codes[[name]])
  cat(name, "_estimate=", sprintf("%.17g", result$est), "\n", sep = "")
  cat(name, "_standard_error=", sprintf("%.17g", result$se), "\n", sep = "")
}
