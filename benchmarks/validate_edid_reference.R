# Cross-language point-estimate check against david-loeb/edid.
#
# Usage:
#   Rscript benchmarks/validate_edid_reference.R PATH_TO_EDID_CHECKOUT
#
# The recorded validation checkout is commit
# f55a4a4aba14f0826f59ad7aa4af3bafaeba529b. This script calls the public
# get_ytilde() and get_influence_func() implementation directly, then applies
# the package's documented inverse-covariance weighting without invoking its
# bootstrap dependency.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("supply exactly one path to the edid source checkout")
}

edid_root <- normalizePath(args[[1]], mustWork = TRUE)
reference_commit <- system2(
  "git", c("-C", edid_root, "rev-parse", "HEAD"), stdout = TRUE
)
expected_commit <- "f55a4a4aba14f0826f59ad7aa4af3bafaeba529b"
if (length(reference_commit) != 1 || reference_commit[[1]] != expected_commit) {
  stop(paste0("edid checkout must be at recorded commit ", expected_commit))
}
source(file.path(edid_root, "R", "get_ytilde.R"))
source(file.path(edid_root, "R", "get_influence_func.R"))

h1 <- c(1, 1, -1, -1)
h2 <- c(1, -1, 1, -1)
h3 <- c(1, -1, -1, 1)
treated <- data.frame(
  id = paste0("treated_", 0:3),
  g = 4,
  y1 = -(10 + h1),
  y2 = -(10 + 2 * h2),
  y3 = -(10 + 4 * h3),
  y4 = 0
)
never <- data.frame(
  id = paste0("never_", 0:3),
  g = 0,
  y1 = 0,
  y2 = 0,
  y3 = 0,
  y4 = 0
)
dat <- rbind(treated, never)
dat$g0 <- as.integer(dat$g == 0)
dat$g4 <- as.integer(dat$g == 4)

ytilde <- get_ytilde(dat)
influence <- get_influence_func(dat, ytilde)
influence_matrix <- matrix(
  unlist(influence), nrow = length(influence), byrow = TRUE
)
candidate_att <- vapply(ytilde, mean, numeric(1))
covariance <- cov(t(influence_matrix))
weights <- as.vector(solve(covariance, rep(1, nrow(covariance))))
weights <- weights / sum(weights)
efficient_influence <- as.vector(weights %*% influence_matrix)
efficient_att <- sum(weights * candidate_att)
standard_error <- sd(efficient_influence) / sqrt(length(efficient_influence))

cat("reference_commit=", reference_commit[[1]], "\n", sep = "")
cat("candidate_names=", paste(names(ytilde), collapse = ","), "\n", sep = "")
cat("candidate_att=", paste(sprintf("%.17g", candidate_att), collapse = ","), "\n", sep = "")
cat("weights=", paste(sprintf("%.17g", weights), collapse = ","), "\n", sep = "")
cat("efficient_att=", sprintf("%.17g", efficient_att), "\n", sep = "")
cat("standard_error_hc1=", sprintf("%.17g", standard_error), "\n", sep = "")
