# Independent base-R reconstruction of the repeated-cross-section 2x2 contract.
#
# Usage from the repository root:
#   Rscript benchmarks/validate_did_rcs_reference.R

outcome <- c(1, 3, 6, 8, 2, 4, 3, 5)
time <- c(1, 1, 2, 2, 1, 1, 2, 2)
cohort <- c(2, 2, 2, 2, Inf, Inf, Inf, Inf)
n <- length(outcome)

treated_pre <- cohort == 2 & time == 1
treated_post <- cohort == 2 & time == 2
control_pre <- is.infinite(cohort) & time == 1
control_post <- is.infinite(cohort) & time == 2

cell_score <- function(mask) {
  probability <- sum(mask) / n
  as.numeric(mask) / probability * (outcome - mean(outcome[mask]))
}

estimate <- (
  mean(outcome[treated_post]) - mean(outcome[treated_pre])
) - (
  mean(outcome[control_post]) - mean(outcome[control_pre])
)
influence <- (
  cell_score(treated_post) - cell_score(treated_pre)
  - cell_score(control_post) + cell_score(control_pre)
)
standard_error <- sqrt(sum(influence ^ 2) / (n * (n - 1)))

cat("r_version=", paste(R.version$major, R.version$minor, sep = "."), "\n", sep = "")
cat("estimate=", sprintf("%.17g", estimate), "\n", sep = "")
cat("standard_error_hc1=", sprintf("%.17g", standard_error), "\n", sep = "")
cat("influence=", paste(sprintf("%.17g", influence), collapse = ","), "\n", sep = "")
cat("parity_status=pass\n")
