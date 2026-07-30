# Independent base-R reconstruction of the direct-odds two-period PT-All score.
#
# Usage:
#   Rscript benchmarks/validate_did_direct_ratio_reference.R

entity <- c(sprintf("g_%02d", 0:11), sprintf("n_%02d", 0:5))
cohort <- c(rep(2, 12), rep(Inf, 6))
change <- c(
  3.0, 3.5, 2.5, 4.0, 2.0, 3.2, 3.8, 2.7, 3.3, 2.2, 4.1, 2.9,
  1.0, 1.5, 0.5, 2.0, 0.0, 1.2
)
# Frozen NumPy stratified-fold assignment for random_state=17 and entity order above.
fold <- c(2, 1, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 2, 2, 0, 0, 1)
n <- length(entity)
pi_g <- mean(cohort == 2)
ratio <- numeric(n)
mean_never <- numeric(n)

for (k in 0:2) {
  train <- fold != k
  holdout <- fold == k
  numerator_n <- sum(train & cohort == 2)
  denominator_n <- sum(train & is.infinite(cohort))
  ratio[holdout] <- numerator_n / denominator_n
  mean_never[holdout] <- mean(change[train & is.infinite(cohort)])
}

treated <- as.numeric(cohort == 2)
never <- as.numeric(is.infinite(cohort))
score <- treated / pi_g * (change - mean_never) -
  ratio * never / pi_g * (change - mean_never)
att <- mean(score)
influence <- score - treated / pi_g * att
standard_error <- sqrt(sum(influence^2) / (n * (n - 1)))

cat("contract=direct_posterior_cohort_odds_two_period_pt_all\n")
cat("generated_on=2026-07-30\n")
cat("generator=benchmarks/validate_did_direct_ratio_reference.R\n")
cat("reproduction_command=Rscript benchmarks/validate_did_direct_ratio_reference.R\n")
cat("r_version=", paste(R.version$major, R.version$minor, sep = "."), "\n", sep = "")
cat("fold=", paste(fold, collapse = ","), "\n", sep = "")
cat("ratio=", paste(sprintf("%.17g", ratio), collapse = ","), "\n", sep = "")
cat("score=", paste(sprintf("%.17g", score), collapse = ","), "\n", sep = "")
cat("influence=", paste(sprintf("%.17g", influence), collapse = ","), "\n", sep = "")
cat("estimate=", sprintf("%.17g", att), "\n", sep = "")
cat("standard_error_hc1=", sprintf("%.17g", standard_error), "\n", sep = "")
cat("parity_status=reference_completed\n")
