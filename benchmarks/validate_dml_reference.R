# Base-R matrix/HC1 parity for CauseKit's partially linear DML2 score.
#
# Usage:
#   Rscript benchmarks/validate_dml_reference.R

outcome_residual <- c(
  -1.5033333333333334,
  1.3033333333333332,
  -3.3066666666666666,
  3.7066666666666666,
  -2.755,
  3.0549999999999997,
  -0.9516666666666667,
  0.45166666666666666
)
treatment_residual <- c(-1, 1, -2, 2, -1.5, 1.5, -0.5, 0.5)
nobs <- length(outcome_residual)
estimate <- sum(treatment_residual * outcome_residual) /
  sum(treatment_residual^2)
residual <- outcome_residual - estimate * treatment_residual
hc1_variance <- (nobs / (nobs - 1)) *
  sum((treatment_residual * residual)^2) /
  sum(treatment_residual^2)^2
standard_error <- sqrt(hc1_variance)

cat("artifact=causekit_partially_linear_dml_r_parity\n")
cat("causekit_version=0.7.0a1\n")
cat("contract=fixed_oof_nuisance_residual_regression_hc1\n")
cat("r_version=", paste(R.version$major, R.version$minor, sep = "."), "\n", sep = "")
cat("nobs=", nobs, "\n", sep = "")
cat("estimate=", sprintf("%.17g", estimate), "\n", sep = "")
cat("standard_error=", sprintf("%.17g", standard_error), "\n", sep = "")
cat("parity_status=reference_completed\n")
