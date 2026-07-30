# Fixed-bandwidth sharp/fuzzy RD parity against official R rdrobust.
# Run from the CauseKit repository root:
#   Rscript benchmarks/validate_rd_reference.R

if (!requireNamespace("rdrobust", quietly = TRUE)) {
  stop("Install the official comparator first: install.packages('rdrobust')")
}

position <- 0:399
running_left <- -2 + position * (1.98 / 399)
running_right <- 0.02 + position * (1.98 / 399)
running <- c(running_left, running_right)
assignment <- as.numeric(running >= 0)

sharp_outcome <- 0.5 + 0.7 * running + 0.3 * running^2 +
  1.8 * assignment + 0.4 * sin(17 * running) + 0.2 * cos(31 * running)
treatment <- c(as.numeric(position %% 5 == 0), as.numeric(position %% 5 <= 2))
fuzzy_outcome <- 0.5 + 0.4 * running + 0.2 * running^2 +
  2.2 * treatment + 0.35 * sin(13 * running) + 0.15 * cos(29 * running)

common_options <- list(
  x = running,
  c = 0,
  h = c(1.2, 1.4),
  b = c(1.6, 1.7),
  p = 1,
  q = 2,
  kernel = "triangular",
  vce = "hc1"
)
sharp <- do.call(rdrobust::rdrobust, c(list(y = sharp_outcome), common_options))
fuzzy <- do.call(
  rdrobust::rdrobust,
  c(list(y = fuzzy_outcome, fuzzy = treatment), common_options)
)

observed <- c(
  sharp_conventional_estimate = unname(sharp$Estimate[1, "tau.us"]),
  sharp_bias_corrected_estimate = unname(sharp$Estimate[1, "tau.bc"]),
  sharp_robust_standard_error = unname(sharp$Estimate[1, "se.rb"]),
  fuzzy_conventional_estimate = unname(fuzzy$Estimate[1, "tau.us"]),
  fuzzy_bias_corrected_estimate = unname(fuzzy$Estimate[1, "tau.bc"]),
  fuzzy_robust_standard_error = unname(fuzzy$Estimate[1, "se.rb"]),
  fuzzy_bias_corrected_treatment_jump = unname(fuzzy$tau_T[2])
)
expected <- c(
  sharp_conventional_estimate = 2.012628078790254,
  sharp_bias_corrected_estimate = 2.149509125727007,
  sharp_robust_standard_error = 0.06729310580994985,
  fuzzy_conventional_estimate = 2.7836636780089483,
  fuzzy_bias_corrected_estimate = 3.0987154201530087,
  fuzzy_robust_standard_error = 0.19880345063046992,
  fuzzy_bias_corrected_treatment_jump = 0.4360805850270855
)
tolerance <- 5e-12
differences <- abs(observed - expected)
status <- ifelse(differences <= tolerance, "pass", "fail")
parity_status <- if (all(status == "pass")) "pass" else "fail"

results_path <- "benchmarks/validate_rd_r_output.txt"
connection <- file(results_path, open = "wt")
writeLines("contract=fixed_bandwidth_triangular_p1_q2_hc1", connection)
writeLines(paste0("r_version=", R.version.string), connection)
writeLines(paste0("rdrobust_version=", as.character(utils::packageVersion("rdrobust"))), connection)
writeLines("nobs=800", connection)
writeLines("bandwidth_left=1.2", connection)
writeLines("bandwidth_right=1.4", connection)
writeLines("bias_bandwidth_left=1.6", connection)
writeLines("bias_bandwidth_right=1.7", connection)
writeLines(paste0("tolerance=", format(tolerance, scientific = TRUE)), connection)
for (name in names(observed)) {
  writeLines(paste0(name, "=", format(observed[[name]], digits = 17)), connection)
  writeLines(
    paste0(name, "_absolute_difference=", format(differences[[name]], digits = 17)),
    connection
  )
  writeLines(paste0(name, "_status=", status[[name]]), connection)
}
writeLines(paste0("parity_status=", parity_status), connection)
close(connection)

cat("results_file=", results_path, "\n", sep = "")
cat("parity_status=", parity_status, "\n", sep = "")
if (parity_status != "pass") {
  stop("R rdrobust parity failed; inspect ", results_path)
}
