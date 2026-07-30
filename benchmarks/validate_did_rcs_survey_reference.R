# Independent R survey reconstruction of CauseKit's stratified-PSU Taylor contract.
#
# Run from the repository root:
#   Rscript benchmarks/validate_did_rcs_survey_reference.R

if (!requireNamespace("survey", quietly = TRUE)) {
  stop("Install the CRAN survey package before running this reference.")
}

outcome <- c(5, 7, 6, 8, 9, 12, 10, 13, 3, 4, 2, 5, 4, 7, 3, 6)
time <- rep(c(1, 2, 1, 2), each = 4)
cohort <- rep(c(2, 2, Inf, Inf), each = 4)
weight <- c(1, 2, 1, 3, 2, 1, 3, 1, 1, 2, 2, 1, 2, 1, 1, 2)
psu <- rep(c("a1", "a2", "b1", "b2"), 4)
stratum <- rep(c("a", "a", "b", "b"), 4)
cell <- rep(c("tb", "tt", "cb", "ct"), each = 4)
data <- data.frame(outcome, time, cohort, weight, psu, stratum, cell)

for (name in c("tt", "tb", "ct", "cb")) {
  selected <- as.numeric(data$cell == name)
  data[[paste0("ny_", name)]] <- data$outcome * selected
  data[[paste0("n_", name)]] <- selected
}

design <- survey::svydesign(
  ids = ~psu,
  strata = ~stratum,
  weights = ~weight,
  data = data,
  nest = TRUE
)
totals <- survey::svytotal(
  ~ny_tt + n_tt + ny_tb + n_tb + ny_ct + n_ct + ny_cb + n_cb,
  design
)
contrast <- survey::svycontrast(
  totals,
  quote(ny_tt / n_tt - ny_tb / n_tb - ny_ct / n_ct + ny_cb / n_cb)
)

survey_estimate <- as.numeric(stats::coef(contrast))
survey_standard_error <- as.numeric(survey::SE(contrast))
design_df <- survey::degf(design)

cell_linearized <- function(name) {
  selected <- data$cell == name
  denominator <- sum(data$weight[selected])
  estimate <- sum(data$weight[selected] * data$outcome[selected]) / denominator
  linearized <- data$weight * selected * (data$outcome - estimate) / denominator
  list(estimate = estimate, linearized = linearized)
}
tt <- cell_linearized("tt")
tb <- cell_linearized("tb")
ct <- cell_linearized("ct")
cb <- cell_linearized("cb")
base_estimate <- tt$estimate - tb$estimate - ct$estimate + cb$estimate
linearized <- tt$linearized - tb$linearized - ct$linearized + cb$linearized

covariance <- 0
for (h in unique(data$stratum)) {
  positions <- data$stratum == h
  totals_h <- tapply(linearized[positions], data$psu[positions], sum)
  covariance <- covariance + length(totals_h) / (length(totals_h) - 1) *
    sum((totals_h - mean(totals_h)) ^ 2)
}
base_standard_error <- sqrt(covariance)

expected_estimate <- 1.7619047619047623
expected_standard_error <- 0.05472797454035001
tolerance <- 1e-12
statuses <- c(
  abs(base_estimate - expected_estimate) <= tolerance,
  abs(base_standard_error - expected_standard_error) <= tolerance,
  abs(survey_estimate - expected_estimate) <= tolerance,
  abs(survey_standard_error - expected_standard_error) <= tolerance,
  design_df == 2
)
parity_status <- ifelse(all(statuses), "pass", "fail")

lines <- c(
  "contract=survey_rcs_component_hajek_stratified_psu_taylor",
  paste0("r_version=", paste(R.version$major, R.version$minor, sep = ".")),
  paste0("survey_version=", as.character(utils::packageVersion("survey"))),
  paste0("estimate=", sprintf("%.17g", base_estimate)),
  paste0("standard_error=", sprintf("%.17g", base_standard_error)),
  paste0("survey_estimate=", sprintf("%.17g", survey_estimate)),
  paste0("survey_standard_error=", sprintf("%.17g", survey_standard_error)),
  paste0("design_df=", design_df),
  paste0("linearized=", paste(sprintf("%.17g", linearized), collapse = ",")),
  paste0("parity_status=", parity_status)
)
output_path <- "benchmarks/validate_did_rcs_survey_reference_output.txt"
writeLines(lines, output_path)
cat(paste0(lines, "\n"), sep = "")
if (parity_status != "pass") {
  stop(paste("Survey R parity failed; inspect", output_path))
}
