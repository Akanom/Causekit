#!/usr/bin/env Rscript

# Official R fixed-effects Panel IV parity using AER::ivreg and sandwich::vcovCL.

source_url <- paste0(
  "https://raw.githubusercontent.com/bashtage/linearmodels/",
  "28af72e/linearmodels/datasets/wage_panel/wage_panel.csv.bz2"
)
source_sha256 <- "ee4f36706491d6348614f06bfcd5b2eef411603590a996ebb061078ae1024e78"
results_path <- "benchmarks/validate_panel_iv_r_output.txt"

if (!requireNamespace("AER", quietly = TRUE)) {
  stop("AER is required: install.packages('AER')")
}
if (!requireNamespace("sandwich", quietly = TRUE)) {
  stop("sandwich is required: install.packages('sandwich')")
}
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("digest is required: install.packages('digest')")
}

source_file <- tempfile(fileext = ".csv.bz2")
on.exit(unlink(source_file), add = TRUE)
download.file(source_url, source_file, mode = "wb", quiet = TRUE)
observed_hash <- digest::digest(file = source_file, algo = "sha256", serialize = FALSE)
if (!identical(observed_hash, source_sha256)) {
  stop(sprintf("Source SHA-256 mismatch: expected %s, observed %s", source_sha256, observed_hash))
}

data <- read.csv(bzfile(source_file))
data <- data[order(data$nr, data$year), ]
data$union_lag <- ave(
  data$union,
  data$nr,
  FUN = function(value) c(NA_real_, value[-length(value)])
)
data$hours_1000 <- data$hours / 1000
roles <- c("lwage", "union", "union_lag", "hours_1000", "married", "nr", "year")
complete <- data[stats::complete.cases(data[, roles]), ]

fit <- AER::ivreg(
  lwage ~ hours_1000 + married + union + factor(nr) + factor(year) |
    hours_1000 + married + union_lag + factor(nr) + factor(year),
  data = complete,
  x = TRUE,
  y = TRUE,
  model = TRUE
)
covariance <- sandwich::vcovCL(
  fit,
  cluster = complete$nr,
  type = "HC1",
  cadjust = TRUE,
  fix = FALSE
)

terms <- c("hours_1000", "married", "union")
estimate <- stats::coef(fit)[terms]
standard_error <- sqrt(diag(covariance))[terms]
expected_estimate <- c(
  hours_1000 = -0.15962613778805484,
  married = 0.06533125127699804,
  union = 0.24962753388340345
)
expected_standard_error <- c(
  hours_1000 = 0.02416893230287323,
  married = 0.02417901182155533,
  union = 0.31401132417362376
)
tolerance <- 1e-8
estimate_difference <- abs(estimate - expected_estimate)
standard_error_difference <- abs(standard_error - expected_standard_error)
estimate_status <- ifelse(estimate_difference <= tolerance, "pass", "fail")
standard_error_status <- ifelse(standard_error_difference <= tolerance, "pass", "fail")
parity_status <- if (
  all(estimate_status == "pass") && all(standard_error_status == "pass")
) "pass" else "fail"

lines <- c(
  "contract=wage_panel_two_way_fe_entity_clustered_cr1",
  paste0("r_version=", R.version.string),
  paste0("aer_version=", as.character(utils::packageVersion("AER"))),
  paste0("sandwich_version=", as.character(utils::packageVersion("sandwich"))),
  paste0("source_sha256=", observed_hash),
  paste0("nobs=", nrow(complete)),
  paste0("n_entities=", length(unique(complete$nr))),
  paste0("n_periods=", length(unique(complete$year))),
  sprintf("tolerance=%.17g", tolerance)
)
for (term in terms) {
  lines <- c(
    lines,
    sprintf("%s_estimate=%.17g", term, estimate[[term]]),
    sprintf("%s_expected_estimate=%.17g", term, expected_estimate[[term]]),
    sprintf("%s_estimate_absolute_difference=%.17g", term, estimate_difference[[term]]),
    paste0(term, "_estimate_status=", estimate_status[[term]]),
    sprintf("%s_standard_error=%.17g", term, standard_error[[term]]),
    sprintf("%s_expected_standard_error=%.17g", term, expected_standard_error[[term]]),
    sprintf(
      "%s_standard_error_absolute_difference=%.17g",
      term,
      standard_error_difference[[term]]
    ),
    paste0(term, "_standard_error_status=", standard_error_status[[term]])
  )
}
lines <- c(lines, paste0("parity_status=", parity_status))
writeLines(lines, results_path, useBytes = TRUE)
cat(paste0("results_file=", results_path, "\n"))
cat(paste0("parity_status=", parity_status, "\n"))
if (parity_status != "pass") {
  stop(sprintf("Panel IV R parity failed; inspect %s", results_path), call. = FALSE)
}
