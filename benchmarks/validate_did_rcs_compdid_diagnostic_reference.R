# Equality-diagnostic parity against the authors' official compdid 0.1.0
# drdid_stationarity_test() implementation.
#
# Usage from the CauseKit repository root:
#   Rscript benchmarks/validate_did_rcs_compdid_diagnostic_reference.R PATH_TO_COMPDID_CHECKOUT

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("supply exactly one path to the pinned compdid checkout")
}

compdid_root <- normalizePath(args[[1L]], mustWork = TRUE)
expected_commit <- "894bd65a952c30f01a4e0005efba4cb335065eb7"
expected_att_core_blob <- "5d7ab30e0c3db8d90112b4d59d7e752ad6f459cf"
expected_dp_grooming_blob <- "9b00850566ce6fff9f2897f7852733da4ede7779"
expected_stationarity_blob <- "e06750e061948b5c02f08741f651b0e8fd4155f5"

git_value <- function(...) {
  value <- system2(
    "git",
    c("-C", shQuote(compdid_root), ...),
    stdout = TRUE,
    stderr = TRUE
  )
  if (!identical(attr(value, "status"), NULL) || length(value) != 1L) {
    stop("could not inspect the pinned compdid checkout")
  }
  value[[1L]]
}

reference_commit <- git_value("rev-parse", "HEAD")
if (!identical(reference_commit, expected_commit)) {
  stop(paste0("compdid checkout must be at recorded commit ", expected_commit))
}
source_blobs <- c(
  att_core = git_value("hash-object", "R/att-core.R"),
  dp_grooming = git_value("hash-object", "R/dp-grooming.R"),
  stationarity = git_value("hash-object", "R/stationarity-test.R")
)
expected_blobs <- c(
  att_core = expected_att_core_blob,
  dp_grooming = expected_dp_grooming_blob,
  stationarity = expected_stationarity_blob
)
if (!identical(source_blobs, expected_blobs)) {
  stop("pinned compdid source blobs do not match the recorded diagnostic sources")
}

description <- read.dcf(file.path(compdid_root, "DESCRIPTION"))
compdid_version <- unname(description[1L, "Version"])
if (!identical(compdid_version, "0.1.0")) {
  stop("pinned compdid checkout must declare version 0.1.0")
}

official <- new.env(parent = globalenv())
sys.source(file.path(compdid_root, "R", "stationarity-test.R"), envir = official)

data <- utils::read.csv(
  file.path("benchmarks", "did_rcs_compdid_diagnostic_input.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
if (length(unique(data$nonstationary_att)) != 1L ||
    length(unique(data$stationary_att)) != 1L) {
  stop("the fixed diagnostic input must contain one estimate for each path")
}
make_drdid <- function(att, influence, type) {
  fit <- list(
    ATT = as.numeric(att),
    att.inf.func = as.numeric(influence),
    argu = list(
      type = type,
      stabilized = TRUE,
      nuisance_method = "fixed_external_influence"
    )
  )
  class(fit) <- c("compdid_att", "drdid")
  fit
}
nonstationary <- make_drdid(
  unique(data$nonstationary_att),
  data$nonstationary_influence,
  "dr,nonstnr"
)
stationary <- make_drdid(
  unique(data$stationary_att),
  data$stationary_influence,
  "dr,stnr"
)
diagnostic <- official$drdid_stationarity_test(
  nonstationary,
  stationary,
  alpha = 0.05
)

influence_diff <- nonstationary$att.inf.func - stationary$att.inf.func
n <- nrow(data)
variance_of_estimate_hc0 <- mean(influence_diff^2) / n
if (!isTRUE(all.equal(
  diagnostic$statistic,
  diagnostic$estimate_diff^2 / variance_of_estimate_hc0,
  tolerance = 1e-14
))) {
  stop("official diagnostic statistic does not match its HC0 influence identity")
}

cat("reference_repository=https://github.com/pedrohcgs/comp_did.git\n")
cat("reference_commit=", reference_commit, "\n", sep = "")
cat("att_core_blob=", source_blobs[["att_core"]], "\n", sep = "")
cat("dp_grooming_blob=", source_blobs[["dp_grooming"]], "\n", sep = "")
cat("stationarity_test_blob=", source_blobs[["stationarity"]], "\n", sep = "")
cat("source_loading=git_pinned_official_R_sources\n")
cat("compdid_version=", compdid_version, "\n", sep = "")
cat("r_version=", paste(R.version$major, R.version$minor, sep = "."), "\n", sep = "")
cat("entry_point=drdid_stationarity_test\n")
cat("input_contract=fixed_aligned_causekit_estimates_and_influences\n")
cat("stabilized=true\n")
cat("n_observations=", n, "\n", sep = "")
cat("nonstationary_att=", sprintf("%.17g", nonstationary$ATT), "\n", sep = "")
cat("stationary_att=", sprintf("%.17g", stationary$ATT), "\n", sep = "")
cat("att_difference=", sprintf("%.17g", diagnostic$estimate_diff), "\n", sep = "")
cat("variance_of_estimate_hc0=", sprintf("%.17g", variance_of_estimate_hc0), "\n", sep = "")
cat("statistic_hc0=", sprintf("%.17g", diagnostic$statistic), "\n", sep = "")
cat("p_value_hc0=", sprintf("%.17g", diagnostic$p.value), "\n", sep = "")
cat("reject_0_05=", tolower(as.character(diagnostic$decision$reject[[1L]])), "\n", sep = "")
cat("hc0_to_causekit_hc1=W_HC0=W_HC1*n/(n-1)\n")
cat("influence_rows=", paste(data$row, collapse = ","), "\n", sep = "")
cat(
  "nonstationary_influence_values=",
  paste(sprintf("%.17g", nonstationary$att.inf.func), collapse = ","),
  "\n",
  sep = ""
)
cat(
  "stationary_influence_values=",
  paste(sprintf("%.17g", stationary$att.inf.func), collapse = ","),
  "\n",
  sep = ""
)
cat(
  "difference_influence_values=",
  paste(sprintf("%.17g", influence_diff), collapse = ","),
  "\n",
  sep = ""
)
cat("parity_status=reference_complete\n")
