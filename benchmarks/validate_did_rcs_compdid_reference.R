# Point-estimate and influence-function parity against the authors' official
# compdid 0.1.0 nonstationary repeated-cross-section score.
#
# The exact exported R implementation is loaded from a commit-pinned checkout;
# compiling unrelated C++ nuisance learners is intentionally unnecessary.
#
# Usage from the CauseKit repository root:
#   Rscript benchmarks/validate_did_rcs_compdid_reference.R PATH_TO_COMPDID_CHECKOUT

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("supply exactly one path to the pinned compdid checkout")
}

compdid_root <- normalizePath(args[[1L]], mustWork = TRUE)
expected_commit <- "894bd65a952c30f01a4e0005efba4cb335065eb7"
expected_att_core_blob <- "5d7ab30e0c3db8d90112b4d59d7e752ad6f459cf"
expected_dp_grooming_blob <- "9b00850566ce6fff9f2897f7852733da4ede7779"

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
att_core_blob <- git_value("hash-object", "R/att-core.R")
dp_grooming_blob <- git_value("hash-object", "R/dp-grooming.R")
if (!identical(att_core_blob, expected_att_core_blob)) {
  stop("pinned compdid R/att-core.R blob does not match the recorded source")
}
if (!identical(dp_grooming_blob, expected_dp_grooming_blob)) {
  stop("pinned compdid R/dp-grooming.R blob does not match the recorded source")
}

description <- read.dcf(file.path(compdid_root, "DESCRIPTION"))
compdid_version <- unname(description[1L, "Version"])
if (!identical(compdid_version, "0.1.0")) {
  stop("pinned compdid checkout must declare version 0.1.0")
}

official <- new.env(parent = globalenv())
sys.source(file.path(compdid_root, "R", "dp-grooming.R"), envir = official)
sys.source(file.path(compdid_root, "R", "att-core.R"), envir = official)
if (!is.function(official$drdid_nonstationary)) {
  stop("the pinned source does not expose drdid_nonstationary")
}

data <- utils::read.csv(
  file.path("benchmarks", "did_rcs_compdid_parity_input.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

# CauseKit persists probabilities as (00, 01, 10, 11), whereas compdid's
# public second-stage contract consumes (11, 10, 01, 00). CauseKit only needs
# (m00, m01, m10): m11 cancels between compdid's target residual and regression
# contrast, but it is supplied because compdid correctly requires an n x 4 fit.
fit.ps <- as.matrix(data[c("p11", "p10", "p01", "p00")])
fit.or <- as.matrix(data[c("m11", "m10", "m01", "m00")])
fitted <- official$drdid_nonstationary(
  y = data$outcome,
  d = data$d,
  post = data$post,
  fit.ps = fit.ps,
  fit.or = fit.or,
  stabilized = TRUE,
  boot = FALSE,
  inffunc = TRUE
)

influence <- as.numeric(fitted$att.inf.func)
if (length(influence) != nrow(data) || any(!is.finite(influence))) {
  stop("compdid returned a malformed influence function")
}
if (abs(mean(influence)) > 1e-12) {
  stop("compdid influence function is not centered on the parity fixture")
}

cat("reference_repository=https://github.com/pedrohcgs/comp_did.git\n")
cat("reference_commit=", reference_commit, "\n", sep = "")
cat("att_core_blob=", att_core_blob, "\n", sep = "")
cat("dp_grooming_blob=", dp_grooming_blob, "\n", sep = "")
cat("source_loading=git_pinned_official_R_sources\n")
cat("compdid_version=", compdid_version, "\n", sep = "")
cat("r_version=", paste(R.version$major, R.version$minor, sep = "."), "\n", sep = "")
cat("entry_point=drdid_nonstationary\n")
cat("stabilized=true\n")
cat("bootstrap=false\n")
cat("influence_requested=true\n")
cat("causekit_probability_order=00,01,10,11\n")
cat("compdid_probability_order=11,10,01,00\n")
cat("causekit_outcome_order=00,01,10\n")
cat("compdid_outcome_order=11,10,01,00\n")
cat("treated_post_outcome_role=algebraically_cancels\n")
cat("n_observations=", nrow(data), "\n", sep = "")
cat("att=", sprintf("%.17g", fitted$ATT), "\n", sep = "")
cat("standard_error=", sprintf("%.17g", fitted$se), "\n", sep = "")
cat("influence_mean=", sprintf("%.17g", mean(influence)), "\n", sep = "")
cat("influence_rows=", paste(data$row, collapse = ","), "\n", sep = "")
cat(
  "influence_values=",
  paste(sprintf("%.17g", influence), collapse = ","),
  "\n",
  sep = ""
)
cat("parity_status=reference_complete\n")
