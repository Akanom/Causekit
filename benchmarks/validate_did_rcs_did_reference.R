# Estimator-level repeated-cross-section parity against bcallaway11/did 2.5.0.
#
# Usage from the repository root:
#   Rscript benchmarks/validate_did_rcs_did_reference.R PATH_TO_DID_V2_5_CHECKOUT

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("supply exactly one path to the pinned did v2.5 checkout")
}

did_root <- normalizePath(args[[1]], mustWork = TRUE)
reference_commit <- system2(
  "git", c("-C", did_root, "rev-parse", "HEAD"), stdout = TRUE
)
expected_commit <- "c449b8ce72029855d2de94b377f131be0e53e53a"
if (length(reference_commit) != 1 || reference_commit[[1]] != expected_commit) {
  stop(paste0("did checkout must be at recorded commit ", expected_commit))
}
if (!requireNamespace("did", quietly = TRUE)) {
  stop("install the pinned did 2.5.0 checkout before running this comparator")
}
if (as.character(utils::packageVersion("did")) != "2.5.0") {
  stop("installed did package must be version 2.5.0")
}

input_path <- file.path("benchmarks", "did_rcs_parity_input.csv")
source_data <- utils::read.csv(input_path, check.names = FALSE)
# did 2.5.0 refuses control groups with fewer than five observations per
# period. Repeat the deterministic support three times so every group clears
# that package-specific guard while preserving every cell mean and estimand.
data <- source_data[rep(seq_len(nrow(source_data)), each = 3), ]
data$observation <- paste0(data$observation, "_r", rep(seq_len(3), nrow(source_data)))
n <- nrow(data)

emit_fit <- function(control_group) {
  r_control <- if (control_group == "never_treated") "nevertreated" else "notyettreated"
  fitted <- did::att_gt(
    yname = "outcome",
    tname = "time",
    gname = "treatment_time",
    xformla = NULL,
    data = data,
    panel = FALSE,
    control_group = r_control,
    anticipation = 0,
    base_period = "varying",
    est_method = "reg",
    bstrap = FALSE,
    cband = FALSE,
    print_details = FALSE
  )
  keep <- which(fitted$t >= fitted$group)
  for (position in keep) {
    adjusted_se <- fitted$se[[position]] * sqrt(n / (n - 1))
    cat(
      "group_time=", control_group, ",",
      sprintf("%.17g", fitted$group[[position]]), ",",
      sprintf("%.17g", fitted$t[[position]]), ",",
      sprintf("%.17g", fitted$att[[position]]), ",",
      sprintf("%.17g", fitted$se[[position]]), ",",
      sprintf("%.17g", adjusted_se), "\n",
      sep = ""
    )
  }

  dynamic <- did::aggte(fitted, type = "dynamic", bstrap = FALSE, cband = FALSE)
  keep_dynamic <- which(dynamic$egt >= 0)
  for (position in keep_dynamic) {
    adjusted_se <- dynamic$se.egt[[position]] * sqrt(n / (n - 1))
    cat(
      "event_time=", control_group, ",",
      sprintf("%.17g", dynamic$egt[[position]]), ",",
      sprintf("%.17g", dynamic$att.egt[[position]]), ",",
      sprintf("%.17g", dynamic$se.egt[[position]]), ",",
      sprintf("%.17g", adjusted_se), "\n",
      sep = ""
    )
  }
  adjusted_overall_se <- dynamic$overall.se * sqrt(n / (n - 1))
  cat(
    "esavg=", control_group, ",",
    sprintf("%.17g", dynamic$overall.att), ",",
    sprintf("%.17g", dynamic$overall.se), ",",
    sprintf("%.17g", adjusted_overall_se), "\n",
    sep = ""
  )

  calendar <- did::aggte(fitted, type = "calendar", bstrap = FALSE, cband = FALSE)
  for (position in seq_along(calendar$egt)) {
    adjusted_se <- calendar$se.egt[[position]] * sqrt(n / (n - 1))
    cat(
      "calendar_time=", control_group, ",",
      sprintf("%.17g", calendar$egt[[position]]), ",",
      sprintf("%.17g", calendar$att.egt[[position]]), ",",
      sprintf("%.17g", calendar$se.egt[[position]]), ",",
      sprintf("%.17g", adjusted_se), "\n",
      sep = ""
    )
  }
}

cat("reference_commit=", reference_commit[[1]], "\n", sep = "")
cat("did_version=", as.character(utils::packageVersion("did")), "\n", sep = "")
cat("r_version=", paste(R.version$major, R.version$minor, sep = "."), "\n", sep = "")
cat("source_observations=", nrow(source_data), "\n", sep = "")
cat("deterministic_replications=3\n")
cat("n_observations=", n, "\n", sep = "")
emit_fit("never_treated")
emit_fit("not_yet_treated")
cat("parity_status=reference_complete\n")
