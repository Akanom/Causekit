# Prepare the authors' hash-pinned Sequeira application data outside the repository.
#
# Usage from the CauseKit repository root:
#   Rscript benchmarks/prepare_did_rcs_composition_real_data.R PATH_TO_COMPDID_CHECKOUT
#
# An optional second argument overrides the output directory.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("supply the pinned compdid checkout and optionally an output directory")
}

compdid_root <- normalizePath(args[[1L]], mustWork = TRUE)
expected_commit <- "894bd65a952c30f01a4e0005efba4cb335065eb7"
expected_blob <- "6a6a8bbe9792bc6385849421a7fd0d76692cd79e"
expected_sha256 <- "37f113f1c706a3b35c325b996884c843beb9a4f773c3928d505ca51b285a7953"

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
source_blob <- git_value("hash-object", "data/sequeira_bribes.rda")
if (!identical(source_blob, expected_blob)) {
  stop("pinned compdid Sequeira data blob does not match the recorded source")
}

sha256_file <- function(path) {
  if (.Platform$OS.type == "windows") {
    raw <- system2("certutil", c("-hashfile", shQuote(path), "SHA256"), stdout = TRUE)
  } else {
    raw <- system2("sha256sum", shQuote(path), stdout = TRUE)
  }
  candidates <- gsub("[[:space:]]", "", raw)
  candidates <- tolower(candidates[nchar(candidates) == 64L & grepl("^[0-9a-fA-F]+$", candidates)])
  if (length(candidates) != 1L) {
    stop("could not compute a unique SHA-256 digest")
  }
  candidates[[1L]]
}

source_path <- file.path(compdid_root, "data", "sequeira_bribes.rda")
source_sha256 <- sha256_file(source_path)
if (!identical(source_sha256, expected_sha256)) {
  stop("pinned compdid Sequeira data SHA-256 does not match the recorded source")
}

if (length(args) == 2L) {
  output_dir <- args[[2L]]
} else {
  cache_root <- Sys.getenv("LOCALAPPDATA", unset = file.path(path.expand("~"), ".cache"))
  output_dir <- file.path(cache_root, "causekit", "parity", "did_rcs_composition_v1")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

source_environment <- new.env(parent = emptyenv())
load(source_path, envir = source_environment)
data <- source_environment$sequeira_bribes
data$lba_value[2783L] <- 0
data$monitor <- data$monitor - 1
data$psi <- 2 - data$psi
data$ltonnage <- log(data$tonnage + 1)

analysis_columns <- c(
  "bp", "lba", "lba_value", "lba_tonnage",
  "post_2008", "tariff_change_2008", "hc_4digits",
  "lvalue_tonnage", "ltonnage", "lvalue_shipment_metical", "tariff2007",
  "differentiated", "agri", "perishable", "dfs", "day_w_arrival",
  "monitor", "psi", "rsa", "term"
)
# These two paper-preparation fields are not analysis covariates, but they are
# included in the complete-case rule to reproduce the authors' exact 1,084 rows.
complete_case_columns <- c(analysis_columns, "clear_agent", "hc_group")
complete <- stats::complete.cases(data[, complete_case_columns])
analysis <- data[complete, analysis_columns, drop = FALSE]
analysis <- cbind(source_row = which(complete), analysis)
if (nrow(analysis) != 1084L || length(unique(analysis$hc_4digits)) != 131L) {
  stop("prepared Sequeira analysis dimensions do not match the pinned contract")
}

csv_path <- file.path(output_dir, "sequeira_bribes_analysis.csv")
old_options <- options(digits = 17L, scipen = 999L)
on.exit(options(old_options), add = TRUE)
csv_lines <- capture.output(
  utils::write.csv(analysis, row.names = FALSE, quote = TRUE, na = "")
)
connection <- file(csv_path, open = "wb")
writeChar(paste0(paste(csv_lines, collapse = "\n"), "\n"), connection, eos = NULL, useBytes = TRUE)
close(connection)
csv_sha256 <- sha256_file(csv_path)

manifest_path <- file.path(output_dir, "manifest.txt")
manifest_lines <- c(
  "schema=causekit_did_rcs_composition_real_data_v1",
  paste0("reference_commit=", reference_commit),
  paste0("source_blob=", source_blob),
  paste0("source_sha256=", source_sha256),
  paste0("analysis_csv_sha256=", csv_sha256),
  paste0("rows=", nrow(analysis)),
  paste0("clusters=", length(unique(analysis$hc_4digits))),
  "source_dataset=Sequeira_2016_tariff_liberalization_and_bribery",
  "preparation=SantAnna_Xu_2026_complete_case_application_sample"
)
connection <- file(manifest_path, open = "wb")
writeChar(paste0(paste(manifest_lines, collapse = "\n"), "\n"), connection, eos = NULL)
close(connection)

cat("analysis_csv=", csv_path, "\n", sep = "")
cat("manifest=", manifest_path, "\n", sep = "")
cat("analysis_csv_sha256=", csv_sha256, "\n", sep = "")
cat("rows=", nrow(analysis), "\n", sep = "")
cat("clusters=", length(unique(analysis$hc_4digits)), "\n", sep = "")
