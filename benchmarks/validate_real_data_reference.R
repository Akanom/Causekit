# Real-data parity for CauseKit's released estimator families.
#
# Usage:
#   Rscript benchmarks/validate_real_data_reference.R \
#     PATH_TO_PREPARED_CSVS PATH_TO_EDID_CHECKOUT PATH_TO_MATCHING_CHECKOUT
#
# Prepare PATH_TO_PREPARED_CSVS with:
#   python benchmarks/prepare_real_data.py --download

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("supply prepared-data, edid-checkout, and Matching-checkout paths")
}

data_root <- normalizePath(args[[1]], mustWork = TRUE)
edid_root <- normalizePath(args[[2]], mustWork = TRUE)
matching_root <- normalizePath(args[[3]], mustWork = TRUE)

git_commit <- function(path) {
  value <- system2("git", c("-C", path, "rev-parse", "HEAD"), stdout = TRUE)
  if (length(value) != 1) stop("could not resolve comparator commit")
  value[[1]]
}

edid_commit <- git_commit(edid_root)
matching_commit <- git_commit(matching_root)
expected_edid <- "f55a4a4aba14f0826f59ad7aa4af3bafaeba529b"
expected_matching <- "1208eaa7bfa888b1fc903481dddfb8c0dffa40d5"
if (edid_commit != expected_edid) stop("edid checkout is not at the pinned commit")
if (matching_commit != expected_matching) stop("Matching checkout is not at the pinned commit")

emit <- function(name, value) {
  if (length(value) == 1 && is.numeric(value)) {
    value <- sprintf("%.17g", value)
  }
  cat(name, "=", paste(value, collapse = ","), "\n", sep = "")
}

hc1 <- function(design, residuals) {
  n <- nrow(design)
  k <- ncol(design)
  bread <- solve(crossprod(design))
  scores <- design * as.numeric(residuals)
  n / (n - k) * bread %*% crossprod(scores) %*% bread
}

influence_se <- function(influence) {
  n <- length(influence)
  sqrt(sum(influence^2) / (n * (n - 1)))
}

emit("artifact", "causekit_real_data_r_parity")
emit("r_version", paste(R.version$major, R.version$minor, sep = "."))
emit("edid_commit", edid_commit)
emit("matching_commit", matching_commit)

# IV2SLS: Stata's 1980 Census housing example, with HC1 covariance.
hsng <- read.csv(file.path(data_root, "hsng.csv"), check.names = FALSE)
region <- model.matrix(~ factor(region), data = hsng)[, -1, drop = FALSE]
colnames(region) <- paste0("region", 2:4)
x_iv <- cbind(const = 1, pcturban = hsng$pcturban, hsngval = hsng$hsngval)
z_iv <- cbind(const = 1, pcturban = hsng$pcturban, faminc = hsng$faminc, region)
z_cross_inverse <- solve(crossprod(z_iv))
x_projected <- z_iv %*% z_cross_inverse %*% crossprod(z_iv, x_iv)
iv_bread <- solve(crossprod(x_projected, x_iv))
iv_beta <- iv_bread %*% crossprod(x_projected, hsng$rent)
iv_residual <- hsng$rent - x_iv %*% iv_beta
iv_scores <- x_projected * as.numeric(iv_residual)
iv_covariance <- nrow(x_iv) / (nrow(x_iv) - ncol(x_iv)) *
  iv_bread %*% crossprod(iv_scores) %*% iv_bread
for (name in rownames(iv_beta)) {
  emit(paste0("iv_", name, "_estimate"), iv_beta[name, 1])
  emit(paste0("iv_", name, "_standard_error"), sqrt(iv_covariance[name, name]))
}

# RandomizedATE: NSW experiment, raw and Lin-adjusted HC1 contracts.
nsw <- read.csv(file.path(data_root, "nsw_mixtape.csv"), check.names = FALSE)
treatment <- nsw$treat
raw_design <- cbind(const = 1, ate = treatment)
raw_beta <- solve(crossprod(raw_design), crossprod(raw_design, nsw$re78))
raw_residual <- nsw$re78 - raw_design %*% raw_beta
raw_covariance <- hc1(raw_design, raw_residual)
emit("randomized_raw_ate_estimate", raw_beta["ate", 1])
emit("randomized_raw_ate_standard_error", sqrt(raw_covariance["ate", "ate"]))

randomized_covariates <- as.matrix(nsw[, c(
  "age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"
)])
centered <- sweep(randomized_covariates, 2, colMeans(randomized_covariates), "-")
lin_design <- cbind(const = 1, ate = treatment, centered, centered * treatment)
lin_beta <- solve(crossprod(lin_design), crossprod(lin_design, nsw$re78))
lin_residual <- nsw$re78 - lin_design %*% lin_beta
lin_covariance <- hc1(lin_design, lin_residual)
emit("randomized_lin_ate_estimate", lin_beta["ate", 1])
emit("randomized_lin_ate_standard_error", sqrt(lin_covariance["ate", "ate"]))

# Supplied-nuisance IPW/AIPW: Cattaneo data with common Logit/OLS nuisances.
cattaneo <- read.csv(file.path(data_root, "cattaneo2.csv"), check.names = FALSE)
observational_design <- cbind(
  const = 1,
  as.matrix(cattaneo[, c("mmarried", "mage", "medu", "fbaby")])
)
assigned <- cattaneo$mbsmoke
outcome <- cattaneo$bweight
propensity_fit <- glm.fit(
  x = observational_design,
  y = assigned,
  family = binomial()
)
propensity <- propensity_fit$fitted.values
mu0_beta <- solve(
  crossprod(observational_design[assigned == 0, , drop = FALSE]),
  crossprod(observational_design[assigned == 0, , drop = FALSE], outcome[assigned == 0])
)
mu1_beta <- solve(
  crossprod(observational_design[assigned == 1, , drop = FALSE]),
  crossprod(observational_design[assigned == 1, , drop = FALSE], outcome[assigned == 1])
)
mu0 <- as.numeric(observational_design %*% mu0_beta)
mu1 <- as.numeric(observational_design %*% mu1_beta)

observational_effect <- function(estimator, estimand) {
  if (estimator == "ipw" && estimand == "ate") {
    score <- assigned * outcome / propensity -
      (1 - assigned) * outcome / (1 - propensity)
    estimate <- mean(score)
    influence <- score - estimate
  } else if (estimator == "ipw" && estimand == "att") {
    wt <- assigned
    wc <- (1 - assigned) * propensity / (1 - propensity)
    mean_t <- sum(wt * outcome) / sum(wt)
    mean_c <- sum(wc * outcome) / sum(wc)
    estimate <- mean_t - mean_c
    influence <- wt * (outcome - mean_t) / mean(wt) -
      wc * (outcome - mean_c) / mean(wc)
  } else if (estimator == "ipw" && estimand == "atc") {
    wt <- assigned * (1 - propensity) / propensity
    wc <- 1 - assigned
    mean_t <- sum(wt * outcome) / sum(wt)
    mean_c <- sum(wc * outcome) / sum(wc)
    estimate <- mean_t - mean_c
    influence <- wt * (outcome - mean_t) / mean(wt) -
      wc * (outcome - mean_c) / mean(wc)
  } else if (estimator == "aipw" && estimand == "ate") {
    score <- mu1 - mu0 + assigned * (outcome - mu1) / propensity -
      (1 - assigned) * (outcome - mu0) / (1 - propensity)
    estimate <- mean(score)
    influence <- score - estimate
  } else if (estimator == "aipw" && estimand == "att") {
    numerator <- assigned * (outcome - mu0) -
      (1 - assigned) * propensity / (1 - propensity) * (outcome - mu0)
    denominator <- assigned
    estimate <- mean(numerator) / mean(denominator)
    influence <- (numerator - estimate * denominator) / mean(denominator)
  } else {
    numerator <- assigned * (1 - propensity) / propensity * (outcome - mu1) -
      (1 - assigned) * (outcome - mu1)
    denominator <- 1 - assigned
    estimate <- mean(numerator) / mean(denominator)
    influence <- (numerator - estimate * denominator) / mean(denominator)
  }
  c(estimate = estimate, standard_error = influence_se(influence))
}

for (estimator in c("ipw", "aipw")) {
  for (estimand in c("ate", "att", "atc")) {
    result <- observational_effect(estimator, estimand)
    emit(paste0(estimator, "_", estimand, "_estimate"), result[["estimate"]])
    emit(
      paste0(estimator, "_", estimand, "_standard_error"),
      result[["standard_error"]]
    )
  }
}

# Matching point estimates on a deterministic real-data subset.
suppressPackageStartupMessages(library(MASS))
source(file.path(matching_root, "R", "Matching.R"))
selected <- sort(c(which(assigned == 1)[1:200], which(assigned == 0)[1:400]))
matching_y <- matrix(outcome[selected], ncol = 1)
matching_t <- matrix(assigned[selected], ncol = 1)
matching_score <- matrix(propensity[selected], ncol = 1)
unit_weights <- matrix(1, nrow = length(selected), ncol = 1)
all_codes <- c(att = 0, ate = 1, atc = 2)
for (estimand in names(all_codes)) {
  result <- Rmatch(
    Y = matching_y,
    Tr = matching_t,
    X = matching_score,
    Z = matching_score,
    V = unit_weights,
    All = all_codes[[estimand]],
    M = 1,
    BiasAdj = 0,
    Weight = 1,
    Weight.matrix = 1,
    Var.calc = 0,
    weight = unit_weights,
    SAMPLE = 0,
    ccc = sqrt(.Machine$double.eps),
    cdd = .Machine$double.eps,
    ecaliper = NULL,
    restrict = NULL
  )
  emit(paste0("matching_", estimand, "_estimate"), result$est)
}
emit("matching_real_subset_rows", length(selected))

# Conventional group-time DiD on the hospital panel.
hospdd <- read.csv(file.path(data_root, "hospdd_panel.csv"), check.names = FALSE)
hospitals <- sort(unique(hospdd$hospital))
periods <- sort(unique(hospdd$month))
outcome_matrix <- xtabs(outcome ~ hospital + month, data = hospdd)
cohort <- tapply(hospdd$treatment_time, hospdd$hospital, unique)[as.character(hospitals)]
treated_group <- cohort == 4
control_group <- cohort == 0
base_period <- 3
post_periods <- periods[periods >= 4]
did_att <- numeric(length(post_periods))
did_influence <- matrix(0, nrow = length(hospitals), ncol = length(post_periods))
for (position in seq_along(post_periods)) {
  period <- post_periods[[position]]
  change <- outcome_matrix[, as.character(period)] -
    outcome_matrix[, as.character(base_period)]
  treated_mean <- mean(change[treated_group])
  control_mean <- mean(change[control_group])
  did_att[[position]] <- treated_mean - control_mean
  did_influence[, position] <-
    treated_group / mean(treated_group) * (change - treated_mean) -
    control_group / mean(control_group) * (change - control_mean)
}
did_overall_influence <- rowMeans(did_influence)
emit("did_group_time_att", did_att)
emit("did_group_time_standard_error", apply(did_influence, 2, influence_se))
emit("did_esavg_estimate", mean(did_att))
emit("did_esavg_standard_error", influence_se(did_overall_influence))

# Efficient DiD from the pinned public edid equations, without invoking bootstrap code.
source(file.path(edid_root, "R", "get_ytilde.R"))
source(file.path(edid_root, "R", "get_influence_func.R"))
edid_data <- data.frame(id = hospitals, g = ifelse(cohort == 0, 0, cohort))
for (period in periods) {
  edid_data[[paste0("y", period)]] <- outcome_matrix[, as.character(period)]
}
edid_data$g0 <- as.integer(edid_data$g == 0)
edid_data$g4 <- as.integer(edid_data$g == 4)
ytilde <- get_ytilde(edid_data)
influence <- get_influence_func(edid_data, ytilde)
efficient_att <- efficient_se <- numeric(length(post_periods))
efficient_influence <- matrix(0, nrow = length(hospitals), ncol = length(post_periods))
for (position in seq_along(post_periods)) {
  prefix <- paste0("4_", post_periods[[position]], "_")
  selected_names <- names(ytilde)[startsWith(names(ytilde), prefix)]
  candidate_att <- vapply(ytilde[selected_names], mean, numeric(1))
  candidate_influence <- matrix(
    unlist(influence[selected_names]),
    nrow = length(selected_names),
    byrow = TRUE
  )
  covariance <- cov(t(candidate_influence))
  weights <- as.vector(solve(covariance, rep(1, nrow(covariance))))
  weights <- weights / sum(weights)
  efficient_influence[, position] <- as.vector(weights %*% candidate_influence)
  efficient_att[[position]] <- sum(weights * candidate_att)
  efficient_se[[position]] <- influence_se(efficient_influence[, position])
}
efficient_overall_influence <- rowMeans(efficient_influence)
emit("edid_group_time_att", efficient_att)
emit("edid_group_time_standard_error", efficient_se)
emit("edid_esavg_estimate", mean(efficient_att))
emit("edid_esavg_standard_error", influence_se(efficient_overall_influence))
emit("parity_status", "reference_completed")
