#!/usr/bin/env Rscript

# Independent base-R reconstruction of CauseKit's fixed honest evaluation moments.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript validate_rlearner_reference.R input.csv output.txt")
}

data <- read.csv(args[[1L]], check.names = FALSE)
required <- c("outcome_residual", "treatment_residual", "cate", "constant_effect", "group")
if (!all(required %in% names(data))) {
  stop("R-learner parity input is missing required columns")
}

u <- data$outcome_residual
v <- data$treatment_residual
s <- data$cate
constant_effect <- unique(data$constant_effect)
if (length(constant_effect) != 1L) {
  stop("constant_effect must be constant across parity rows")
}
n <- length(u)
r_loss <- mean((u - v * s)^2)
constant_r_loss <- mean((u - v * constant_effect)^2)
r_loss_gain <- 1 - r_loss / constant_r_loss

center <- sum(v^2 * s) / sum(v^2)
calibration_design <- cbind(level = v, heterogeneity = v * (s - center))
calibration_beta <- solve(crossprod(calibration_design), crossprod(calibration_design, u))
calibration_error <- as.vector(u - calibration_design %*% calibration_beta)
calibration_scores <- calibration_design * calibration_error
calibration_bread <- solve(crossprod(calibration_design))
calibration_covariance <- calibration_bread %*%
  (n / (n - ncol(calibration_design)) * crossprod(calibration_scores)) %*%
  calibration_bread

group <- as.integer(data$group)
groups <- sort(unique(group))
if (!identical(groups, seq_along(groups))) {
  stop("group labels must be consecutive positive integers")
}
group_design <- matrix(0, nrow = n, ncol = length(groups))
group_design[cbind(seq_len(n), group)] <- v
group_beta <- solve(crossprod(group_design), crossprod(group_design, u))
group_error <- as.vector(u - group_design %*% group_beta)
group_scores <- group_design * group_error
group_bread <- solve(crossprod(group_design))
group_covariance <- group_bread %*%
  (n / (n - ncol(group_design)) * crossprod(group_scores)) %*%
  group_bread

values <- c(
  honest_r_loss = r_loss,
  honest_constant_r_loss = constant_r_loss,
  r_loss_gain = r_loss_gain,
  calibration_center = center,
  calibration_level = calibration_beta[[1L]],
  calibration_heterogeneity = calibration_beta[[2L]],
  calibration_level_variance = calibration_covariance[1L, 1L],
  calibration_heterogeneity_variance = calibration_covariance[2L, 2L],
  calibration_covariance = calibration_covariance[1L, 2L]
)
for (position in seq_along(groups)) {
  values[[paste0("group_", position, "_effect")]] <- group_beta[[position]]
  values[[paste0("group_", position, "_variance")]] <- group_covariance[position, position]
}

lines <- paste(names(values), sprintf("%.17g", values), sep = "=")
writeLines(c("parity_status=pass", paste0("r_version=", R.version.string), lines), args[[2L]])
