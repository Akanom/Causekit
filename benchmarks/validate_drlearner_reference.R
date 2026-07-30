#!/usr/bin/env Rscript

# Independent base-R reconstruction of CauseKit's fixed honest DR evaluation moments.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript validate_drlearner_reference.R input.csv output.txt")
}

data <- read.csv(args[[1L]], check.names = FALSE)
required <- c("dr_score", "cate", "constant_effect", "group")
if (!all(required %in% names(data))) {
  stop("DR-learner parity input is missing required columns")
}

phi <- data$dr_score
s <- data$cate
constant_effect <- unique(data$constant_effect)
if (length(constant_effect) != 1L) {
  stop("constant_effect must be constant across parity rows")
}
n <- length(phi)
dr_loss <- mean((phi - s)^2)
constant_dr_loss <- mean((phi - constant_effect)^2)
dr_loss_gain <- 1 - dr_loss / constant_dr_loss

center <- mean(s)
calibration_design <- cbind(level = 1, heterogeneity = s - center)
calibration_beta <- solve(crossprod(calibration_design), crossprod(calibration_design, phi))
calibration_error <- as.vector(phi - calibration_design %*% calibration_beta)
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
group_design[cbind(seq_len(n), group)] <- 1
group_beta <- solve(crossprod(group_design), crossprod(group_design, phi))
group_error <- as.vector(phi - group_design %*% group_beta)
group_scores <- group_design * group_error
group_bread <- solve(crossprod(group_design))
group_covariance <- group_bread %*%
  (n / (n - ncol(group_design)) * crossprod(group_scores)) %*%
  group_bread

values <- c(
  honest_dr_loss = dr_loss,
  honest_constant_dr_loss = constant_dr_loss,
  dr_loss_gain = dr_loss_gain,
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
