# Independent base-R reconstruction of CauseKit's fixed-OOF covariate RCS score.
#
# This is deliberately not a DRDID package wrapper. It freezes the normalized
# Sant'Anna-Zhao (2020) equation (3.4) score, ratio influence contributions, and
# CauseKit HC1 mapping without fitting or selecting a nuisance model in R.

x <- c(0, .2, .6, 1, .1, .4, .7, .9, 0, .3, .5, .8, .2, .4, .6, 1)
d <- c(rep(1, 8), rep(0, 8))
post <- rep(c(rep(0, 4), rep(1, 4)), 2)
m0_pre <- 1 + x
m0_post <- 2 + 2 * x
m1_pre <- 3 + .5 * x
m1_post <- 6 + 2.5 * x
y <- ifelse(
  d == 1,
  ifelse(post == 1, m1_post, m1_pre),
  ifelse(post == 1, m0_post, m0_pre)
)
p <- .25 + .5 * x
m0 <- ifelse(post == 1, m0_post, m0_pre)

ratio_component <- function(weight, value) {
  denominator <- mean(weight)
  eta <- weight * value / denominator
  component <- mean(eta)
  influence <- eta - weight * component / denominator
  list(component = component, influence = influence)
}

inputs <- list(
  list(d * post, y - m0, 1),
  list(d * (1 - post), y - m0, -1),
  list(p / (1 - p) * (1 - d) * post, y - m0, -1),
  list(p / (1 - p) * (1 - d) * (1 - post), y - m0, 1),
  list(d, m1_post - m0_post, 1),
  list(d * post, m1_post - m0_post, -1),
  list(d, m1_pre - m0_pre, -1),
  list(d * (1 - post), m1_pre - m0_pre, 1)
)

estimate <- 0
influence <- numeric(length(y))
for (input in inputs) {
  value <- ratio_component(input[[1]], input[[2]])
  estimate <- estimate + input[[3]] * value$component
  influence <- influence + input[[3]] * value$influence
}
standard_error_hc1 <- sqrt(sum(influence^2) / (length(y) * (length(y) - 1)))
expected_influence <- c(-.975, -.575, .225, 1.025, -.775, -.175, .425, .825, rep(0, 8))
passed <- abs(estimate - 2.4875) <= 1e-12 &&
  max(abs(influence - expected_influence)) <= 1e-12 &&
  abs(sum(influence)) <= 1e-12

cat("contract=fixed_oof_santanna_zhao_rc_efficient_hc1\n")
cat(sprintf("estimate=%.17g\n", estimate))
cat(sprintf("standard_error_hc1=%.17g\n", standard_error_hc1))
cat("influence=", paste(sprintf("%.17g", influence), collapse = ","), "\n", sep = "")
cat("parity_status=", ifelse(passed, "pass", "fail"), "\n", sep = "")
if (!passed) {
  quit(status = 9)
}
