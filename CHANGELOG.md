# Changelog

All notable changes to `causalkit` are recorded here. The project follows
[Semantic Versioning](https://semver.org/) once a public contract is released. Alpha
versions may refine APIs, but breaking changes must still be documented explicitly.

## [0.2.0a1] - Unreleased

### Added

- `RandomizedATE` for two-arm experiments, with an exact difference-in-means path and
  fully interacted, mean-centered Lin regression adjustment.
- HC1 and one-way CR1 inference, strict binary-assignment and alignment checks, arm
  counts, covariate-balance diagnostics, explicit design assumptions, and OutputHub
  adaptation.
- Analytical, deterministic simulation, refusal-path, and Statsmodels parity tests for
  randomized-experiment estimation and covariance.

### Known limitations

- Supports individual-level two-arm assignment only. Blocking/stratification weights,
  unequal assignment probabilities, cluster-level estimands, randomization inference,
  repeated outcomes, attrition correction, and multi-arm experiments are not yet covered.

## [0.1.0a1] - Unreleased

### Added

- Initial `IV2SLS` public estimator with an excluded-instrument API.
- Support for multiple endogenous regressors, exogenous controls, and explicit intercept
  handling.
- Homoskedastic (`"unadjusted"`), HC1 (`"robust"`), and one-way cluster-robust CR1
  (`"clustered"`) covariance estimators.
- Labelled result fields for parameters, covariance, standard errors, test statistics,
  p-values, residuals, fitted values, observation counts, residual degrees of freedom,
  and covariance metadata.
- Classical and covariance-aware first-stage diagnostics, including partial R-squared and
  joint excluded-instrument tests.
- Sargan overidentification diagnostics for overidentified homoskedastic models only.
- Strict validation for missing and non-finite data, pandas index alignment, schema,
  dimensions, rank, and identification conditions, with an explicit joint complete-case
  `missing="drop"` option.
- Confidence intervals, prediction, linear combinations, Wald tests, summary frames,
  Markdown rendering, and optional OutputHub adaptation following compatible
  `systemgmmkit` and `limiteddepkit` result conventions.
- Methodology, scope, architecture, validation, migration, contribution, security, and
  citation documentation.

### Migration

- Established `causalkit.IV2SLS` as the destination for ordinary linear 2SLS workflows
  formerly represented by the out-of-scope `limiteddepkit.TreatmentEffect` snapshot.
- Replaced the legacy full-instrument-matrix convention with an explicit
  excluded-instrument contract. This is not a drop-in API rename.
- Implemented the package from public econometric definitions and new package contracts;
  no private implementation code was copied.

### Known limitations

- No formula API, automatic categorical encoding, imputation, or silent row dropping.
- No weak-IV-robust confidence sets or complete weak-instrument testing suite.
- No heteroskedasticity-robust overidentification test; Sargan is deliberately restricted
  to homoskedastic inference.
- No multiway clustering, few-cluster correction beyond CR1, sampling weights, or panel
  IV estimator in this release.
- IPW/AIPW, matching, difference-in-differences and
  event studies, regression discontinuity, and panel IV remain roadmap items.
- DADPLM and BDCPM are outside the current scope.

The release date will be assigned when `0.1.0a1` is published. Entries above describe the
target release surface and are not a claim that a particular checkout has passed the full
verification gate.
