# Changelog

All notable changes to `causalkit` are recorded here. The project follows
[Semantic Versioning](https://semver.org/) once a public contract is released. Alpha
versions may refine APIs, but breaking changes must still be documented explicitly.

## [0.5.0a1] - Unreleased

### Added

- `NearestNeighborMatch` and `NearestNeighborMatchResult` for supplied
  logit-propensity ATT, ATC, and bidirectional-imputation ATE point estimation.
- Replacement matching with inclusive automatic/numeric calipers, optional explicit
  caliper opt-out, intersection common support, deterministic fractional boundary ties,
  exact retained/excluded indices, target relabeling, effect weights, reuse counts, and
  weighted balance diagnostics.
- Hand-computed estimand identities, support/caliper/tie/reuse/balance tests,
  deterministic heterogeneous-effect recovery, refusal-path coverage, and a reproducible
  100,000-row sorted-scalar benchmark harness.

### Architecture

- The matcher consumes supplied propensity predictions and records their provenance; it
  does not copy nuisance estimators from `limiteddepkit`.
- Neighbor search sorts scalar arm scores and expands locally, avoiding a quadratic
  treated-by-control distance matrix.

### Known limitations

- This is a point-estimation alpha. `inference="none"` is the default and inferential
  result fields are undefined.
- The reserved Abadie–Imbens path refuses until reuse-aware conditional variance and the
  estimated-propensity first-step adjustment are implemented and validated. Ordinary
  bootstrap and clustered matching inference also refuse.
- Matching without replacement, alternative metrics, bias correction, external parity,
  OutputHub adaptation, and the full promotion benchmark matrix remain deferred.

## [0.4.0a1] - Unreleased

### Added

- `CrossFitter` with fresh per-fold propensity and arm-specific outcome factories,
  deterministic treatment-stratified folds, exact index preservation, and complete
  out-of-fold prediction records.
- Public nuisance estimator, propensity-result, and outcome-result protocols plus explicit
  prediction adapters for non-standard model APIs.
- `estimand="att"` and `estimand="atc"` for IPW and AIPW, with normalized target-
  population weighting and estimand-specific influence-function inference.
- A repository `HANDOVER.md` recording completed milestones, sibling-package reuse rules,
  performance constraints, validation gates, and the next development order.

## [0.3.0a1] - Unreleased

### Added

- `IPWATE` and `AIPWATE` for the observational-population ATE using externally supplied,
  preferably cross-fitted nuisance predictions.
- Strict propensity support checks, optional explicit clipping, effective-sample-size and
  overlap diagnostics, influence-function HC inference, and one-way clustered inference.
- Vectorized large-sample execution, exact score-identity tests, oracle recovery,
  clustered aggregation checks, and OutputHub adaptation.

### Architecture

- Nuisance estimation remains external: existing `limiteddepkit` binary/outcome models
  can generate predictions without being copied into `causalkit`. This package owns the
  causal score, estimand, diagnostics, and inference.

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
