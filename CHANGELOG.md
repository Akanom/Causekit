# Changelog

All notable changes to `causalkit` are recorded here. The project follows
[Semantic Versioning](https://semver.org/) once a public contract is released. Alpha
versions may refine APIs, but breaking changes must still be documented explicitly.

## [0.6.0a4] - Unreleased

### Added

- Public `FittedPropensityMLEProtocol` and `inference="abadie_imbens_estimated"` for
  ATT, ATC, and ATE matching on a validated full-sample unpenalized Logit MLE.
- Abadie–Imbens first-step covariance, target-derivative, and Fisher-information
  corrections with separately reported known-score and adjustment components.
- Hand-computed three-estimand contracts, malformed-model and unsupported-design
  refusals, independent `statsmodels.Logit` parity, direct `limiteddepkit.BinaryLogitResult`
  interoperability verification, a seeded coverage smoke, and a 100,000-row benchmark.
- A manual Stata `teffects psmatch` parity harness for the estimated-score contract.

### Architecture

- CausalKit consumes the fitted nuisance result and never imports or duplicates the
  binary Logit estimator owned by `limiteddepkit`.
- The estimated-score path is intentionally narrower than `CrossFitter`: it validates
  convergence, sample/feature alignment, Logit predictions, likelihood stationarity, and
  nonsingular normalized information. Cross-fitted, penalized, probit, and generic scores
  retain `inference="none"`.
- Scalar local moments use sorted searches, and ATT/ATC target derivatives use `cKDTree`;
  no treated-by-control or covariate pairwise distance matrix is allocated.

### Known limitations

- The estimated-score Stata harness requires a manual Stata run before parity can be
  recorded. R `Matching` treats a supplied score as fixed and is non-comparable for the
  fitted-Logit first-step correction.
- Support/caliper selection, expanded ties, clustered/paired/survey uncertainty, bias
  correction, and arbitrary machine-learning first steps remain unsupported analytically.

## [0.6.0a3] - Unreleased

### Added

- Fixed/known-score Abadie-Imbens analytical variance for scalar nearest-neighbor ATT,
  ATC, and ATE matching with replacement, including same-arm conditional-variance
  matching, comparison-reuse adjustments, normal inference, and confidence intervals.
- Machine-readable `propensity_score_status=` and `variance_neighbors=` contracts plus
  result-level normalized variance, sampling variance, conditional variances, and
  conditional/effect variance components.
- Hand-computed ATT/ATC/ATE variance identities, a deterministic coverage smoke test,
  expanded refusal coverage, OutputHub model/design-table adaptation, and a 100,000-row
  inference benchmark scenario.
- Pinned R `Matching` 4.10-15 reference parity for estimates and standard errors, plus a
  Stata/MP 17 `teffects nnmatch` parity pass for all three estimands with its reviewed
  machine-readable output retained in the repository.

### Architecture

- The analytical path is intentionally limited to a declared fixed score with no
  caliper or common-support selection and no expanded cross-arm or same-arm boundary
  ties. The default remains `inference="none"`.
- A provenance string is audit metadata, not proof that a score is fixed. Estimated and
  cross-fitted scores continue to refuse analytical inference until a supported
  first-step-specific variance contract is implemented.
- Ordinary bootstrap and generic clustered covariance remain prohibited for fixed-
  neighbor matching.

### Known limitations

- Estimated-propensity adjustment, matching after target-changing support/caliper rules,
  bias correction, and clustered/paired/survey uncertainty are not implemented.
- The fixed-score fixture has Python, R, and Stata parity, but estimated-propensity
  first-step uncertainty and broader publication-scale sensitivity evidence remain open.

## [0.6.0a2] - Unreleased

### Added

- Reusable `CrossFitter.fit_predict_class_probabilities` and
  `CrossFitter.fit_predict_tasks` operations for multiclass probabilities and arbitrary
  masked scalar nuisance regressions on one deterministic fold plan.
- A covariate-adjusted `EfficientDiD` path that obtains cohort probabilities,
  group-specific outcome-change regressions, and residual-product conditional
  covariances through public `CrossFitter` factories.
- Observation-specific Chen-Sant'Anna-Xie equation (3.12) covariance systems and
  efficient weights, with aligned fold, probability, candidate-score, and conditional-
  weight audit records.
- Reproducible entity-level or declared-cluster Rademacher multiplier-bootstrap max-t
  simultaneous event-study bands for both conventional and efficient DiD.
- Exact nuisance recovery, multi-moment conditional-covariance, singular-refusal,
  hand-reconstructed robust/cluster band, and seeded coverage-smoke tests.

### Architecture

- `EfficientDiD` still owns no regression or classification model. Fresh nuisance models
  come from user-supplied factories, and every reported nuisance prediction is aligned
  and out of fold.
- Estimated cohort probabilities below the declared floor are refused, not clipped.
  Singular observation-specific covariance systems are refused without a ridge,
  eigenvalue repair, or pseudoinverse.
- Pointwise influence-function standard errors remain in the main result tables;
  simultaneous intervals and their realized critical value are exposed separately.

### Known limitations

- The covariate path requires a balanced short panel and numeric, entity-constant
  covariates. Repeated cross-sections and sampling weights are not implemented.
- The efficiency claim is conditional on PT-All and the overlap, consistency,
  weighting, and nuisance-rate conditions in Assumption C.1 of Chen, Sant'Anna, and Xie.
- Broader covariate parity, pre-trend/Hausman diagnostics, and publication-scale Monte
  Carlo evidence remain promotion work.

## [0.6.0a1] - Unreleased

### Added

- `DifferenceInDifferences` for conventional no-covariate cohort-time effects using
  never-treated or not-yet-treated comparisons, explicit anticipation windows,
  cohort-share event/calendar aggregations, and analytic robust or clustered inference.
- `EfficientDiD` implementing the no-covariate PT-All estimator of Chen, Sant'Anna, and
  Xie (2025), including generated-outcome candidates, inverse-covariance efficiency
  weights, event-study aggregation, and complete influence-function audit records.
- `DiDResult` with scalar ESavg summaries, group-time/event/calendar tables, candidate and
  aggregate influence functions, realized efficiency weights, pointwise confidence
  intervals, and OutputHub tables.
- Hand-computed conventional effects and aggregations, exact non-uniform efficient-weight
  identities, anticipation/control/cluster tests, refusal coverage, and optional pinned
  parity against the public R `edid` implementation.

### Architecture

- Conventional and efficient DiD share one strict balanced-panel/treatment-timing
  validator and one aggregation/inference layer while retaining separate identifying
  assumptions and public estimator classes.
- Fixed-T cohort/period loops contain vectorized entity-level NumPy operations. Cluster
  labels are normalized once and influence scores are summed without row-level loops.
- The efficient path solves covariance systems and refuses singular designs; it does not
  hide instability with a ridge, pseudoinverse, or clipped weights.

### Known limitations

- The alpha is for balanced short panels without covariates or sampling weights.
  Covariate-adjusted efficient estimation must use the public nuisance/cross-fitting
  protocols and implement conditional covariance estimation before promotion.
- Confidence intervals are pointwise. Multiplier-bootstrap simultaneous event-study
  bands, pre-trend/Hausman diagnostics, repeated cross-sections, and coverage simulations
  remain open gates.

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
