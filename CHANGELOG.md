# Changelog

All notable changes to CauseKit are recorded here. The project follows
[Semantic Versioning](https://semver.org/) once a public contract is released. Alpha
versions may refine APIs, but breaking changes must still be documented explicitly.

## [0.7.0a6] - Unreleased

### Added

- Public `RepeatedCrossSectionDiD`, `RepeatedCrossSectionDiDResult`, and
  `RepeatedCrossSectionPretrendDiagnostic` for no-covariate conventional cohort-time DiD
  under an explicit stationary-composition restriction. The estimator uses four
  independent cohort-period means, fixes comparison-cohort membership at target and
  baseline, and retains never-treated or valid not-yet-treated choices.
- Full observation-level group/event/calendar/ESavg influence records, pooled estimated-
  cohort-share terms, HC1 observation inference, one-way CR1 PSU inference, finite-cluster
  references, cell-count audits, and independent-cell adjacent pre-trend placebos without
  ridge or pseudoinverse repair.
- Hand-computed 2-by-2 estimate/influence/HC1 tests observed failing before the public
  estimator existed, staggered aggregation and anticipation identities, strict refusal
  tests, base-R 4.5.1 and reviewed Stata 17 hand parity, OutputHub tables,
  a hash-verified public hospital-data execution smoke, and a 100,000-row fixed-period
  performance harness.
- A preregistered 1,000-replication-per-design coverage certificate for balanced and
  unequal period sizes, never-treated and not-yet-treated controls, and every maintained
  group-time, event-time, calendar-time, and ES-average target. All 32 cells pass bias,
  analytical-SE calibration, coverage, Monte Carlo uncertainty, and zero-refusal gates.
- Estimator-level parity against pinned R `did` 2.5.0 at commit
  `c449b8ce72029855d2de94b377f131be0e53e53a`. All point estimates and analytical standard
  errors agree after the recorded `sqrt(n/(n-1))` HC0-to-HC1 mapping. The reviewed
  Stata/IC 17 `csdid` artifact exactly matches all 16 point estimates and six group-time
  analytical standard errors; aggregate SEs are recorded as non-comparable because
  `csdid` propagates period-specific cell-share rather than pooled cohort-share influence.
- Opt-in covariate-adjusted repeated-cross-section DiD through the public `CrossFitter`
  boundary. The path cross-fits one comparison propensity and four group-period outcome
  regressions on shared observation/whole-PSU folds, implements the normalized locally
  efficient doubly robust score, refuses overlap violations without clipping, and exposes
  nuisance predictions, folds, task diagnostics, and aligned conditional pre-trends.
- Failing-first hand score/influence, leakage, overlap, conditional-pretrend, and refusal
  contracts; both double-robustness legs; staggered aggregation identities; seeded
  simulation; fixed-OOF base-R parity; OutputHub diagnostics; and a hash-verified 7,368-row
  real-data/PSU execution smoke.
- A hash-bound covariate publication certificate with 1,000 replications per design,
  both control rules, 4,000 estimator fits, and 240,000 fold-local nuisance fits. All 44
  effect/placebo coverage cells and four joint conditional-pre-trend size cells pass the
  fixed calibration, bias, overlap, availability, audit-count, and zero-refusal gates.
  The initial undersized stressed design's fold-support failures are documented; the
  promoted design preserves its probabilities and unequal-size ratio while enforcing a
  minimum expected period-X-cohort cell size.

### Changed

- The package version and public documentation now identify `0.7.0a6`. Balanced-panel
  `DifferenceInDifferences` and `EfficientDiD` remain separate and unchanged; there is no
  `panel=False` alias or synthesized entity identifier.
- The repeated-cross-section design contract now records the implemented first-slice and
  completed no-covariate and covariate pointwise promotion evidence. Simultaneous-band
  behavior is unchanged and remains closed.

### Known limitations

- Stationary composition is declared but not testable from the estimator.
  Compositional-change-robust estimation, sampling or survey weights, and simultaneous
  event-study bands remain unimplemented and refuse. Covariate estimator-level parity is
  unavailable where reviewed R/Stata public paths target different moments.
- Stata `csdid` aggregate standard errors are non-comparable to the maintained pooled-
  cohort-share uncertainty contract. Observation/PSU simultaneous bands remain separately
  unimplemented.

## [0.7.0a5] - Unreleased

### Added

- Public `DiDPretrendDiagnostic` results with adjacent uncontaminated cohort-period
  placebos, full entity influence and covariance records, HC1 or cluster-summed joint
  inference, explicit no-lead/singular availability status, and OutputHub reporting.
- Public `did_hausman_test` and `DiDHausmanDiagnostic` for the common post-treatment
  event-study vector under aligned no-covariate PT-Post and PT-All results. The test uses
  the covariance of the difference influence function and refuses mismatched samples or
  singular systems without rank changes or numerical repair.
- A separately reasoned repeated-cross-section DiD contract covering the estimand,
  stationary-composition declaration, observation/PSU sampling unit, planned CrossFitter
  boundary, refusals, parity targets, performance, and promotion tests. No estimator
  placeholder is exported.
- A backward-compatible Hillstrom benchmark selector can run only the native spline stage,
  avoiding unnecessary repetition of the settled linear row. The `0.7.0a5` nonlinear-only
  visit smoke reproduced the frozen metrics and selected the zero-knot linear submodel;
  the new conversion-outcome comparison selected one knot but did not improve honest
  R-loss over linear ridge or the construction-fitted constant.
- A design-only `NativeOrthogonalStackedCATE` promotion contract freezes direct R-loss,
  construction-side cross-fitted simplex stacking, constant/linear/additive/interaction
  candidates, hard complexity limits, known-truth semisynthetic gates, external comparator
  evidence, and strict honest-evaluation non-leakage. No placeholder is exported.

### Changed

- DiD results now retain a deterministic design fingerprint and aligned inference-cluster
  labels so cross-estimator diagnostics cannot silently compare different samples.
- The package version and public documentation now identify `0.7.0a5`; conventional and
  efficient DiD point estimators and matching inference remain unchanged.

### Known limitations

- The covariate-adjusted PT-All path does not yet expose a conditional pre-trend score,
  and the maintained Hausman diagnostic is limited to the aligned no-covariate,
  never-treated PT-Post comparison.
- Repeated-cross-section DiD remains design-complete but unimplemented. Publication-scale
  DiD coverage, larger covariate performance, and broader covariate parity remain open.

## [0.7.0a4] - Unreleased

### Added

- A preregistered publication-scale matching certificate covering ATT, ATC, and ATE for
  both maintained analytical inference paths under favorable and stressed overlap. The
  runner records bias, empirical sampling variation, mean analytical standard errors,
  coverage with Monte Carlo uncertainty, interval width, arm counts, runtime, software
  versions, seeds, configuration, and pass/fail gates without resampling the estimator.
- A separate hash-verified Cattaneo design-sensitivity certificate for no support rule,
  intersection support, and narrower/default/wider logit-score calipers. Every row retains
  the realized target label, attrition, tie/reuse, and multivariate balance summaries.

### Changed

- The package version and public documentation now identify `0.7.0a4`. Matching
  publication evidence is promoted without changing either analytical estimator or the
  default `inference="none"` boundary.
- Caliper/support sensitivity remains point-estimation-only. CauseKit does not attach the
  maintained no-selection variance after a rule changes the retained target population.

### Known limitations

- Generic, cross-fitted, penalized, probit, trimmed-target, tied-boundary, clustered,
  paired, survey, and multiway matching inference remain unsupported. Ordinary bootstrap
  remains prohibited for fixed-neighbor matching.
- The promotion certificate evaluates two declared one-dimensional data-generating
  designs and one real observational dataset. It does not prove identification, universal
  nominal coverage, or robustness to unmeasured confounding.

## [0.7.0a3] - Unreleased

### Added

- Public `CATEEstimatorProtocol`, `DRLearner`, and `DRLearnerResult` under a contract
  separate from the R-learner's weighted final stage.
- Immutable treatment-stratified row or whole-cluster construction/evaluation roles;
  shared cluster-preserving propensity and arm-outcome folds; strict arm-specific fitting;
  fresh full-construction evaluation refits; augmented inverse-probability scores without
  hidden clipping; and a CauseKit-native unweighted ridge-GCV default CATE stage.
- Honest DR-score and construction-constant loss, intercept/heterogeneity calibration,
  tie-preserving mean-score groups, HC1/CR1 inference, seeded max-t group bands, influence
  records, graph data/optional plots, future-data prediction, and OutputHub tables.
- Observed-failing hand/leakage/refusal contracts, exact score/loss/covariance/max-t
  identities, both one-nuisance-side-correct simulations, native CATE recovery, base-R
  4.5.1 parity, a reproducible manual Stata HC1 fixture, a hash-verified NSW test, and a
  one-run NSW final-learner comparison.

### Changed

- The package version and public documentation now identify `0.7.0a3` and describe R- and
  DR-learning as separate estimator objectives sharing honest infrastructure.
- On the one-run NSW DR comparison, native ridge-GCV and optional scikit-learn RidgeCV
  produced identical displayed predictions, loss, and gain. Native used about 39% less
  Python-managed peak memory and was about 5.5% slower; boosting and forest were worse
  than the construction constant. Existing R-learner benchmark artifacts were not rerun.

### Known limitations

- DR inference is conditional on one recorded honest split. Unit-level CATE intervals,
  repeated-split aggregation, RATE, policy value, deployment refitting, and publication-
  scale coverage are not implemented.
- Double robustness requires a correct propensity or both correct arm outcome regressions,
  plus identification and nuisance-rate conditions; it does not repair unmeasured
  confounding or arbitrary final-stage misspecification.
- Repeated-split and publication-scale uncertainty evidence remain open even though the
  fixed-evaluation Python/base-R/reviewed-Stata parity row passes.

## [0.7.0a2] - Unreleased

### Added

- Public `NuisanceDiagnosticsProtocol` and provider-neutral per-task/per-fold diagnostic
  tables on every CrossFitter result.
- Native ridge audit fields for selected penalty, effective degrees of freedom, GCV score,
  training RMSE, numerical rank, grid size, and boundary selection; the complete table is
  retained on `PartiallyLinearDMLResult` and exported through OutputHub.
- An honest R-learner design contract covering construction/evaluation separation,
  overlap, weighted fitting, held-out R-loss, differential calibration, group effects,
  simultaneous bands, graph-data parity, refusals, and promotion evidence.
- Public provider-neutral `WeightedCATEEstimatorProtocol` and `CATEResultProtocol`, plus
  internal native stratified-CV penalized Logit and weighted ridge-GCV prerequisites with
  training-index/tuning audit state and strict malformed-provider refusals.
- `RLearner` and `RLearnerResult` with immutable treatment-stratified row or
  whole-cluster roles, cluster-preserving outer folds, fresh-factory enforcement,
  construction-only nuisance/CATE fitting, held-out R-loss and constant comparison,
  differential calibration, tie-preserving group effects, HC1/CR1 covariance, seeded
  max-t bands, graph data/optional plots, future-data prediction, and an exact identity
  between direct and transformed weighted R-objectives.
- R-learner OutputHub tables, linear/null/power/band-coverage simulation gates, independent
  base-R evaluation parity, a reproducible manual Stata HC1 fixture, a 2,000-row/500-
  cluster performance smoke, and a hash-pinned NSW CATE benchmark against aligned
  weighted RidgeCV, histogram boosting, and random forest comparators.
- Public opt-in `NativeSplineRidgeCATE` and `NativeSplineRidgeCATEResult`. Construction-
  only weighted GCV selects a bounded zero/one/three-knot additive linear-spline basis and
  ridge penalty; the result exposes the exact basis, tuning path, selected complexity, and
  prediction schema. Pairwise interactions are explicit and basis growth is capped.
- A hash-pinned 42,613-customer Hillstrom randomized-email benchmark comparing the native
  linear and adaptive spline CATE stages on one identical honest split. Kaggle remains an
  external download client, not a CauseKit dependency.

### Changed

- Candidate GCV residual sums of squares now use the SVD shrinkage identity without
  reconstructing a fitted vector for each penalty. A denser 41-point grid was evaluated
  but rejected as the default after it slightly worsened the recorded real-data OOF errors;
  the proven six-point default remains.
- Revalidated the five-fold Cattaneo workflow at `theta=-225.350625` with robust standard
  error `22.439969`; all ten nuisance fits selected interior penalties. The frozen
  `0.7.0a1` external-learner rows were not rerun or overwritten.
- Extended the one-run ML harness to the hash-pinned NSW job-training data. Native
  ridge-GCV had the lowest outcome OOF RMSE, runtime, and Python-managed peak memory;
  standardized scikit-learn RidgeCV had a 0.180% lower treatment OOF RMSE. The nonlinear
  configurations were worse on both nuisance targets, so the native default remains.
- On the separate honest NSW CATE run, standardized scikit-learn RidgeCV had the lowest
  R-loss but improved on the construction-fitted constant by only 0.0034%. Native
  ridge-GCV was 0.99% worse than constant; boosting and random forest were materially
  worse. No learner showed significant differential calibration. The earlier scalar-DML
  benchmarks were not rerun or overwritten.
- The adaptive spline row was added without rerunning the original NSW comparator rows; it
  was 1.43% worse than the constant baseline and remains opt-in. On the separate Hillstrom
  RCT it matched the native linear stage to displayed precision, consistent with safe
  fallback to the zero-knot candidate.
- The reviewed Stata/IC 17 fixed-evaluation R-learner artifact passes every loss,
  calibration, covariance, and group assertion at `1e-8`; the maximum absolute difference
  is `1.55e-15`.

### Known limitations

- Weakly lower training-fold GCV under a denser supplied grid did not improve the recorded
  held-out errors and does not imply stronger causal identification.
- R-learner inference is conditional on one recorded honest split. Repeated-split
  aggregation, unit-level CATE intervals, RATE, targeting/policy value, and deployment
  refitting are not provided. Native spline evidence covers one known nonlinear simulation
  and two real-data splits, not universal superiority or publication-scale coverage.

## [0.7.0a1] - Unreleased

### Added

- `PartiallyLinearDML` and `PartiallyLinearDMLResult` for DML2 estimation of the scalar
  treatment coefficient in a declared partially linear structural model.
- A CauseKit-owned, dependency-free standardized ridge nuisance learner with
  generalized-cross-validation penalty selection performed separately inside every
  outer training fold.
- Shared-fold outcome/treatment nuisance tasks through `CrossFitter`, exact binary-arm
  stratification, aligned prediction/residual/fold audit records, HC1 and one-way CR1
  influence inference, and scale-aware residual-treatment identification refusal.
- Hand-reconstructed score/influence/variance contracts, deterministic simulation,
  strict refusal tests, Statsmodels, base-R 4.5.1, and reviewed Stata/IC 17
  residual-stage parity, OutputHub adaptation, and a pinned real-data workflow smoke.
- A one-run hash-verified Cattaneo real-data benchmark comparing the native learner with
  scikit-learn RidgeCV, histogram gradient boosting, and random forest without adding a
  runtime dependency.

### Architecture

- The native causal-ML path owns its default nuisance learner and does not depend on a
  third-party ML package. Existing public nuisance factories remain optional escape
  hatches for designs that need another learner.
- The scalar coefficient is labelled `theta`, not automatically `ATE`; its causal
  interpretation requires the constant-effect partially linear model plus the documented
  exchangeability, variation, and nuisance-rate assumptions.
- Heterogeneous-effect R/DR learners and graphing remain later contract-first milestones.

### Known limitations

- The alpha does not provide heterogeneous treatment effects, dose-response curves,
  endogenous-treatment DML, repeated cross-fitting, multiway clustering, sample weights,
  bootstrap inference, or native causal forests.
- The deterministic Python, base-R, and manually executed Stata parity rows pass for the
  aligned fixed-OOF residual stage; they do not compare nuisance-learning algorithms.
- A single comparator benchmark is descriptive, not a model ranking or coverage study.

## [0.6.0a4] - Unreleased

### Breaking change

- Renamed the distribution and import namespace from the unreleased `causalkit` identity
  to `causekit`. The former name is occupied on the public Python package index by an
  unrelated project, so this release deliberately provides no compatibility shim under
  that namespace.

### Added

- Public `FittedPropensityMLEProtocol` and `inference="abadie_imbens_estimated"` for
  ATT, ATC, and ATE matching on a validated full-sample unpenalized Logit MLE.
- Abadie–Imbens first-step covariance, target-derivative, and Fisher-information
  corrections with separately reported known-score and adjustment components.
- Hand-computed three-estimand contracts, malformed-model and unsupported-design
  refusals, independent `statsmodels.Logit` parity, a seeded coverage smoke, and a
  100,000-row benchmark.
- A manual Stata `teffects psmatch` parity harness that persists point estimates,
  uncorrected variance components, first-step corrections, and final standard errors.
- Reviewed Stata/IC 17 ATT/ATC/ATE parity, including separate known-score variance and
  fitted-propensity first-step components, with maximum absolute standard-error
  difference `2.62713550913674e-9` at tolerance `1e-8`.
- An opt-in, SHA-256-verified real-data registry and runnable workflow spanning IV,
  randomized, observational, matching, conventional-DiD, and efficient-DiD models.
- A deterministic real-data parity preparation pipeline, 14 passing Python/R comparisons
  against pinned `edid` and `Matching` checkouts, and a manual Stata saved-output harness
  for all estimand-aligned rows available in Stata.
- Reviewed Stata/IC 17 real-data parity passes for IV, randomized ATE, conditional-
  nuisance IPW/AIPW, matching, and conventional DiD; PT-All efficient DiD is recorded as
  unavailable rather than replaced by a non-aligned estimator.

### Architecture

- CauseKit consumes fitted nuisance results through provider-neutral protocols and has
  no LimitedDepKit runtime, validation, or benchmark dependency.
- Completed the legacy `TreatmentEffect` ownership migration: `IV2SLS` is the sole
  maintained causal/IV implementation and reconstructs the old homoskedastic numerical
  contract in a package-owned migration test without retaining duplicate source.
- The estimated-score path is intentionally narrower than `CrossFitter`: it validates
  convergence, sample/feature alignment, Logit predictions, likelihood stationarity, and
  nonsingular normalized information. Cross-fitted, penalized, probit, and generic scores
  retain `inference="none"`.
- Scalar local moments use sorted searches, and ATT/ATC target derivatives use `cKDTree`;
  no treated-by-control or covariate pairwise distance matrix is allocated.

### Known limitations

- R `Matching` treats a supplied score as fixed and is non-comparable for the fitted-Logit
  first-step correction.
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
  does not own nuisance estimators.
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

- Nuisance estimation remains external through provider-neutral public predictions. This
  package owns the causal score, estimand, diagnostics, and inference.

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

- Established `causekit.IV2SLS` as the destination for ordinary linear 2SLS workflows
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

The earlier alpha milestones above are incorporated into the `0.6.0a4` release history;
their retained headings describe the staged implementation sequence rather than separate
public-package uploads.
