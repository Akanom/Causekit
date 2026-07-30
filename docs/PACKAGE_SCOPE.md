# Package scope

`causekit` contains estimators and workflows whose central problem is causal
identification. Inclusion requires more than a method being common in applied economics:
the package must be able to state the estimand, identifying assumptions, supported data
structure, inference target, diagnostics, and validation boundary.

## Public `0.7.0a6` alpha surface

The surface is estimator-specific. Cross-sectional linear instrumental variables retain
the following contract:

| Capability | Supported contract |
| --- | --- |
| Estimator | `IV2SLS` with one or more endogenous regressors |
| Instruments | `instruments=` contains excluded instruments only; the intercept and exogenous regressors enter the full instrument set internally |
| Exogenous controls | Optional explicit numeric design; a constant is added by default |
| Covariance | `"unadjusted"` (homoskedastic), `"robust"` (HC1), or `"clustered"` (one-way CR1) |
| Data policy | Missing and non-finite values raise by default; `missing="drop"` applies one explicit joint complete-case mask; pandas indices and schemas must align exactly |
| First stage | Ordinary and partial R-squared, classical F, and covariance-aware joint exclusion test for each endogenous regressor |
| Overidentification | Sargan only for overidentified `"unadjusted"` fits |
| Results | Labelled estimates and inference, fitted values, residuals, diagnostics, prediction, tables, Markdown, and optional reporting adaptation |

The randomized-experiment family supports two-arm individual-level assignment through
`RandomizedATE`: exact difference in means or fully interacted Lin adjustment, HC1/CR1
inference, arm counts, and pre-treatment covariate-balance diagnostics. It requires both
arms coded exactly 0/1 and does not infer or validate the assignment mechanism from data.

The stable label describes the intended API surface within this alpha, not a claim that
instrument validity or causal identification can be automated.

The observational surface includes supplied-nuisance `IPWATE` and `AIPWATE`, reusable
`CrossFitter` orchestration, and ATT/ATC/ATE target choices. The matching alpha adds
`NearestNeighborMatch` for supplied logit-propensity distance with replacement, explicit
support/caliper/tie rules, bidirectional ATE imputation, audit weights, reuse counts, and
balance diagnostics. `inference="none"` remains the default. A maintained Abadie-Imbens
analytical path is available for a declared fixed score without support/caliper selection
or expanded ties. A separately validated full-sample unpenalized Logit-MLE path consumes
`FittedPropensityMLEProtocol` results and applies the Abadie–Imbens estimated-score
correction. Arbitrary/cross-fitted/penalized scores, bootstrap, and clustered alternatives
refuse rather than returning a methodologically incomplete standard error.
The publication-scale certificate exercises ATT, ATC, and ATE for both analytical paths
under favorable and stressed overlap, while a separate Cattaneo caliper/support grid keeps
`inference="none"` and exposes every target change and exclusion.

The DiD surface adds `DifferenceInDifferences` and `EfficientDiD` for balanced short
panels. The conventional class retains never-treated/not-yet-treated comparison choices;
the efficient class implements both no-covariate and cross-fitted covariate-adjusted
Chen-Sant'Anna-Xie PT-All generated-outcome weighting. Both expose cohort-time,
event-time, calendar-time, and ESavg effects, entity influence functions, robust/one-way-
clustered pointwise inference, optional multiplier-bootstrap simultaneous event-study
bands, uncontaminated pre-trend placebos, and OutputHub tables. The public Hausman
diagnostic compares aligned no-covariate PT-All and PT-Post event-study paths from the
difference influence function. Efficient DiD is an opt-in stronger-assumption estimator,
not a replacement default.

The separate `RepeatedCrossSectionDiD` surface implements conventional no-covariate and
opt-in cross-fitted covariate-adjusted cohort-time effects under declared stationary
composition. The adjusted path consumes public `CrossFitter` factories for one propensity
and four group-period outcome regressions per comparison, implements the locally efficient
doubly robust repeated-cross-section score, and aligns conditional pre-trend placebos to
that score. Both paths retain fixed comparison membership, pooled cohort-share
aggregation, cell-count audits, and HC1 observation or one-way CR1 PSU inference. It has
no entity role and still refuses sampling weights and composition-change-robust labels.
Its opt-in multiplier path draws once per observation or indivisible PSU and supplies
studentized max-t simultaneous event-study bands without nuisance refitting. Separate
1,000-replication publication certificates cover the
no-covariate and covariate-adjusted pointwise paths; the latter includes conditional
placebo coverage and joint pre-trend size with genuinely fitted fold-local nuisances. A
third 16-cell certificate covers the complete event vector under observation/PSU
sampling for both paths and both control rules with zero refusals.

The causal-ML surface adds `PartiallyLinearDML` for the scalar DML2 coefficient in a
declared constant-effect partially linear model. CauseKit owns its default standardized
ridge-GCV nuisance learner, performs selection independently inside each outer training
fold, and exposes aligned predictions, residuals, fold assignments, fold-level tuning
diagnostics, orthogonal scores, influence functions, HC1/CR1 inference,
residual-treatment diagnostics, and OutputHub metadata/tables. Optional public nuisance
factories remain available only when a design needs a different learner; they do not
create a runtime dependency.

The heterogeneous-effect surface adds `RLearner`. It creates immutable treatment-
stratified construction/evaluation roles, or indivisible cluster roles when clustered
inference is declared. Outcome and propensity nuisances are cross-fitted only inside
construction, the CATE stage minimizes the exact weighted R-objective, and evaluation is
used once for held-out R-loss, differential calibration, and tie-preserving overlap-
weighted group effects. Pointwise HC1/CR1 inference and seeded max-t group bands are
split-conditional. The result provides no unit-level CATE intervals, RATE, policy value,
or repeated-split aggregation.

The separately contracted `DRLearner` cross-fits the propensity and two arm-specific
outcome regressions in construction, forms the augmented inverse-probability score without
clipping, and fits an unweighted CATE stage. Evaluation-only DR loss, calibration, mean-
score groups, HC1/CR1 covariance, and max-t bands preserve the same immutable honest
boundary without relabelling the R-objective. It exposes the same unit-level, RATE, policy,
and repeated-split refusals.

## Deliberate boundaries in this release

The current alpha does not provide:

- a formula parser or stored preprocessing pipeline;
- automatic categorical coding, interaction generation, scaling, or imputation; row
  deletion occurs only through the explicit joint `missing="drop"` policy;
- ordinary least squares as a general regression package;
- limited-dependent-variable likelihoods;
- panel fixed effects, dynamic-panel GMM, or panel IV;
- weak-IV-robust confidence sets or a complete identification-robust testing suite;
- heteroskedasticity-robust overidentification tests;
- multiway clustering, general-purpose bootstrap inference, sampling weights, or survey design;
- nonlinear IV, GMM beyond linear 2SLS, or control-function estimators;
- automatic discovery, selection, or validation of instruments;
- native causal forests, dose-response curves, or policy learning beyond the implemented
  honest R-/DR-learners and opt-in adaptive spline R-learner CATE stage; or
- repeated cross-fitting, multiway clustered DML, survey weights, or DML bootstrap
  inference.

These exclusions protect a clear claim boundary. A feature is not silently approximated by
a different statistic merely to fill a result field.

## Relationship to sibling packages

The three packages are separated by their main estimand and data structure:

- `causekit` owns identification-aware causal estimators and cross-sectional IV;
- `limiteddepkit` owns models defined by limited outcomes, censoring, truncation, duration,
  choice, or another limited observation rule; and
- `systemgmmkit` owns panel estimators and dynamic-panel GMM workflows.

Shared conventions are intentional: labelled pandas results, explicit covariance metadata,
strict alignment, diagnostic objects, post-estimation tables, and optional OutputHub
adaptation. Where future work overlaps panel validation, entity/time indexing, fixed
effects, clustered covariance, post-estimation, plotting, or reporting, the design should
reuse applicable `systemgmmkit` contracts instead of creating a disconnected parallel
framework. Reuse does not require importing private sibling internals or adding a runtime
dependency.

Binary logit/probit, count, censoring/truncation, duration, ordinal, and related
limited-outcome likelihoods do not move into `causekit` merely to obtain nuisance
predictions. IPW/AIPW and heterogeneous-outcome workflows use small provider-neutral
public nuisance protocols. Causal estimands, assignment/ignorability assumptions,
overlap, cross-fitting, influence-function inference, and treatment-effect diagnostics
remain the responsibility of `causekit`.

The first observational slice therefore consumes supplied propensity and potential-
outcome predictions. `IPWATE` uses the Horvitz-Thompson ATE score; `AIPWATE` uses the
augmented influence-function score. Both enforce exact row alignment and strict propensity
support and expose weight effective sample sizes. Built-in propensity and arm-specific
outcome fitting for IPW/AIPW remains separate from the specialized DML default; the latter
owns only the native regression nuisance path required by its declared partially linear
score.

`CrossFitter` now provides that protocol boundary. It generates out-of-fold propensity and
arm-specific outcome predictions from fresh model factories and supports fitted results
that either expose the default prediction methods or use explicit adapters. IPW/AIPW now
support the analysis-population ATE, treated-population ATT, and control-population ATC.
This is orchestration, not ownership transfer: nuisance estimators remain in their proper
packages.

`PartiallyLinearDML` is the deliberate exception for a complete native causal-ML path. Its
small ridge-GCV learner is implemented in CauseKit, selected within each outer fold, and
subordinate to the DML score. This does not move sibling-package likelihood models into
CauseKit or turn the package into a general-purpose prediction library.

`RLearner` is the corresponding heterogeneous-effect exception. Its native penalized
probability and weighted ridge-GCV components remain subordinate to the honest R-objective
and evaluation contract. Custom factories replace only declared learner roles; CauseKit
continues to own splitting, overlap, residualization, evaluation, covariance, and graph
semantics.

`DRLearner` is a separate heterogeneous-effect exception rather than an alias or wrapper
around `RLearner`. It owns the augmented inverse-probability score, two arm-specific
outcome roles, unweighted public CATE protocol, honest loss/calibration/groups, and strict
no-clipping overlap boundary. It reuses CauseKit's native ridge-GCV only as a subordinate
default; optional external factories are comparator/provider escape hatches and do not
become package dependencies.

Matching inference uses a second, deliberately narrower protocol because the
Abadie–Imbens first-step formula requires a regular full-sample parametric MLE rather than
an arbitrary prediction learner. CauseKit validates the provider-neutral public fit
result and owns matching, the causal estimand, diagnostics, and uncertainty correction.

## `limiteddepkit.TreatmentEffect` migration provenance

The historical `TreatmentEffect` class implemented ordinary homoskedastic 2SLS inside
`limiteddepkit`. Because endogeneity does not make an outcome limited, the class was
removed from that package's public and installable namespaces. The migration is now
complete and the obsolete source snapshot has also been removed from LimitedDepKit.

`causekit.IV2SLS` is a new destination with a stronger contract. The migration rationale
and expected result conventions informed the design, but no private implementation code
was copied. The estimator is implemented from the public matrix definition of 2SLS and
package-owned validation, covariance, diagnostics, and result layers. A maintained
CauseKit test reconstructs the legacy homoskedastic matrix contract and checks the new
estimator after mapping the old full-`Z` interface to excluded instruments.

The APIs differ materially. Legacy `TreatmentEffect.fit(..., Z=...)` expected a full
instrument matrix that already spanned the exogenous regressors. New
`IV2SLS.fit(..., instruments=...)` expects only excluded instruments and constructs the
full instrument matrix internally. The new class also defaults to an intercept and HC1
inference, exposes unified parameter ordering, and refuses silent index or missing-data
changes. Migration therefore requires an explicit design audit, not a search-and-replace.

## Roadmap candidates

The following families are credible later additions, subject to estimator-specific design
and validation gates:

| Family | Required design questions before promotion |
| --- | --- |
| Matching extensions | Publication-scale sensitivity/coverage and available Python/R/Stata evidence pass; generic-score, selected-target, tie-expanded, clustered, survey, and bootstrap inference remain separately prohibited or deferred |
| DiD promotion | Covariate-adjusted repeated-cross-section DiD, aligned conditional pre-trends, publication-scale pointwise and observation/PSU joint-band coverage are implemented; composition robustness and survey weights retain separate gates |
| Causal ML promotion | Implement the frozen construction-cross-fitted native orthogonal-stack contract, then harden R-/DR-learners with repeated-split and publication-scale evidence; unit-level intervals, RATE, and policy evaluation retain separate contracts |
| Regression discontinuity | Sharp/fuzzy design, running-variable support, bandwidth and polynomial choice, manipulation checks, bias correction, and local estimand |
| Panel IV | Entity/time indexing, fixed effects, within transformations, serial dependence, instrument variation, clustered inference, and compatibility with `systemgmmkit` |

Roadmap status is not an implementation promise. A family remains experimental or absent
until its public contract, failure behavior, tests, independent reference evidence, and
documentation are complete.

The normative matching decisions and fixed-score inferential boundary are recorded in
[Nearest-neighbor matching contract](MATCHING_CONTRACT.md). The exported estimator has an
audited point path plus narrow fixed-score and full-sample Logit-MLE analytical paths; this
is not a claim that arbitrary estimated or machine-learning scores have supported inference.

The conventional/efficient distinction, timing rules, formulas, covariate nuisance path,
and inference boundaries are recorded in [Difference-in-differences contract](DID_CONTRACT.md).

## Out of current scope

DADPLM and BDCPM are separate research/modeling lines and are outside the current
`causekit` scope. This release does not reserve public imports, placeholder estimators, or
compatibility claims for either project. Any future scope proposal must begin with a clear
estimand and architecture review rather than assume inclusion from thematic proximity.

Generic prediction models, neural networks, unrestricted finite mixtures, Markov-switching
models, and limited-outcome likelihoods are also not added merely because they can appear
inside a causal workflow. Nuisance models may eventually support an in-scope causal
estimand, but their contract must be subordinate to that estimand and validated as part of
the complete procedure. The native ridge-GCV implementation is therefore an internal
component of `PartiallyLinearDML` and `RLearner`, not a general regression API.

## Promotion criteria

A new stable family must have:

1. an explicit estimand and target population;
2. auditable identifying assumptions and appropriate causal-language limits;
3. a typed, strict, and documented input contract;
4. inference matched to the sampling and assignment structure;
5. diagnostics that do not overstate what observed data can establish;
6. analytical, simulation, refusal-path, and independent-reference tests;
7. reproducible evidence with versions, mappings, seeds, and tolerances;
8. compatible result, post-estimation, and reporting conventions; and
9. a documented migration path for any affected public API.

Model count is not a success criterion. A narrower surface with honest assumptions and
maintained evidence is preferable to unsupported causal labels.
