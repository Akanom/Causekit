# Architecture

`causekit` is organized around small, auditable estimation paths. The `0.7.0a6`
architecture keeps causal assumptions visible, separates numerical estimation from
inference and diagnostics, and returns frozen labelled result containers suitable for
reporting. The pandas objects stored inside a result should be treated as read-only; helper
functions return defensive copies where applicable.

## Execution path

```text
public IV2SLS specification
        |
        v
input normalization and strict alignment
        |
        v
structural design S = [constant, exogenous, endogenous]
instrument design Q = [constant, exogenous, excluded instruments]
        |
        v
rank and identification checks
        |
        v
2SLS point estimate, fitted values, structural residuals
        |
        +----------------------+
        |                      |
        v                      v
covariance and inference       first-stage and overidentification diagnostics
        |                      |
        +-----------+----------+
                    v
labelled result, prediction, summaries, optional adapters
```

The ordering is deliberate. The estimator establishes one aligned analysis sample before
building either structural or instrument matrices. Covariance and diagnostics consume the
same fitted sample and residual definitions, preventing each layer from silently selecting
different rows.

## Public boundary

The root package exports the supported estimator and result/diagnostic types. The IV
entry point is:

```python
IV2SLS(
    covariance="robust",
    add_constant=True,
    missing="raise",
).fit(
    y,
    endogenous=...,
    instruments=...,
    exogenous=None,
    clusters=None,
)
```

Only excluded instruments belong in `instruments`. This semantic boundary is more
important than an internal matrix name: the estimator owns construction of the full
instrument set and can therefore check duplicate roles, order, and rank consistently.

The accepted covariance labels are exactly `"unadjusted"`, `"robust"`, and
`"clustered"`. Adding aliases or formula syntax would expand the public parsing contract
and is not part of this release.

## Source map

The installed source tree assigns one primary responsibility to each module:

| Module | Responsibility |
| --- | --- |
| `causekit.__init__` | Stable root exports and package version |
| `causekit._data` | Input coercion, labels, exact pandas alignment, joint missing-data policy, and prediction schema |
| `causekit._covariance` | Homoskedastic, HC1, and one-way CR1 covariance kernels and inference metadata |
| `causekit.diagnostics` | First-stage diagnostic records and homoskedastic Sargan testing |
| `causekit.iv` | Public `IV2SLS`, 2SLS execution path, and fitted `IV2SLSResult` |
| `causekit.panel_iv` | Long-panel validation, compact fixed-effect absorption, Panel 2SLS, absorbed-rank HC1/CR1 inference, and fixed-effect-adjusted diagnostics |
| `causekit.randomized` | Two-arm difference-in-means and Lin-adjusted ATE execution path, balance records, and fitted result |
| `causekit.observational` | Supplied-nuisance IPW/AIPW scores, overlap diagnostics, vectorized influence-function and cluster inference |
| `causekit.crossfit` | Public nuisance protocols, deterministic stratified fold orchestration, binary/multiclass probabilities, masked scalar tasks, fresh-model fitting, prediction adaptation, and aligned out-of-fold records |
| `causekit.matching` | Supplied-score ATT/ATC/ATE matching, sorted scalar neighbor search, support/caliper rules, fractional ties, weights, reuse, balance, and separate fixed-score or validated Logit-MLE analytical inference |
| `causekit.did` | Balanced-panel validation, conventional group-time DiD, cross-fitted covariate PT-All scores/conditional weights, pointwise and simultaneous influence inference, and cohort/event/calendar aggregation |
| `causekit.did_rcs` | Stationary repeated-section validation, marginal/cross-fitted doubly robust scores, observation/PSU pointwise and multiplier max-t inference, conditional placebos, and cohort/event/calendar aggregation |
| `causekit.ml` | Native ridge-GCV nuisance fitting, shared-fold partially linear DML2 score, influence inference, fold-level tuning diagnostics, and fitted result |
| `causekit.rd` | Sharp/fuzzy local-polynomial RD, bounded native bandwidth selection, robust bias correction, local score inference, support/manipulation diagnostics, and fitted result |
| `causekit.postestimation` | Summary, covariance, confidence interval, prediction, residual, fitted-value, linear-combination, and Wald helpers |
| `causekit.integrations.outputhub` | Lazy optional conversion and insertion into Universal Output Hub |
| `causekit.datasets` | Opt-in HTTPS-only, SHA-256-pinned real-data cache used by examples and parity; source datasets are not redistributed |

Underscored modules are internal. Users should import estimators, result types, diagnostics,
and post-estimation helpers from `causekit`; internal module paths may change during the
alpha series.

## Layer responsibilities

### Specification and validation

The public model stores covariance, intercept, and missing-data choices. Validation then:

- converts accepted numeric inputs without losing pandas labels;
- establishes a single observation index;
- enforces one-dimensional outcomes and cluster labels and two-dimensional designs;
- rejects missing and non-finite numeric data;
- rejects duplicate column labels, mismatched rows, and misaligned indices;
- adds and names the constant when requested;
- preserves parameter order and fitted schema; and
- checks dimensions, residual degrees of freedom, full column rank, the IV order condition,
  and the rank condition.

Validation produces a normalized internal bundle rather than making downstream layers
repeat coercion. `missing="drop"` is the only row-removal path: it applies one mask to every
input and records the removal count. No downstream component may drop or reorder rows.

Panel IV owns a separate long-form bundle because entity/time identity is part of its
estimand and covariance contract. It sorts one unique entity-time index, applies one joint
complete-case mask, factors entity/time/cluster labels once, checks two-way graph
connectivity, and residualizes the outcome plus every structural and excluded-instrument
column together. Balanced two-way data use double demeaning; connected unbalanced data use
deterministic alternating projections with an explicit convergence audit. The estimator
never constructs fixed-effect dummies or an observation projection matrix.

### 2SLS core

The numerical core receives validated structural and full instrument designs. It computes
the projection-based 2SLS solution, fitted values, and **structural** residuals. Covariance
must use those structural residuals, not residuals from a regression that substitutes
first-stage fitted endogenous variables into the outcome equation.

The core should solve linear systems or use a stable decomposition. Explicit matrix
inverses and a materialized `n x n` projection matrix are conceptual definitions, not a
required runtime strategy. Avoiding `P_Q` materialization prevents an obvious quadratic
memory cost on large samples.

### Covariance and coefficient inference

Covariance is a separate policy layer with a common output containing the labelled matrix,
reference distribution, degrees of freedom, and cluster count when relevant.

- `unadjusted` uses the homoskedastic residual variance and t inference with `n-k` degrees
  of freedom;
- `robust` uses observation-score HC1 with multiplier `n/(n-k)` and normal inference; and
- `clustered` aggregates scores by one cluster dimension, applies CR1, and uses t inference
  with `G-1` degrees of freedom.

Keeping reference-distribution metadata with the covariance prevents summary, confidence
interval, and p-value code from guessing based on a display label.

### Diagnostics

First-stage diagnostics refit each endogenous column on the common full instrument design.
They retain both the classical F comparison and a joint excluded-instrument test using the
selected covariance. Each diagnostic carries statistics, degrees of freedom, p-values,
distribution metadata, partial fit evidence, and a limited weak-instrument warning.

Overidentification is a separate optional diagnostic. The result contains a Sargan object
only when the model is overidentified and coefficient covariance is unadjusted. It remains
absent for exactly identified, HC1, and clustered fits rather than relabeling a
homoskedastic statistic as robust.

### Result and post-estimation

The fitted result is a data object, not a live optimizer. It stores:

- labelled `params`, `covariance`, `standard_errors`, `test_statistics`, and `pvalues`;
- aligned `fitted_values` and structural `residuals`;
- `first_stage` and optional `overidentification` diagnostics;
- `nobs`, `df_resid`, and `covariance_type`; and
- the fitted schema needed for prediction.

`summary_frame()`, `conf_int()`, `predict()`, and `to_markdown()` are views over those
stored results. They must not refit the model, alter the covariance choice, or silently
reconstruct preprocessing. Prediction enforces the fitted exogenous/endogenous schema and
preserves a valid pandas index.

### Optional integrations

Reporting adapters live below `causekit.integrations` and translate a fitted result into
the external reporting contract. `universal-output-hub` is optional; importing and fitting
the core estimator must not require it. When unavailable, the adapter should raise an
actionable installation error only when called.

Adapters may rename fields for an external schema but must not recompute estimates,
covariance, degrees of freedom, or diagnostics. The native result remains the source of
truth.

## Compatibility strategy

`systemgmmkit` and `limiteddepkit` established useful conventions for labelled econometric
results. `causekit` follows compatible meanings for parameters, covariance, standard
errors, test statistics, p-values, observation counts, confidence intervals, prediction,
and table/report adapters.

Compatibility is structural rather than inheritance-based:

- no sibling package is a required runtime dependency;
- no private sibling module is imported;
- no result class is promised to be interchangeable where estimator semantics differ; and
- shared conventions are verified through public fields and adapter behavior.

`PanelIV2SLS` follows the reviewed SystemGMMKit panel-validation, indexing, compact-within,
and clustering meanings without importing sibling code. CauseKit strengthens the causal
boundary with excluded-instrument-only roles, strict absorbed/rank refusals, full
absorbed-rank covariance corrections, fixed-effect-adjusted first stages, and no silent
column dropping. It is static Panel IV, not a second dynamic-panel GMM implementation.
An ownership audit confirmed that LimitedDepKit exposes no active causal estimator to
migrate. Its stale archived 2SLS snapshot and test were removed after CauseKit's maintained
IV contract passed promotion. Historical cross-package comparison text is not part of
either package's executable API.

Limited-outcome estimators are not copied into this package. `CrossFitter` targets a
provider-neutral fit/predict protocol with no sibling-package import or validation
dependency. The causal procedure owns sample splitting, estimand construction,
diagnostics, and valid uncertainty propagation.

The current observational fast path performs one strict alignment pass, constructs the
IPW/AIPW score with vectorized array operations, and aggregates clustered influence sums
with normalized integer codes. It does not loop over observations, refit nuisance models,
or materialize quadratic matrices. This follows the high-throughput implementation
patterns maintained in the sibling packages while keeping causal score semantics local.

The matching fast path sorts each treatment arm once and uses binary search plus local
left/right expansion for every focal unit. It materializes only selected match rows, not
an arm-by-arm distance matrix. Matching decisions consume treatment, supplied propensity,
and declared design settings; outcome values are used only after the match design is fixed
to form observed-minus-imputed contrasts. Fractional tie weights and comparison reuse
remain explicit in the result.

Matching uncertainty is intentionally separate from generic covariance code. The alpha
defaults to `inference="none"`. Its maintained Abadie–Imbens path estimates same-arm
conditional variances, applies estimand-specific comparison-reuse formulas, and is exposed
only for a declared fixed score without support/caliper selection or expanded ties.
The separate `FittedPropensityMLEProtocol` consumes a provider-neutral public fitted
result; it does not fit or copy a binary model. The estimated-score path validates the
full-sample unpenalized Logit score and normalized Fisher information, then applies the
Abadie–Imbens first-step correction. Generic, cross-fitted, penalized, or unverifiable
predictions still refuse analytical inference. The matcher does not reuse IV/ATE sandwich
or CR1 kernels merely to populate standard-error fields.

The DiD path performs one long-to-wide balanced-panel validation and keeps the entity as
the sampling unit. `DifferenceInDifferences` and `EfficientDiD` share this panel bundle,
cohort-share aggregation, labelled influence-function results, and robust/clustered
inference. Their point estimators remain separate: the conventional class uses one
pre-treatment baseline with never-treated or not-yet-treated comparisons, while the
efficient class constructs the PT-All generated outcomes and solves their covariance
system. Fixed-T cohort/period loops are permitted; all entity-level arithmetic is
vectorized and cluster scores are aggregated after one factorization of the labels.

Panel pre-trend diagnostics reuse the validated wide outcome bundle but build a separate
matrix of adjacent changes that end before the anticipation boundary. Joint covariance is
formed from the complete influence matrix, with cluster vector sums performed once.
`did_hausman_test` aligns common post-treatment event-study coordinates and tests the
PT-All minus PT-Post influence vector; a result fingerprint prevents comparisons across
different outcomes, samples, timing, or cluster designs.

Repeated cross sections do not enter this panel bundle. `did_rcs.py` owns their separate
four-cell means, observation-level influence records, pooled cohort-share aggregation,
cell audit table, and observation/PSU inference. It accepts unequal period and cell sizes,
requires an explicit composition declaration, and never synthesizes a panel entity. Only
the generic score-covariance and joint-Wald kernels are shared with `did.py`; the panel
validator, within-entity changes, and panel pre-trend helper are not reused.

Its covariate path plans every group-time and conditional-placebo comparison together,
then sends one propensity and four masked group-period outcome tasks per comparison through
one public `CrossFitter` call. A single cohort-period-stratified fold vector is therefore
shared across tasks; declared PSUs remain whole. `did_rcs.py` owns the normalized locally
efficient doubly robust score, ratio influence contributions, overlap refusal,
aggregation, and uncertainty. It does not own a nuisance learner, import the panel
outcome-change machinery, clip probabilities, or expose irrelevant comparison-row
predictions as meaningful values.

The composition-robust branch preserves the pairwise score instead of building one
design-wide multinomial. `CrossFitter.fit_predict_class_probability_tasks` plans every
`(cohort, baseline, target)` task on one immutable global row/PSU fold vector, trains only
the pair-masked four-class rows, and predicts only relevant held-out pair rows. Coordinated
masked outcome tasks fit `m_00`, `m_01`, and `m_10`; `m_11` is not unused work. The class-
probability boundary realigns labelled permutations but refuses missing, extra,
duplicate, or unlabelled schemas.

`did_rcs.py` hard-refuses overlap failures, constructs normalized pair weights, embeds
each pair influence on the full sample with exact zeros outside the pair, and retains the
support/scale/task audit in a pair ledger. Event and calendar aggregation use target-
period treated-cell shares plus their estimated-share influence. The public composition
diagnostic subtracts aligned robust and stationary influence matrices before covariance
assembly; marginal covariance subtraction, pair-local resplitting, numerical rank repair,
and diagnostic-driven estimator selection are prohibited.

Survey support remains outside the runtime architecture and requires an explicit design
object, weighted nuisance protocol, and design-based variance. The balanced-panel direct-
ratio route replaces `EfficientDiD` multiclass probability ratios with calibrated ordered
pairwise cohort odds and refactors its conditional covariance only by a scale that cancels
in normalized efficient weights. `CrossFitter` owns fresh pair/fold fitting, all-holdout
prediction, immutable fold reuse, alignment, and support audits; `did.py` owns the PT-All
candidate graph, score, `Omega_tilde` assembly, thresholds, and cross-score refusals.

No nuisance learner lives in `did.py`. The covariate-adjusted efficient path expresses
cohort classification, group-specific outcome changes, and conditional residual products
as public `CrossFitter` operations. `did.py` owns the causal score, equation (3.12)
conditional covariance assembly, normalized solve, aggregation, and uncertainty. A low
cohort probability or singular weight system refuses rather than silently applying
clipping, a ridge, or a pseudoinverse.

The specialized causal-ML path lives in `ml.py`. `PartiallyLinearDML` sends outcome and
treatment conditional-mean tasks through the same `CrossFitter` plan, residualizes once,
and evaluates the pooled DML2 orthogonal score with vectorized arrays. When no factories
are supplied, every fold receives a fresh CauseKit-native standardized ridge learner; its
penalty is selected by generalized cross-validation using only that fold's training rows.
The SVD path evaluates candidate residual sums of squares without reconstructing fitted
vectors. A denser candidate grid did not improve the recorded real-data out-of-fold errors,
so the original six-point default remains. The optional public
`NuisanceDiagnosticsProtocol` carries scalar tuning diagnostics through `CrossFitter` into
the DML result and OutputHub.
The residual second stage reuses the one-column HC1/CR1 covariance kernel, for which the
linear score and influence-function formulas coincide exactly. Zero or numerically weak
residual treatment variation refuses before inference rather than receiving ridge repair.

The public honest R-learner reuses the same provider-neutral boundary. Its
`WeightedCATEEstimatorProtocol` requires genuine
`sample_weight` support and a separate `CATEResultProtocol` prediction surface. Internal
native prerequisites provide stratified training-only-CV penalized Logit probabilities
and weighted ridge-GCV for the algebraically exact R-loss transformation. Both retain
training indices and tuning diagnostics.

`NativeSplineRidgeCATE` extends only the weighted-CATE stage. It builds bounded additive
linear-spline candidates from construction-only quantiles and selects knot count plus
ridge penalty by weighted GCV. The zero-knot candidate is always available by default,
pairwise interactions require explicit opt-in, and a hard basis-size ceiling prevents
quadratic feature growth from becoming an implicit runtime path. It remains a specialized
R-learner component rather than a general regression API.

The internal construction layer deterministically assigns treatment-stratified rows or
whole clusters to construction and evaluation, and clustered outer folds never split a
cluster. Role state is retained in immutable tuple storage and exposed only through copies.
Only construction data enter cross-fitted nuisance and weighted-CATE fitting; fresh
full-construction nuisance refits and the construction-fitted CATE model provide evaluation
predictions. Direct and transformed weighted R-objectives are retained as an exact identity.
`RLearner` then consumes the locked evaluation role once for R-loss, differential
calibration, and tie-preserving overlap-weighted group moments. Calibration uses the
shared HC1/CR1 OLS covariance kernel; group max-t draws use the corresponding observation-
or cluster-summed influence matrix. `RLearnerResult` owns OutputHub tables, exact graph
data, optional plotting, and future-data prediction through the construction-fitted CATE
model. Unit-level intervals, repeated-split aggregation, RATE, and policy evaluation stay
outside this alpha.

The honest DR learner reuses role assignment, shared cluster-preserving outer folds,
fresh-state auditing, covariance, tie grouping, multiplier bands, and result/reporting
conventions, while keeping a separate statistical objective. Two masked `CrossFitter`
tasks fit control and treated outcome regressions only on their arms; a third task fits the
propensity on the same fold plan. Construction OOF predictions form the augmented
inverse-probability score. The public `CATEEstimatorProtocol` then fits an unweighted
construction-only regression, distinct from the R-learner's weighted protocol. Fresh
full-construction nuisance refits and the construction CATE model predict evaluation rows.
`DRLearner` consumes evaluation outcomes/treatments once for fixed score loss, calibration,
and group inference and never feeds them back into selection.

The historical `limiteddepkit.TreatmentEffect` migration is complete. `IV2SLS` owns the
replacement and a maintained numerical migration contract; the obsolete source snapshot
has been removed from LimitedDepKit. Migration details belong in the README and package-
scope guide, not in compatibility shims that preserve an ambiguous full-`Z` API.

## Dependencies

The core runtime is deliberately small:

- NumPy for linear algebra and array representation;
- pandas for labelled inputs and results; and
- SciPy for probability distributions and inferential tails.

OutputHub support is an optional integration. `linearmodels` and `statsmodels` are optional
validation comparators, not estimation backends. The public estimator must behave without
them installed.

## Extension rules

New estimators should reuse, where substantively valid:

- normalization and strict alignment;
- schema and missing-data validation;
- labelled result and inference conventions;
- covariance metadata;
- diagnostics and reporting containers;
- optional integration boundaries; and
- testing markers and evidence records.

Reuse must not erase estimator-specific assumptions. IPW/AIPW needs nuisance-model and
influence-function contracts; DiD needs treatment-timing and comparison-cohort contracts;
RDD needs bandwidth and local-polynomial contracts; panel IV needs entity/time indexing and
within-panel covariance rules. A common result shape is useful only when its fields retain
the same meaning.

For eventual panel work, review applicable `systemgmmkit` infrastructure before creating
new entity/time validation, fixed effects, clustered covariance, diagnostics,
post-estimation, plotting, or OutputHub paths. DADPLM and BDCPM remain outside the current
architecture.

## Performance and security invariants

- Do not materialize an `n x n` projection matrix when equivalent factorized operations are
  available.
- Do not materialize a treated-by-control matching distance matrix for scalar scores; sort
  once and inspect local neighbor groups.
- For estimated-score matching, use sorted scalar searches for local score moments and a
  `cKDTree` for ATT/ATC covariate-neighbor derivatives; do not allocate pairwise matrices.
- Avoid unnecessary copies of large designs and residual arrays.
- Aggregate cluster scores in one pass over normalized cluster codes.
- Keep DML nuisance selection inside each outer training fold and pool only aligned
  out-of-fold scores; never tune against an outer held-out row.
- Keep nuisance diagnostics scalar and fold-labelled; do not expose raw training rows or
  fitted provider objects through the result surface.
- Fail before expensive factorization when shapes, missing data, or exact rank make the
  model invalid.
- Never log raw observations, instrument values, or cluster identifiers by default.
- Keep core estimation offline and deterministic for fixed numeric inputs.
- Treat optional adapters as trust boundaries and avoid hidden network or file writes.

Any performance optimization must preserve parameter labels, sample alignment, numerical
tolerances, and refusal behavior. Benchmarks should record sample dimensions, instrument
count, platform, versions, and peak memory as well as elapsed time.
