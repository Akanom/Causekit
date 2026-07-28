# Architecture

`causalkit` is organized around small, auditable estimation paths. The `0.6.0a1`
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
| `causalkit.__init__` | Stable root exports and package version |
| `causalkit._data` | Input coercion, labels, exact pandas alignment, joint missing-data policy, and prediction schema |
| `causalkit._covariance` | Homoskedastic, HC1, and one-way CR1 covariance kernels and inference metadata |
| `causalkit.diagnostics` | First-stage diagnostic records and homoskedastic Sargan testing |
| `causalkit.iv` | Public `IV2SLS`, 2SLS execution path, and fitted `IV2SLSResult` |
| `causalkit.randomized` | Two-arm difference-in-means and Lin-adjusted ATE execution path, balance records, and fitted result |
| `causalkit.observational` | Supplied-nuisance IPW/AIPW scores, overlap diagnostics, vectorized influence-function and cluster inference |
| `causalkit.crossfit` | Public nuisance protocols, deterministic stratified fold orchestration, fresh-model fitting, prediction adaptation, and aligned out-of-fold records |
| `causalkit.matching` | Supplied-score ATT/ATC/ATE matching, sorted scalar neighbor search, support/caliper rules, fractional ties, weights, reuse, balance, and inference refusals |
| `causalkit.did` | Balanced-panel validation, conventional group-time DiD, PT-All efficient generated outcomes/weights, influence inference, and cohort/event/calendar aggregation |
| `causalkit.postestimation` | Summary, covariance, confidence interval, prediction, residual, fitted-value, linear-combination, and Wald helpers |
| `causalkit.integrations.outputhub` | Lazy optional conversion and insertion into Universal Output Hub |

Underscored modules are internal. Users should import estimators, result types, diagnostics,
and post-estimation helpers from `causalkit`; internal module paths may change during the
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

Reporting adapters live below `causalkit.integrations` and translate a fitted result into
the external reporting contract. `universal-output-hub` is optional; importing and fitting
the core estimator must not require it. When unavailable, the adapter should raise an
actionable installation error only when called.

Adapters may rename fields for an external schema but must not recompute estimates,
covariance, degrees of freedom, or diagnostics. The native result remains the source of
truth.

## Compatibility strategy

`systemgmmkit` and `limiteddepkit` established useful conventions for labelled econometric
results. `causalkit` follows compatible meanings for parameters, covariance, standard
errors, test statistics, p-values, observation counts, confidence intervals, prediction,
and table/report adapters.

Compatibility is structural rather than inheritance-based:

- no sibling package is a required runtime dependency;
- no private sibling module is imported;
- no result class is promised to be interchangeable where estimator semantics differ; and
- shared conventions are verified through public fields and adapter behavior.

Limited-outcome estimators are not copied into this package. `CrossFitter` targets a
documented fit/predict protocol so public `limiteddepkit` estimators can participate
optionally without becoming a core dependency. The causal procedure still owns sample
splitting, estimand construction, diagnostics, and valid uncertainty propagation.

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

Matching uncertainty is intentionally separate from generic covariance code. The point
alpha defaults to `inference="none"`; the reserved Abadie–Imbens path refuses until both
the reuse-aware conditional variance and propensity-estimation adjustment contracts are
implemented and validated. It must not reuse the IV/ATE sandwich or CR1 kernels merely to
populate standard-error fields.

The DiD path performs one long-to-wide balanced-panel validation and keeps the entity as
the sampling unit. `DifferenceInDifferences` and `EfficientDiD` share this panel bundle,
cohort-share aggregation, labelled influence-function results, and robust/clustered
inference. Their point estimators remain separate: the conventional class uses one
pre-treatment baseline with never-treated or not-yet-treated comparisons, while the
efficient class constructs the PT-All generated outcomes and solves their covariance
system. Fixed-T cohort/period loops are permitted; all entity-level arithmetic is
vectorized and cluster scores are aggregated after one factorization of the labels.

No nuisance learner lives in `did.py`. The covariate-adjusted efficient path remains
closed until it can consume public nuisance predictions/cross-fitting and implement the
paper's conditional covariance estimator. A singular weight system refuses rather than
silently applying a ridge or pseudoinverse.

The historical `limiteddepkit.TreatmentEffect` snapshot is provenance for migration, not a
code dependency. `IV2SLS` was designed around the explicit excluded-instrument contract and
new package-owned layers. Migration details belong in the README and package-scope guide,
not in compatibility shims that preserve an ambiguous full-`Z` API.

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
- Avoid unnecessary copies of large designs and residual arrays.
- Aggregate cluster scores in one pass over normalized cluster codes.
- Fail before expensive factorization when shapes, missing data, or exact rank make the
  model invalid.
- Never log raw observations, instrument values, or cluster identifiers by default.
- Keep core estimation offline and deterministic for fixed numeric inputs.
- Treat optional adapters as trust boundaries and avoid hidden network or file writes.

Any performance optimization must preserve parameter labels, sample alignment, numerical
tolerances, and refusal behavior. Benchmarks should record sample dimensions, instrument
count, platform, versions, and peak memory as well as elapsed time.
