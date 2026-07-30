# Identification and interpretation

## Randomized experiments

`RandomizedATE` targets the average treatment contrast in the retained analysis sample.
Its causal interpretation requires a genuine randomized assignment mechanism, treatment
consistency, no interference, an analysis population fixed independently of outcomes,
and inference that matches assignment and outcome dependence. Covariates used for Lin
adjustment must be pre-treatment. The fully interacted specification avoids imposing a
common covariate slope across treatment arms; centering makes the treatment coefficient
the sample-covariate-average adjusted contrast.

The estimator cannot diagnose whether assignment was actually randomized. Covariate
balance is descriptive and should not be used as a pass/fail test of randomization or as
an outcome-driven model-selection rule. Clustered covariance permits within-cluster
outcome dependence but does not, by itself, implement a cluster-randomized estimand or
few-cluster randomization inference.

## Observational IPW and AIPW

`IPWATE` and `AIPWATE` target the population ATE represented by the analysis sample.
Identification requires consistency, no interference, conditional exchangeability given
the pre-treatment information used by the nuisance functions, and positivity. These are
substantive assumptions; fitted propensity support cannot detect omitted confounders.

Supplied nuisance predictions preserve provider-neutral ownership boundaries and make
data leakage auditable. Adaptive nuisance fits should be cross-fitted. The AIPW score has
the usual double-robust property only under regularity conditions and valid inference also
depends on nuisance convergence rates or justified low-complexity fitting. Clipping limits
extreme weights but changes the score; report unclipped and alternative-bound sensitivity
results rather than treating clipping as an invisible numerical repair.

With `estimand="att"`, the target population is the treated group; controls are weighted
by propensity odds. With `estimand="atc"`, the target is the control group; treated units
are weighted by inverse propensity odds. These estimands can differ materially under
effect heterogeneity. The software therefore labels the result with the requested target
rather than treating ATT/ATC as aliases for ATE.

This guide states what `IV2SLS` estimates, the conditions under which it is identified,
and the limits of its diagnostics. It is not a substitute for a design-specific argument.

## Model and instrument contract

For observation `i`, the structural equation is

```text
y_i = x_i' beta + d_i' gamma + u_i,
```

where `x_i` contains exogenous controls, `d_i` contains endogenous regressors, and `u_i`
is the structural disturbance. Let `z_i` contain excluded instruments. With an intercept,
the structural design and full instrument design are

```text
S = [1, X, D]
Q = [1, X, Z].
```

`IV2SLS.fit(..., instruments=Z, exogenous=X)` accepts **excluded instruments only**. It
adds the intercept and exogenous regressors to `Q` internally. Repeating `X` in
`instruments` changes the intended design and is not the supported API.

With `P_Q = Q (Q'Q)^{-1} Q'`, the 2SLS point estimate is

```text
theta_hat = (S' P_Q S)^{-1} S' P_Q y,
```

where `theta = (beta, gamma)` and reported parameter order is constant, supplied
exogenous columns, then endogenous columns. The implementation uses stable linear-algebra
operations rather than requiring users to construct `P_Q` explicitly.

`add_constant=True` is the default. Set `add_constant=False` for a no-intercept model or
when the supplied exogenous matrix already contains the intended constant. There is no
formula API in this release, so categorical encodings, transformations, and interactions
must be created and archived by the analyst.

## Identification conditions

The matrix calculation requires more than matching dimensions.

### Order condition

If there are `m` endogenous regressors and `q` excluded instruments, then `q >= m` is
necessary. Equality gives a just-identified model; `q > m` gives an overidentified model.
The order condition is not sufficient.

### Rank condition

After removing the linear contribution of the exogenous design, the excluded instruments
must span `m` independent directions in the endogenous regressors. Informally,

```text
rank(Z' M_X D) = m,
```

with the intercept included in `X` when requested and `M_X` denoting residualization with
respect to that exogenous design. Perfectly collinear instruments, irrelevant instruments,
or endogenous regressors without distinct excluded variation violate this condition.

The estimator rejects exact rank failure. Near-rank failure may still produce unstable
estimates and weak-instrument behavior; numerical invertibility is not substantive
identification.

### Instrument relevance

Excluded instruments must predict the endogenous regressors conditional on the exogenous
controls. First-stage partial R-squared and joint-exclusion statistics describe sample
evidence about this condition. They do not guarantee adequate finite-sample behavior.

### Independence and exclusion

The identifying moment condition requires the full instrument set to be orthogonal to the
structural disturbance in the target population. An excluded instrument must affect the
outcome only through the modeled endogenous pathway, apart from effects already controlled
by `X`. These are design assumptions. They cannot be proved from a first-stage F statistic
or from model fit.

### Treatment definition, interference, and population

A causal analysis must specify the treatment version, outcome, timing, unit, population,
and interference assumptions. If one unit's instrument or treatment changes another
unit's outcome, ordinary observation-level IV may not represent the desired estimand.
Clustering changes the covariance calculation; it does not resolve interference or redefine
the unit of treatment.

### Effect heterogeneity

In a constant-coefficient structural model, `gamma` is interpreted under that model and
the IV moment restrictions. With heterogeneous treatment effects, the coefficient need not
be the population average treatment effect. For binary instruments and treatments, a local
average treatment effect interpretation additionally relies on conditions such as
monotonicity and a clearly defined complier population. With multiple or continuous
instruments, the weighting and target require separate justification.

## First-stage diagnostics

The result provides one first-stage diagnostic for each endogenous regressor. Each
diagnostic reports:

- `r_squared` from the unrestricted first-stage regression on the full instrument set;
- `partial_r_squared` for the excluded instruments after conditioning on exogenous
  regressors;
- `classical_f_statistic`, its numerator and denominator degrees of freedom, and p-value;
- `excluded_instrument_statistic`, degrees of freedom, p-value, and reference distribution
  using the selected covariance contract; and
- `weak_instrument_warning`, a deliberately limited screening flag.

The classical F statistic uses the homoskedastic first-stage comparison. It remains useful
as a familiar descriptive statistic, but it is not made heteroskedasticity- or
cluster-robust by fitting the structural equation with a different covariance.

The covariance-aware joint statistic tests that the coefficients on the excluded
instruments are jointly zero using `"unadjusted"`, HC1, or one-way CR1 first-stage
covariance as requested. Read the diagnostic's recorded distribution and degrees of
freedom rather than assuming every reported value is the same kind of F statistic.

Common rules of thumb based on a single first-stage F were derived for narrower settings
than many empirical applications. They can fail with heteroskedasticity, clustering,
multiple endogenous regressors, many instruments, or nonstandard estimands. A warning is
not a formal weak-IV-robust test, and absence of a warning is not a certificate of strong
identification.

## Overidentification

When `q > m` and `covariance="unadjusted"`, `result.overidentification` contains a Sargan
test of the overidentifying restrictions. The reference chi-squared degrees of freedom are
`q - m` under the homoskedastic null and the maintained IV assumptions.

Sargan is not reported for:

- a just-identified model, because there are no overidentifying restrictions to test;
- `covariance="robust"`, because the homoskedastic statistic is not a robust J test; or
- `covariance="clustered"`, for the same reason and because cluster dependence changes the
  relevant moment covariance.

An absent diagnostic is therefore `not applicable` or `not implemented`, not `passed`.
A large p-value is also not proof that every instrument is valid: the test evaluates the
restrictions jointly, may have low power, and depends on at least one valid identifying
path. Instrument validity still requires substantive evidence.

## Covariance and reference distributions

All three covariance choices use the same 2SLS point estimate but answer different
sampling questions.

### Unadjusted

`covariance="unadjusted"` assumes homoskedastic structural disturbances and uses the
residual variance with residual degrees of freedom. Coefficient tests and confidence
intervals use a Student t reference with `n - k` degrees of freedom, where `k` is the
number of structural parameters.

### Robust HC1

`covariance="robust"` forms the observation-level sandwich meat from structural residual
scores and applies the HC1 multiplier `n / (n - k)`. Inference uses the asymptotic normal
reference. HC1 permits conditional heteroskedasticity but still relies on independent
observations and a large-sample approximation.

### One-way clustered CR1

`covariance="clustered"` sums structural residual scores within each of `G` clusters and
applies the CR1 multiplier

```text
[G / (G - 1)] * [(n - 1) / (n - k)].
```

Coefficient inference uses Student t with `G - 1` degrees of freedom. This permits
arbitrary within-cluster score dependence under independence across clusters. CR1 can be
unreliable with few clusters, severe imbalance, or influential clusters; the package does
not claim a few-cluster-robust correction. Cluster labels must be observed and aligned for
every fitted row.

## Matching identification boundary

`NearestNeighborMatch` defines ATT, ATC, or ATE through observed-minus-imputed potential
outcome contrasts. A causal interpretation requires consistency, no interference,
conditional exchangeability given the measured pre-treatment covariates used to construct
the supplied propensity, and positivity in the realized target population. The package
cannot verify that the propensity model included all confounders or that its predictions
are correctly specified.

Support and caliper exclusions can change the target population. The result therefore
separates `requested_estimand` from `realized_estimand`, exposes excluded indices and arm
counts, and uses a `_matched_support` label when focal observations are removed. Better
observed balance is a design diagnostic, not evidence that unmeasured confounding was
eliminated. Caliper sensitivity and covariate-level balance should be reported together
with focal attrition and comparison reuse.

`inference="none"` is not a claim that a matching design is uncertainty-free. The
maintained `inference="abadie_imbens"` path applies only to a declared known/fixed scalar
score, without caliper/support selection or expanded ties, and accounts for comparison
reuse through same-arm conditional-variance matches. The separate
`inference="abadie_imbens_estimated"` path applies only to a validated regular full-sample
unpenalized Logit MLE on the matching sample. It incorporates the model-specific first-step
variance adjustment; for ATE that adjustment is nonpositive asymptotically, while for
ATT/ATC it can have either sign because the target itself depends on the propensity
parameter. A fitted or cross-fitted propensity is not eligible merely because predictions
were supplied: unverifiable, penalized, cross-fitted, and unsupported-link scores continue
to require `inference="none"`. Ordinary bootstrap is not substituted because fixed-neighbor
matching is nonsmooth.

## Difference-in-differences and event studies

`DifferenceInDifferences` targets cohort-time average treatment effects on the treated,
`ATT(g,t)`, and cohort-share-weighted event/calendar summaries. Identification requires
an absorbing treatment definition, a clean baseline, no anticipation outside the declared
window, overlap with the chosen comparisons, consistency, no interference, and parallel
untreated trends for each reported comparison. The estimator cannot establish these
conditions from fitted pre-period outcomes.

With `control_group="never_treated"`, the conventional estimator compares each treated
cohort's change from the period immediately before its effective treatment boundary with
the same change among never-treated entities. With `control_group="not_yet_treated"`, it
also uses entities whose effective treatment boundary is after the target time. An entity
is never retained as a control after its anticipation window begins. These choices can
target the same `ATT(g,t)` under their respective assumptions but need not be equally
credible in a given application.

`EfficientDiD` imposes the stronger PT-All condition used by Chen, Sant'Anna, and Xie
(2025): conditional mean untreated trends are common across every retained period and
cohort, optionally conditional on the declared baseline covariates. This overidentifies
each `ATT(g,t)`, allowing the estimator to combine admissible pre-period/auxiliary-cohort
moments using unconditional or observation-specific inverse covariance. The resulting weights may be negative without creating the
heterogeneous-effect contamination associated with TWFE weights, because every weighted
moment identifies the same cohort-time effect under PT-All. If PT-All is false, the
precision claim and possibly consistency fail; smaller standard errors are not evidence
that the stronger restriction is true.

The event-study table uses observed cohort shares among cohorts available at each event
time. Calendar summaries use shares among adopted cohorts and exclude declared
anticipation periods. `ESavg` is the simple average of non-anticipation event-time
effects. These are explicit target populations, not interchangeable labels.

Analytic intervals are pointwise. Robust inference treats the panel entity as the random
sampling unit; higher-level clustered inference assumes independent clusters and many
clusters. The optional multiplier max-t path supplies a simultaneous band across the
reported event-study coordinates, with multipliers drawn at the declared sampling level.

The no-covariate panel results expose adjacent, uncontaminated cohort-period pre-trend
placebos and a joint test. A two-period design can have no testable placebo, and a singular
joint covariance is reported as unavailable without repairing or dropping moments.
Failure to reject is not evidence that parallel trends holds. The separate
`did_hausman_test` compares aligned no-covariate PT-All and PT-Post post-treatment
event-study vectors using the influence function of their difference. Rejection weighs
against the additional PT-All restrictions; non-rejection does not prove them or justify
mechanical estimator selection.

Neither panel class has a repeated-cross-section interpretation. The separate
`RepeatedCrossSectionDiD` first slice targets the same cohort-time ATT logic using four
independently sampled means: treated target minus treated baseline, less fixed-comparison
target minus fixed-comparison baseline. It requires stationary composition of the
relevant cohort populations across samples, enough observations in every used cell, and
the same consistency, no-interference, overlap, no-anticipation, and repeated-cross-
section parallel-trends conditions. Stationarity is recorded as an assumption, not
inferred from cell counts or a pre-trend test. Observation-level HC1 or declared-PSU CR1
scores replace entity-level panel changes. Composition-change-robust and covariate-
adjusted scores are not approximated by the no-covariate estimator.

## Honest heterogeneous effects

`RLearner` targets `tau(x) = E[Y(1)-Y(0) | X=x]` for an exact binary treatment. A causal
interpretation requires consistency, no interference, conditional exchangeability given
the declared pre-treatment covariates, overlap, and adequate nuisance rates. These are
assumptions, not conclusions from a flexible learner or a wide CATE distribution.

CauseKit assigns observations—or whole declared clusters—to immutable construction and
evaluation roles. Only construction outcomes and treatments enter nuisance or CATE
fitting. Evaluation is used once for residual R-loss, differential calibration, and
tie-preserving groups. The R-loss gain compares against a constant effect fitted on
construction and is not predictive R-squared. The differential heterogeneity coefficient
tests whether the held-out proxy contains effect-ranking signal and whether its scale is
near one; it does not establish pointwise CATE truth.

Group effects solve `sum(v_i u_i) / sum(v_i^2)` inside fixed score groups. They are overlap-
weighted residual-moment effects, not automatically ordinary group ATEs. HC1 or one-way
CR1 inference is conditional on the recorded split, and the multiplier max-t band covers
the complete reported group path under the declared sampling assumptions. The alpha does
not provide unit-level CATE intervals, repeated-split aggregation, RATE, targeting curves,
or policy value.

`DRLearner` targets the same binary-treatment CATE through the augmented
inverse-probability score. Its conditional score mean identifies `tau(x)` when the
propensity is correct or both treatment-arm outcome regressions are correct, subject to
the same consistency, no-interference, conditional-exchangeability, overlap, and
nuisance-rate conditions. “Doubly robust” does not cover unmeasured confounding, overlap
failure, a single correct arm regression, or arbitrary final-stage approximation error.

Construction-only cross-fitting and full-construction refits preserve the immutable honest
boundary. Evaluation DR-score loss is a noisy model-comparison signal, not observed
unit-level effect error. Calibration regresses the fixed evaluation score on an intercept
and centered CATE prediction; tie-preserving groups average the DR score and therefore
target score-defined group ATEs under the declared assumptions. Their HC1/CR1 and max-t
uncertainty remains split-conditional and does not imply unit-level CATE intervals.

## Missing values, indices, and sample definition

The default policy is `missing="raise"`. Missing or non-finite numeric values trigger an
error. The explicit `missing="drop"` policy constructs one complete-case mask across all
estimation inputs, drops those rows jointly, preserves the retained index, and records the
count in `result.dropped_rows`. Prediction inputs are never dropped automatically.

All pandas inputs must have exactly matching row-index values in the same order; the
estimator does not sort, intersect, or silently reindex them. An index is not required to
be unique, but every labelled input must carry the same sequence. Unlabelled array-like
inputs are treated positionally and must have the same row count. Column names must be
unique after string conversion, role names must not overlap, and the fitted schema is
preserved.

These checks protect the estimand. If the outcome, endogenous regressor, instrument,
controls, and cluster labels refer to different rows, the moment conditions no longer
describe the proposed sample. Prefer constructing and auditing the analysis sample before
fitting; if `missing="drop"` is used, report the joint exclusion count and retained index.
Perform imputation explicitly before fitting and reuse the same index across all inputs.

## What to report

At minimum, an IV analysis should report:

1. the causal question, treatment, outcome, unit, timing, target population, and estimand;
2. every endogenous regressor, exogenous control, and excluded instrument;
3. why each instrument should be relevant, independent, and excluded;
4. the sample construction, missing-data policy, transformations, and intercept choice;
5. observations, clusters where applicable, and parameter/instrument counts;
6. covariance type, finite-sample correction, and inference reference distribution;
7. per-endogenous first-stage partial R-squared and both classical and covariance-aware
   joint-exclusion diagnostics;
8. the Sargan result only when its homoskedastic conditions apply, with its limitations;
9. sensitivity to instrument sets, controls, functional form, influential observations,
   and clustering choices; and
10. the exact `causekit` version and reproduction command.

Use phrases such as “the 2SLS coefficient is consistent under the stated relevance,
independence, and exclusion assumptions,” not “the package proves a causal effect.”
