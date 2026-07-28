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
10. the exact `causalkit` version and reproduction command.

Use phrases such as “the 2SLS coefficient is consistent under the stated relevance,
independence, and exclusion assumptions,” not “the package proves a causal effect.”
