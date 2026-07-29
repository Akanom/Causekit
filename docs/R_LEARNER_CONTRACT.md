# Honest R-learner contract

Status: public alpha implemented. `RLearner` and `RLearnerResult` provide honest role
splitting, cross-fitted construction, held-out R-loss, differential calibration,
tie-preserving group inference, simultaneous bands, graph data/optional plotting,
OutputHub integration, simulation evidence, base-R parity, and one real-data CATE
comparison. RATE, policy value, unit-level intervals, deployment refitting, and
repeated-split aggregation remain deliberately out of scope. The reviewed fixed-evaluation
Stata/IC 17 harness passes at the declared `1e-8` tolerance.

This contract defines what CauseKit identifies, fits, evaluates, exposes, and refuses for
heterogeneous-effect learning. It deliberately separates CATE prediction
from evidence that a learned ranking or calibration is useful. Training fit, ordinary
prediction error, or a visually wide CATE distribution is not evidence of treatment-effect
heterogeneity.

## Differentiation without false novelty

The R-loss itself is not a CauseKit invention; it follows Nie and Wager. Cross-fitted
nuisances and configurable final learners are also available in
[EconML's R-learner machinery](https://www.pywhy.org/EconML/_autosummary/econml.dml._rlearner.html),
and honest-prediction calibration is available for causal forests through
[`grf::test_calibration`](https://grf-labs.github.io/grf/reference/test_calibration.html).
CauseKit therefore does not describe the estimator as a new statistical model or claim
that no other package has any individual component.

CauseKit's differentiation is the integrated public contract: a dependency-free native
path; immutable row or cluster construction/evaluation roles; cluster-preserving outer
folds; provider-neutral nuisance and weighted-CATE boundaries; copy-out tuning and
leakage audit records; held-out R-loss against a construction-fitted constant; HC1 or CR1
differential calibration; tie-preserving overlap-weighted groups; and seeded simultaneous
group bands. This combination is a package engineering and validation contribution. Any
stronger priority or uniqueness claim requires a versioned systematic comparison and is
not made here.

## Target and identification

The first R-learner alpha is limited to a binary treatment `W` coded exactly `0/1` and the
conditional average treatment effect

```text
tau(x) = E[Y(1) - Y(0) | X = x].
```

Its causal interpretation requires consistency, no interference, conditional
exchangeability given the declared pre-treatment covariates, and overlap. These assumptions
are not learned from the data. Clustered samples also require independent sampling across
declared clusters for the supported cluster-level inference.

For

```text
m(x) = E[Y | X = x]
e(x) = P(W = 1 | X = x)
u_i  = Y_i - m_hat(X_i)
v_i  = W_i - e_hat(X_i),
```

the construction sample fits `tau` by minimizing the R-loss

```text
sum_i [u_i - v_i tau(X_i)]^2 + regularization(tau).
```

This follows the residualized objective in Nie and Wager's
[R-learner paper](https://arxiv.org/abs/1712.04912). CauseKit will not relabel an S-, T-,
or X-learner as an R-learner merely because both produce unit-level predictions.

## Honest sample roles

A single observation or cluster may not serve both model-selection and evaluation roles.
The public API makes these roles machine-readable:

1. **Construction sample:** tunes and fits the outcome, propensity, and CATE learners.
   Nuisance predictions used to fit the CATE learner are cross-fitted within this sample.
2. **Evaluation sample:** receives predictions from models that never used its outcomes or
   treatments. It is used once for the declared R-loss and calibration diagnostics.
3. **Deployment refit:** an optional model refit after evaluation may use all observations
   for future prediction. Its predictions must never replace the honest evaluation records.

The construction/evaluation split is randomized with a recorded seed and stratified by
treatment. If clusters are declared, entire clusters—not rows—are assigned to roles and
outer folds. The splitter must refuse a requested split that leaves too few observations,
arms, clusters, or unique CATE values for a declared diagnostic. It may not silently change
the evaluation fraction, seed, fold count, or number of calibration groups.

Repeated splitting is not part of the first default. A single-split result is explicitly
labelled split-conditional. If repeated-split aggregation is later promoted, its interval
and p-value aggregation must be separately contracted; the repeated-splitting approach in
[Chernozhukov, Demirer, Duflo, and Fernández-Val](https://www.nber.org/papers/w24678)
is the relevant benchmark, not an ad hoc average of CATE predictions.

### Implemented role and construction boundary

The internal construction layer now creates one deterministic randomized split for a
recorded seed. Without clusters, it rounds the requested evaluation count separately
within each exact `0/1` treatment arm. With clusters, a randomized greedy allocation
balances treatment-arm counts and requested role sizes while assigning every cluster as
one indivisible unit. The result retains both requested and realized evaluation fractions;
they can differ under cluster splitting. Any allocation that cannot support both arms,
the requested outer folds, native inner propensity tuning, or at least two evaluation
clusters refuses instead of modifying the design.

Row and cluster roles are stored as immutable tuples behind copy-out accessors. Clustered
outer folds use the same whole-cluster rule, and every fold must retain both treatment
arms. Outcome and propensity nuisances use the identical treatment-stratified
`CrossFitter` plan inside the construction sample. Factory identity is audited across all
outer-fold fits and the fresh full-construction refits used for evaluation prediction.

Only construction outcomes and treatments enter nuisance or CATE fitting. The fresh
full-construction nuisance refits and the construction-fitted weighted CATE model produce
evaluation predictions without using evaluation outcomes or treatments as fit targets.
The result retains `u`, `v`, `u/v`, `v^2`, the direct R-objective, and its algebraically
identical weighted-transformation value. The construction audit is retained inside the
public result and never recomputed from evaluation outcomes.

## Learner protocols

Outcome and propensity nuisances continue to use fresh public CrossFitter factories. The
propensity result must provide `predict_proba(X)` and every honest prediction must lie
inside the declared overlap interval `[overlap_floor, 1 - overlap_floor]`. CauseKit refuses
violations; it does not clip them and call the same target identified.

The CATE factory must create fresh estimators supporting

```text
fit(X, pseudo_outcome, sample_weight=weight)
predict(X)
```

where, for binary treatment under enforced overlap,

```text
pseudo_outcome_i = u_i / v_i
weight_i         = v_i^2.
```

This weighted regression is algebraically identical to the direct R-loss. The fitted
result must return one finite prediction per row and preserve labelled schema. CauseKit
owns the objective, weights, split roles, evaluation, and refusal rules even when a custom
prediction backend is supplied.

A package-owned weighted ridge-GCV CATE learner is the first default. Its native
probability learner has a separately tested penalized-logit contract, so the full path is
independent of external ML packages. Reusing unconstrained
linear ridge probabilities without refusal, silently clipping them, or importing another
package's model as the default are not acceptable shortcuts.

### Opt-in native nonlinear CATE stage

`NativeSplineRidgeCATE` is CauseKit's first specialized nonlinear weighted-CATE stage. It
standardizes the declared construction covariates, forms additive continuous linear-spline
bases at construction-only empirical quantiles, and uses the same weighted R-objective GCV
to select jointly among the requested knot counts and ridge penalties. The default
candidate set `(0, 1, 3)` deliberately includes the zero-knot linear basis, so the learner
can decline nonlinear complexity without consulting honest evaluation outcomes.

The result exposes the exact prediction basis, selected knot count, basis dimension,
penalty, weighted GCV, and complete candidate tuning path. Pairwise standardized linear
interactions are explicit opt-in. Every requested basis must remain below
`max_basis_features`; CauseKit refuses oversized bases rather than constructing an
unbounded polynomial or pairwise expansion.

This is a package-owned nonlinear implementation, but piecewise-linear splines and ridge
GCV are established methods. CauseKit's contribution is their leakage-audited integration
with the weighted R-objective and honest evaluation contract, not a claim that the basis
family is mathematically novel. Seeded nonlinear recovery passes. On real data, the model
is worse than linear ridge on the NSW split and safely matches it on the Hillstrom email
RCT, so it remains opt-in.

### Implemented prerequisite boundary

CauseKit now exposes the provider-neutral `WeightedCATEEstimatorProtocol` and
`CATEResultProtocol`. A custom CATE factory must create an estimator whose bound method
accepts `fit(X, pseudo_outcome, sample_weight=weight)`; the fitted result must provide
`predict(X)`. CauseKit validates positive finite aligned weights, prediction length,
finiteness, and pandas evaluation-index preservation. A method that merely has `fit` but
does not accept `sample_weight` refuses before fitting.

The package-owned probability prerequisite is standardized ridge-penalized binary Logit.
For every candidate `alpha`, it minimizes

```text
sum_i [log(1 + exp(eta_i)) - W_i eta_i] + alpha ||beta||^2 / 2,
```

with an unpenalized intercept. Selection uses mean held-out log loss from deterministic,
stratified inner folds created only from the supplied training sample. Scaling is refitted
inside each inner training fold. The selected model is then refitted on that supplied
training sample and retains its training index and inner-fold assignments for the
honesty audit. It requires both exact `0/1` arms and enough observations in each arm for
every inner fold. It returns mathematical Logit probabilities without clipping; the
public R-learner enforces its declared overlap interval.

The package-owned CATE prerequisite is weighted standardized ridge with an unpenalized
weighted intercept and GCV-selected penalty. It fits the declared `u/v` pseudo-outcome
using exactly `v^2` weights, so its weighted residual sum of squares equals
`sum_i [u_i - v_i tau(X_i)]^2`. It retains training indices and scalar tuning diagnostics.
Both native estimators remain internal components, not general regression APIs. Only the
structural protocols are public until the complete honest estimator is promoted.

## Honest evaluation metrics

Every metric below uses only the evaluation sample and nuisance/CATE predictions trained
without that sample. Hyperparameters may not be revised after viewing these values.

### R-loss and constant-effect baseline

The primary model-selection-free score is

```text
honest_r_loss = mean_i [u_i - v_i tau_hat(X_i)]^2.
```

The comparator is a constant effect fitted on the construction sample, not on evaluation
outcomes. Report

```text
r_loss_gain = 1 - honest_r_loss / honest_constant_r_loss.
```

The gain may be negative and is not ordinary predictive `R^2`. Lower R-loss is preferable,
but it does not by itself prove identification or validate unit-level causal effects.

### Differential calibration

Let the fixed evaluation proxy be `s_i = tau_hat(X_i)` and define its residual-treatment-
squared weighted center

```text
s_bar = sum_i(v_i^2 s_i) / sum_i(v_i^2).
```

On the evaluation sample, fit the no-intercept calibration regression

```text
u_i = beta_level v_i + beta_heterogeneity v_i (s_i - s_bar) + error_i.
```

Report both coefficients, their joint covariance, confidence intervals, and tests of
`beta_heterogeneity = 0` and `beta_heterogeneity = 1`. A nonzero differential coefficient
indicates that the held-out proxy captures effect variation in its ranking/direction; a
coefficient near one is a scale-calibration check. Neither test establishes pointwise CATE
truth. Robust HC1 or declared one-way CR1 inference must reuse CauseKit covariance
conventions. A constant or rank-deficient proxy makes differential calibration unavailable
and must be reported, not regularized into existence.

### Honest calibration groups

For a declared number of groups, assign evaluation observations by the fixed CATE proxy.
Equal predictions stay together. CauseKit refuses fewer unique score groups than requested
and reports realized group sizes and treatment counts. Within group `g`, report the
overlap-weighted residual-moment effect

```text
tau_g = sum_{i in g}(v_i u_i) / sum_{i in g}(v_i^2).
```

This is an overlap-weighted group effect, not automatically the ordinary group ATE. Group
influence values must support pointwise HC1/CR1 intervals and seeded max-t simultaneous
bands over the complete calibration path. No unit-level CATE confidence intervals are
promised by this contract.

### Ranking and policy metrics

RATE/AUTOC-style prioritization metrics are valuable but are not part of the first
R-learner alpha. Their inference requires an independent, estimand-aligned score—normally a
doubly robust score—and a separate weighting/uncertainty contract. The reference is the
[RATE framework](https://doi.org/10.1080/01621459.2024.2393466). CauseKit will not report
an in-sample Qini curve from raw outcomes as causal validation.

## Graphing contract

Graphs are added only with the evaluation surface, never as training-set decoration:

- **Calibration plot:** honest group mean predicted CATE on the horizontal axis and the
  overlap-weighted group effect on the vertical axis, with simultaneous bands and a
  clearly labelled 45-degree calibration reference.
- **CATE distribution:** distribution of honest evaluation predictions with the sample
  role, split seed, and no unit-level uncertainty claim in metadata.
- **Targeting plot:** deferred until the RATE/DR-score contract is implemented.

Every graph returns its underlying labelled table and records construction/evaluation
counts, fold/split seed, covariance, group/tie rule, and whether inference is
split-conditional.

## Result and audit surface

The result exposes, without recomputation:

- construction, evaluation, and optional deployment indices/roles;
- nuisance and CATE fold assignments and model names;
- fold-level tuning diagnostics available through `NuisanceDiagnosticsProtocol`;
- honest CATE predictions, residuals, propensities, overlap diagnostics, and weights;
- honest and constant R-loss plus the gain;
- calibration coefficients, covariance, tests, and availability status;
- calibration-group tables, influence values, and simultaneous-band configuration;
- assumptions, warnings, unavailable diagnostics, and exact refusal reasons; and
- OutputHub model metadata and the same calibration/tuning tables used by graphs.

## Required refusals

The implementation must refuse:

- nonbinary treatment, missing/nonfinite inputs, duplicate covariate columns, or index drift;
- post-treatment covariates when explicitly marked by the formula/data-role layer;
- insufficient treatment-arm or cluster capacity in either sample role or any outer fold;
- nonfresh nuisance/CATE factories, unsupported weighted fitting, or malformed predictions;
- propensity predictions outside the declared overlap interval, without clipping;
- zero or numerically weak residual-treatment variation;
- evaluation rows or clusters used in any fitting/tuning operation;
- calibration with a constant/rank-deficient CATE proxy;
- requested calibration groups that cannot preserve score ties and minimum arm counts; and
- unit-level interval, policy-value, RATE, or ordinary-ATE claims not implemented by the
  declared moments.

## Validation and promotion evidence

Implementation started with observed-failing contracts. The public alpha now has:

1. a hand-computed weighted R-objective and prediction fixture;
2. deterministic role/fold alignment and explicit leakage sentinels;
3. hand-computed honest R-loss, constant baseline, and calibration regression;
4. robust and clustered covariance plus simultaneous group-band identities;
5. all required refusal tests;
6. seeded linear and piecewise-nonlinear recovery plus null/power and simultaneous-band
   coverage smoke tests; publication-scale Monte Carlo remains release hardening;
7. independent base-R parity for fixed honest loss/calibration/group moments and reviewed
   Stata/IC 17 HC1 parity; Python hand contracts cover the seeded max-t algorithm;
8. NSW and Hillstrom real-data honest evaluations against constant-effect and aligned CATE
   comparators, while retaining the existing scalar-DML benchmark as historical evidence
   rather than rerunning it as if it were a CATE comparison;
9. a 2,000-row/500-cluster performance smoke without pairwise matrices or observation
   loops; and
10. public documentation, OutputHub tables, graph-data parity, build, security, and package
    quality gates. Exact executed status is recorded in the release handover.

## Pre-mortem

| Failure mode | Consequence | Prevention |
| --- | --- | --- |
| Evaluation leakage | Optimistic heterogeneity evidence | Immutable row/cluster roles and leakage sentinels |
| Propensity clipping | Hidden target/design change | Declared overlap floor and strict refusal |
| Flexible in-sample CATE spread | Noise presented as heterogeneity | Honest R-loss and differential calibration |
| Treating scalar DML as a CATE baseline | Invalid model ranking | Constant R-loss and aligned CATE comparators |
| Pointwise intervals from generic ML | Unsupported precision claims | Group/calibration inference only in the first alpha |
| Split-sensitive conclusions | Unstable substantive claims | Recorded split-conditional status; later contracted aggregation |
| Attractive but invalid targeting graph | Policy claim without score inference | Defer targeting plot until RATE/DR-score promotion |

## Implementation order

Completed prerequisite milestone:

1. Promoted the weighted-fit/CATE prediction protocols and native probability contract.
2. Added hand-computed weighted-objective/penalized-score, construction-only audit, and
   malformed-input/provider refusal tests.

Completed construction milestone:

1. Added deterministic treatment-stratified row roles and whole-cluster roles with
   immutable copy-out accessors.
2. Added cluster-preserving outer folds, fresh-factory auditing, cross-fitted construction
   nuisances, exact R-objective construction, and construction-only evaluation predictions.

Completed public evaluation milestone:

1. Added honest R-loss, a construction-fitted constant baseline, and differential
   calibration with HC1/CR1 inference.
2. Added tie-preserving overlap-weighted group effects, influence records, seeded max-t
   bands, graph-data parity, optional plots, OutputHub, and future-data prediction.
3. Added seeded linear/nonlinear simulations, base-R and reviewed Stata parity,
   performance evidence, and two hash-pinned real-data comparisons.

Later enhancements require new contracts rather than changes to this result:

1. repeated-split aggregation and its p-value/interval rules;
2. RATE/DR-score ranking inference and policy evaluation; and
3. unit-level CATE uncertainty only if a learner-specific valid method is implemented.
