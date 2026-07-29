# Honest R-learner contract

Status: design approved for a future alpha; no public `RLearner` implementation or
placeholder import exists yet.

This contract defines what CauseKit must identify, fit, evaluate, expose, and refuse
before adding heterogeneous-effect learning. It deliberately separates CATE prediction
from evidence that a learned ranking or calibration is useful. Training fit, ordinary
prediction error, or a visually wide CATE distribution is not evidence of treatment-effect
heterogeneity.

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
The initial API must make these roles machine-readable:

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

A package-owned weighted ridge-GCV CATE learner is the intended first default. A native
probability learner with a separately tested penalized-logit contract is a prerequisite
before the full R-learner can claim an entirely native default. Reusing unconstrained
linear ridge probabilities without refusal, silently clipping them, or importing another
package's model as the default are not acceptable shortcuts.

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

The future result must expose, without recomputation:

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

## Validation and promotion gates

Implementation starts with failing contracts and is not promoted until it has:

1. a hand-computed weighted R-objective and prediction fixture;
2. deterministic role/fold alignment and explicit leakage sentinels;
3. hand-computed honest R-loss, constant baseline, and calibration regression;
4. robust and clustered covariance plus simultaneous group-band identities;
5. all required refusal tests;
6. simulations with known constant, linear, nonlinear, null, and weak-overlap CATEs,
   reporting PEHE, ranking error, R-loss gain, calibration size/power, and band coverage;
7. aligned external parity where the same objective, split, and weighting can be fixed;
8. one real-data honest evaluation against a constant-effect baseline and appropriate CATE
   comparators, while retaining the existing scalar-DML benchmark as historical evidence
   rather than rerunning it as if it were a CATE comparison;
9. performance evidence without pairwise matrices or per-observation Python loops; and
10. public documentation, OutputHub tables, graph-data parity, build, security, and package
    quality gates.

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

1. Promote the weighted-fit/CATE prediction protocol and native probability contract.
2. Write hand and leakage/refusal tests.
3. Implement construction/evaluation splitting and cross-fitted R-objective fitting.
4. Implement honest R-loss and differential calibration.
5. Add group effects, simultaneous bands, and calibration graph-data parity.
6. Run simulations, aligned parity, and one real-data comparison before publication.
