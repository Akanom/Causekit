# Causal machine-learning contract

This document defines the CauseKit causal-ML boundary before adding specialized
estimators. Machine learning is a nuisance-estimation tool inside an identified causal
procedure; predictive accuracy alone is not a causal estimand or an identification
argument.

## Scope decision

CauseKit owns causal scores, target parameters, sample splitting, diagnostics, inference,
refusal behavior, and the native nuisance learner used by its default execution path. The
first native learner is a dependency-free, standardized ridge regression with
generalized-cross-validation selection performed separately inside every outer training
fold. CauseKit does not import another package's ML implementation. The existing public
nuisance-model protocols remain an escape hatch for a design that genuinely requires a
different learner, without making that provider a CauseKit dependency.

The initial specialized estimator is `PartiallyLinearDML`. It implements the DML2
orthogonal-score estimator for the partially linear model

```text
Y = theta D + g(X) + U,       E[U | X, D] = 0
D = m(X) + V,                 E[V | X] = 0.
```

For out-of-fold predictions `l_hat(X) = E[Y | X]` and
`m_hat(X) = E[D | X]`, define

```text
u_i = Y_i - l_hat(X_i)
v_i = D_i - m_hat(X_i)

theta_hat = sum_i(v_i u_i) / sum_i(v_i^2)
psi_i     = v_i (u_i - theta_hat v_i)
IF_i      = psi_i / mean_i(v_i^2).
```

The estimator pools the orthogonal score across folds (DML2). Every nuisance prediction
must be out of fold, and both nuisance tasks use the same deterministic fold assignment.
For an exactly binary treatment, folds are stratified by treatment arm; for a continuous
treatment, folds are shuffled without discretizing or inventing strata.

Chernozhukov et al. establish the orthogonal-score and cross-fitting framework and treat
the partially linear regression parameter as a principal DML example:
[The Econometrics Journal article](https://doi.org/10.1111/ectj.12097) and
[open preprint](https://arxiv.org/abs/1608.00060).

## Estimand and causal claim

The fitted coefficient is the constant treatment-effect parameter `theta` in the declared
partially linear structural model. It can be interpreted as an average treatment effect
only when that constant-effect model, consistency, no interference, conditional
exchangeability, sufficient residual treatment variation, and the required nuisance-rate
and regularity conditions are credible. CauseKit does not relabel an arbitrary predictive
partial association as an ATE.

The first alpha accepts binary or continuous scalar treatments. It does not implement a
nonparametric dose-response curve, an endogenous-treatment DML score, instruments,
multiple treatments, treatment-policy learning, or conditional treatment effects.

## Nuisance-model protocol

With no factories supplied, the public constructor uses CauseKit's native ridge-GCV
nuisance learner for both the outcome and treatment regressions. Its centering, scaling,
penalty grid, tie behavior, and numerical solve are package-owned and tested. Selection is
nested inside each outer fold, so it never evaluates a penalty on that fold's held-out
observations. A tested 41-point candidate grid weakly lowered training-fold GCV but did not
improve the one-run real-data out-of-fold errors, so the proven six-point default remains.
Users can supply a denser ordered grid explicitly; lower training GCV is not represented as
a guarantee of lower held-out error or better causal identification.

The GCV path computes residual sums of squares from the SVD shrinkage factors without
reconstructing a fitted length-`n` vector for every candidate. Each fitted native result
implements the public `NuisanceDiagnosticsProtocol`. `CrossFitter` exposes one row per
task/fold with training and holdout counts, selected penalty, effective degrees of freedom,
GCV score, training RMSE, numerical rank, grid size, and boundary-selection flag. Custom
models without the optional protocol retain a row with `diagnostics_available=False`;
malformed declared diagnostics refuse rather than disappearing silently.

The constructor also accepts fresh factories for the outcome and treatment regressions
when the native learner is substantively inadequate. Each factory must return an object
with `fit(X, y)`. A fitted result must expose `predict(X)` unless a task-specific prediction
adapter is supplied. The package records the fitted result class for each task, fold
assignments, aligned out-of-fold predictions, and residuals.

Factories, not pre-fitted objects, are required because fold-specific model state must not
leak across the held-out observations. CauseKit neither tunes the supplied learner nor
asserts that its convergence rate is adequate. Hyperparameter tuning must itself be
confined to each training fold, for example by returning a pipeline with internal
cross-validation from the factory.

## Inference contract

`covariance="robust"` uses the one-regressor, no-intercept HC1 sandwich from the
residual-on-residual score and normal reference inference. Equivalently,

```text
Var(theta_hat) = sum_i(IF_i^2) / [n(n - 1)].
```

`covariance="clustered"` aggregates the same influence contributions once by a declared
one-way cluster, applies `G / (G - 1)`, and uses a `t(G - 1)` reference distribution. The
cluster vector must align exactly with every labelled estimation input and contain at
least two clusters. Multiway clustering, repeated cross-fitting, bootstrap inference,
sample weights, and survey corrections are not silently approximated.

The first alpha reports conventional asymptotic DML uncertainty. It does not claim valid
finite-sample inference merely because cross-fitting ran successfully.

## Required refusals

The estimator refuses:

- nonnumeric, nonfinite, empty, or duplicate-column covariates;
- nonnumeric or nonfinite outcome/treatment data;
- inconsistent lengths or any pandas index/order mismatch;
- fewer observations than folds, or fewer observations in a binary arm than folds;
- invalid covariance/cluster combinations or fewer than two clusters;
- nuisance factories/results that violate the public fit/predict contract;
- prediction length, finiteness, or alignment failures; and
- zero or numerically unidentified residual treatment variation.

No ridge, pseudoinverse, treatment jitter, propensity clipping, fold merging, or in-sample
fallback repairs these failures.

## Why this estimator comes first

Credible alternatives were assessed before implementation:

| Candidate | Strength | Reason not first |
| --- | --- | --- |
| S/T/X meta-learners | Simple CATE baselines and broad learner compatibility | Plug-in heterogeneity estimates do not provide this scalar orthogonal-inference contract |
| R-learner | Orthogonal residual objective for CATE estimation | Requires a weighted final-stage learner contract and separate honest CATE evaluation/inference |
| DR-learner | Doubly robust pseudo-outcome for CATE estimation | Requires propensity overlap rules plus an honest second-stage and CATE-specific uncertainty contract |
| Native honest causal forest | Strong adaptive heterogeneity workflow | Requires a separate splitting, honesty, treatment-overlap, prediction, and inference contract; it will not be delegated to another runtime package |
| Partially linear DML | Orthogonal scalar target, cross-fitting, auditable influence inference | Selected as the smallest complete specialized causal-ML model |

The next heterogeneous-effect candidate is governed by the separate
[honest R-learner contract](R_LEARNER_CONTRACT.md), based on the residual objective of
[Nie and Wager](https://arxiv.org/abs/1712.04912). A doubly robust learner whose two-stage
contract follows [Kennedy](https://arxiv.org/abs/2004.14497) comes later. Neither is a
placeholder public import in this milestone.

CauseKit has implemented the R-learner's internal native prerequisites: stratified
training-only-CV ridge Logit for binary probabilities and weighted ridge-GCV for the exact
`u/v`, `v^2` CATE transformation. Public `WeightedCATEEstimatorProtocol` and
`CATEResultProtocol` keep custom weighted learners provider-neutral. This foundation does
not expose an `RLearner`, honest evaluation result, or heterogeneity claim; every remaining
gate is maintained in the separate contract.

## Validation and promotion gates

Before promotion, the implementation must have:

1. a hand-reconstructed coefficient, score, influence function, HC1 standard error, and
   confidence interval;
2. exact shared-fold and original-index preservation checks;
3. strict factory, prediction, covariance, clustering, and weak-residual-variation
   refusals;
4. deterministic binary- and continuous-treatment simulations;
5. independent residual-regression parity with explicitly aligned HC1 conventions;
6. R and Stata status recorded only for estimand-aligned implementations;
7. a one-run hash-verified real-data timing, memory, and OOF-prediction benchmark against
   optional external learners, with no quadratic allocation or runtime dependency;
8. a runnable provider example and complete result/OutputHub documentation; and
9. package-wide lint, format, type, test, build, artifact, and security gates.

## Pre-mortem

| Failure mode | Consequence | Prevention/evidence |
| --- | --- | --- |
| Training leakage | Regularization bias is presented as debiased estimation | Fresh factory per fold; stored fold and OOF prediction audit records |
| Weak residual treatment signal | Unstable or unidentified denominator | Scale-aware refusal plus reported residual second moment |
| Poor nuisance rates | Nominal intervals undercover | Explicit assumption/limitation; simulation and comparator evidence, never an automatic validity claim |
| Hidden effect heterogeneity | PLR coefficient is over-described as the population ATE | Result is labelled `theta`; causal interpretation states the constant-effect requirement |
| Cluster/sample drift | Incorrect covariance and degrees of freedom | Exact index alignment and one normalized cluster aggregation |
| Provider lock-in | Causal API becomes tied to one ML ecosystem | CauseKit-native default learner; protocols are optional escape hatches, not a dependency |
