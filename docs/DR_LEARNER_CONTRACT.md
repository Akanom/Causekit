# Honest doubly robust learner contract

Status: public alpha implemented. This contract is separate from the R-learner contract.
It targets the same binary-treatment CATE but uses an augmented inverse-probability
pseudo-outcome and ordinary second-stage regression. The honest construction/evaluation
boundary is unchanged. Python hand/simulation/real-data gates, base-R fixed-evaluation
parity, and the reviewed Stata/IC 17 HC1 fixture pass.

## Target and score

For an exact binary treatment `W`, CauseKit targets

```text
tau(x) = E[Y(1) - Y(0) | X = x].
```

Let `e(x) = P(W=1 | X=x)` and `mu_w(x) = E[Y | W=w, X=x]`. Construction-fold
out-of-fold nuisance predictions form

```text
phi_i = mu1_i - mu0_i
      + W_i (Y_i - mu1_i) / e_i
      - (1-W_i) (Y_i - mu0_i) / (1-e_i).
```

The final CATE model regresses `phi` on `X` using construction rows only. Under
consistency, no interference, conditional exchangeability, overlap, and suitable
nuisance/final-stage rates, `E[phi | X=x] = tau(x)`. The score is doubly robust in the
specific sense that its conditional mean remains correct if the propensity model is
correct or both treatment-arm outcome regressions are correct. It is not protection
against unmeasured confounding, overlap failure, arbitrary final-stage misspecification,
or invalid honest inference.

## Immutable honest roles

CauseKit reuses the R-learner's treatment-stratified row split and whole-cluster split:

1. Construction rows tune and fit all nuisance and CATE models. Propensity and both
   arm-specific outcomes are cross-fitted on one shared, cluster-preserving outer-fold
   plan. Every factory call returns fresh state.
2. Fresh full-construction nuisance refits and the construction-fitted CATE model predict
   evaluation rows. Evaluation outcomes and treatments enter no fitting or tuning call.
3. The evaluation role is consumed once for the declared loss, calibration, and grouped
   effects. Results are labelled split-conditional. Repeated splits and deployment refits
   require later contracts.

Propensities must lie in `[overlap_floor, 1-overlap_floor]` on construction and
evaluation predictions. CauseKit refuses violations and never silently clips them.

## Public learner boundary

The DR final stage has a distinct provider-neutral contract:

```text
CATEEstimatorProtocol.fit(X, pseudo_outcome)
CATEResultProtocol.predict(X)
```

It is deliberately not the weighted `WeightedCATEEstimatorProtocol` used by the
R-learner. The dependency-free default is CauseKit's existing standardized ridge-GCV
regression fitted only on the supplied construction pseudo-outcomes. Custom factories
remain responsible for fold-internal tuning and must preserve schema, index, length, and
finite-value contracts.

## Honest evaluation and uncertainty

The primary score is evaluation-sample mean squared error of the fixed prediction against
the construction-fitted DR score. Its comparator is the construction mean pseudo-outcome:

```text
honest_dr_loss          = mean[(phi_eval - tau_hat(X_eval))^2]
honest_constant_dr_loss = mean[(phi_eval - mean(phi_construction))^2]
dr_loss_gain            = 1 - honest_dr_loss / honest_constant_dr_loss.
```

The noisy score is a model-evaluation signal, not observed unit-level treatment-effect
truth. Hyperparameters may not be revised after reading it.

Differential calibration fits the evaluation-only regression

```text
phi_eval = beta_level + beta_heterogeneity * (tau_hat - mean(tau_hat)) + error.
```

CauseKit reports HC1 or declared one-way CR1 covariance and tests
`beta_heterogeneity=0` and `beta_heterogeneity=1`. A constant/rank-deficient prediction
refuses.

Tie-preserving score groups report arithmetic means of the evaluation DR score. Each group
must retain both treatment arms and, for CR1, at least two clusters. Pointwise HC1/CR1
intervals and seeded Rademacher max-t simultaneous bands use the retained evaluation
influence matrix. These group effects target score-defined group ATEs under the declared
identification and nuisance conditions; they are not the R-learner's overlap-weighted
residual moments.

No unit-level CATE interval, RATE/AUTOC, policy value, or ordinary full-sample ATE is
provided. Each needs a separate estimand and uncertainty contract.

## Required refusals

The implementation refuses nonbinary treatment; missing/nonfinite or misaligned inputs;
duplicate covariate columns; insufficient role/fold/arm/cluster capacity; recycled
estimators; malformed nuisance or CATE predictions; overlap violations without clipping;
arm-specific outcome fits that receive the wrong treatment arm; evaluation leakage;
rank-deficient calibration; score groups that split ties or lack both arms/clusters;
zero constant-comparator loss; and unsupported unit-level or policy inference.

## Validation gate

Promotion required observed-failing hand contracts before code, exact pseudo-outcome and
loss identities, row/cluster leakage sentinels, robust/cluster covariance and max-t
identities, deterministic recovery and one-model-correct simulations, public API and
OutputHub tests, fixed-evaluation Python/base-R parity, an honest real-data comparison that
does not rerun settled R-learner benchmarks, a linear-memory performance smoke, docs,
build, lint, typing, and security checks. The implementation passes the hand score/loss,
HC1/CR1, group influence, and seeded max-t identities; both one-nuisance-side-correct
simulations; native linear CATE recovery; base-R 4.5.1 fixed-evaluation parity; and the
hash-verified NSW smoke. The DR-only NSW comparator was run once: native ridge-GCV and
scikit-learn RidgeCV matched exactly, while boosting and forest lost to the construction
constant. The Stata harness reconstructs the aligned fixed evaluation because Stata has no
identified native command for CauseKit's complete learner; the reviewed Stata/IC 17
artifact passes at `1e-8` with maximum absolute difference `4.44e-16`.

## Differentiation and pre-mortem

The DR pseudo-outcome is established methodology, including Kennedy's two-stage analysis
and EconML's DRLearner. CauseKit's contribution is an independent, dependency-free,
provider-neutral implementation with immutable row/cluster roles, shared cluster-safe
cross-fitting, explicit no-clipping overlap refusal, copy-out audit records, and honest
split-conditional evaluation/inference. No mathematical novelty claim is made.

| Failure mode | Consequence | Prevention and validation |
| --- | --- | --- |
| Evaluation data tune a nuisance or CATE model | optimistic loss and invalid inference | immutable roles plus fit-index sentinels |
| Extreme propensities dominate `phi` | unstable targets and misleading groups | explicit overlap floor, diagnostics, no clipping |
| Only one arm outcome model is correct | false double-robustness claim | document that both arm regressions form the outcome side |
| Flexible final stage overfits noisy scores | poor honest generalization | construction-only tuning and held-out constant comparison |
| Pointwise CATE intervals inferred from group HC1/CR1 | unsupported uncertainty claim | group/calibration-only result surface and refusals |
| Cluster rows cross roles or folds | leakage and understated variance | whole-cluster splitting and cluster-summed influence |
