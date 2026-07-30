# Fixed-effects Panel IV contract

## Status and scope

This document freezes the first CauseKit Panel IV contract. The public estimator is
`PanelIV2SLS`: static linear two-stage least squares after absorbing entity fixed effects
and, optionally, time fixed effects. It is not dynamic-panel GMM, does not construct lag
instruments, and does not classify variables as predetermined or endogenous on the user's
behalf.

The implementation follows the public panel-validation, compact within-transformation,
entity/time-indexing, and one-way clustered-inference conventions reviewed in
`systemgmmkit`, but CauseKit owns its estimator and does not import a sibling package at
runtime.

## Public API

```python
result = PanelIV2SLS(
    covariance="clustered",
    time_effects=True,
    missing="raise",
).fit(
    data,
    outcome="y",
    endogenous=["treatment"],
    instruments=["encouragement"],
    exogenous=["control"],
    entity="unit",
    time="period",
    cluster=None,
)
```

`data` must be a long-form pandas `DataFrame`. `instruments` contains excluded
instruments only; exogenous regressors are included in both the structural and instrument
designs. Entity effects are always absorbed. `time_effects=False` selects one-way entity
effects. With clustered covariance, `cluster=None` means entity clustering. A named
higher-level cluster is permitted only when it is constant within entity.

## Estimand and identifying moments

Let `M_F` residualize against the requested fixed effects, `X=[W,D]` contain exogenous
and endogenous structural regressors, and `Z=[W,Q]` contain exogenous and excluded
instruments. The estimator solves

```text
beta = (X~' Z~ (Z~' Z~)^-1 Z~' X~)^-1
       X~' Z~ (Z~' Z~)^-1 Z~' y~,
```

where tildes denote `M_F`-transformed variables. The implementation solves small normal
systems and never materializes an observation-by-observation projection matrix or a
fixed-effect dummy matrix.

A structural coefficient has a causal interpretation only under instrument relevance,
conditional independence/exogeneity after the included covariates and fixed effects,
exclusion, correct linear specification, and the estimand-specific assumptions needed for
the intended interpretation. Unit fixed effects remove time-invariant additive
confounding; they do not repair time-varying confounding or an invalid instrument.

## Panel and role validation

The estimator:

- requires distinct outcome, endogenous, exogenous, excluded-instrument, entity, and time
  roles, except that the default cluster role is the entity itself;
- refuses duplicate entity-time rows and missing entity/time identifiers;
- sorts entity and time labels deterministically and exposes a unique panel `MultiIndex`;
- applies one joint complete-case mask only when `missing="drop"`;
- requires at least two entities and two retained observations per entity;
- requires at least two observed periods when time effects are requested;
- refuses a disconnected entity-time incidence graph under two-way effects;
- refuses nonnumeric or nonfinite model variables; and
- records balance, entity/period counts, the retained index, and removed-row count.

Missing-data deletion is not an identification strategy. Users must assess whether the
complete-case panel remains scientifically defensible.

## Fixed effects

One-way effects use exact entity demeaning. Balanced two-way panels use exact double
demeaning. Unbalanced connected panels use deterministic alternating projections with a
fixed numerical tolerance and iteration ceiling. Failure to converge refuses estimation.

Every outcome, structural regressor, and excluded instrument is transformed by the same
operator. A regressor or instrument with no remaining variation is refused as absorbed.
Collinear structural designs, collinear instrument designs, and deficient cross-rank are
refused. CauseKit does not silently drop named variables, add a ridge penalty, or replace a
failed inverse with an unidentified pseudo-solution.

The absorbed rank is `N_entity` for entity effects and
`N_entity + N_time - 1` for connected two-way effects. It is included in residual degrees
of freedom and finite-sample covariance corrections.

## Inference

The supported covariance labels remain `"unadjusted"`, `"robust"`, and `"clustered"`:

- `unadjusted` uses the within structural residual variance, residual degrees of freedom,
  and t inference;
- `robust` uses HC1 with the absorbed-rank degrees-of-freedom correction and normal
  inference; it does not address within-entity serial dependence; and
- `clustered` aggregates projected 2SLS scores once by entity or a declared higher-level
  cluster, applies CR1 using the full absorbed rank, and uses t inference with `G-1`
  degrees of freedom.

Entity clustering is the default because row-level HC1 is generally not a credible default
for longitudinal outcomes. Multiway clustering, Driscoll-Kraay covariance, bootstrap
inference, survey designs, and few-cluster corrections beyond CR1 are outside this first
contract.

## Diagnostics

Each endogenous regressor receives a fixed-effect-adjusted first-stage record containing
within R-squared, partial R-squared, the classical excluded-instrument F statistic, and a
covariance-aligned excluded-instrument Wald test. The familiar `F < 10` flag is descriptive,
not a universal weak-identification test. With multiple endogenous regressors it is not a
substitute for conditional weak-IV diagnostics.

The result also exposes raw and within standard deviations for every endogenous regressor
and excluded instrument. Homoskedastic Sargan testing is reported only for overidentified
`unadjusted` fits. It is not relabelled as robust or cluster valid.

## Result and prediction boundary

The fitted result exposes labelled coefficients, covariance, inference, level-scale fitted
values and structural residuals, the absorbed component, within fitted values, panel
metadata, first stages, instrument-variation diagnostics, assumptions, and notes.
Level-scale fitted values reconstruct the estimated additive fixed-effect component so
`fitted + residual == outcome` on the retained sample.

Generic out-of-sample prediction is intentionally unavailable in the first contract: new
entity effects are unidentified and silently assigning zero or training means would change
the prediction estimand. In-sample fitted and residual helpers remain available.

## Refusal boundary and later contracts

The first contract does not provide:

- pooled Panel IV without entity effects (use `IV2SLS` with an explicitly chosen
  covariance instead);
- dynamic-panel Difference/System GMM;
- internally generated lags or leads;
- random-effects IV or Hausman-Taylor estimators;
- multiway, time-only, spatial, or cross-sectional-dependence covariance;
- weak-IV-robust Anderson-Rubin, CLR, or conditional-LR confidence sets;
- automatic instrument selection, regularization, or many-instrument repair; or
- formula parsing, categorical expansion, or automatic imputation.

Each extension requires a separate estimand, failure, inference, and parity contract.

## Promotion gates

Promotion requires all of the following:

1. exact hand-computed one-way FE point, residual, CR1, and first-stage identities;
2. balanced and unbalanced two-way deterministic recovery;
3. role, missingness, indexing, absorption, rank, cluster, and disconnected-panel refusals;
4. row-permutation and no-quadratic-matrix tests;
5. aligned Python, R, and Stata evidence where public estimators share the contract;
6. a hash-pinned real-data comparison;
7. deterministic simulation recovery/coverage with zero unexpected refusals;
8. fixed-size runtime and peak-memory evidence;
9. post-estimation, OutputHub, example, and packaging verification; and
10. complete documentation of assumptions and unavailable extensions.
