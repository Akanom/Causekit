# Repeated-cross-section difference-in-differences design contract

## Status

This document freezes the identification, data, API, inference, and validation decisions
for repeated-cross-section DiD. The no-covariate stationary-composition first slice is now
implemented as `RepeatedCrossSectionDiD`; it was exported only after the hand identities
and refusal tests had been written and observed failing. Publication-scale coverage and
available estimator-level R/Stata parity now pass. Stata aggregate standard errors remain
non-comparable because their estimated-share influence differs. Covariate adjustment,
composition-change robustness, and simultaneous bands remain promotion gates rather than
implicit features.

Repeated cross sections are not an option on the balanced-panel classes. The observations,
influence functions, nuisance tasks, and composition assumptions differ materially from a
panel. A separate public class is therefore the implemented surface.

## Chosen first slice

The first implementation will target conventional cohort-time ATT effects under repeated
sampling and a stationary-composition restriction. It will retain both never-treated and
valid not-yet-treated comparisons and the existing anticipation semantics. It will not
claim the short-panel Chen-Sant'Anna-Xie PT-All efficiency bound.

The implemented constructor is:

```python
RepeatedCrossSectionDiD(
    control_group="never_treated",
    composition="stationary",
    anticipation=0,
    covariance="robust",
    inference="analytic",
)
```

The implemented `fit` roles are `outcome`, `time`, and `treatment_time`, with optional
`cluster`. `covariates` and `sampling_weights` are accepted only to return explicit
first-slice refusals. There is no `entity` role. Supplying an entity identifier cannot
silently turn repeated observations into a panel or change the sampling unit.

## Estimand and identification

For adoption cohort `g`, target period `t`, clean baseline `b`, and a fixed eligible
comparison population `C(g,t)`, the no-covariate repeated-cross-section estimand is

```text
ATT_rcs(g,t)
  = [E(Y | G=g, T=t) - E(Y | G=g, T=b)]
    - [E(Y | C(g,t), T=t) - E(Y | C(g,t), T=b)].
```

The same cohort-membership rule defining `C(g,t)` is applied in the target and baseline
samples. `control_group="never_treated"` uses only the declared never-treated population.
`control_group="not_yet_treated"` additionally uses cohorts whose effective treatment
boundary is later than `t`; already-treated or anticipation-window observations are never
comparisons.

A causal ATT interpretation requires consistency, no interference, overlap in group and
period cells, an absorbing conceptual treatment-adoption definition, no anticipation
outside the declared window, repeated-cross-section conditional parallel trends, and the
declared composition restriction. These conditions are not established by a pre-trend
test or by flexible nuisance fitting.

`composition="stationary"` declares that the joint distribution of the relevant baseline
characteristics and treatment cohort is stable across the repeated samples. This is an
identifying restriction, not a balance diagnostic. CauseKit will record it in every result
and refuse to relabel it as verified. A later `composition="robust"` path may implement the
Sant'Anna-Xu estimand under compositional changes, but it requires its own score,
diagnostics, external evidence, and promotion gate; it will not be approximated by adding
time controls to the stationary-composition estimator.

## Data contract

Each row is one sampled observation in one period. Required conditions are:

- finite numeric outcomes and time labels;
- one explicit treatment-cohort value per row, with a declared never-treated sentinel;
- at least one clean pre-period and one target period for each reported treated cohort;
- at least two observations in every treated-period and comparison-period cell used for
  analytical inference;
- finite observed adoption times that coincide with an observed time label;
- no sampling weights in the first slice; and
- at least two independent clusters when clustered covariance is requested.

Repeated rows and repeated real-world people are not detectable without a sampling
identifier. If a cluster or primary-sampling-unit label is supplied, it defines the
independent inference unit and may recur across periods. A person identifier is not
accepted as a cosmetic argument: data that follow the same individuals belong in the
existing balanced-panel estimators.

The validator will not require equal period sizes, balanced cohort-period cells, or a
rectangular entity-period grid. It will validate every cell actually used and retain a
public cell-count table so attrition and weak support are auditable.

## Influence functions and uncertainty

The random sampling unit is an observation, or the declared higher-level cluster. Every
cell mean has an observation-level influence contribution that includes its group-period
probability. The four aligned cell contributions are added to obtain each
`ATT_rcs(g,t)` influence function. Cohort-share aggregation will include estimated-share
terms, just as the panel path does, but the shares are estimated from repeated samples and
cannot reuse entity-level panel formulas.

The first slice will use the package's HC1-style scaling for independent observations and
one-way CR1 score aggregation for declared clusters. Pointwise normal or finite-cluster
`t` references will follow the existing public convention. Multiplier event-study bands
are a later inference gate and must draw one multiplier per observation or cluster—not one
per nonexistent panel entity.

Pre-trend placebos will compare independent cohort-period cell means and expose their full
observation-level influence matrix. They cannot call the panel pre-trend helper. The
Chen-Sant'Anna-Xie PT-All/PT-Post Hausman diagnostic is a short-panel result and remains
unavailable on this surface unless a repeated-cross-section theorem and aligned estimator
pair are separately contracted.

## Covariate-adjusted phase

Covariate adjustment will be a second phase, after the no-covariate contract passes. It
will consume provider-neutral nuisance factories through `CrossFitter`; no classifier or
regression class will be copied into the DiD module. Folds will be assigned at the
observation level or kept whole by declared cluster. Required nuisances include the
relevant group-period probabilities and period/group outcome regressions for the chosen
doubly robust repeated-cross-section score.

Every public nuisance prediction must be out of fold. Overlap failures will refuse rather
than clip. The score must use the repeated-cross-section efficient influence function and
variance contribution; the panel outcome-change regression and panel PT-All conditional
covariance machinery are not valid substitutes.

## Refusals

The estimator refuses unsupported composition labels, panel/entity arguments,
finite adoption times outside observed periods, missing or non-finite roles, absent clean
baselines, empty or undersized group-period cells, invalid anticipation, unsupported
sampling weights or survey designs, malformed clusters, covariates without `CrossFitter`,
own-observation nuisance predictions, weak group-period overlap, and requests for the
panel PT-All efficiency or Hausman labels.

No implicit row deletion, weight normalization, propensity clipping, variance repair,
ridge, pseudoinverse, or generic bootstrap will be introduced merely to produce a result.

## Validation and promotion gates

The validation sequence started with tests that failed before estimator code existed.
Current first-slice status is:

1. The hand-computed two-period 2-by-2 ATT, full observation influence, and HC1 identity
   pass in Python, an independent base-R 4.5.1 reconstruction, and a reviewed Stata 17
   reconstruction.
2. Staggered never-treated/not-yet-treated group-time identities, unequal period-size and
   row-permutation invariance, anticipation/placebo identities, HC1/CR1 reconstruction,
   strict refusals, and event/calendar/ESavg share-influence identities pass.
3. The preregistered 1,000-replication balanced/unequal-period publication-scale coverage
   and SE-calibration certificate passes all 32 maintained cells.
4. The hash-pinned public `hospdd` workflow exercises patient-level repeated samples with
   hospital PSUs. Its Stata source label says the data are artificial, so this is an
   execution smoke rather than substantive empirical evidence.
5. The 100,000-row, six-period benchmark completes through vectorized row arithmetic
   without an observation-distance or entity-period matrix. Exact runtime and memory are
   environment-specific and are reproduced by `benchmarks/benchmark_did_rcs.py`.
6. OutputHub, namespace, documentation, package build, lint, type, and security checks are
   release gates for each candidate. Estimator-level parity with pinned R
   `did::att_gt(panel = FALSE)` passes for both control rules after the explicit HC0-to-HC1
   mapping. Reviewed Stata `csdid` matches group-time estimates/SEs and aggregate points;
   aggregate SEs remain explicitly non-comparable. Unavailable cells must stay explicit.

## Alternatives considered

- Adding `panel=False` to `DifferenceInDifferences` was rejected because it would put two
  sampling-unit and influence-function contracts behind one deceptively small switch.
- Treating synthetic entity IDs as a panel was rejected because it creates invalid
  within-entity changes and uncertainty.
- Implementing the covariate doubly robust score first was rejected because the simpler
  cell-mean identities are needed to validate the data and aggregation layer independently.
- Assuming away compositional changes without an explicit result field was rejected. The
  first slice may impose stationarity, but it must say so at construction and in output.

## Pre-mortem

The most likely failure modes are mixing period-specific comparison populations, using
panel influence scaling, hiding weak group-period cells inside an aggregate, and treating
failure to reject a composition or pre-trend test as validation. Tests will reconstruct
every cell score, freeze comparison membership across the two periods, expose all cell
counts, and keep diagnostic language non-confirmatory. A second risk is accidental API
convergence with the panel class; the explicit absence of an entity argument and separate
result metadata are release gates.

## Primary methodology

- Brantly Callaway and Pedro H. C. Sant'Anna (2021), [*Difference-in-Differences with
  Multiple Time Periods*](https://doi.org/10.1016/j.jeconom.2020.12.001).
- Pedro H. C. Sant'Anna and Jun Zhao (2020), [*Doubly Robust Difference-in-Differences
  Estimators*](https://doi.org/10.1016/j.jeconom.2020.06.003).
- Pedro H. C. Sant'Anna and Qi Xu (2023), [*Difference-in-Differences with Compositional
  Changes*](https://arxiv.org/abs/2304.13925).
