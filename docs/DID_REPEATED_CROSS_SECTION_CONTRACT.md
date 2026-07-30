# Repeated-cross-section difference-in-differences design contract

## Status

This document freezes the identification, data, API, inference, and validation decisions
for repeated-cross-section DiD. The no-covariate stationary-composition slice and the
opt-in cross-fitted covariate-adjusted slice are implemented as
`RepeatedCrossSectionDiD`. Each was added only after its hand identities and refusal tests
had been written and observed failing. The covariate path implements the locally efficient
doubly robust repeated-cross-section score of Sant'Anna and Zhao (2020), not the balanced-
panel PT-All machinery. Observation/PSU simultaneous bands are promoted. A narrow
two-group, two-period composition-change-robust covariate score is implemented under an
explicit opt-in; its staggered extension and diagnostic remain separate promotion gates
with a frozen [influence-alignment contract](DID_RCS_COMPOSITION_DIAGNOSTIC_ALIGNMENT_CONTRACT.md).
Survey designs remain design-only.

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
    nuisance_probability_floor=1e-6,
)
```

The implemented `fit` roles are `outcome`, `time`, and `treatment_time`, with optional
`cluster`. Supplying `covariates` requires an explicit provider-neutral `CrossFitter` with
fresh `propensity_factory` and `outcome_factory` products. `sampling_weights` still
refuse. There is no `entity` role. Supplying an entity identifier cannot silently turn
repeated observations into a panel or change the sampling unit.

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
and refuse to relabel it as verified. `composition="robust"` now implements only the
Sant'Anna-Xu two-group, two-period estimand for treated observations in the target-period
population. It requires covariates, a four-cell generalized propensity, three outcome
regressions, and its own influence function; it is not approximated by adding time
controls to the stationary estimator. Staggered effects and a composition diagnostic
remain unavailable; the next-stage document contracts them without exporting a
placeholder.

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

The implemented pointwise path uses the package's HC1-style scaling for independent
observations and one-way CR1 score aggregation for declared clusters. Pointwise normal or
finite-cluster `t` references follow the existing public convention. With
`inference="multiplier_bootstrap"`, the simultaneous path retains the same event-study
estimate, standard error, and influence matrix, then draws one Rademacher multiplier per
observation or declared PSU—not one per nonexistent panel entity. For `n` observations,
observation draws are

```text
sqrt(n/(n-1)) * sum_i xi_i * psi_i(event) / n.
```

For `G` declared PSUs, CauseKit first sums `psi_i(event)` within PSU and uses
`sqrt(G/(G-1))` with one indivisible multiplier per PSU. Each draw is studentized by the
matching HC1/CR1 event standard error; the band critical value is the requested
higher-quantile of the maximum absolute statistic. For `U` sampling units, batch size is
`min(256, remaining, max(1, floor(8,000,000/U)))`, so large samples do not create a full
iterations-by-observations matrix. Batching changes memory use, not the seeded multiplier
stream. This is a multiplier approximation from fixed influence scores, not an ordinary
row-resampling bootstrap and not nuisance refitting.

Pre-trend placebos will compare independent cohort-period cell means and expose their full
observation-level influence matrix. They cannot call the panel pre-trend helper. The
Chen-Sant'Anna-Xie PT-All/PT-Post Hausman diagnostic is a short-panel result and remains
unavailable on this surface unless a repeated-cross-section theorem and aligned estimator
pair are separately contracted.

## Covariate-adjusted phase

The implemented covariate phase consumes provider-neutral nuisance factories through
`CrossFitter`; no classifier or regression class is copied into the DiD module. A single
global fold assignment is shared by every reported group-time and conditional-placebo
task. Folds are stratified by observed cohort-period cell and assigned at observation
level, or kept whole by the declared cluster. Consequently, each cohort-period cell must
occur in every requested fold (and in enough distinct clusters for clustered fitting).

For each fixed two-group, two-period comparison, CauseKit cross-fits:

- `p(X) = P(G=g | G in {g,C(g,t)}, X)` over the pooled two periods; and
- four outcome regressions `m(d,s,X) = E[Y | D=d,T=s,X]` for treated/comparison by
  baseline/target period.

Let `D=1` identify cohort `g`, `S=1` the target period, `w=D`, and
`q=p(X)/(1-p(X))`. With `m0 = S*m(0,1,X) + (1-S)*m(0,0,X)`, the maintained estimate is the
sample mean of the eight normalized components

```text
 D*S*(Y-m0) / E[D*S]
-D*(1-S)*(Y-m0) / E[D*(1-S)]
-q*(1-D)*S*(Y-m0) / E[q*(1-D)*S]
+q*(1-D)*(1-S)*(Y-m0) / E[q*(1-D)*(1-S)]
+D*(m(1,1,X)-m(0,1,X)) / E[D]
-D*S*(m(1,1,X)-m(0,1,X)) / E[D*S]
-D*(m(1,0,X)-m(0,0,X)) / E[D]
+D*(1-S)*(m(1,0,X)-m(0,0,X)) / E[D*(1-S)].
```

Every ratio contributes its centered numerator/denominator influence term. The pair score
is rescaled back to the full observation sample before group/event/calendar/ESavg
aggregation and HC1/CR1 inference. Orthogonality plus cross-fitting removes first-order
nuisance-estimation terms; this is not a claim of exact finite-sample covariance parity
with a particular parametric nuisance fit.

Every relevant public nuisance prediction is out of fold; irrelevant comparison rows are
stored as missing rather than presented as meaningful extrapolations. Probabilities at or
beyond the declared floor refuse. CauseKit does not clip probabilities, trim rows, repair
denominators, or silently change the estimand. Adjacent pre-treatment placebos use the
same five cross-fitted nuisances and eight-component score, so the diagnostic tests the
declared conditional parallel-trends restriction rather than an inconsistent marginal
one. Failure to reject remains non-confirmatory.

## Refusals

The estimator refuses unsupported composition labels, panel/entity arguments,
finite adoption times outside observed periods, missing or non-finite roles, absent clean
baselines, empty or undersized group-period cells, invalid anticipation, unsupported
sampling weights or survey designs, malformed clusters, covariates without `CrossFitter`,
own-observation nuisance predictions, weak group-period overlap, and requests for the
panel PT-All efficiency or Hausman labels. Simultaneous inference additionally refuses
unknown inference labels, fewer than 99 multiplier draws, malformed seeds or confidence
levels, missing clustered roles, and any event coordinate with a nonpositive or nonfinite
standard error.

No implicit row deletion, weight normalization, propensity clipping, variance repair,
ridge, pseudoinverse, or generic bootstrap will be introduced merely to produce a result.

## Validation and promotion gates

The validation sequence started with tests that failed before estimator code existed.
Current promotion status is:

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
7. The covariate hand contract reproduces all eight normalized score components, the full
   influence vector, and HC1 in Python and an independent base-R reconstruction. Tests
   verify both double-robustness legs, no row/PSU leakage, hard overlap refusal, shared
   folds, conditional placebos, and never/not-yet-treated staggered aggregation.
8. A seeded 40-replication conditional-score smoke checks bias and HC1 coverage. The
   hash-verified 7,368-row `hospdd` application exercises two-fold PSU-preserving fitting,
   60 fold/task audit rows, conditional placebos, and a real observed covariate. The source
   labels these data artificial, so this remains execution evidence rather than a
   substantive causal result.
9. The covariate publication certificate uses 1,000 replications in each of two fixed
   designs and both control rules: 4,000 estimator fits, 240,000 fresh nuisance fold fits,
   44 group/aggregate/placebo coverage cells, and four joint conditional-pre-trend size
   cells. All pass the preregistered bias, coverage, SE-calibration, Monte Carlo error,
   overlap, audit-count, availability, and zero-refusal gates. The stressed design retains
   severe conditional cohort probabilities and unequal periods while requiring at least
   32 expected observations in every period-X-cohort cell; realized failures are never
   repaired or omitted from the refusal ledger.
10. Failing-first observation and PSU max-t tests independently reconstruct every seeded
    Rademacher draw, finite-sample factor, studentization, higher-quantile critical value,
    and band endpoint. Seeded repeatability, analytic-path emptiness, indivisible PSU
    roles, invalid configuration, and zero-standard-error refusals pass. The covariate
    integration test verifies that bands reuse retained OOF influence scores without an
    extra nuisance fit. A 100-replication, two-event observation-level joint-coverage
    smoke passes its frozen `>= 88/100` gate.
11. The hash-bound simultaneous certificate fixes 1,000 replications per cell, 999 max-t
    draws, two designs, both control rules, observation/PSU sampling, and unadjusted or
    genuinely cross-fitted covariate scores. All 16 complete-event-vector joint-coverage
    cells pass at `0.931–0.961`; Monte Carlo SE is at most `0.0081`, all 16,000 fits and
    480,000 nuisance fold fits complete, the narrowest realized PSU cell spans 31
    clusters, and the refusal ledger is empty. Seeds and indivisible PSU roles are audited.

## Separately contracted next stages

Two repeated-section capabilities have independent contracts:

- [Composition-change robustness](DID_RCS_COMPOSITION_CHANGE_CONTRACT.md) now has its
  pairwise score, influence, weights, strict refusals, and OutputHub transport. Staggered
  aggregation, its diagnostic, parity, simulation, and publication evidence remain open;
  the diagnostic cannot be used for pretest-based estimator selection.
- [Survey designs](DID_RCS_SURVEY_DESIGN_CONTRACT.md) change the population measure and
  design-based uncertainty. Bare `sampling_weights` remain insufficient; a validated
  design, explicit population target, weighted nuisance protocol, and survey-specific
  variance gate are required.

These options cannot be combined by multiplying weights or reusing a score. Their
Cartesian combination requires its own theorem, hand influence contract, refusals, and
coverage certificate before it can be exported. The separately contracted
[direct cohort-ratio nuisance](DID_DIRECT_RATIO_CONTRACT.md) belongs first to balanced-
panel `EfficientDiD`; its calibrated PT-All odds must not be passed into this score.

## Alternatives considered

- Adding `panel=False` to `DifferenceInDifferences` was rejected because it would put two
  sampling-unit and influence-function contracts behind one deceptively small switch.
- Treating synthetic entity IDs as a panel was rejected because it creates invalid
  within-entity changes and uncertainty.
- Implementing the covariate doubly robust score before the no-covariate slice was rejected
  because the simpler cell-mean identities were needed to validate the data and
  aggregation layer independently. The covariate score now reuses that promoted layer.
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
- Pedro H. C. Sant'Anna and Qi Xu (2026), [*Difference-in-Differences with Compositional
  Changes*](https://doi.org/10.1016/j.jeconom.2025.106147).
