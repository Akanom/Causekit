# Difference-in-differences contract

## Status and decision

The `0.6.0a1` milestone adds two separate estimators. The efficient estimator is an
addition, not a replacement for conventional difference-in-differences (DiD).

| Public estimator | Identifying restriction | Comparison observations | Weighting |
| --- | --- | --- | --- |
| `DifferenceInDifferences` | post-treatment parallel trends for each treated cohort | never-treated, or never-treated plus units not yet treated at the target time | cohort mean changes and observed cohort shares |
| `EfficientDiD` | parallel trends in every retained period and across all cohorts (`PT-All`) | never-treated plus the paper's admissible auxiliary treated cohorts | inverse covariance of the non-collinear generated-outcome influence functions |

The conventional estimator is the safer default when the stronger pre-period and
cross-cohort restrictions needed for efficiency are not substantively justified.
`EfficientDiD` must never silently weaken or relabel those assumptions.

The initial alpha implements the paper's closed-form no-covariate short-panel path. It
does not copy nuisance estimators from `limiteddepkit`. A later covariate-adjusted path
must consume causalkit's public nuisance protocols and cross-fitting orchestrator, and
must also implement the paper's conditional covariance estimation contract.

## Data and timing

Both estimators accept a long pandas data frame and explicit column names for outcome,
entity, time, and first treatment time. The contract requires:

- exactly one row per entity-period;
- a balanced panel with a finite outcome in every cell;
- a treatment-time value that is constant within entity;
- an explicit `never_treated` sentinel, defaulting to positive infinity;
- at least one never-treated group and at least two entities in every cohort used for
  analytic inference;
- absorbing binary treatment encoded by the first-treatment-time column; and
- at least one uncontaminated baseline period for every treated cohort.

Time values are finite numeric labels. Event time is the number of ordered panel periods
relative to adoption, so irregular numeric spacing does not change the event-time count.
Finite treatment times must equal an observed time label. A date after the observed panel
is not silently recoded as never treated.

`anticipation=A` moves the effective treatment boundary back by `A` ordered periods.
Group-time results retain the reported adoption cohort, while event times from `-A` to
`-1` identify the declared anticipation window. Comparisons use the effective boundary,
so a unit is never used as a control once its anticipation window begins. Calendar-time
aggregates omit anticipation-period effects; the event-study table retains them.

## Conventional group-time effects

For cohort `g`, target period `t`, baseline `b=g-A-1`, and comparison set `C(g,t)`,
the no-covariate estimator is

```text
ATT(g,t) = mean(Y_t - Y_b | G=g)
           - mean(Y_t - Y_b | entity in C(g,t)).
```

`control_group="never_treated"` uses only the declared never-treated cohort.
`control_group="not_yet_treated"` also uses cohorts whose effective treatment boundary
is after `t`. Already-treated observations are never controls.

The entity-level influence function is the centered treated mean-change score minus the
centered comparison mean-change score. The result exposes every group-time influence
column so the estimate, uncertainty, and aggregations can be reconstructed.

## Efficient PT-All effects

`EfficientDiD` implements the no-covariate specialization of Chen, Sant'Anna, and Xie
(2025), equations (3.9)-(3.14) and Section 4.1. For each `ATT(g,t)`, it constructs the
admissible generated outcomes indexed by auxiliary treated cohort `g'` and bridge period
`t_pre`. With earliest retained baseline `t1`, each candidate sample estimand is

```text
mean(Y_t - Y_t1 | G=g)
- [mean(Y_t - Y_tpre | never treated)
   + mean(Y_tpre - Y_t1 | G=g')].
```

The candidate influence functions form rows of a matrix with sample covariance `V`.
The efficient weights are computed by solving linear systems, not by forming a matrix
inverse:

```text
w = solve(V, 1) / (1' solve(V, 1)).
```

Weights must sum to one but need not be non-negative. Negative efficiency weights combine
homogeneous identifying moments; they are not TWFE treatment-effect weights. The result
reports the candidate estimands, auxiliary cohorts, bridge periods, and realized weights.
A singular or numerically unidentified covariance system is refused rather than repaired
with an undocumented ridge or pseudoinverse.

`pre_periods="all"` uses every available period before each cohort's effective treatment
boundary. A positive integer uses that many immediately preceding periods. The same
choice is recorded in the result.

## Aggregation and target populations

Group-time effects target the observed members of cohort `g`. Event-study effects use
observed cohort shares among cohorts observed at a given event time. Calendar-time effects
use observed cohort shares among cohorts already adopted by that calendar period. The
scalar `ESavg` result is the unweighted average of non-anticipation event-study effects,
matching the paper's definition.

Cohort-share estimation is included in the event/calendar influence functions. The result
therefore does not treat random aggregation weights as fixed.

## Uncertainty

The alpha supports pointwise analytic influence-function inference:

- `covariance="robust"` uses the package's HC1-style finite-sample scaling
  `sum(IF_i^2) / [n(n-1)]` with a normal reference distribution;
- `covariance="clustered"` sums entity influence functions once by the declared cluster,
  applies `G/(G-1)`, and uses a `t(G-1)` reference distribution.

The paper's random-sampling unit is the panel entity. A higher-level cluster must be
constant within entity and have at least two levels. Pointwise intervals do not control
family-wise error across the event-study path. The paper's semiparametric efficiency-bound
claim is retained only for independent entities; requesting higher-level clustered
uncertainty does not establish efficiency under a cluster-dependent model.
Multiplier-bootstrap simultaneous bands remain a promotion gate and are refused in this
alpha.

## Refusals

The public estimators refuse duplicate entity-time rows, unbalanced panels, non-finite
outcomes or time values, varying treatment time or cluster within entity, an unobserved
finite adoption time, missing never-treated observations, cohorts without a clean
baseline, undersized cohorts, unsupported repeated cross-sections, sampling weights,
covariate adjustment, invalid control labels, non-analytic inference, and singular
efficient covariance systems.

## Validation and promotion gates

The maintained tests begin with hand-computed conventional `ATT(g,t)`, event-time,
calendar-time, and `ESavg` identities. A separate eight-entity fixture yields three
orthogonal generated-outcome scores with variances proportional to `1`, `4`, and `16`, so
the efficient weights must be exactly `16/21`, `4/21`, and `1/21`. Refusal behavior is
part of this contract.

Promotion beyond the no-covariate alpha requires covariate-adjusted nuisance integration,
cross-fitting tests, conditional covariance estimation, multiplier-bootstrap simultaneous
bands, coverage simulations, larger performance fixtures, and aligned external parity.

Primary methodology:

- Xiaohong Chen, Pedro H. C. Sant'Anna, and Haitian Xie (2025), [*Efficient
  Difference-in-Differences and Event Study Estimators*](https://arxiv.org/abs/2506.17729),
  arXiv:2506.17729.
- Brantly Callaway and Pedro H. C. Sant'Anna (2021), [*Difference-in-Differences with
  Multiple Time Periods*](https://doi.org/10.1016/j.jeconom.2020.12.001), Journal of
  Econometrics 225(2), 200-230.
