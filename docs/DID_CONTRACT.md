# Difference-in-differences contract

## Status and decision

The maintained panel milestone keeps two separate estimators. The efficient estimator is an
addition, not a replacement for conventional difference-in-differences (DiD).

| Public estimator | Identifying restriction | Comparison observations | Weighting |
| --- | --- | --- | --- |
| `DifferenceInDifferences` | post-treatment parallel trends for each treated cohort | never-treated, or never-treated plus units not yet treated at the target time | cohort mean changes and observed cohort shares |
| `EfficientDiD` | parallel trends in every retained period and across all cohorts (`PT-All`) | never-treated plus the paper's admissible auxiliary treated cohorts | inverse covariance of the non-collinear generated-outcome influence functions |

The conventional estimator is the safer default when the stronger pre-period and
cross-cohort restrictions needed for efficiency are not substantively justified.
`EfficientDiD` must never silently weaken or relabel those assumptions.

The efficient estimator supports both the paper's closed-form no-covariate path and a
covariate-adjusted path. It does not own or copy nuisance estimators.
Covariate adjustment consumes user-owned model factories through causekit's public
`CrossFitter` protocol and implements the paper's conditional covariance contract.

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

## Covariate-adjusted efficient path

Passing `covariates=` to `EfficientDiD.fit` requires an explicit `cross_fitter=`. Each
covariate must be numeric, finite, and constant within entity. `EfficientDiD` supplies
targets and group masks to the orchestrator; it never instantiates a regression or
classifier itself.

The first nuisance stage cross-fits one multiclass cohort model and group-specific
conditional outcome-change regressions. For candidate `(g', t_pre)`, equation (4.4) is
implemented as

```text
Gg/pi_g * (Yt-Y1 - m_inf,t,tpre(X) - m_g',tpre,1(X))
- [p_g(X)/p_inf(X)] * Ginf/pi_g * (Yt-Ytpre - m_inf,t,tpre(X))
- [p_g(X)/p_g'(X)] * Gg'/pi_g * (Ytpre-Y1 - m_g',tpre,1(X)).
```

The density ratios are formed from aligned out-of-fold multiclass probabilities. This is
a supported implementation route in the paper, although direct ratio regression may be
more stable near weak overlap. Every probability used in a ratio must exceed
`nuisance_probability_floor`; the implementation refuses instead of clipping.

For equation (3.12), conditional covariances are estimated as cross-fitted regressions of
products of out-of-fold outcome-change residuals. A dedicated
`second_moment_factory=` may be supplied on `CrossFitter`; otherwise its outcome factory
is reused. For every entity and group-time cell, the resulting symmetric conditional
covariance matrix must be finite and positive definite under `singularity_tolerance`.
No ridge, diagonal clipping, eigenvalue repair, or pseudoinverse is applied.

The observation-specific weights are

```text
w_i = solve(Omega_i, 1) / (1' solve(Omega_i, 1)).
```

`conditional_efficiency_weights` exposes every `w_i`; `efficiency_weights` reports its
sample mean, minimum, maximum, and standard deviation by candidate. The result also
exposes the shared nuisance fold and cohort-probability matrix. The semiparametric
efficiency label is justified only when PT-All and the second-moment, proper-weighting,
overlap, nuisance consistency, and product-rate conditions in the paper's Assumption C.1
hold. Cross-fitting prevents own-observation training leakage; it does not prove those
population conditions.

## Aggregation and target populations

Group-time effects target the observed members of cohort `g`. Event-study effects use
observed cohort shares among cohorts observed at a given event time. Calendar-time effects
use observed cohort shares among cohorts already adopted by that calendar period. The
scalar `ESavg` result is the unweighted average of non-anticipation event-study effects,
matching the paper's definition.

Cohort-share estimation is included in the event/calendar influence functions. The result
therefore does not treat random aggregation weights as fixed.

## Uncertainty

Both estimators support pointwise analytic influence-function inference:

- `covariance="robust"` uses the package's HC1-style finite-sample scaling
  `sum(IF_i^2) / [n(n-1)]` with a normal reference distribution;
- `covariance="clustered"` sums entity influence functions once by the declared cluster,
  applies `G/(G-1)`, and uses a `t(G-1)` reference distribution.

The paper's random-sampling unit is the panel entity. A higher-level cluster must be
constant within entity and have at least two levels. Pointwise intervals do not control
family-wise error across the event-study path. `inference="multiplier_bootstrap"`
additionally constructs a studentized Rademacher max-t band over all reported event
times. Multipliers are drawn at the entity level for robust inference and once per
declared cluster for clustered inference. The existing HC1/cluster finite-sample scale is
applied before studentization. The realized critical value, level, iteration count, seed,
and lower/upper limits are recorded separately from the pointwise table. A band refuses
if any included event-time standard error is zero or non-finite; no dimension is silently
dropped.

The paper's semiparametric efficiency-bound claim is retained only for independent
entities; requesting higher-level clustered uncertainty does not establish efficiency
under a cluster-dependent model.

## Pre-trend diagnostics

Every no-covariate panel result exposes `result.pretrend`. For cohort `g`, the diagnostic
uses adjacent changes ending strictly before the effective treatment boundary. With
anticipation `A`, every retained placebo therefore has event time at most `-A-1`; the
declared anticipation window is never tested as though it were untreated.

Each placebo compares the treated cohort with the estimator's declared never-treated or
not-yet-treated comparison population and reports its entity influence function. The
joint null sets every retained cohort-period placebo to zero. Robust covariance is the
full HC1-style cross-product of the influence matrix and uses a chi-square reference;
clustered covariance sums the entire vector once per declared cluster and uses the
package's finite-cluster F reference. Singular joint covariance leaves the individual
placebos visible but marks the joint test unavailable. No restriction is dropped and no
ridge or pseudoinverse is applied.

Two-period designs can have no uncontaminated placebo change; this is recorded as
unavailable rather than as a passing test. The current covariate-adjusted efficient path
also records the diagnostic as unavailable because an unadjusted placebo would not test
its conditional PT-All restriction. Failure to reject any pre-trend diagnostic is not
evidence that parallel trends holds.

## PT-All versus PT-Post Hausman diagnostic

`did_hausman_test(pt_post, pt_all)` implements the event-study comparison in Theorem A.1
of Chen, Sant'Anna, and Xie for the maintained no-covariate specialization. It requires a
conventional never-treated PT-Post result and a no-covariate PT-All result using every
admissible pre-period moment, fitted to exactly the same outcome/timing sample and
covariance design.

The tested vector is the common post-treatment event-study path, not only `ESavg`. The
finite-sample covariance is computed directly from the difference between the aligned
PT-All and PT-Post influence functions. This positive-semidefinite construction avoids
subtracting two estimated covariance matrices. A singular difference covariance is
refused without a pseudoinverse or effective-rank change. Rejection is evidence against
the extra PT-All restrictions; failure to reject does not prove PT-All and is not an
automatic estimator-selection rule.

The maintained Hausman slice refuses not-yet-treated PT-Post comparisons, covariate-
adjusted results, reduced `pre_periods`, different samples, roles, timing, covariance, or
clusters. Those comparisons require separately aligned efficient influence functions.

## Repeated-cross-section boundary

Repeated cross sections require observation-level rather than entity-level influence
functions and an explicit composition restriction. They are therefore not a mode switch
on either panel estimator. The public implementation contract, phased nuisance design,
strict refusals, parity targets, and promotion gates are frozen in
[`DID_REPEATED_CROSS_SECTION_CONTRACT.md`](DID_REPEATED_CROSS_SECTION_CONTRACT.md). No
placeholder estimator is exported yet.

## Refusals

The public panel estimators refuse duplicate entity-time rows, unbalanced panels, non-finite
outcomes or time values, varying treatment time, covariates, or cluster within entity, an unobserved
finite adoption time, missing never-treated observations, cohorts without a clean
baseline, undersized cohorts, unsupported repeated cross-sections, sampling weights,
missing nuisance factories, invalid or near-zero cohort probabilities, invalid control
labels, malformed multiplier settings, degenerate event paths requested for simultaneous
bands, and singular unconditional or conditional efficient covariance systems.

## Validation and promotion gates

The maintained tests begin with hand-computed conventional `ATT(g,t)`, event-time,
calendar-time, and `ESavg` identities. A separate eight-entity fixture yields three
orthogonal generated-outcome scores with variances proportional to `1`, `4`, and `16`, so
the efficient weights must be exactly `16/21`, `4/21`, and `1/21`. Refusal behavior is
part of this contract.

The maintained promotion evidence now covers covariate nuisance integration, shared-fold
cross-fitting, conditional covariance inversion and refusal, robust/clustered multiplier
band identities, a seeded coverage smoke, hand-computed uncontaminated pre-trend placebos,
clustered joint tests, and the influence-difference PT-All/PT-Post Hausman diagnostic.
Remaining gates include repeated-cross-section implementation, publication-scale Monte
Carlo studies, larger covariate-performance fixtures, and aligned external parity for the
covariate path.

Primary methodology:

- Xiaohong Chen, Pedro H. C. Sant'Anna, and Haitian Xie (2025), [*Efficient
  Difference-in-Differences and Event Study Estimators*](https://arxiv.org/abs/2506.17729),
  arXiv:2506.17729.
- Brantly Callaway and Pedro H. C. Sant'Anna (2021), [*Difference-in-Differences with
  Multiple Time Periods*](https://doi.org/10.1016/j.jeconom.2020.12.001), Journal of
  Econometrics 225(2), 200-230.
