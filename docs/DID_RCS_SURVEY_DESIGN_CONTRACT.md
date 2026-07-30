# Survey-design contract for repeated-cross-section DiD

## Status and boundary

This contract is implemented for stationary-composition repeated cross sections with and
without covariates. CauseKit continues to refuse bare `sampling_weights` because a numeric
column alone does not identify its meaning, target population, sampling design,
nuisance-fit behavior, or variance estimator. Public
`RepeatedCrossSectionSurveyDesign` and `WeightedNuisanceEstimatorProtocol` freeze those
roles. Composition-robust combinations and survey-valid simultaneous bands remain outside
this promoted slice.

Survey design is separate from composition-change robustness. Sampling weights transport
from sampled observations to a declared finite or superpopulation target; they do not by
themselves balance the four group-period covariate distributions or rescue an invalid
stationary-composition assumption. Balanced-panel direct cohort ratios are also separate.
Every future combination must receive an explicit score and inference contract rather
than multiplying weights from independent features.

## Weight semantics and target population

The public survey path accepts inverse inclusion or calibrated analysis weights,
not frequency weights, exposure weights, precision weights, matching weights, or
user-invented causal weights. The result must distinguish:

- `target_population="sample"`, the current unweighted observation/PSU estimand; and
- `target_population="survey_population"`, a design-weighted population estimand.

For the population target, every effect also declares its reference population. The first
supported choice is the treated target-period population for each cohort-time effect,
matching the composition-robust ATT definition when that later combination is promoted.
No option may silently pool pre- and post-period treated populations. The result, summary,
OutputHub, graph data, and exported benchmark metadata must record the weight meaning,
target, wave, and normalization convention.

Multiplying all analysis weights by a common finite positive constant must leave every
Hájek point estimate, normalized component, target share, and Taylor-linearized influence
unchanged. Weight normalization is part of each estimand component; CauseKit does not
overwrite the user column or pretend normalized weights are known inclusion probabilities.

## Public design object

The immutable public object and fit role are:

```python
RepeatedCrossSectionSurveyDesign(
    weights=...,
    psu=...,
    strata=None,
    weight_type="inverse_inclusion",
    singleton_psu="raise",
)

RepeatedCrossSectionDiD(...).fit(
    ...,
    survey_design=design,
    target_population="survey_population",
)
```

The name was frozen by failing API tests before implementation. Inputs may be aligned
column names or labelled one-dimensional objects following the package's existing index
contract. The object validates and records roles; the estimator owns the target-specific
score and inference. `weight_type` initially accepts only documented inverse-inclusion or
calibrated analysis weights. A raw `sampling_weights=` vector continues to refuse because
it cannot express the design. `target_population="sample"` remains the existing
unweighted path and cannot be paired with a survey design merely to change its variance.

The base estimator is no-covariate, `composition="stationary"`, and
`target_population="survey_population"`. Its inference uses a with-replacement,
one-stage stratified PSU Taylor-linearization design. An observation is its own PSU only
when the user explicitly requests the independent-observation design. Replicate-weight
designs, finite-population corrections, multistage designs, certainty PSUs, and lonely-
PSU adjustment policies remain absent until separately derived and tested; their fields
are not added as placeholders.

## Point estimation and nuisance fitting

Every survey-weighted cell mean or augmented score component uses explicit Hájek
normalization under the declared target. Cohort, event-time, calendar-time, and ES-average
aggregation uses design-weighted target-population shares and includes the influence of
estimating those shares. Unweighted sample shares are not reusable.

When covariates are used, nuisance training honors the survey analysis weights in the
outer training partition. The existing provider protocol `fit(X, y)` cannot silently drop
them. The runtime-checkable protocol is:

```python
class WeightedNuisanceEstimatorProtocol(Protocol):
    def fit(self, X: object, y: object, *, sample_weight: object) -> object: ...
```

Probability and scalar-prediction methods continue to follow the current public nuisance
protocols. `CrossFitter` passes aligned training-only weights and records a weight hash,
sum, effective sample size, fold, task, and provider diagnostics. The required keyword
signature proves that weights were passed, not that arbitrary provider internals used
them correctly; provider compliance remains an explicit assumption and sentinel models
test the integration. CauseKit never catches a `TypeError` and retries without weights.
Inner tuning is training-only and weighted under the same declared design.

The stationary covariate score has its own eight-component survey-weighted ratio influence
and passes weights through construction-only `CrossFitter` tasks. Direct density ratios and
the composition-robust four-cell score remain separate. Each future combination needs a
hand-derived weighted influence function and its own rate conditions; naively multiplying
an existing influence vector by survey weights is prohibited.

## Design-based uncertainty

A sampling-weighted HC1 or ordinary PSU CR1 sandwich is not automatically a survey-design
variance. For the first Taylor slice, CauseKit forms the estimator's linearized variable,
aggregates it within PSU inside stratum, and applies the preregistered with-replacement
stratum-by-PSU variance formula and design degrees of freedom. The result reports stratum
and PSU counts, design degrees of freedom, and singleton handling. A stratum with fewer
than two noncertainty PSUs refuses in the first slice.

Replicate-weight inference is a later independent gate. It must name and validate the
replication method, full-sample and replicate scaling constants, Fay factor where
applicable, and whether nuisances are refitted for every replicate. BRR, Fay BRR,
jackknife, and bootstrap weights are not interchangeable labels.

The current observation/PSU Rademacher max-t bands are model-based multiplier bands from
fixed influence scores. They cannot be relabeled survey-valid. Survey simultaneous bands
must use a design-justified joint replication or linearization procedure, preserve PSU and
stratum roles, and pass their own fixed-seed joint-coverage certificate.

## Required refusals

The survey path must refuse:

- bare weights without a validated survey-design object;
- unknown weight semantics or target population;
- missing, nonnumeric, complex, nonfinite, zero, or negative analysis weights;
- row/index drift among data, weights, strata, PSUs, FPCs, or replicate columns;
- a zero weighted denominator, zero weighted target share, or inadequate weighted support
  in any group-period component or training fold;
- inconsistent PSU nesting within strata, fewer than two noncertainty PSUs per stratum,
  or a requested singleton policy other than the implemented strict refusal;
- unsupported FPC, multistage, certainty-PSU, replicate method, or replicate scaling;
- unweighted nuisance providers, ignored `sample_weight`, holdout-weight leakage, or
  fold-specific rescaling that changes the training target;
- use of ordinary HC1/CR1 or the current Rademacher band under a survey-valid label;
- attempts to combine the survey design with composition robustness or direct-ratio
  nuisances before the relevant combined contract is promoted; and
- claims that weights establish sampling ignorability, conditional parallel trends,
  treatment overlap, or correct population coverage.

There is no automatic weight trimming, winsorization, calibration, nonresponse repair,
singleton-stratum adjustment, FPC inference, or fallback to unweighted estimation.

## Diagnostics

Every weighted result exposes:

- raw and normalized weight summaries, Kish effective sample size, maximum normalized
  share, and design effect diagnostics overall and by used group-period cell;
- unweighted and weighted cell counts and target-population shares;
- stratum, PSU, singleton, and design-degrees-of-freedom audits;
- fold-level weighted support and nuisance-consumption records when covariates are used;
- the exact point and variance normalization formulas; and
- sensitivity rows comparing the sample and survey-population targets when both are
  substantively identified and preregistered.

Diagnostics never prove representativeness or justify trimming. Severe concentration is
a visible refusal under preregistered thresholds, not a hidden adjustment.

## Completed failing-first promotion sequence

Before implementation, tests were observed failing for:

1. a hand-computed weighted two-by-two population ATT, each normalized component, target
   share, full linearized variable, and stratified-PSU variance;
2. exact invariance to positive global rescaling of analysis weights;
3. a deliberately informative-sampling example where the sample and population targets
   differ in the expected direction;
4. immutable index/PSU/stratum roles, whole-PSU fold assignment, and training-only weight
   consumption by nuisance providers;
5. every invalid weight, empty weighted cell, singleton stratum, unsupported design, and
   forbidden-combination refusal; and
6. independent survey-variance and design-degrees-of-freedom reconstruction.

Promotion evidence uses nonlinear simulations with informative and noninformative
sampling, unequal wave sizes, treatment-effect heterogeneity, weight concentration, and
multiple strata/PSUs. Bias, empirical/analytical SE calibration, pointwise coverage,
target recovery, scale invariance, support, refusal rate, runtime, and memory gates are
preregistered. A later simultaneous-inference phase adds joint coverage and fixed-seed
replication gates.

External evidence must use an aligned survey target and design. Base R plus the R `survey`
ecosystem and reviewed Stata `svy` calculations are candidates for the Taylor slice, but
each row must record exact strata/PSU/FPC/weight mapping and finite-sample correction.
The R `compdid` `i.weights` option is a useful point-score comparator, not proof of full
complex-survey inference. A versioned public survey example must include data provenance,
weight documentation, target-population interpretation, and reproducible download hashes;
restricted microdata are not committed.

The maintained R 4.5.1 / `survey` 4.5 fixture reconstructs the four component totals,
nonlinear Hájek contrast, complete linearized variable, stratified-PSU covariance, and
design degrees of freedom. The reviewed Stata/IC 17 `.do` file passes the same official
`svy: total` plus `nlcom` mapping and writes results before assertions. The pinned
YRBS sensitivity exactly reproduces the authors' sampling-weight-only four-mean point
calculation. Their processed public file omits PSU identifiers, so CauseKit explicitly
uses independent rows within the retained strata and does not claim that sensitivity's
standard error reproduces the paper's bootstrap. The paper's proposed composition-balancing
IPW target is also distinct from CauseKit's stationary-composition survey score.

The exact publication-scale coverage, hash-pinned YRBS sensitivity, and fixed-size
performance results are maintained in
[`DID_RCS_SURVEY_PROMOTION_EVIDENCE.md`](DID_RCS_SURVEY_PROMOTION_EVIDENCE.md).

## Pre-mortem

Likely failures are accepting a generic `weight` column with unknown meaning, changing the
estimand through silent normalization, fitting nuisances without weights, treating CR1 as
a survey variance, and multiplying composition and sampling weights without deriving the
combined influence. The explicit design object, target metadata, weighted-provider
protocol, hand linearization, and combination refusals are the required defenses.

## Primary methodology

- Kerry Ye, Alyssa Bilinski, and Youjin Lee (2025), [*Difference-in-differences analysis
  with repeated cross-sectional survey data*](https://doi.org/10.1007/s10742-025-00364-7).
- Pedro H. C. Sant'Anna and Qi Xu (2026), [*Difference-in-Differences with Compositional
  Changes*](https://doi.org/10.1016/j.jeconom.2025.106147), for the distinct
  composition-robust sample score and influence benchmark.
