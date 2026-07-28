# Nearest-neighbor matching contract

Status (2026-07-28): design approved for implementation planning; no public matching
estimator exists yet. This document is normative for the first implementation slice.

## Problem and claim boundary

Nearest-neighbor matching is a design and imputation procedure for binary-treatment
observational studies. It does not discover confounders, prove conditional
exchangeability, repair unmeasured confounding, or turn poor overlap into population-wide
identification. Matching decisions must be based on treatment and pre-treatment data,
without inspecting outcomes.

The identifying assumptions are treatment consistency, no interference, conditional
exchangeability given the declared pre-treatment covariates, and positivity for the
requested target population. Matching can improve observed balance and reduce reliance on
outcome extrapolation; balance is not evidence about unobserved confounding.

## Proposed public boundary

The first estimator should be named `NearestNeighborMatch`, with a fitted
`NearestNeighborMatchResult`. The point-estimation API should accept supplied distance
inputs rather than fit a duplicate propensity model:

```python
NearestNeighborMatch(
    estimand="att",
    metric="propensity_logit",
    neighbors=1,
    replacement=True,
    caliper="auto",
    common_support="intersection",
    ties="all",
    bias_correction="none",
    inference="abadie_imbens",
).fit(
    y,
    treatment=...,
    propensity=...,
    covariates=...,
)
```

The propensity input may come from `CrossFitter` or another auditable out-of-sample
workflow. The matcher owns matching, matched-sample weights, diagnostics, estimand labels,
and valid uncertainty; it does not own binary-response nuisance estimation.

## Estimand and target population

Accepted estimands are exactly `"att"`, `"atc"`, and `"ate"`; the default is `"att"`.

- `att`: treated observations are focal units and controls provide counterfactual matches.
- `atc`: controls are focal units and treated observations provide counterfactual matches.
- `ate`: both treatment arms are focal in separate directions and missing potential
  outcomes are imputed for every retained unit.

The result must expose both `requested_estimand` and `realized_estimand`. If common-support
or caliper rules remove focal observations, the result must not continue to label the
estimate as the full-sample ATT, ATC, or ATE. It becomes the corresponding effect for the
explicit retained overlap population, for example `att_matched_support`. The result must
report retained and excluded counts and percentages by treatment arm plus the exact
retained index.

ATE matching must not be implemented as an undocumented average of separate ATT and ATC
estimates. It uses unit-level imputation in both directions over the retained overlap
population, with weights implied by that construction.

## Distance metric and scaling

The first stable metric is `"propensity_logit"`: absolute distance on
`log(p / (1 - p))` from a supplied propensity strictly between zero and one. Matching on
the raw propensity may be added as `"propensity"` for sensitivity analysis, but it must
not be an alias because raw and logit scales imply different neighborhoods and calipers.

Mahalanobis distance is deferred from the first implementation. It requires a separate
contract for covariance estimation, singular or high-dimensional covariates, mixed data
types, exact constraints, and scaling. The first matcher may still accept `covariates` for
balance diagnostics and bias correction; those values do not silently alter the matching
metric.

Propensity values are never clipped by the matcher. Invalid values at zero or one raise.
If a caller used clipped propensities upstream, that provenance must remain visible in the
analysis record.

## Replacement

The stable first slice supports `replacement=True` only. Replacement is order-invariant
apart from the declared tie rule, permits the closest comparison to serve multiple focal
units, and matches the fixed-neighbor asymptotic framework selected for inference.

`replacement=False` must raise `NotImplementedError`, not silently switch algorithms.
Without-replacement greedy matching depends on focal-unit order, while globally optimal
matching is a distinct assignment problem with different computational and inferential
contracts. It should be designed as a later estimator or explicit algorithm choice.

The result must report comparison-unit reuse counts, the maximum reuse count, the number
of unique comparison observations, and matching weights.

## Caliper

`caliper` is defined on the selected metric scale and is inclusive: a candidate is
eligible when `distance <= caliper` within a documented floating-point tolerance.

- `caliper="auto"` is the proposed default for `metric="propensity_logit"`: `0.2` times
  the sample standard deviation of the supplied logit propensity, computed before support
  exclusions. This is a simulation-supported starting rule, not a universal optimum.
- A positive finite numeric value is interpreted directly on the selected metric scale.
- `caliper=None` is allowed only as an explicit opt-out and adds a result warning that
  arbitrarily distant matches were permitted.

The stored result must include the requested caliper rule, realized numeric width,
caliper scale, standard-deviation convention, and unmatched counts. Auto-caliper analyses
must be accompanied by sensitivity results over at least one narrower and one wider value
before publication-facing claims.

If fewer than `neighbors` eligible matches exist, the focal unit is unmatched; the
algorithm does not silently reduce the requested neighbor count. If no focal units remain,
fit raises. A configurable partial-neighbor policy is deferred.

## Ties

The stable point-estimation rule is `ties="all"`. When multiple candidates tie at the
`neighbors` boundary within the declared numeric tolerance, all boundary ties are retained
and divide the boundary match weight equally. This is deterministic and invariant to input
row order. The result reports the requested neighbor count, realized match-count
distribution, number of boundary-tie events, and maximum tie multiplicity.

`ties="first"` is rejected because row-order or identifier-based selection makes the
estimate depend on an arbitrary ordering. Random tie-breaking is also excluded from the
stable slice because a seed does not make the resulting estimand scientifically less
arbitrary.

This tie rule creates an inference restriction: fixed-neighbor Abadie–Imbens analytical
variance is available only when no boundary tie expands the realized neighbor count.
When boundary ties occur, `inference="abadie_imbens"` must refuse with an actionable
message. The user may report a point estimate with `inference="none"`; the package must
not substitute an unvalidated standard error.

## Common support

`common_support="intersection"` is the default. On the matching metric, the eligible
interval is

```text
[max(min(metric | T=0), min(metric | T=1)),
 min(max(metric | T=0), max(metric | T=1))].
```

This is a transparent convex-hull rule on the scalar metric, not proof of multivariate
overlap. The result stores interval bounds and exclusions by arm. `common_support=None`
is permitted only explicitly and adds a warning.

Support restriction happens before neighbor search. Caliper failure is recorded
separately from support exclusion. The package never trims observations silently or
relabels a restricted-support estimate as the original population effect. More elaborate
optimal trimming changes the target population and belongs in a later, explicitly named
design contract.

## Balance diagnostics

Balance uses supplied pre-treatment `covariates`; outcome data never enters matching or
balance selection. The result reports before and after matching:

- weighted means by treatment arm;
- standardized mean differences using the pre-match pooled standard deviation as the
  fixed denominator;
- variance ratios where both variances are positive;
- maximum absolute empirical-CDF difference for numeric covariates;
- missing/constant-variable status; and
- aggregate maximum and mean absolute standardized differences.

An absolute standardized difference above `0.1` may trigger a heuristic warning but is
not a pass/fail identification test. Balance p-values are not reported: they confound
balance with sample size and encourage significance-driven design selection.

The matched weights used for the effect estimate must also be used for post-match balance.
Diagnostics must not pretend reused comparison observations are independent duplicates.

## Point estimate and optional bias correction

For each focal unit, the missing potential outcome is the weighted mean outcome among its
eligible matches. The uncorrected effect is the requested target-population average of
unit-level observed-minus-imputed contrasts, with sign normalized as `Y(1) - Y(0)`.

`bias_correction="none"` is the stable initial default. Abadie–Imbens regression bias
correction may be added only with a separate supplied/fitted outcome-regression protocol,
cross-fitting or a justified fixed-complexity contract, and independent recovery tests.
The matcher must not import or duplicate regression estimators from `limiteddepkit`.

## Uncertainty

Accepted values are `"abadie_imbens"` and `"none"`; the proposed default is
`"abadie_imbens"`.

The analytical variance implementation must follow the fixed-neighbor, replacement
framework and account for comparison-unit reuse. It is available only when its maintained
conditions hold: fixed realized neighbor count, no expanded boundary ties, independent
sampling units, and adequate observations in both arms for conditional variance
estimation. The result stores the variance method, neighbor count, reuse diagnostics,
reference distribution, and every finite-sample convention.

Ordinary nonparametric bootstrap is prohibited for the stable fixed-neighbor estimator.
Abadie and Imbens show that it is generally invalid even where the estimator is root-N
consistent and asymptotically normal. Subsampling or other modified resampling requires a
separate method and validation contract; it is not exposed under the name `bootstrap`.

Clustered, paired, survey, and multiway uncertainty are deferred. Supplying cluster labels
must not merely wrap matched outcomes in the package's generic CR1 kernel because matching
selection and reuse affect dependence. Until derived and validated, clustered matching
inference raises `NotImplementedError`.

## Strict data and failure contract

- Treatment is one-dimensional, contains both arms, and is coded exactly `0/1`.
- Outcome, treatment, propensity, covariates, and any future cluster input have identical
  row counts and exact pandas index order.
- Missing or non-finite values raise in the stable first slice; no implicit complete-case
  deletion or imputation occurs.
- Propensities are finite and strictly between zero and one.
- `neighbors` is a positive integer and cannot exceed available opposite-arm candidates
  after support restrictions.
- Covariate names are unique and reserved internal labels cannot collide.
- Constant covariates remain visible in diagnostics but receive undefined SMD/variance
  ratio with an explicit status rather than division by zero.
- Fit fails when no focal observations, no opposite-arm candidates, or no caliper-eligible
  matches remain.

## Result and audit surface

`NearestNeighborMatchResult` must include:

- requested and realized estimand and target-population text;
- effect, valid uncertainty when available, confidence interval, and inference metadata;
- original, support-eligible, matched, and unmatched counts by arm;
- exact retained and excluded indices with exclusion reasons;
- focal-to-comparison match table containing distance, rank, tie group, and fractional
  match weight;
- observation-level analysis weights and comparison reuse counts;
- metric, propensity provenance, caliper, replacement, tie, support, neighbor, and bias-
  correction metadata;
- before/after balance diagnostics;
- assumptions, warnings, and refusal-relevant notes; and
- dependency-free Markdown plus optional OutputHub adaptation without re-estimation.

Returned pandas data are defensive copies. Match tables and diagnostics must not include
outcome values by default, which limits accidental disclosure when exported.

## Performance contract

For scalar propensity distance, sort treated and control metrics once and use binary
search/local expansion; do not materialize the full `n_treated x n_control` distance
matrix. The target is approximately `O(n log n + n * local_candidates)` time and `O(n +
matches)` memory. Tie and caliper logic must operate on sorted local neighborhoods.

Benchmarks must include balanced and highly imbalanced arms, weak and strong overlap,
heavy ties, caliper failure, and comparison-unit reuse. Record sample sizes, realized
matches, elapsed time, and peak memory. A 100,000-row scalar-score smoke path is required
before promotion.

## Validation and promotion gates

Implementation cannot be called complete until all gates pass:

1. hand-computed tiny examples for ATT, ATC, and bidirectional ATE;
2. invariance to row permutation under `ties="all"`;
3. exact caliper-boundary, support-boundary, tie-weight, and reuse identities;
4. deterministic simulations with constant and heterogeneous effects;
5. analytical-variance identity tests and coverage simulations in maintained conditions;
6. explicit refusal tests for invalid bootstrap, no replacement, expanded ties with
   analytical inference, clustered inference, bad propensities, and empty matches;
7. parity against an aligned implementation such as Stata `teffects nnmatch` or R
   `Matching`, with exact mapping of estimand, metric, replacement, neighbors, ties, bias
   correction, and variance conventions;
8. weighted balance identities and no outcome access during design construction;
9. 100,000-row runtime/memory smoke without a quadratic distance matrix;
10. OutputHub, artifact-content, clean-wheel, lint, format, typing, and full-suite gates.

## Pre-mortem

| Likely failure | Early warning | Mitigation/required response |
| --- | --- | --- |
| Estimate is labelled full-sample ATT after focal exclusions | Matched focal count is below support-eligible count | Store requested and realized estimand separately; block misleading summary label |
| Arbitrary row order changes tied matches | Permutation test changes effect or reuse counts | Stable `ties="all"`; reject first/random tie modes |
| Generic bootstrap produces plausible but invalid intervals | Bootstrap option appears without method-specific theory | Prohibit ordinary bootstrap; implement and certify analytical variance first |
| Caliper improves balance by discarding most focal units | Low matched focal fraction or sensitivity across widths | Report attrition prominently and require caliper sensitivity evidence |
| Propensity balance hides multivariate imbalance | Good propensity SMD but poor covariate SMD/eCDF | Require covariate-level before/after diagnostics and preserve warnings |
| Implementation allocates a quadratic distance matrix | Memory rises with product of arm sizes | Sorted scalar search, peak-memory benchmark, and 100k-row gate |
| Without-replacement result depends on greedy order | Estimate changes after row permutation | Do not support it in the first stable slice; design optimal assignment separately |

## Methodological basis

- Abadie and Imbens establish large-sample properties and analytical variance for fixed-
  neighbor matching: <https://doi.org/10.1111/j.1468-0262.2006.00655.x>.
- Abadie and Imbens show ordinary bootstrap failure for fixed-neighbor matching:
  <https://doi.org/10.3982/ECTA6474>.
- Abadie and Imbens develop regression bias correction for matching estimators:
  <https://economics.mit.edu/sites/default/files/publications/Bias-Corrected%20Matching%20Estimators%20for.pdf>.
- Rosenbaum and Rubin develop propensity-score-informed matched sampling:
  <https://doi.org/10.1080/00031305.1985.10479383>.
- Austin's simulations motivate `0.2` standard deviations of the logit propensity as a
  starting caliper for mean/risk-difference settings, not a universal rule:
  <https://doi.org/10.1002/pst.433>.
- Crump et al. show that limited overlap affects precision and specification sensitivity
  and that restricting the population changes the estimand:
  <https://doi.org/10.1093/biomet/asn055>.
