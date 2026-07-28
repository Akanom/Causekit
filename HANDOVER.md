# Development handover

This file records the active development sequence and implementation constraints so a
future task can resume without reconstructing intent from chat history.

## Completed milestones

1. `0.1.0a1`: cross-sectional `IV2SLS`, strict excluded-instrument semantics, robust and
   clustered inference, identification checks, first-stage diagnostics, and OutputHub.
2. `0.2.0a1`: two-arm `RandomizedATE`, raw difference in means, fully interacted Lin
   adjustment, balance diagnostics, and robust/clustered inference.
3. `0.3.0a1`: supplied-nuisance `IPWATE` and `AIPWATE`, overlap/weight diagnostics,
   influence-function inference, and explicit clipping.
4. `0.4.0a1`: reusable deterministic `CrossFitter`, public nuisance fit/prediction
   protocols, and ATE/ATT/ATC support.

## Active milestone

5. `0.5.0a1` matching point alpha: contract tests were written and observed failing before
   the engine existed. The current implementation covers supplied logit-propensity
   ATT/ATC/ATE matching with replacement, sorted scalar neighbor search, support and
   caliper attrition, deterministic fractional ties, effect/reuse weights, balance, strict
   refusals, deterministic recovery tests, and a reproducible benchmark harness.

   This milestone is not complete. `inference="none"` is intentionally the default.
   Abadie–Imbens reuse-aware variance, the estimated-propensity first-step adjustment,
   external Stata/R parity, OutputHub adaptation, and the remaining promotion matrix are
   open. Do not substitute a generic sandwich, ordinary bootstrap, or cluster wrapper.

## Required implementation patterns

- Reuse the strict alignment, labelled-result, covariance, post-estimation, OutputHub,
  release, and parity conventions established by `limiteddepkit` and `systemgmmkit`.
- Keep numerical fast paths vectorized. Normalize cluster labels once and aggregate scores
  by integer code; do not introduce observation-level Python loops or quadratic matrices.
- A fold loop is permitted where model fitting is inherently fold-specific. Each fold must
  receive fresh nuisance estimators from factories, and every reported nuisance prediction
  must be out of fold.
- Do not copy binary, count, censoring, duration, or ordinal estimators from
  `limiteddepkit`. Integrate their public fit/predict results through protocols or explicit
  prediction adapters. Move model ownership only through a separately reviewed migration.
- Do not claim that overlap diagnostics prove exchangeability, that AIPW repairs unmeasured
  confounding, or that clipping is an innocuous numerical operation.
- Every model addition requires analytical identities, deterministic recovery,
  refusal-path tests, aligned external/reference evidence where available, performance
  smoke coverage, documentation, packaging checks, and an intentional commit.

## Next development order

1. Finish matching promotion from the implemented point alpha in
   `docs/MATCHING_CONTRACT.md`: settle score-provenance-specific analytical inference,
   then add variance identities/coverage, aligned external parity, OutputHub adaptation,
   and the full benchmark matrix without weakening the current refusal boundaries.
2. Difference-in-differences and event studies: define treatment timing, comparison
   cohorts, anticipation, staggered adoption, weighting, and clustered inference.
3. Regression discontinuity: sharp/fuzzy design, bandwidth, polynomial order,
   manipulation checks, bias correction, and local estimand.
4. Panel IV: reuse public `systemgmmkit` panel validation, entity/time indexing, fixed
   effects, and clustered covariance contracts.

Matching implementation began with contract tests, as required. Continue to preserve the
tie/inference, target-population, and no-quadratic-matrix decisions; do not relabel the
point alpha as a completed inferential estimator merely to make an early demo run.
