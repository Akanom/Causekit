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
5. `0.5.0a1` matching point alpha: supplied logit-propensity ATT/ATC/ATE matching with
   replacement, sorted scalar neighbor search, support/caliper attrition, deterministic
   fractional ties, effect/reuse weights, balance, strict refusals, deterministic recovery
   tests, and a reproducible benchmark harness.

## Continued completed milestones

6. `0.6.0a1` DiD alpha: conventional DiD remains a separate first-class estimator with
   never-treated or not-yet-treated comparisons. The Chen–Sant'Anna–Xie efficient
   estimator is opt-in under PT-All. Both use one strict balanced-panel/treatment-timing
   validator, explicit anticipation, cohort/event/calendar effects, entity influence
   functions, HC1-style or higher-level clustered pointwise inference, and OutputHub.

   Contract and hand-computed tests were written and observed failing before `did.py`
   existed. The efficient fixture recovers weights `16/21`, `4/21`, and `1/21`, and matches
   the public R `edid` implementation at commit
   `f55a4a4aba14f0826f59ad7aa4af3bafaeba529b` for candidate effects, weights, the combined
   ATT, and HC1 standard error.

   This milestone is a no-covariate short-panel alpha, not the full paper implementation.
   Covariate-adjusted public nuisance integration, the conditional covariance estimator,
   multiplier-bootstrap simultaneous bands, pre-trend/Hausman diagnostics, repeated
   cross-sections, coverage simulations, and broader parity remain open.

7. `0.6.0a2` DiD nuisance/inference promotion: `CrossFitter` now provides reusable
   multiclass class-probability and masked scalar-regression tasks. The covariate-adjusted
   `EfficientDiD` path uses those tasks for cohort probabilities, group-specific outcome
   changes, and residual-product conditional covariances; it then solves the paper's
   observation-specific equation (3.12) systems without hidden repair. Both DiD classes
   support reproducible entity- or declared-cluster Rademacher max-t simultaneous
   event-study bands. Exact recovery, multi-moment, robust/cluster band, singular-refusal,
   and seeded coverage-smoke tests are maintained.

8. `0.6.0a3` matching known-score inference promotion: the Abadie-Imbens path now
   estimates same-arm conditional variances and applies exact ATT/ATC/ATE comparison-
   reuse formulas for a declared fixed scalar score. It reports normal inference and full
   variance audit fields, refuses estimated/cross-fitted scores and target-changing
   support/caliper selection, exports OutputHub design tables, matches pinned CRAN
   `Matching` 4.10-15 and Stata/MP 17 `teffects nnmatch`, and retains both comparator
   harnesses plus the reviewed Stata output. The 100,000-row inference scenario completes
   in linear memory without a distance matrix.

9. `0.6.0a4` matching estimated-Logit inference promotion: the public
   `FittedPropensityMLEProtocol` consumes a provider-neutral regular full-sample
   unpenalized Logit result. CauseKit validates fit/sample/schema/score/information
   identities and applies separate Abadie–Imbens ATE/ATT/ATC first-step
   corrections. Hand contracts, strict refusals, independent `statsmodels` parity,
   seeded coverage smoke, OutputHub audit fields, and a 100,000-row 85.20 MiB benchmark
   are implemented. Reviewed Stata/IC 17 parity
   establishes that `vce(robust, nn(2))` maps to CauseKit's one-neighbor known-score
   variance and two leave-own-out local first-step regression neighbors. ATT/ATC/ATE
   point estimates, variance components, and final standard errors pass at `1e-8`.

10. `0.6.0a4` ownership and real-data release gate: the unreleased distribution/import
    identity is now `causekit`, with no shim under the occupied `causalkit` name. CauseKit
    owns all migrated causal models and has no LimitedDepKit runtime or validation
    dependency. Four opt-in HTTPS datasets are pinned by SHA-256, a runnable example covers
    every model family, and the real-data Python/R certificate passes 14 estimand-aligned
    comparisons. The reviewed Stata/IC 17 saved-output certificate also passes every
    comparable family; efficient PT-All DiD remains explicitly unavailable in Stata.

11. `0.7.0a1` native causal-ML alpha: `PartiallyLinearDML` implements the pooled DML2
    score with shared `CrossFitter` folds, a CauseKit-owned standardized ridge-GCV default
    fitted separately inside every outer training fold, HC1/one-way CR1 inference, strict
    residual-treatment identification, full OOF audit records, OutputHub adaptation, and
    a hash-verified Cattaneo real-data workflow. Hand, refusal, simulation, Statsmodels,
    base-R 4.5.1, and manually executed Stata/IC 17 contracts pass. A one-run real-data
    performance comparison records native and optional external learner results without
    adding an ML dependency.

12. `0.7.0a2` ridge audit and heterogeneous-effect design: the native GCV path now uses
    SVD shrinkage identities without fitted-vector reconstruction. A denser grid lowered
    training GCV but slightly worsened the same real-data OOF errors and was rejected as
    the default. `NuisanceDiagnosticsProtocol` carries scalar per-task/fold
    tuning records through CrossFitter, DML results, and OutputHub. The five-fold Cattaneo
    smoke was revalidated without rerunning external comparators. The honest R-learner
    construction/evaluation, R-loss, calibration, grouping, uncertainty, graphing, and
    refusal contract is recorded; no placeholder estimator is exported.

    A separate one-run NSW job-training benchmark now exercises the same scalar DML path
    on 445 observations and eight pre-treatment covariates. CauseKit native ridge-GCV had
    the lowest outcome OOF RMSE (6608.467270), runtime (0.0456 seconds), and
    Python-managed peak memory (0.210 MiB); fold-local standardized scikit-learn RidgeCV
    had a 0.180% lower treatment OOF RMSE. Histogram boosting and random forest were worse
    on both nuisance targets. This supports keeping native ridge-GCV as the efficient
    default, not a universal superiority or CATE claim. The Cattaneo rows were not rerun.

13. `0.7.0a2` R-learner prerequisites: public `WeightedCATEEstimatorProtocol` and
    `CATEResultProtocol` define the provider-neutral weighted-fit/prediction boundary.
    Internal native components implement standardized ridge-penalized binary Logit with
    deterministic stratified training-only CV, plus weighted ridge-GCV whose transformed
    loss is exactly the direct R-objective. Both retain construction indices and tuning
    diagnostics. Hand score/objective, integration, leakage-audit, weight, schema, index,
    malformed-provider, and prediction refusals pass. No `RLearner` placeholder is
    exported; honest splitting and every evaluation/promotion gate remain open.

14. `0.7.0a2` honest R-construction layer: deterministic treatment-stratified row roles
    and balanced whole-cluster roles are retained behind immutable copy-out accessors.
    Cluster-aware outer folds keep every cluster intact and retain both treatment arms.
    Outcome and propensity nuisances are cross-fitted on identical construction-only
    folds; fresh full-construction refits provide evaluation nuisance predictions, and the
    weighted CATE model is fit only on construction. The direct R-objective and transformed
    `u/v`, `v^2` objective pass an exact identity contract. This remains internal: honest
    R-loss, calibration, group inference, graphs, and all promotion evidence are open.

15. `0.7.0a2` public honest R-learner alpha: `RLearner` consumes the immutable evaluation
    role once for held-out R-loss against a construction-fitted constant, differential
    calibration, and tie-preserving overlap-weighted groups. HC1/CR1 covariance, seeded
    max-t group bands, influence records, exact graph data, optional plots, OutputHub,
    and future-data prediction are integrated. Linear/null/power/band-coverage simulations,
    base-R fixed-evaluation parity, a manual Stata HC1 fixture, and a 2,000-row/500-cluster
    smoke are maintained. A one-run NSW comparison changed only the weighted CATE learner:
    external RidgeCV narrowly minimized R-loss but improved on constant by just 0.0034%;
    native ridge-GCV was 0.99% worse than constant, both tree learners were materially
    worse, and no differential calibration test supported heterogeneity. The earlier
    scalar-DML records were not rerun.

16. `0.7.0a2` native nonlinear CATE stage and final evaluation parity:
    `NativeSplineRidgeCATE` is an opt-in, dependency-free additive linear-spline stage.
    Construction-only weighted GCV selects zero/one/three knots and the ridge penalty;
    exact basis data, selected complexity, candidate path, schema, and a strict feature
    ceiling are public. A known piecewise CATE recovery/refusal contract passes. The new
    NSW row was 1.43% worse than constant and was added without rerunning old rows. On the
    42,613-customer hash-pinned Hillstrom randomized-email contrast, spline and linear
    stages had identical displayed honest metrics and a 0.0334% gain over constant. The
    reviewed Stata/IC 17 fixed-evaluation artifact passes at `1e-8` with maximum absolute
    difference `1.55e-15`. Linear ridge remains the default.

17. `0.7.0a3` separately contracted honest DR learner: `DRLearner` cross-fits propensity
    and both arm-specific outcome regressions inside immutable construction roles, forms
    the canonical augmented inverse-probability score without clipping, and fits a
    separate unweighted `CATEEstimatorProtocol`. Fresh full-construction nuisances and the
    construction CATE model predict the untouched evaluation role. HC1/CR1 differential
    calibration, tie-preserving mean-score groups, seeded max-t bands, graph data,
    OutputHub, future-data prediction, score/loss/influence audit records, exact hand and
    double-robustness simulations, base-R parity, and a hash-verified NSW comparison are
    implemented. Native ridge-GCV exactly matched optional scikit-learn RidgeCV on that
    split while using less Python-managed peak memory; boosting and forest were worse.
    The reviewed Stata/IC 17 fixed-evaluation artifact passes every field at `1e-8`; the
    maximum absolute difference is `4.44e-16`.

18. `0.7.0a4` matching evidence promotion: the maintained known-score and regular
    full-sample unpenalized Logit-MLE analytical paths are unchanged, but now carry a
    preregistered publication-scale certificate. One thousand replications in each of
    favorable and stressed overlap exercise ATT, ATC, and ATE for both contracts (12,000
    fits); every cell passes with coverage `0.943–0.955`, maximum absolute bias `0.0073`,
    analytical-SE/empirical-SD ratios `0.963–1.019`, zero refusals, and coverage Monte
    Carlo SE at most `0.0073`. A separate 15-row, hash-verified Cattaneo grid compares no
    support rule, intersection support, and `0.1`/`0.2`/`0.3` logit-score-SD calipers with
    `inference="none"`. It retains every target relabel and attrition count; support removes
    10 controls and the narrow caliper removes three more control focal units. Settled
    R/Stata parity and 100,000-row performance artifacts were not rerun.

19. `0.7.0a5` DiD diagnostic contract: no-covariate panel results now expose adjacent
    uncontaminated cohort-period pre-trend placebos, complete entity influence/covariance
    records, and robust chi-square or cluster-summed finite-cluster F joint tests. No-lead
    and singular cases are explicit and receive no numerical repair. The public Hausman
    diagnostic compares the common post-treatment event-study vector under aligned
    never-treated PT-Post and no-covariate PT-All fits, using the covariance of the
    difference influence function and strict design fingerprints. A separate repeated-
    cross-section contract freezes observation/PSU scores, stationary-composition
    semantics, refusals, phased CrossFitter integration, parity targets, coverage, and
    performance gates without exporting a placeholder estimator. A nonlinear-only
    Hillstrom real-data smoke was rerun without the settled linear row: the adaptive spline
    stage reproduced the frozen honest metrics and explicitly selected the zero-knot
    linear submodel. A new conversion-outcome comparison selected one knot but had 0.0019%
    higher honest R-loss than linear ridge; neither stage beat the construction-fitted
    constant, so no nonlinear performance gain is claimed.

20. `0.7.0a6` repeated-cross-section DiD first slice: public
    `RepeatedCrossSectionDiD` is separate from both balanced-panel classes and implements
    conventional no-covariate cohort-time effects under declared stationary composition.
    Four independent group-period means, fixed comparison membership, unequal cell sizes,
    pooled estimated-share aggregation, full observation influences, HC1/CR1 inference,
    independent-cell placebos, cell audits, and OutputHub are integrated. Hand/refusal
    tests were observed failing before implementation; staggered, permutation,
    anticipation, covariance, aggregation, base-R 4.5.1 hand parity, small coverage,
    public-data, and 100,000-row smokes pass. The reviewed Stata 17 hand artifact also
    passes the estimate and HC1 standard error at `1e-12`. A 1,000-replication-per-
    design promotion certificate now passes all 32 group/event/calendar/ESavg cells for
    both control rules, and pinned R `did` 2.5.0 estimator-level parity passes after the
    explicit HC0-to-HC1 finite-sample mapping. The reviewed Stata/IC 17 `csdid` artifact
    exactly matches all 16 point estimates and all six group-time analytical standard
    errors. Its aggregate standard errors remain explicitly non-comparable because Stata
    propagates period-specific cell-share influence while CauseKit/R use pooled cohort
    shares. Covariates, compositional-change robustness, survey weights, and simultaneous
    bands remain open.

21. `0.7.0a6` covariate repeated-cross-section DiD promotion: the same separate public
    surface now accepts covariates only with an explicit provider-neutral `CrossFitter`.
    One propensity and four group-period outcome regressions are fitted out of fold on a
    shared cohort-period-stratified observation/whole-PSU split for every group-time and
    conditional-placebo comparison. The normalized eight-component Sant'Anna-Zhao locally
    efficient score, ratio influence terms, pooled-share aggregation, HC1/CR1 inference,
    strict overlap refusal, diagnostics, and OutputHub are integrated. Hand tests were
    observed failing first. Both double-robustness legs, row/PSU leakage, staggered
    never/not-yet-treated identities, conditional pre-trends, seeded reproducibility, a
    40-replication bias/coverage smoke, independent base-R score parity, and a hash-
    verified 7,368-row hospital/PSU covariate smoke pass. A separate 100,000-row,
    six-period, 100-fit benchmark completes in 1.270 seconds with 244.743 MiB Python-
    managed peak memory on the recorded environment. The hospital source is artificial
    and is execution evidence, not a substantive causal result.

22. `0.7.0a6` covariate repeated-cross-section publication promotion: a hash-bound,
    deterministic 1,000-replication certificate exercises favorable balanced and
    stressed-overlap unequal-period designs under both control rules. All 44
    group/aggregate/conditional-placebo pointwise cells pass with coverage `0.940–0.964`,
    mean-SE/empirical-SD ratios `0.954–1.048`, maximum absolute bias `0.0089`, and zero
    refusals across 4,000 estimator and 240,000 fold-local nuisance fits. All four joint
    conditional-pre-trend size cells pass at `0.049–0.057`. The initial one-third-size
    stress design exposed 11 fold-support refusals; the promoted design holds conditional
    probabilities, unequal-period ratio, targets, seed, and gates fixed while requiring
    at least 32 expected observations per period-X-cohort cell. No estimator or
    simultaneous-band code changed.

23. `0.7.0a6` repeated-cross-section simultaneous-band alpha: opt-in
    `inference="multiplier_bootstrap"` reuses the retained event-study influence matrix and
    draws one Rademacher multiplier per observation or per indivisible declared PSU after
    cluster summation. HC1/CR1 finite-sample factors, studentization, higher-quantile max-t
    critical values, endpoints, seed/configuration metadata, and OutputHub transport are
    public. Observation and PSU hand identities, repeat-seed equality, invalid-setting and
    zero-SE refusals, covariate no-refit integration, and a 100-replication two-event joint-
    coverage smoke pass. The bounded-batch 100,000-row/four-event/999-draw benchmark takes
    2.117 seconds and 208.581 MiB Python-managed peak memory on the recorded environment.
    No panel entity or ordinary row-resampling bootstrap is reused.

24. `0.7.0a6` repeated-cross-section simultaneous-inference promotion: a hash-bound
    fixed-seed certificate crosses two designs, unadjusted and genuinely cross-fitted
    covariate scores, observation and indivisible-PSU sampling, and both control rules.
    All 16 complete event-vector cells pass joint coverage at `0.931–0.961`, with Monte
    Carlo SE at most `0.0081` and zero refusals across 16,000 estimator fits, 480,000
    nuisance fold fits, and 15,984,000 default-setting max-t draws. Every band metadata,
    coordinate, cluster-count, and whole-PSU fold-role audit passes; minimum realized PSU
    cell support is 31. No estimator code or public default changed.

25. `0.7.0a6` pairwise composition-change-robust repeated-section alpha: explicit
    `composition="robust"` now targets the treated target-period population for exactly
    one treated cohort and two periods. It cross-fits one ordered four-cell generalized
    propensity plus `m00`, `m01`, and `m10` on identical immutable row/whole-PSU folds,
    retains the four normalized cell weights, and reuses HC1/CR1 and multiplier inference.
    Four failing-first tests now pass the exact hand estimate, EIF, HC1, overlap, scope,
    leakage, PSU-role, nuisance-schema, and OutputHub contracts. No `m11` nuisance is fit.
    A later 32-row composition-shift gate recovers target ATT `5` while the stationary
    score targets pooled-treated value `4`. Two additional failing-first protocol gaps
    were closed: class predictions now refuse missing, extra, duplicate, or unlabeled
    schemas, safely realign labelled permutations, and the complete robust result is row-
    permutation invariant. Official R `compdid` 0.1.0 point, standard-error, and all 16
    influence-coordinate comparisons pass at pinned commit
    `894bd65a952c30f01a4e0005efba4cb335065eb7`; the relevant R source blobs, nuisance
    column mappings, fixture, and saved output are pinned and audited.
    Hash-pinned Sequeira robust/stationary sensitivity, the 100,000-row eight-task
    performance gate, and eight observation/PSU publication-scale coverage cells now
    pass. The certificate records 8,000 estimator fits, 72,000 fold-level nuisance fits,
    zero refusals, stationary efficiency costs, and the analytical shift bias. Staggered
    aggregation, longer-design conditional pre-trends/bands, the composition diagnostic,
    and survey combinations remain open.
    The repaired project-scoped `python -m pip_audit .` now completes and reports no
    known vulnerabilities; the broader host environment separately flags Pillow and
    Starlette versions that are not CauseKit project dependencies.

## Open promotion gates

- Matching still defaults to `inference="none"`. Known-score reuse-aware inference and a
  separately validated full-sample Logit-MLE correction now pass hand, parity,
  publication-scale coverage, real-data sensitivity, and performance gates. Generic,
  cross-fitted, penalized, unsupported-link, target-selected, tie-expanded, and clustered
  paths still refuse. R `Matching` remains non-comparable for the first-step correction
  because it conditions on the supplied score. Do not substitute a generic sandwich,
  ordinary bootstrap, or cluster wrapper.
- Efficient DiD owns no nuisance model classes. The implemented covariate path must keep
  consuming public cross-fitting factories. Its direct cohort-ratio replacement for
  multiclass probability ratios is now frozen in `docs/DID_DIRECT_RATIO_CONTRACT.md`; it
  remains design-only and must preserve calibrated pair orientation and the PT-All score.
- Repeated-cross-section DiD's no-covariate and explicit CrossFitter covariate-adjusted
  stationary-composition paths are public. Publication-scale no-covariate and covariate
  pointwise coverage and
  available estimator-level R `did`/Stata `csdid` parity pass, with Stata aggregate SEs
  recorded as non-comparable. The covariate score has independent fixed-OOF base-R parity;
  pinned R/Stata estimator-level cells remain unavailable because their maintained public
  paths do not implement the same score. Observation/PSU multiplier mechanics and the
  16-cell publication-scale joint-band certificate pass. The narrow pairwise composition-
  robust score passes its hand/refusal contracts, official R `compdid` point/influence
  parity, hash-pinned real-data sensitivity, fixed-size performance, and pairwise
  publication-scale pointwise coverage. Its staggered, diagnostic, and longer-design
  simultaneous gates remain open. Survey designs remain a separate design-only contract
  and continue to refuse.
- The R-learner alpha is public with honest evaluation, calibration, groups, bands, graph
  data, simulations, base-R/Stata parity, performance, native nonlinear support, and two
  real-data CATE records. The next native nonlinear stage is frozen as a separate
  construction-cross-fitted orthogonal-stack contract; no placeholder is exported and the
  current linear default is unchanged. Repeated splits, unit-level intervals, RATE, policy
  evaluation, and deployment refitting remain separate future contracts. The partially linear
  DML residual-stage parity row passes in Python, base R 4.5.1, and reviewed Stata/IC 17.
- The DR-learner alpha is a separate public estimator, not an R-learner alias. Its Python
  hand/simulation gates and base-R fixed-evaluation parity pass. One-run NSW evidence is
  recorded without rerunning settled R-learner comparators. Repeated splits, unit-level
  intervals, RATE, policy evaluation, deployment refitting, and publication-scale coverage
  remain separate gates. The Python/base-R/reviewed-Stata fixed-evaluation parity row
  passes.

## Required implementation patterns

- Reuse the strict alignment, labelled-result, covariance, post-estimation, OutputHub,
  release, and parity conventions established by `limiteddepkit` and `systemgmmkit`.
- Keep numerical fast paths vectorized. Normalize cluster labels once and aggregate scores
  by integer code; do not introduce observation-level Python loops or quadratic matrices.
- A fold loop is permitted where model fitting is inherently fold-specific. Each fold must
  receive fresh nuisance estimators from factories, and every reported nuisance prediction
  must be out of fold.
- Do not copy binary, count, censoring, duration, or ordinal estimators into CauseKit.
  General nuisance integration stays provider-neutral through protocols or explicit
  prediction adapters. The small native ridge-GCV learner is owned only as the default
  component of `PartiallyLinearDML` and the R-learner, not as a general regression surface.
- LimitedDepKit is owned by a separate agent/location. Do not inspect, edit, test, build,
  commit, clean, or release that repository from this CauseKit worktree.
- Do not claim that overlap diagnostics prove exchangeability, that AIPW repairs unmeasured
  confounding, or that clipping is an innocuous numerical operation.
- Every model addition requires analytical identities, deterministic recovery,
  refusal-path tests, aligned external/reference evidence where available, performance
  smoke coverage, documentation, packaging checks, and an intentional commit.

## Next development order

1. Extend the promoted pairwise [composition-change](docs/DID_RCS_COMPOSITION_CHANGE_CONTRACT.md)
   score only through separately contracted diagnostics and longer/staggered designs.
   Official `compdid` parity, real-data sensitivity, performance, and pairwise
   publication-scale coverage now pass; do not rerun them for an unrelated extension.
   Keep the separate
   [survey-design](docs/DID_RCS_SURVEY_DESIGN_CONTRACT.md) contract design-only until that
   base evidence is complete. The balanced-panel
   [direct-ratio](docs/DID_DIRECT_RATIO_CONTRACT.md) nuisance is a separate PT-All option;
   do not combine or reuse these paths before their base gates pass.
2. Regression discontinuity: sharp/fuzzy design, bandwidth, polynomial order,
   manipulation checks, bias correction, and local estimand.
3. Panel IV: reuse public `systemgmmkit` panel validation, entity/time indexing, fixed
   effects, and clustered covariance contracts.

Continue to preserve the matching tie/inference, target-population, and no-quadratic-
matrix decisions. For DiD, preserve the conventional/efficient separation, PT-All label,
entity-level influence records, no-hidden-regularization refusal, and fixed-T/vectorized-n
execution path. Do not relabel either alpha as promoted merely to make an early demo run.

## Cross-software parity gate

Before a model family is release-complete, maintain estimand-aligned Python, R, and Stata
comparisons where all three ecosystems expose the same estimator. Record software and
package versions, source or fixture, option and parameter mapping, covariance corrections,
seed, tolerances, and maximum discrepancies. If an ecosystem lacks the estimator or uses
different identifying moments, record the row as non-comparable or unavailable rather
than manufacturing a parity pass. The existing no-covariate efficient DiD R comparator is
pinned; Python/R/Stata coverage for every model remains a package-wide release gate.
