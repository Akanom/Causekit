# Direct cohort-odds PT-All promotion evidence

## Status

The provider-neutral direct cohort-odds nuisance route is implemented and promoted for
covariate-adjusted balanced-panel `EfficientDiD`. It remains an alternative nuisance
route for the same Chen-Sant'Anna-Xie PT-All estimand, not a separate estimator or a
weaker identification claim. Multiclass probabilities remain supported, and each fit
must select exactly one route.

## Analytical and refusal evidence

The maintained hand contracts cover:

- calibrated posterior cohort odds with unequal prior shares and the exact omitted-prior
  error in the scale-sensitive score;
- ordered-pair and reciprocal orientation, exact labelled alignment, provider adapters,
  fresh pair/fold factories, pair-only training, all-holdout prediction, immutable entity
  or higher-level-cluster folds, and row-role hashes;
- a two-cohort candidate and a three-cohort non-identity auxiliary candidate reproducing
  the multiclass score, candidate influence, `Omega_tilde = p_g Omega` conditional
  weights, group/event influence, HC1, and CR1 results;
- ratio floor/ceiling, denominator importance effective-size and maximum-share, pair PSU
  support, missing protocol, declared density-ratio, reversed orientation, complex,
  nonfinite, nonpositive, length/index, duplicate/unknown-pair, weak-support, ambiguous-
  route, and singular conditional-covariance refusals; and
- a frozen irrelevant-class softmax-underflow example. The two score-relevant
  multiclass probabilities become numerically zero even though their restricted binary
  posterior odds remain finite. Direct estimation recovers those odds without clipping,
  rescaling, or using the irrelevant class. This is the required documented stability
  benefit; the publication designs below verify no material degradation elsewhere.

## Publication-scale nonlinear and weak-overlap certificate

`benchmarks/validate_did_direct_ratio_promotion.py` fixes 1,000 replications per design,
600 entities, three outer folds, 199 max-t draws, a constant effect of `2`, and identical
folds/outcome nuisances across direct, binary-multiclass, and oracle odds. Both designs
use a nonlinear quadratic posterior-odds surface. The stressed design raises the linear
log-odds slope from `0.45` to `1.0`.

Every one of the 6,000 fits completed with zero refusals. The direct route reported:

| Design | Absolute bias | Empirical SD / mean SE | Pointwise coverage | Simultaneous coverage |
| --- | ---: | ---: | ---: | ---: |
| Nonlinear favorable | 0.00241 | 1.006 | 0.947 | 0.944 |
| Nonlinear weak overlap | 0.00019 | 1.014 | 0.951 | 0.943 |

Across all replications, the largest direct-versus-multiclass paired estimate difference
was `3.79e-08` and the largest paired standard-error difference was `2.15e-08`, both below
the frozen `1e-7` gates. Oracle coverage ranged from `0.930` to `0.946`; all direct,
multiclass, and oracle bias, SE-calibration, pointwise, simultaneous, identical-fold, and
zero-refusal gates passed.

The first stressed exploratory slice used 480 entities and slope `1.35`. It was not
promoted: one direct fit refused and its empirical-SD/mean-SE ratio was `1.82`. The final
stressed design was frozen at 600 entities and slope `1.0` only after a bounded 60-run
support diagnostic showed zero refusals and acceptable calibration. The complete
1,000-replication certificate above is the publication record; the failed predecessor is
retained here so overlap diagnostics are not hidden.

## Real-data sensitivity

`benchmarks/validate_did_direct_ratio_real_data.py` uses the hash-pinned public `hospdd`
source (`SHA-256 e3ae6451e89cb915c546ab772410046726f280ad7d117611376beb4f46a521bb`).
The processed balanced panel has 322 rows, 46 hospitals, seven periods, 18 treated
hospitals, and 28 never-treated hospitals. Baseline outcome and pre-trend are fixed before
either route is fit. Both routes use the same three folds, candidates, mean outcome and
second-moment nuisances, and seed.

Direct and multiclass estimates were `0.8491271618976050` and
`0.8491271618976048`; their standard errors were `0.03444789236987216` and
`0.03444789236987217`. Maximum candidate-influence and conditional-weight differences
were `8.88e-16` and `1.55e-15`. Direct odds ranged from `0.324` to `1.553`, minimum
denominator importance effective size was `7.52`, and maximum share was `0.202`. This is a
route-sensitivity execution record, not evidence that conditional PT-All holds or that
the estimate is causal.

## External score parity

`benchmarks/validate_did_direct_ratio_reference.R` independently reconstructs the fixed-
fold two-period posterior-odds score in base R 4.5.1. All 18 odds and influence
coordinates, the estimate `2.066666666666667`, and HC1 standard error
`0.3343855940231219` match CauseKit within `2e-14`. No maintained reviewed R or Stata
estimator accepts the same direct pairwise odds nuisances and PT-All moment. Estimator-
level cells are therefore recorded as unavailable, not approximated with a different DiD
command.

## Performance

`benchmarks/benchmark_did_direct_ratio.py` runs the fixed-two-period fast path on 200,000
panel rows and 100,000 entities with one fitted ordered pair and five folds. On the
recorded Windows/Python 3.14 environment it completed in `8.474` seconds under
`tracemalloc`, used `42.09 MiB` Python peak memory, recovered the effect exactly, and
retained denominator effective size `10,000` with maximum share `0.0001`. The frozen
gates are 10 seconds, 300 MiB, and absolute bias `1e-4`.

## Reproduction

From the repository root:

```text
python -m pytest tests/test_did_direct_ratio.py -q
Rscript benchmarks/validate_did_direct_ratio_reference.R
python -m pytest tests/validation/test_did_direct_ratio_parity.py -q
python benchmarks/validate_did_direct_ratio_real_data.py
python benchmarks/benchmark_did_direct_ratio.py
python benchmarks/validate_did_direct_ratio_promotion.py
python -m pytest tests/validation/test_did_direct_ratio_promotion.py -q
```

The publication harness is intentionally not part of the ordinary unit-test run; its
hashable JSON evidence is regenerated only when this nuisance contract changes.

Recorded artifact SHA-256 values are:

- publication JSON: `47ff5d731199408b6e7d9bb79e9526bcd417be78e05decd9a1caf2f753f40ecd`;
- real-data JSON: `f53b42010525a499a14d294fc48ce2ad61bd233ab44777fe602d42b2db8f3b9f`;
- performance JSON: `ee5cc3661d2d298b4ea84361275a5493c7098e0919ec1a937f958cf39f7b1c84`; and
- saved base-R output: `f43302e7a47d59b829e45c9c22c366b55229525d4ef92d0a8b0ad5c5d97859e9`.

## Limitations

Direct odds do not establish overlap, exchangeability, no anticipation, or PT-All. They
do not authorize survey, repeated-section, matching, or causal-ML score reuse. A provider
is responsible for returning calibrated posterior odds from its declared
`predict_odds_ratio` method and for keeping inner tuning inside the supplied training
rows. CauseKit validates explicit calibration/orientation metadata when supplied and
refuses a declared raw density ratio, but no runtime can detect a provider that falsely
labels its own predictions. Thresholds diagnose the fitted sample; they do not prove
population support.
