# Fixed-effects Panel IV promotion evidence

## Status

The native Python implementation, independent Python explicit-dummy reference, official R
and Stata references, hash-pinned real-data sensitivity, publication-scale coverage, and
fixed-size performance gates pass. The frozen Panel IV promotion contract is complete.

## Frozen contract

- estimator: static 2SLS after mandatory entity and optional time absorption;
- excluded instruments only in `instruments=`;
- compact one-/two-way within transformation, including converged alternating projections
  for connected unbalanced panels;
- covariance: homoskedastic, HC1, or one-way CR1 with absorbed-rank finite-sample
  correction;
- default inference: entity-clustered CR1 with `t(G-1)` reference;
- diagnostics: fixed-effect-adjusted first stages and raw/within instrument variation; and
- no dynamic-panel, lag-construction, regularization, silent variable dropping, dummy
  matrix, or observation projection matrix.

## Evidence register

| Gate | Command | Result |
| --- | --- | --- |
| Hand/refusal/integration | `python -m pytest tests/test_panel_iv.py tests/test_api_surface.py tests/test_outputhub.py` | Pass |
| Independent Python parity | `python -m pytest tests/validation/test_panel_iv_parity.py` | Pass across 12 one-/two-way, balanced/unbalanced, and covariance cells |
| Hash-pinned real data | `python benchmarks/validate_panel_iv_real_data.py` | Pass |
| Official R parity | `Rscript benchmarks/validate_panel_iv_reference.R` | Pass |
| Recovery and coverage | `python benchmarks/validate_panel_iv_promotion.py` | Pass, 1,500 fits and zero refusals |
| Performance | `python benchmarks/benchmark_panel_iv.py` | Pass, 200,000 rows |
| Official Stata parity | `do "benchmarks/validate_panel_iv_stata.do"` | Pass, six estimate/standard-error fields at tolerance `1e-8` |

## Independent Python parity

`linearmodels` 7.0 receives an explicit intercept plus entity/time dummy design and
`debiased=True`. CauseKit uses the algebraically equivalent compact within design. Across
balanced/unbalanced panels, entity/two-way effects, and unadjusted/HC1/entity-CR1
covariance, slope estimates, covariance matrices, structural residuals, observation
counts, and absorbed residual degrees of freedom agree within `1e-10`.

## Real-data sensitivity

The source is the Vella-Verbeek wage panel distributed by `linearmodels` 7.0 at pinned
commit `28af72e`. Its compressed source SHA-256 is
`ee4f36706491d6348614f06bfcd5b2eef411603590a996ebb061078ae1024e78`.
The fixed numerical specification uses log wage as outcome, union status as endogenous,
one-period lagged union as excluded, hours/1,000 and marriage as exogenous, person/year
effects, and person-clustered CR1.

The retained panel has 3,815 rows, 545 people, and seven periods. CauseKit estimates the
union coefficient as `0.24962753388340345` with standard error
`0.31401132417362376`; the adjusted first-stage classical F is `22.080626214224417`.
The maximum coefficient difference from explicit-dummy `linearmodels` is `7.73e-13`, and
the maximum covariance difference is `7.29e-14` at tolerance `1e-9`.

R 4.5.1 with `AER` 1.2.16 and `sandwich` 3.1.1 independently fits the explicit dummy
model and one-way HC1/cluster adjustment. All three estimates and standard errors pass at
`1e-8`; the largest difference is `3.90e-13`.

This is numerical sensitivity, not an empirical identification certificate. A lagged
union indicator is not automatically independent of wage shocks or excluded from current
wages. No causal union-wage claim follows from the parity record.

## Publication-scale recovery and coverage

Three fixed-seed designs use a valid excluded instrument, an endogenous first-stage
disturbance, additive fixed effects, and serially dependent structural errors:

| Design | Fits | Absolute bias | Mean-SE / empirical-SD | 95% coverage | Refusals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced entity FE | 500 | 0.00120 | 1.0910 | 0.952 | 0 |
| Balanced two-way FE | 500 | 0.00065 | 1.0877 | 0.962 | 0 |
| Unbalanced two-way FE | 500 | 0.00062 | 1.0915 | 0.972 | 0 |

The preregistered gates require zero refusals, absolute bias at most `0.05`, SE ratio
`0.82–1.18`, coverage `0.91–0.985`, and median first-stage F at least `25`.

## Performance

The fixed 200,000-row, 20,000-entity, ten-period two-way design completed in approximately
`0.503` seconds with `52.66 MiB` peak Python allocation. Its coefficient error was
`0.00399`. This is a machine-specific smoke, not a general speed guarantee. The design
asserts that production estimation constructs neither fixed-effect dummies nor an
observation-by-observation projection matrix.

## Reviewed Stata parity

Run:

```text
python benchmarks/prepare_panel_iv_stata.py --download
do "benchmarks/validate_panel_iv_stata.do"
```

The do-file uses official Stata/IC 17 `ivregress 2sls`, explicit person/year indicators,
entity-clustered covariance, double precision, and saves all six estimate/standard-error
checks before asserting. The reviewed artifact passes every field. Its maximum absolute
difference is `1.2146e-12` for the union standard error, and its SHA-256 is
`1ac883346889ff0d9c651864b8fdf17570cb663d392fc56e883d832961f70704`.
