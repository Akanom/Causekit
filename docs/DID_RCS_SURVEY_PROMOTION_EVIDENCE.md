# Survey-population repeated-section DiD promotion evidence

This record promotes the stationary-composition survey-population path of
`RepeatedCrossSectionDiD`. It covers component-wise Hájek point estimation,
one-stage with-replacement stratified-PSU Taylor inference, weighted nuisance routing,
official R parity, a reviewed Stata harness, real-data sensitivity, and fixed-size
performance. It does not promote finite-population corrections, replicate weights,
singleton adjustment, composition-robust survey scores, or survey-valid simultaneous
bands.

## Hand and external-reference contracts

The hand fixture reconstructs all four weighted component means, the complete linearized
variable, target-period treated aggregation shares, stratified PSU totals, Taylor
covariance, and design degrees of freedom. Refusal tests cover invalid weights, alignment
drift, PSU/stratum roles, inadequate cell support, concentration, singleton strata,
unweighted nuisance providers, holdout leakage, and unsupported feature combinations.

Base R 4.5.1 and the official R `survey` 4.5 package independently recover estimate
`1.7619047619047623`, standard error `0.0547279745403282` (absolute difference from the
hand Taylor calculation `2.18e-14`), and design degrees of freedom `2`. The saved R output
SHA-256 is `9a493c94299895704215636fa293b8ef31b805869ea8ffd054e7c63933726d56`.
The reviewed Stata/IC 17 harness maps the same totals through `svy: total` and `nlcom`,
independently reconstructs the Taylor variance, and passes every saved assertion. It
retains `e(df_r)` before
`nlcom, post` replaces the survey estimation results. Exact hand and point comparisons use
`1e-12`; only Stata's numerically differentiated `nlcom` standard error uses `1e-9`. The
official standard error differs from the independent analytical reconstruction by
`4.56e-11`; the saved output SHA-256 is
`e75e9737567923fa6203c9f6eb0a2f017581479fbf0c3aac61b687e5da0e0a1f`.

Reproduce with:

```bash
Rscript benchmarks/validate_did_rcs_survey_reference.R
pytest tests/validation/test_did_rcs_survey_parity.py -q
```

## Publication-scale pointwise inference

The fixed seed `20260730` crosses independent-observation and indivisible-PSU sampling,
informative and noninformative analysis weights, and balanced and unequal wave sizes.
Each of the eight cells uses 1,000 replications. All 8,000 estimator fits succeed with
zero refusals.

| Sampling | Weights | Waves | Bias | SE ratio | Coverage |
| --- | --- | --- | ---: | ---: | ---: |
| Observation | Noninformative | Balanced | -0.00366 | 0.99709 | 0.951 |
| Observation | Noninformative | Unequal | -0.00533 | 0.95741 | 0.939 |
| Observation | Informative | Balanced | -0.00000 | 0.99187 | 0.940 |
| Observation | Informative | Unequal | -0.00161 | 0.95571 | 0.948 |
| PSU | Noninformative | Balanced | 0.00380 | 0.98700 | 0.954 |
| PSU | Noninformative | Unequal | -0.00016 | 1.00303 | 0.951 |
| PSU | Informative | Balanced | -0.00722 | 0.99530 | 0.942 |
| PSU | Informative | Unequal | -0.01895 | 0.97302 | 0.941 |

The preregistered gates are coverage `0.90–0.99`, mean-analytical-SE to empirical-SD
ratio `0.75–1.25`, absolute bias at most `0.08`, and zero refusals. Realized coverage is
`0.939–0.954`, SE ratio is `0.95571–1.00303`, maximum absolute bias is `0.01895`, and the
maximum coverage Monte Carlo standard error is `0.00757`. These results validate the
declared simulation designs; they do not establish sampling ignorability or parallel
trends in an application.

## Hash-pinned real-data sensitivity

The public processed Youth Risk Behavior Surveillance beverage-tax file is pinned to
source commit `d0e1959d1e0dcc27f30b38a38f31d31fb3959799` and SHA-256
`7aa7b27284d3a1108bd4d7334470b2825bd68e72470f7347e8dac0d9ea3bd58f`. It contains
60,084 observations and 57 strata. CauseKit's survey estimate is `-0.9226631515407981`,
agreeing with the authors' sampling-weight-only four-mean mapping to
`4.44e-16`; the unweighted sample estimate is `-0.5892322561247982`.

The processed file omits PSU identifiers. This sensitivity therefore declares each row
as its own PSU within stratum and reports standard error `0.2551806933716691`; it does not
claim parity with the paper's bootstrap inference or its distinct composition-balancing
IPW estimator.

Run the base and covariate-adjusted examples with:

```bash
python examples/survey_repeated_cross_section_did.py
python examples/survey_repeated_cross_section_did.py --covariate-adjusted
```

## Fixed-size performance and canonical certificate

The recorded Python 3.14.6 / Windows 11 run fits 100,000 rows, 2,500 PSUs, and 20 strata
in `0.1682` seconds with `22.23` MiB Python-managed peak memory. The frozen gates are 10
seconds and 350 MiB; timings and allocation peaks remain machine-specific diagnostics.

Reproduce the complete certificate with:

```bash
python benchmarks/validate_did_rcs_survey_promotion.py
pytest tests/validation/test_did_rcs_survey_promotion.py -q
```

The canonical JSON is `benchmarks/did_rcs_survey_promotion_evidence.json`, SHA-256
`f275ebe25adbf62fc1f65af8037106713ac5065ae6151712083adccde85ebd5f`. Its generator
SHA-256 is `81ca2d62e3b331e193dc0171be7cb46521b550d04b076df12ef47e0fe4251c10`.
