# Cross-software parity register

This register is the package-wide Python/R/Stata promotion gate. A row is a parity pass
only when the external implementation targets the same estimand on the same estimation
sample with aligned options and finite-sample corrections. Similar labels or estimates
are not enough.

## Required evidence record

Every comparator harness must record:

- the causal estimand and identifying moments;
- data source or generator, seed, and retained sample;
- CausalKit commit/version and external software/package versions;
- treatment, outcome, covariate, instrument, cohort, time, and cluster mapping;
- intercept, weighting, support, caliper, tie, anticipation, nuisance, and aggregation
  options as applicable;
- covariance definition, degrees-of-freedom or debiasing correction, and reference
  distribution;
- compared fields, absolute/relative tolerances, and observed maximum discrepancies; and
- a one-command reproduction path.

Use `pass`, `fail`, `pending`, `unavailable`, or `non-comparable`. `Unavailable` means the
ecosystem has no identified implementation. `Non-comparable` means an apparent analogue
uses different estimands or moments. Neither counts as a pass.

## Current matrix

| CausalKit family | Python comparator | R comparator | Stata comparator | Current status |
| --- | --- | --- | --- | --- |
| `IV2SLS` | `linearmodels` aligned coefficient/covariance tests | pending | pending | partial |
| `RandomizedATE` | `statsmodels` regression/covariance identities | pending | pending | partial |
| `IPWATE` / `AIPWATE` ATE/ATT/ATC | analytical score identities; external comparator pending | pending | pending | pending |
| `CrossFitter` | protocol and leakage/alignment tests; not itself an estimand | non-comparable | non-comparable | internal protocol |
| `NearestNeighborMatch` fixed-score ATT/ATC/ATE | hand/reuse/variance identities pass | CRAN `Matching` 4.10-15 fixture passes | Stata/MP 17 `teffects nnmatch` fixture passes | pass for recorded fixture |
| `NearestNeighborMatch` estimated-Logit ATT/ATC/ATE | hand formulas, `statsmodels.Logit`, and `limiteddepkit.BinaryLogitResult` pass | `Matching` conditions on supplied scores: non-comparable | `teffects psmatch` harness written; manual run pending | partial |
| Conventional staggered DiD | hand identities; external comparator pending | pending | pending | pending |
| Efficient DiD, no covariates | native result checked against pinned fixture | public `edid` commit `f55a4a4aba14f0826f59ad7aa4af3bafaeba529b` | pending/unavailable pending audit | partial |
| Efficient DiD, covariate adjusted | deterministic and simulation evidence only | pending | pending/unavailable pending audit | pending |

The no-covariate efficient-DiD R harness is
`benchmarks/validate_edid_reference.R`; its maintained fixture compares every candidate
effect, the inverse-covariance weights, combined ATT, and HC1 standard error. This is not
evidence for the covariate-adjusted path or for Stata.

The fixed-score matching R harness is `benchmarks/validate_matching_reference.R`. It
pins CRAN `Matching` commit `1208eaa7bfa888b1fc903481dddfb8c0dffa40d5`
(version 4.10-15) and compares ATT, ATC, and ATE estimates plus Abadie-Imbens standard
errors on the no-tie, one-neighbor contract fixture. The maintained Stata script is
`benchmarks/validate_matching_stata.do`. A manual Windows run with Stata/MP 17 passed;
the reviewed output is `benchmarks/validate_matching_stata_17_output.txt`. The Stata ATC
mapping reverses treatment, estimates ATET, and negates the coefficient while retaining
its standard error. Stata requires at least two same-treatment neighbors in
`vce(robust, nn(#))`, so that harness compares CausalKit's otherwise identical
`variance_neighbors=2` contract; `nneighbor(1)` still governs the effect match.

The estimated-Logit Python fixture independently fits the treatment model with
`statsmodels.Logit`, passes that fitted result through `FittedPropensityMLEProtocol`, and
compares all three effects and adjusted standard errors with hand-recorded values. The
same protocol has been exercised directly with the current public
`limiteddepkit.BinaryLogitResult`, without a CausalKit dependency or copied estimator;
`benchmarks/validate_matching_limiteddepkit.py` retains that reproduction path.
CRAN `Matching` accepts a supplied score but conditions on it for uncertainty, so it is
not comparable to the fitted-score first-step correction. The maintained Stata harness is
`benchmarks/validate_matching_estimated_stata.do`; it uses `teffects psmatch`, one effect
neighbor, and `vce(robust, nn(2))`. Its status remains pending until the machine-readable
output from a manual Stata run is reviewed and retained.

## Completion rule

Before declaring all planned models release-complete, resolve every `pending` cell where
an estimand-aligned implementation exists. When none exists, retain evidence for the
ecosystem search and mark the cell `unavailable`; do not replace it with a different
estimator. Keep generated comparator outputs outside source control unless they are small,
reviewed, versioned fixtures produced by a committed harness.
