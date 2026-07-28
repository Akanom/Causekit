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
| `NearestNeighborMatch` ATT/ATC/ATE | pending | pending | pending | blocked on inference promotion |
| Conventional staggered DiD | hand identities; external comparator pending | pending | pending | pending |
| Efficient DiD, no covariates | native result checked against pinned fixture | public `edid` commit `f55a4a4aba14f0826f59ad7aa4af3bafaeba529b` | pending/unavailable pending audit | partial |
| Efficient DiD, covariate adjusted | deterministic and simulation evidence only | pending | pending/unavailable pending audit | pending |

The no-covariate efficient-DiD R harness is
`benchmarks/validate_edid_reference.R`; its maintained fixture compares every candidate
effect, the inverse-covariance weights, combined ATT, and HC1 standard error. This is not
evidence for the covariate-adjusted path or for Stata.

## Completion rule

Before declaring all planned models release-complete, resolve every `pending` cell where
an estimand-aligned implementation exists. When none exists, retain evidence for the
ecosystem search and mark the cell `unavailable`; do not replace it with a different
estimator. Keep generated comparator outputs outside source control unless they are small,
reviewed, versioned fixtures produced by a committed harness.
