# CauseKit examples

Run these files from the repository root after installing the package. The examples keep
research-design assumptions and unsupported inference claims visible rather than treating
an estimator call as proof of identification.

## Start here

| Example | Purpose | Data/network | Additional dependency |
| --- | --- | --- | --- |
| `00_quickstart.py` | Randomized ATE, robust 2SLS, and conventional DiD | Deterministic simulation; offline | None |
| `real_world_causal_workflow.py` | IV, randomized, observational, matching, native causal ML, and DiD | Five hash-pinned public datasets; `--download` is explicit | `validation` extra |

```bash
python examples/00_quickstart.py
python -m pip install -e ".[validation]"
python examples/real_world_causal_workflow.py --download
```

Downloaded public data are cached outside the repository and accepted only after an exact
SHA-256 check. A successful run establishes reproducible execution on the declared data and
specification; it does not establish the empirical identifying assumptions.

## Design-specific workflows

| Example | Contract demonstrated |
| --- | --- |
| `panel_iv.py` | Fixed-effects Panel IV with entity-clustered inference and instrument-variation audits |
| `regression_discontinuity.py` | Sharp and fuzzy fixed-bandwidth RD, bandwidth diagnostics, and optional plotting |
| `covariate_repeated_cross_section_did.py` | Cross-fitted covariate-adjusted repeated-section DiD |
| `survey_repeated_cross_section_did.py` | Survey-population repeated-section DiD with strata and PSUs |
| `composition_robust_repeated_cross_section_did.py` | Pairwise composition-change-robust repeated-section DiD |
| `longer_composition_robust_repeated_cross_section_did.py` | Longer/staggered composition-robust aggregation and diagnostics |
| `direct_ratio_efficient_did.py` | Direct cohort-odds PT-All nuisance route |

Formal Python/R/Stata parity remains in `tests/validation`, `benchmarks`, and
`docs/PARITY.md`; example output is not promoted as independent parity evidence.
