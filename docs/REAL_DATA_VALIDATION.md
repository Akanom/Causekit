# Real-data validation and example provenance

CauseKit does not redistribute the source datasets used by its applied example and
cross-software checks. Acquisition is opt-in and HTTPS-only; every cached file must match
the release-pinned SHA-256 digest before pandas reads it. The runnable workflow and parity
preparation use the same registry in `causekit.datasets`.

| Registry name | Real-data role | Source | SHA-256 |
| --- | --- | --- | --- |
| `hsng` | 1980 U.S. Census state housing IV example | `https://www.stata-press.com/data/r19/hsng.dta` | `d19cd25299af57569d93d8f16b4f72d5ffc7f897c9247d8a8e5fddddef43ad11` |
| `nsw_mixtape` | National Supported Work randomized job-training experiment | `https://raw.githubusercontent.com/scunning1975/mixtape/master/nsw_mixtape.dta` | `fc424cfc9d7861f4b95a6612f27c7e842671fea5a8612edcfe0273ee62e6f0a4` |
| `cattaneo2` | Maternal smoking, birthweight, supplied-nuisance effects, matching, and native partially linear DML | `https://www.stata-press.com/data/r19/cattaneo2.dta` | `631e926eb9981828ba2e542b32c16ae08f336b9efa10621651a8a185405e0577` |
| `hospdd` | Hospital procedure-adoption conventional and efficient DiD | `https://www.stata-press.com/data/r19/hospdd.dta` | `e3ae6451e89cb915c546ab772410046726f280ad7d117611376beb4f46a521bb` |

Stata Press lists `hsng` for `ivregress`, `cattaneo2` for the `teffects` family, and
`hospdd` for `didregress`. The NSW file is the 445-observation experimental sample used in
the matching chapter of *Causal Inference: The Mixtape*. Dataset availability and a
familiar command example do not validate the identifying assumptions of a particular
analysis.

## Applied workflow

Install the optional validation dependency and authorize the first download explicitly:

```bash
python -m pip install -e ".[validation]"
python examples/real_world_causal_workflow.py --download
```

The workflow fits:

- robust `IV2SLS` on the housing data;
- unadjusted and Lin-adjusted `RandomizedATE` on NSW;
- five-fold supplied-nuisance IPW/AIPW, point matching, and CauseKit-native partially
  linear DML on `cattaneo2`; and
- conventional and PT-All efficient DiD on an equal-hospital-weight panel constructed
  from `hospdd`.

It prints explicit interpretation boundaries. In particular, instrument diagnostics do
not establish exclusion; the smoking analysis remains observational; cross-fitted
matching is not given an unsupported analytical standard error; and PT-All efficient DiD
is displayed beside the conventional estimator rather than replacing it.

The `0.7.0a2` real-data causal-ML workflow returned `theta=-225.350625` with robust standard
error `22.439969` under the documented five-fold native ridge-GCV specification. All ten
outcome/treatment fold fits selected interior penalties. This is a
deterministic software smoke, not evidence that smoking is conditionally exchangeable or
that a constant treatment effect is scientifically credible. The separate one-run
performance record is [Causal-ML real-data performance](ML_BENCHMARK.md).

## Cross-language preparation

Generate deterministic, 17-digit CSVs and a JSON manifest outside the repository:

```bash
python benchmarks/prepare_real_data.py --download
```

The default raw cache is `%LOCALAPPDATA%/causekit/datasets` on Windows (or
`~/.cache/causekit/datasets` when `LOCALAPPDATA` is absent). Cross-language CSVs and their
source/output hashes are written below `causekit/parity/real_data_v1` in that cache.

Run the pinned R certificate with exact comparator checkouts:

```bash
Rscript benchmarks/validate_real_data_reference.R \
  PATH_TO_PREPARED_CSVS PATH_TO_EDID_CHECKOUT PATH_TO_MATCHING_CHECKOUT
```

The reviewed R 4.5.1 run passed all 14 Python assertions. Maximum absolute differences
were below `8.7e-7` for IV, `8.1e-7` for randomized ATE, `1.4e-5` for supplied-nuisance
IPW/AIPW, `1e-9` for matching, and `2e-14` for both conventional and efficient DiD. The
larger observational difference is from the stopping tolerance of R `glm.fit` versus
`statsmodels.Logit`; it is below `0.00002` birthweight grams (`0.02` milligrams).

Stata must be run manually from the repository root:

```stata
do "benchmarks/validate_real_data_stata.do"
```

The `.do` file writes `benchmarks/validate_real_data_stata_output.txt` before asserting.
Native Stata commands cover IV, randomized HC1 regression, and fixed-score matching.
Portable influence equations cover CauseKit's explicitly conditional supplied-nuisance
IPW/AIPW contract and group-time DiD aggregation. Stata `teffects` first-step correction
and `didregress` common-effect aggregation are not silently substituted for those targets.

The reviewed Stata/IC 17 run passes every comparable family and the overall certificate.
Maximum absolute differences are `8.65e-7` for IV, `8.07e-7` for randomized ATE,
`5.64e-7` for supplied-nuisance IPW/AIPW, `7.39e-13` for matching, and `1.11e-16` for
conventional DiD. The saved artifact records the software flavor, tolerances, estimates,
standard errors, source hashes, per-family statuses, and unavailable efficient-DiD cell.

## External implementation boundary

The pinned public R `edid` checkout explicitly states that it does not support covariates.
Stata 19 provides heterogeneous DiD estimators, but they do not implement the
Chen–Sant'Anna–Xie PT-All optimal weighting contract. Covariate-adjusted efficient DiD and
CauseKit's exact Rademacher simultaneous-band algorithm therefore retain deterministic
equation, refusal, and coverage tests while R/Stata are recorded as unavailable or
non-comparable—not as fabricated parity passes.
