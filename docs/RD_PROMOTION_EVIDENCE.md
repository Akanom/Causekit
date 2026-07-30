# Regression discontinuity promotion evidence

This record identifies the reproducible evidence used to promote CauseKit's first regression
discontinuity (RD) contract. It supplements the methodological definition in
[`RD_CONTRACT.md`](RD_CONTRACT.md); it does not turn a numerical validation result into evidence
for any applied study's identifying assumptions.

## Frozen artifacts

| Evidence | Generator | SHA-256 | Status |
|---|---|---|---|
| Hash-pinned Head Start sensitivity and fixed-contract parity | `python benchmarks/validate_rd_real_data.py` | `d685c56317a2425f405498e09d65e4d9e0caea2cf1197932a51010502ef81085` | Pass |
| Nonlinear sharp/fuzzy simulation recovery and coverage | `python benchmarks/validate_rd_promotion.py` | `55f29c00d60156226b434cdf42722308dda61f4a8c31c28018dd06bcb7480122` | Pass |
| 200,000-row runtime and peak-memory benchmark | `python benchmarks/benchmark_rd.py` | `26036ef9b1dbd1f13e679d8ef8562190e2c199530ddf162ab0fd9a780c1f8257` | Pass |
| Official R `rdrobust` fixed-contract parity | `Rscript benchmarks/validate_rd_reference.R` | `7f58149410b15202cea8b0c36671fe629bad3c4cdd6705d5d27408e09e840ab7` | Pass |
| Official Stata `rdrobust` fixed-contract parity | `do "benchmarks/validate_rd_stata.do"` | `29591cdf4a4ea3c4269838aafb02fec7942df8b870774c3aa221e04fbf429a0d` | Pass |

The JSON artifacts record the CauseKit and comparator versions, platform, fixed seeds,
specifications, gates, and reproduction commands. Regenerating an artifact can legitimately
change timing or platform metadata; review the semantic results and update this hash ledger
intentionally rather than silently replacing it.

## Results frozen on 2026-07-30

- Official Python `rdrobust` 2.0.0 and R `rdrobust` 4.0.0 fixed-bandwidth sharp/fuzzy
  conventional estimates, robust bias-corrected estimates, robust standard errors, and fuzzy
  treatment jump agree within `1e-9`; observed differences are below `1e-13`.
- Across 1,000 fixed-bandwidth replications per design, absolute bias is `0.0057` for sharp RD
  and `0.0104` for fuzzy RD, while 95% robust interval coverage is `0.936` and `0.947`.
- Across 250 native-selector replications per design, absolute bias is `0.0064` and `0.0248`,
  coverage is `0.948` and `0.944`, and there are zero unexpected refusals. Frequent boundary
  selection is retained as a visible sensitivity diagnostic, not treated as a success criterion.
- The hash-pinned Ludwig-Miller Head Start fixture matches official Python `rdrobust` within
  `8.1e-14` under the fixed contract. Its native selection reaches the left grid boundary, so
  the artifact is a numerical sensitivity record and not an identification certificate.
- The 200,000-row benchmark completes in `0.567` seconds with `101.44 MiB` peak traced memory
  on the recorded Windows/Python environment and does not allocate an observation-by-observation
  projection matrix.
- Reviewed Stata/IC 17 with official `rdrobust` 11.1.0 passes all seven sharp/fuzzy point,
  robust-standard-error, and corrected-first-stage fields. The maximum absolute difference is
  `2.3092638912203256e-14` against tolerance `1e-8`.

## Promotion decision

The sharp/fuzzy alpha passes every declared hand/refusal, official Python/R/Stata parity,
hash-pinned real-data, nonlinear recovery/coverage, and performance gate. This decision does not
promote the explicitly unavailable RD extensions or validate identification in an applied study.
