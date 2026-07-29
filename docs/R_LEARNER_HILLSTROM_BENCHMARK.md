# Hillstrom randomized-email R-learner benchmark

This is a one-run honest-evaluation record on real randomized data. It is not a repeated
benchmark, a model-selection proof, or evidence that real individual treatment effects are
observed. The comparison changes only CauseKit's weighted CATE stage.

## Provenance and design

- Original study: Kevin Hillstrom's 2008
  [MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/05/best-answer-e-mail-analytics-challenge.html).
- Reviewed Kaggle mirror:
  `bofulee/kevin-hillstrom-minethatdata-e-mailanalytics` (Apache-2.0 metadata).
- Reviewed CSV SHA-256:
  `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.
- Full source: 64,000 customers randomly assigned to Men's Email, Women's Email, or No
  Email. The original challenge and the
  [winning uplift analysis](https://stochasticsolutions.com/pdf/HillstromChallenge.pdf)
  describe the randomized three-arm design.
- CauseKit contrast: 21,307 Men's Email customers versus 21,306 No Email controls; the
  Women's Email arm is excluded before the binary-treatment fit.
- Outcome: visit during the post-email outcome window.
- Declared pre-treatment covariates: recency, purchase history, prior men's/women's
  merchandise indicators, new-customer status, and fixed one-hot zip/channel fields.
- Honest design: 21,306 construction and 21,307 evaluation observations; evaluation-index
  SHA-256 `73b51b0f103e0554663a46c02fd3ece0e7238fa4579649258a3912fa46a75068`.
- Every model uses the same native outcome/propensity nuisances, roles, three outer folds,
  four calibration groups, seed `20260729`, and 999 max-t draws.

The data were downloaded through a temporary Kaggle client. Kaggle is not a CauseKit
dependency, credentials are not read by the benchmark script, and the raw data remain in
the external application-data cache rather than the repository.

## Results

| Native weighted CATE stage | Honest R-loss | Constant R-loss | R-loss gain | Differential calibration | p-value vs 0 | p-value vs 1 | Seconds | Python peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear ridge-GCV | 0.1182661234 | 0.1183056356 | 0.0334% | 0.6483 | 0.00681 | 0.1422 | **5.536** | 135.793 |
| Adaptive spline-ridge GCV | 0.1182661234 | 0.1183056356 | 0.0334% | 0.6483 | 0.00681 | 0.1422 | 6.013 | **135.747** |

Both fits produced the same evaluation metrics to displayed precision. Given the adaptive
learner's candidate set includes the zero-knot linear basis and selects complexity only on
construction, this is consistent with safe fallback to the linear candidate. The saved
benchmark did not retain the selected-knot scalar, so that statement is an inference from
the aligned output, not a fabricated diagnostic.

The positive differential-calibration test supplies split-conditional evidence that the
learned ranking contains some heterogeneity signal on this contrast. The slope is not
statistically distinguishable from one at 5%, and group effects range from 0.0628 to 0.0893.
These are honest overlap-weighted residual moments, not observed customer-level causal
effects or a policy-value claim. Repeated-split and RATE evidence remain unimplemented.

The external JSON is
`%LOCALAPPDATA%/causekit/benchmarks/causekit-rlearner-hillstrom-v1.json`; its SHA-256 is
`1174b6c732826f3a7d5c201536c59cab024a81fb43b6b3b0e3614f7e6fc82869`.

## Reproduction

After obtaining the reviewed CSV from the declared Kaggle dataset:

```bash
python benchmarks/benchmark_rlearner_hillstrom.py \
  --data path/to/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv \
  --output causekit-rlearner-hillstrom-v1.json
```

The script refuses any source hash other than the reviewed CSV. It does not download data,
read credentials, or add Kaggle to CauseKit's dependency graph.
