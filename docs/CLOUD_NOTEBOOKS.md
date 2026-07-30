# Kaggle and Google Colab usage

CauseKit uses one reviewed notebook for Kaggle and Google Colab. Cloud notebooks are
reproducibility and adoption aids; formal parity claims remain in the package's
Python/R/Stata validation harnesses.

## Package boundary

- Use CauseKit notebooks for randomized, observational, matching, causal-ML, DiD, RD,
  and causal IV workflows owned by CauseKit.
- Do not copy LimitedDepKit likelihood models or SystemGMMKit's general panel
  orchestration into this notebook.
- A comparison belongs in formal parity only when the estimand, sample, nuisance roles,
  covariance, finite-sample correction, comparator version, and tolerance align.

## Credentials and data

- Never commit `kaggle.json`, Google credentials, API tokens, cookies, or notebook
  secrets.
- Keep Kaggle credentials in the runtime secret manager or `~/.kaggle/kaggle.json` on
  the publishing machine.
- The quickstart uses CauseKit's public-data registry. Downloads are explicit,
  HTTPS-only, cached outside the repository, and verified against pinned SHA-256 hashes.
- Do not commit downloaded third-party data or executed notebook output without a
  separate provenance, license, privacy, and artifact review.

## Repository notebook

The shared notebook is `notebooks/kaggle/causekit_quickstart.ipynb`. It installs the
reviewed release directly from PyPI with an exact version pin:

```bash
python -m pip install "causekit[validation,plot,outputhub]==0.7.0a6"
```

The cell installs the validation, plotting, and OutputHub extras from the official PyPI
index with a three-minute timeout, then clears stale imported modules before re-import.
It never clones the private development repository, embeds a credential, or requests a
GitHub token. Internet access is required for the package, public dependencies, and
registered public datasets; a GPU is not required.

To upload it to Kaggle from an authenticated machine:

```powershell
cd notebooks/kaggle
kaggle kernels push -p .
```

The public kernel id in `kernel-metadata.json` is `akanom/causekit-quickstart`.

For Google Colab, use the badge in the notebook, open the tracked file from GitHub, or
upload the same `.ipynb` directly. Keeping one source notebook prevents Kaggle and Colab
examples from drifting into different estimators or claims.

## Local structural verification

Notebook contract tests verify valid JSON, an immutable PyPI version pin, stale-module
eviction, real-data hash-registry use, fixed seeds, empty committed outputs, and the
identification checklist:

```bash
python -m pytest tests/test_cloud_notebook.py
```

This structural gate does not replace a fresh hosted execution after publication. Record
the Kaggle/Colab run URL, runtime date, installed package version, and dataset hashes when
promoting hosted output as release evidence.
