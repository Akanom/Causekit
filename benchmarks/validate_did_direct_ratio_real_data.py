"""Hash-pinned real-data sensitivity for both PT-All cohort-weighting routes."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from causekit import CrossFitter, EfficientDiD, __version__
from causekit.datasets import REAL_DATASETS, load_real_dataset

try:
    from benchmarks._did_direct_ratio_support import (
        BinaryClassProbability,
        BinaryLogitOdds,
        MeanRegression,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/script.py`` execution
    from _did_direct_ratio_support import (  # type: ignore[no-redef]
        BinaryClassProbability,
        BinaryLogitOdds,
        MeanRegression,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "did_direct_ratio_real_data_evidence.json"


def _fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _panel() -> pd.DataFrame:
    raw = load_real_dataset("hospdd")
    panel = (
        raw.groupby(["hospital", "month"], as_index=False)
        .agg(outcome=("satis", "mean"), treated=("procedure", "max"))
        .sort_values(["hospital", "month"], kind="stable")
        .reset_index(drop=True)
    )
    first_treated = panel.loc[panel["treated"].eq(1)].groupby("hospital")["month"].min()
    panel["treatment_time"] = panel["hospital"].map(first_treated).fillna(np.inf)
    wide = panel.pivot(index="hospital", columns="month", values="outcome")
    panel["baseline_outcome"] = panel["hospital"].map(wide.iloc[:, 0])
    panel["pretrend"] = panel["hospital"].map(wide.iloc[:, 2] - wide.iloc[:, 0])
    return panel[
        [
            "hospital",
            "month",
            "outcome",
            "treatment_time",
            "baseline_outcome",
            "pretrend",
        ]
    ]


def _fit(panel: pd.DataFrame, *, direct: bool):
    weighting = (
        {"cohort_ratio_factory": BinaryLogitOdds}
        if direct
        else {"propensity_factory": BinaryClassProbability}
    )
    return EfficientDiD(pre_periods="all").fit(
        panel,
        outcome="outcome",
        entity="hospital",
        time="month",
        treatment_time="treatment_time",
        covariates=["baseline_outcome", "pretrend"],
        cross_fitter=CrossFitter(
            **weighting,
            outcome_factory=MeanRegression,
            second_moment_factory=MeanRegression,
            n_splits=3,
            random_state=20260730,
        ),
    )


def main() -> None:
    panel = _panel()
    direct = _fit(panel, direct=True)
    multiclass = _fit(panel, direct=False)
    if not direct.nuisance_fold.equals(multiclass.nuisance_fold):
        raise AssertionError("Weighting routes did not retain identical folds.")
    evidence = {
        "schema": "causekit_did_direct_ratio_real_data_v1",
        "causekit_version": __version__,
        "generated_on": "2026-07-30",
        "generator": "benchmarks/validate_did_direct_ratio_real_data.py",
        "reproduction_command": "python benchmarks/validate_did_direct_ratio_real_data.py",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source": "hospdd",
        "source_url": REAL_DATASETS["hospdd"].url,
        "source_sha256": REAL_DATASETS["hospdd"].sha256,
        "processed_fingerprint": _fingerprint(panel),
        "rows": len(panel),
        "entities": int(panel["hospital"].nunique()),
        "periods": int(panel["month"].nunique()),
        "treated_entities": int(
            panel.loc[np.isfinite(panel["treatment_time"]), "hospital"].nunique()
        ),
        "never_treated_entities": int(
            panel.loc[np.isinf(panel["treatment_time"]), "hospital"].nunique()
        ),
        "fold_seed": 20260730,
        "folds_identical": True,
        "direct_estimate": direct.estimate,
        "direct_standard_error": direct.standard_error,
        "multiclass_estimate": multiclass.estimate,
        "multiclass_standard_error": multiclass.standard_error,
        "estimate_absolute_difference": abs(direct.estimate - multiclass.estimate),
        "standard_error_absolute_difference": abs(
            direct.standard_error - multiclass.standard_error
        ),
        "candidate_influence_max_absolute_difference": float(
            np.max(
                np.abs(
                    direct.candidate_influence_functions.to_numpy()
                    - multiclass.candidate_influence_functions.to_numpy()
                )
            )
        ),
        "conditional_weight_max_absolute_difference": float(
            np.max(
                np.abs(
                    direct.conditional_efficiency_weights.to_numpy()
                    - multiclass.conditional_efficiency_weights.to_numpy()
                )
            )
        ),
        "direct_ratio_min": float(direct.cohort_ratios.min().min()),
        "direct_ratio_max": float(direct.cohort_ratios.max().max()),
        "direct_min_importance_effective_n": float(
            direct.cohort_ratio_diagnostics["denominator_importance_effective_n"].min()
        ),
        "direct_max_importance_share": float(
            direct.cohort_ratio_diagnostics["denominator_importance_max_share"].max()
        ),
        "interpretation": (
            "Sensitivity comparison only; the observed panel does not validate conditional "
            "PT-All or establish a causal effect."
        ),
        "stata_parity": "unavailable_same_direct_pairwise_pt_all_nuisance_not_exposed",
        "status": "pass",
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
