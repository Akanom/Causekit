from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "00_quickstart.py"
REAL_WORLD_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "real_world_causal_workflow.py"
)
COMPOSITION_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "composition_robust_repeated_cross_section_did.py"
)


def test_quickstart_runs_and_keeps_design_boundaries_visible(capsys) -> None:
    namespace = runpy.run_path(str(EXAMPLE), run_name="__main__")

    output = capsys.readouterr().out
    assert "Randomized trial — Lin-adjusted ATE" in output
    assert "Instrumental variables — robust 2SLS" in output
    assert "Conventional DiD — event-study effects" in output
    assert "diagnostics do not prove them" in output

    randomized = namespace["randomized_example"]()
    iv = namespace["iv_example"]()
    did = namespace["did_example"]()
    assert abs(randomized.estimate - 1.5) < 0.2
    assert abs(iv.params["treatment"] - 2.0) < 0.15
    assert iv.first_stage["treatment"].partial_r_squared > 0.2
    assert np.isfinite(did.estimate)
    assert did.pretrend.available


def test_real_world_example_labels_structural_unavailability_without_nan() -> None:
    namespace = runpy.run_path(str(REAL_WORLD_EXAMPLE))

    matching = namespace["_matching_point_table"](SimpleNamespace(estimate=-2.5))
    assert matching.loc["matching_att", "estimate"] == -2.5
    assert matching.loc["matching_att", "inference"].startswith("not requested")
    assert not matching.isna().any().any()

    source = pd.DataFrame(
        {"effect": [-1.0, 2.0], "std_err": [0.2, 0.3], "n_clusters": [np.nan, np.nan]},
        index=pd.Index([1, 2], name="group"),
    )
    result = SimpleNamespace(calibration_plot_data=lambda: source.copy())
    calibration = namespace["_observation_level_calibration_table"](result)
    assert calibration.columns.tolist() == ["effect", "std_err"]
    assert not calibration.isna().any().any()

    with pytest.raises(RuntimeError, match="point estimate must be finite"):
        namespace["_matching_point_table"](SimpleNamespace(estimate=np.nan))

    invalid_calibration = source.assign(effect=[-1.0, np.nan])
    invalid_result = SimpleNamespace(calibration_plot_data=lambda: invalid_calibration.copy())
    with pytest.raises(RuntimeError, match="calibration statistics must be finite"):
        namespace["_observation_level_calibration_table"](invalid_result)


def test_composition_example_labels_task_specific_class_count_without_nan() -> None:
    namespace = runpy.run_path(str(COMPOSITION_EXAMPLE))
    source = pd.DataFrame(
        {
            "nuisance": ["generalized_propensity", "outcome_control_pre"],
            "declared_class_count": [4.0, np.nan],
        }
    )

    displayed = namespace["_displayable_nuisance_diagnostics"](source)

    assert displayed["declared_class_count"].tolist() == ["4", "not applicable"]
    assert not displayed.isna().any().any()

    invalid_source = source.assign(holdout_nobs=[100.0, np.nan])
    with pytest.raises(RuntimeError, match="unexplained gap"):
        namespace["_displayable_nuisance_diagnostics"](invalid_source)
