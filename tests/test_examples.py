from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "00_quickstart.py"


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
