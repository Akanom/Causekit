"""Optional parity against the public R edid implementation at a pinned commit."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalkit import EfficientDiD


def _hand_panel() -> pd.DataFrame:
    h1 = np.array([1.0, 1.0, -1.0, -1.0])
    h2 = np.array([1.0, -1.0, 1.0, -1.0])
    h3 = np.array([1.0, -1.0, -1.0, 1.0])
    rows: list[dict[str, float | str]] = []
    for position in range(4):
        path = (
            -(10.0 + h1[position]),
            -(10.0 + 2 * h2[position]),
            -(10.0 + 4 * h3[position]),
            0.0,
        )
        for period, outcome in enumerate(path, start=1):
            rows.append(
                {
                    "entity": f"treated_{position}",
                    "time": period,
                    "treatment_time": 4.0,
                    "outcome": outcome,
                }
            )
    for position in range(4):
        for period in range(1, 5):
            rows.append(
                {
                    "entity": f"never_{position}",
                    "time": period,
                    "treatment_time": np.inf,
                    "outcome": 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.validation
def test_efficient_did_matches_pinned_r_edid_checkout() -> None:
    reference_root = os.environ.get("CAUSALKIT_EDID_REFERENCE")
    rscript = shutil.which("Rscript")
    if reference_root is None or rscript is None:
        pytest.skip("set CAUSALKIT_EDID_REFERENCE and install Rscript for R edid parity")

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            rscript,
            str(repository_root / "benchmarks" / "validate_edid_reference.R"),
            reference_root,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    reference = dict(
        line.split("=", maxsplit=1) for line in completed.stdout.splitlines() if "=" in line
    )
    assert reference["reference_commit"] == "f55a4a4aba14f0826f59ad7aa4af3bafaeba529b"
    reference_weights = np.fromstring(reference["weights"], sep=",")

    native = EfficientDiD().fit(
        _hand_panel(),
        outcome="outcome",
        entity="entity",
        time="time",
        treatment_time="treatment_time",
    )
    native_weights = native.efficiency_weights.sort_values("bridge_period")["weight"].to_numpy()
    np.testing.assert_allclose(native_weights, reference_weights, rtol=2e-14, atol=2e-14)
    assert native.estimate == pytest.approx(float(reference["efficient_att"]), abs=2e-14)
    assert native.standard_error == pytest.approx(
        float(reference["standard_error_hc1"]), rel=2e-14, abs=2e-14
    )
