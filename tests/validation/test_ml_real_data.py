"""Hash-verified real-data smoke for CauseKit's native causal-ML path."""

from __future__ import annotations

import numpy as np
import pytest

from causekit import PartiallyLinearDML
from causekit.datasets import load_real_dataset


@pytest.mark.validation
def test_real_cattaneo_native_partially_linear_dml_smoke() -> None:
    try:
        data = load_real_dataset("cattaneo2")
    except FileNotFoundError:
        pytest.skip("download the hash-pinned cattaneo2 dataset to run real-data DML smoke")
    covariates = data[["mmarried", "mage", "medu", "fbaby"]].astype(float)
    native = PartiallyLinearDML(n_splits=5, random_state=20_260_729).fit(
        data["bweight"].astype(float),
        treatment=data["mbsmoke"].astype(float),
        covariates=covariates,
    )

    assert native.nobs == len(data) == 4642
    assert native.native_nuisance is True
    assert native.treatment_kind == "binary"
    assert np.isfinite(native.estimate)
    assert np.isfinite(native.standard_error)
    assert native.standard_error > 0.0
    assert native.estimate == pytest.approx(-225.350625, abs=2e-6)
    assert native.standard_error == pytest.approx(22.439969, abs=2e-6)
    assert len(native.nuisance_diagnostics) == 10
    assert not native.nuisance_diagnostics["alpha_at_boundary"].any()
