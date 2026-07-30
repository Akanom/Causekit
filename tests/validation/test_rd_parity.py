"""Estimator-level parity against the maintained Python rdrobust implementation."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import pytest

from causekit import RegressionDiscontinuity

rdrobust_module = pytest.importorskip("rdrobust")
from rdrobust import rdrobust  # noqa: E402

pytestmark = pytest.mark.validation


def _reference_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = np.arange(400)
    left = -2.0 + position * (1.98 / 399.0)
    right = 0.02 + position * (1.98 / 399.0)
    running = np.r_[left, right]
    assignment = (running >= 0.0).astype(float)
    sharp_outcome = (
        0.5
        + 0.7 * running
        + 0.3 * running**2
        + 1.8 * assignment
        + 0.4 * np.sin(17.0 * running)
        + 0.2 * np.cos(31.0 * running)
    )
    treatment = np.r_[
        (position % 5 == 0).astype(float),
        (position % 5 <= 2).astype(float),
    ]
    fuzzy_outcome = (
        0.5
        + 0.4 * running
        + 0.2 * running**2
        + 2.2 * treatment
        + 0.35 * np.sin(13.0 * running)
        + 0.15 * np.cos(29.0 * running)
    )
    return running, sharp_outcome, treatment, fuzzy_outcome


def test_fixed_bandwidth_sharp_rbc_matches_python_rdrobust_2() -> None:
    assert version("rdrobust") == "2.0.0"
    running, outcome, _, _ = _reference_fixture()
    ours = RegressionDiscontinuity(
        bandwidth=(1.2, 1.4),
        bias_bandwidth=(1.6, 1.7),
    ).fit(outcome, running=running)
    reference = rdrobust(
        outcome,
        running,
        c=0.0,
        h=[1.2, 1.4],
        b=[1.6, 1.7],
        p=1,
        q=2,
        kernel="tri",
        vce="hc1",
    )

    assert ours.conventional_estimate == pytest.approx(
        float(reference.Estimate.iloc[0, 0]), abs=5e-13
    )
    assert ours.bias_corrected_estimate == pytest.approx(
        float(reference.Estimate.iloc[0, 1]), abs=5e-13
    )
    assert ours.robust_standard_error == pytest.approx(
        float(reference.Estimate.iloc[0, 3]), abs=5e-13
    )


def test_fixed_bandwidth_fuzzy_ratio_bias_and_rbc_match_python_rdrobust_2() -> None:
    running, _, treatment, outcome = _reference_fixture()
    ours = RegressionDiscontinuity(
        design="fuzzy",
        bandwidth=(1.2, 1.4),
        bias_bandwidth=(1.6, 1.7),
    ).fit(outcome, running=running, treatment=treatment)
    reference = rdrobust(
        outcome,
        running,
        c=0.0,
        fuzzy=treatment,
        h=[1.2, 1.4],
        b=[1.6, 1.7],
        p=1,
        q=2,
        kernel="tri",
        vce="hc1",
    )

    assert ours.conventional_estimate == pytest.approx(
        float(reference.Estimate.iloc[0, 0]), abs=5e-13
    )
    assert ours.bias_corrected_estimate == pytest.approx(
        float(reference.Estimate.iloc[0, 1]), abs=5e-13
    )
    assert ours.robust_standard_error == pytest.approx(
        float(reference.Estimate.iloc[0, 3]), abs=5e-13
    )
    assert ours.treatment_jump == pytest.approx(float(reference.tau_T.iloc[1, 0]), abs=5e-13)
