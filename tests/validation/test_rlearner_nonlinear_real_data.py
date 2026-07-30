"""Frozen one-run real-data evidence for the native nonlinear CATE stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


def _artifact(name: str) -> tuple[Path, dict]:
    local_data = os.environ.get("LOCALAPPDATA")
    if local_data is None:
        pytest.skip("LOCALAPPDATA is unavailable on this platform")
    path = Path(local_data) / "causekit" / "benchmarks" / name
    if not path.exists():
        pytest.skip(f"external one-run benchmark artifact is absent: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.validation
def test_frozen_nsw_native_spline_row_retains_negative_evidence() -> None:
    path, report = _artifact("causekit-rlearner-nsw-native-spline-v1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "693a45a2dfa6317a95fbf1f9b43c1c8794792ac5aa663f45f995a263c458b955"
    )
    assert report["dataset_source_sha256"] == (
        "fc424cfc9d7861f4b95a6612f27c7e842671fea5a8612edcfe0273ee62e6f0a4"
    )
    result = report["results"][0]
    assert result["model"] == "causekit_native_spline_ridge_gcv"
    assert result["evaluation_index_sha256"] == (
        "f2ce0c338c4331a4ba211e0e32f44aeaa6a775d78b3f821e20bebbaa4977032f"
    )
    assert result["r_loss_gain"] == pytest.approx(-0.014285055070692998, abs=2e-15)
    assert result["heterogeneity_zero_pvalue"] == pytest.approx(
        0.8331503605139978,
        abs=2e-15,
    )


@pytest.mark.validation
def test_frozen_hillstrom_native_rows_share_the_honest_split() -> None:
    path, report = _artifact("causekit-rlearner-hillstrom-v1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "1174b6c732826f3a7d5c201536c59cab024a81fb43b6b3b0e3614f7e6fc82869"
    )
    assert report["dataset_source_sha256"] == (
        "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
    )
    assert report["nobs"] == 42_613
    linear, spline = report["results"]
    assert linear["model"] == "causekit_native_weighted_ridge_gcv"
    assert spline["model"] == "causekit_native_spline_ridge_gcv"
    assert linear["evaluation_index_sha256"] == spline["evaluation_index_sha256"]
    assert linear["honest_r_loss"] == pytest.approx(spline["honest_r_loss"], abs=2e-15)
    assert linear["r_loss_gain"] == pytest.approx(0.0003339839622894525, abs=2e-15)
    assert linear["heterogeneity_zero_pvalue"] == pytest.approx(
        0.00681006738164153,
        abs=2e-15,
    )
