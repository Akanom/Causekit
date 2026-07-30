"""Point and influence parity against the pinned official R ``compdid`` score."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causekit import CrossFitter, RepeatedCrossSectionDiD

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "did_rcs_compdid_parity_input.csv"
OUTPUT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_compdid_output.txt"
SCRIPT_PATH = REPOSITORY_ROOT / "benchmarks" / "validate_did_rcs_compdid_reference.R"
REFERENCE_COMMIT = "894bd65a952c30f01a4e0005efba4cb335065eb7"
ATT_CORE_BLOB = "5d7ab30e0c3db8d90112b4d59d7e752ad6f459cf"
DP_GROOMING_BLOB = "9b00850566ce6fff9f2897f7852733da4ede7779"
INPUT_SHA256 = "eafc591363edf693b42adea3f56782def6addbb7bed67dc993260b1b2b17bf2f"
OUTPUT_SHA256 = "680802078b1634cb5f5e43096f94867ea9ed1b4f9c95fa384d9c30b51fc057a0"


def _canonical_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _parse_reference(text: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        if key in parsed:
            raise AssertionError(f"Duplicate comparator field: {key}")
        parsed[key] = value.split(",")
    return parsed


class _KnownGeneralizedPropensityResult:
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X["x"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                0: 0.40 - 0.03 * x,
                1: 0.20 + 0.02 * x,
                2: 0.25 - 0.01 * x,
                3: 0.15 + 0.02 * x,
            },
            index=X.index,
        )


class _KnownGeneralizedPropensity:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownGeneralizedPropensityResult:
        del X, y
        return _KnownGeneralizedPropensityResult()


class _KnownCellOutcomeResult:
    def __init__(self, cell: str) -> None:
        self.cell = cell

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X["x"].to_numpy(dtype=float)
        if self.cell == "d0-s0":
            return 1.0 + 0.50 * x
        if self.cell == "d0-s1":
            return 2.0 + x
        if self.cell == "d1-s0":
            return 3.0 + 0.25 * x
        raise AssertionError(f"Unexpected outcome cell {self.cell!r}.")


class _KnownCellOutcome:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> _KnownCellOutcomeResult:
        del y
        cells = {str(label).rsplit("-r", 1)[0] for label in X.index}
        if len(cells) != 1:
            raise AssertionError("Each outcome nuisance must train within one cell.")
        return _KnownCellOutcomeResult(cells.pop())


def _cross_fitter(
    *,
    propensity_factory: Callable[[], object] = _KnownGeneralizedPropensity,
    outcome_factory: Callable[[], object] = _KnownCellOutcome,
) -> CrossFitter:
    return CrossFitter(
        propensity_factory=propensity_factory,
        outcome_factory=outcome_factory,
        n_splits=2,
        random_state=19,
    )


def _native_result():
    data = pd.read_csv(INPUT_PATH).set_index("row")
    data["time"] = data["post"] + 1.0
    data["treatment_time"] = np.where(data["d"].eq(1), 2.0, np.inf)
    return RepeatedCrossSectionDiD(composition="robust").fit(
        data,
        outcome="outcome",
        time="time",
        treatment_time="treatment_time",
        covariates=["x"],
        cross_fitter=_cross_fitter(),
    )


def _assert_parity(reference: dict[str, list[str]]) -> None:
    assert reference["reference_repository"] == ["https://github.com/pedrohcgs/comp_did.git"]
    assert reference["reference_commit"] == [REFERENCE_COMMIT]
    assert reference["att_core_blob"] == [ATT_CORE_BLOB]
    assert reference["dp_grooming_blob"] == [DP_GROOMING_BLOB]
    assert reference["source_loading"] == ["git_pinned_official_R_sources"]
    assert reference["compdid_version"] == ["0.1.0"]
    assert reference["entry_point"] == ["drdid_nonstationary"]
    assert reference["stabilized"] == ["true"]
    assert reference["bootstrap"] == ["false"]
    assert reference["influence_requested"] == ["true"]
    assert reference["causekit_probability_order"] == ["00", "01", "10", "11"]
    assert reference["compdid_probability_order"] == ["11", "10", "01", "00"]
    assert reference["causekit_outcome_order"] == ["00", "01", "10"]
    assert reference["compdid_outcome_order"] == ["11", "10", "01", "00"]
    assert reference["treated_post_outcome_role"] == ["algebraically_cancels"]
    assert reference["n_observations"] == ["16"]
    assert reference["parity_status"] == ["reference_complete"]

    result = _native_result()
    assert result.estimate == pytest.approx(float(reference["att"][0]), abs=2e-14)
    assert result.standard_error == pytest.approx(float(reference["standard_error"][0]), abs=2e-14)

    rows = reference["influence_rows"]
    values = np.asarray(reference["influence_values"], dtype=float)
    assert rows == result.overall_influence.index.tolist()
    np.testing.assert_allclose(
        values,
        result.overall_influence.to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-14,
    )
    assert float(reference["influence_mean"][0]) == pytest.approx(0.0, abs=2e-14)


def test_r_compdid_comparator_pins_source_and_explicitly_maps_nuisances() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert REFERENCE_COMMIT in script
    assert ATT_CORE_BLOB in script
    assert DP_GROOMING_BLOB in script
    assert 'c("p11", "p10", "p01", "p00")' in script
    assert 'c("m11", "m10", "m01", "m00")' in script
    assert "drdid_nonstationary" in script
    assert "stabilized = TRUE" in script
    assert "boot = FALSE" in script
    assert "inffunc = TRUE" in script


@pytest.mark.validation
def test_saved_r_compdid_point_and_influence_parity_passes() -> None:
    assert _canonical_text_sha256(INPUT_PATH) == INPUT_SHA256
    assert _canonical_text_sha256(OUTPUT_PATH) == OUTPUT_SHA256
    _assert_parity(_parse_reference(OUTPUT_PATH.read_text(encoding="utf-8")))


@pytest.mark.validation
def test_live_r_compdid_point_and_influence_parity_when_configured() -> None:
    reference_root = os.environ.get("CAUSEKIT_COMPDID_REFERENCE")
    rscript = shutil.which("Rscript")
    if reference_root is None or rscript is None:
        pytest.skip("set CAUSEKIT_COMPDID_REFERENCE and install Rscript for live R compdid parity")
    completed = subprocess.run(
        [rscript, str(SCRIPT_PATH), reference_root],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    _assert_parity(_parse_reference(completed.stdout))


@pytest.mark.validation
def test_r_compdid_comparator_refuses_an_unpinned_checkout_when_r_is_available() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("install Rscript to exercise the unpinned-source refusal")
    completed = subprocess.run(
        [rscript, str(SCRIPT_PATH), str(REPOSITORY_ROOT)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "compdid checkout must be at recorded commit" in completed.stdout + completed.stderr
