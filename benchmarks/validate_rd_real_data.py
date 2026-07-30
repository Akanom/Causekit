"""Hash-pinned Head Start RD sensitivity and official Python parity record."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import urllib.request
from datetime import date
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from rdrobust import rdrobust

import causekit
from causekit import RegressionDiscontinuity

SOURCE_URL = (
    "https://raw.githubusercontent.com/rdpackages-replication/CT_2021_NBER/master/headstart.csv"
)
SOURCE_SHA256 = "27e18a6ec3c15aa3a53aaa96c83024ce07841e9a6f21a7d850264d46711fd70b"
OUTPUT = Path("benchmarks/rd_real_data_evidence.json")


def _download() -> tuple[pd.DataFrame, str]:
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:  # noqa: S310
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"Head Start source hash changed: expected {SOURCE_SHA256}, observed {digest}."
        )
    return pd.read_csv(io.BytesIO(payload)), digest


def main() -> None:
    data, digest = _download()
    outcome = data["mort_age59_related_postHS"]
    running = data["povrate60"]
    cutoff = 59.1984

    fixed = RegressionDiscontinuity(
        cutoff=cutoff,
        bandwidth=9.0,
        bias_bandwidth=9.0,
        polynomial_order=1,
        bias_order=2,
        kernel="uniform",
        missing="drop",
    ).fit(outcome, running=running)
    reference = rdrobust(
        outcome,
        running,
        c=cutoff,
        h=9.0,
        b=9.0,
        p=1,
        q=2,
        kernel="uni",
        vce="hc1",
    )
    native = RegressionDiscontinuity(
        cutoff=cutoff,
        bandwidth="native_mse",
        bandwidth_candidates=11,
        missing="drop",
    ).fit(outcome, running=running)

    reference_conventional = float(reference.Estimate.iloc[0, 0])
    reference_bias_corrected = float(reference.Estimate.iloc[0, 1])
    reference_robust_se = float(reference.Estimate.iloc[0, 3])
    differences = {
        "conventional_estimate": abs(fixed.conventional_estimate - reference_conventional),
        "bias_corrected_estimate": abs(fixed.bias_corrected_estimate - reference_bias_corrected),
        "robust_standard_error": abs(fixed.robust_standard_error - reference_robust_se),
    }
    tolerance = 1e-9
    parity_status = "pass" if max(differences.values()) <= tolerance else "fail"
    evidence = {
        "artifact": "causekit_regression_discontinuity_real_data_v1",
        "generated_on": date.today().isoformat(),
        "generator": "benchmarks/validate_rd_real_data.py",
        "reproduction_command": "python benchmarks/validate_rd_real_data.py",
        "causekit_version": causekit.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "comparator": {"package": "rdrobust", "version": version("rdrobust")},
        "source": {
            "name": "Ludwig-Miller Head Start replication data",
            "url": SOURCE_URL,
            "sha256": digest,
            "rows_raw": int(len(data)),
            "outcome": "mort_age59_related_postHS",
            "running": "povrate60",
            "cutoff": cutoff,
        },
        "fixed_contract": {
            "bandwidth": [9.0, 9.0],
            "bias_bandwidth": [9.0, 9.0],
            "polynomial_order": 1,
            "bias_order": 2,
            "kernel": "uniform",
            "covariance": "HC1",
            "causekit": {
                "nobs": fixed.nobs,
                "n_effective": fixed.n_effective,
                "conventional_estimate": fixed.conventional_estimate,
                "bias_corrected_estimate": fixed.bias_corrected_estimate,
                "robust_standard_error": fixed.robust_standard_error,
                "mass_points_detected": fixed.mass_points_detected,
            },
            "rdrobust": {
                "conventional_estimate": reference_conventional,
                "bias_corrected_estimate": reference_bias_corrected,
                "robust_standard_error": reference_robust_se,
            },
            "absolute_differences": differences,
            "tolerance": tolerance,
            "parity_status": parity_status,
        },
        "native_sensitivity": {
            "method": native.bandwidth_selection.method,
            "bandwidth": [native.bandwidth_left, native.bandwidth_right],
            "bias_bandwidth": [native.bias_bandwidth_left, native.bias_bandwidth_right],
            "bias_corrected_estimate": native.bias_corrected_estimate,
            "robust_standard_error": native.robust_standard_error,
            "n_effective": native.n_effective,
            "mass_points_detected": native.mass_points_detected,
            "manipulation_available": native.manipulation.available,
            "manipulation_log_density_jump": native.manipulation.log_density_jump,
            "manipulation_pvalue": native.manipulation.pvalue,
            "selected_grid_boundary": native.bandwidth_selection.selected_at_boundary,
            "selected_grid_boundary_sides": list(native.bandwidth_selection.boundary_sides),
        },
        "interpretation": (
            "Sensitivity and numerical parity only. The record does not establish smoothness, "
            "absence of sorting, exclusion of other cutoff changes, or a causal effect."
        ),
        "status": parity_status,
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"results_file={OUTPUT.as_posix()}")
    print(f"parity_status={parity_status}")
    if parity_status != "pass":
        raise SystemExit(9)


if __name__ == "__main__":
    main()
