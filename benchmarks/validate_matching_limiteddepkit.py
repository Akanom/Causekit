"""Validate the fitted-propensity protocol with LimitedDepKit's public Logit result.

Run from a development environment containing both source checkouts or installed packages:

    python benchmarks/validate_matching_limiteddepkit.py
"""

from __future__ import annotations

import limiteddepkit
import numpy as np
import pandas as pd
from limiteddepkit import BinaryLogit

import causalkit
from causalkit import FittedPropensityMLEProtocol, NearestNeighborMatch


def main() -> None:
    design = pd.DataFrame(
        {
            "const": 1.0,
            "x": [-2.60, -2.00, -1.35, -0.95, -0.62, -0.18, 0.13, 0.49, 0.91, 1.38, 1.92, 2.57],
        }
    )
    treatment = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], dtype=int)
    outcome = pd.Series([0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 11.0, 10.0, 14.0])
    expected = {
        "att": (2.5, 0.8286781699117679),
        "atc": (13.0 / 6.0, 0.5402752634692672),
        "ate": (7.0 / 3.0, 0.8689422298747904),
    }

    propensity_fit = BinaryLogit().fit(design, treatment)
    if not isinstance(propensity_fit, FittedPropensityMLEProtocol):
        raise AssertionError("BinaryLogitResult does not satisfy FittedPropensityMLEProtocol.")

    results = {}
    for estimand, (expected_estimate, expected_standard_error) in expected.items():
        result = NearestNeighborMatch(
            estimand=estimand,
            metric="propensity",
            caliper=None,
            common_support=None,
            inference="abadie_imbens_estimated",
            variance_neighbors=2,
        ).fit(
            outcome,
            treatment=treatment,
            propensity_model=propensity_fit,
            propensity_design=design,
            propensity_score_status="estimated",
            propensity_provenance="limiteddepkit.BinaryLogit full-sample MLE",
        )
        if not np.isclose(result.estimate, expected_estimate, rtol=0.0, atol=2e-12):
            raise AssertionError(f"{estimand.upper()} estimate parity failed.")
        if not np.isclose(
            result.standard_error,
            expected_standard_error,
            rtol=0.0,
            atol=2e-12,
        ):
            raise AssertionError(f"{estimand.upper()} standard-error parity failed.")
        results[f"{estimand}_estimate"] = result.estimate
        results[f"{estimand}_standard_error"] = result.standard_error

    print(f"causalkit_version={causalkit.__version__}")
    print(f"limiteddepkit_version={limiteddepkit.__version__}")
    print("contract=public_binary_logit_result_full_sample_mle")
    for name, value in results.items():
        print(f"{name}={value:.17g}")
    print("parity_status=pass")


if __name__ == "__main__":
    main()
