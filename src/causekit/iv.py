"""Linear instrumental-variables estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from ._covariance import CovarianceType, iv_covariance
from ._data import MissingPolicy, prepare_iv_data, prepare_prediction_data
from .diagnostics import (
    FirstStageDiagnostic,
    SarganTest,
    first_stage_diagnostics,
    sargan_test,
)


@dataclass(frozen=True)
class IV2SLSResult:
    """Fitted two-stage least-squares result with labeled inference."""

    params: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    test_statistics: pd.Series
    pvalues: pd.Series
    residuals: pd.Series
    fitted_values: pd.Series
    first_stage: dict[str, FirstStageDiagnostic]
    overidentification: SarganTest | None
    nobs: int
    rank: int
    df_resid: int
    covariance_type: CovarianceType
    inference_distribution: str
    inference_df: float | None
    n_clusters: int | None
    y_name: str
    endogenous_names: tuple[str, ...]
    exogenous_names: tuple[str, ...]
    instrument_names: tuple[str, ...]
    estimation_index: pd.Index
    dropped_rows: int
    add_constant: bool
    notes: tuple[str, ...]
    assumptions: tuple[str, ...]
    converged: bool = True
    backend: str = "native-2sls"

    @property
    def all_params(self) -> pd.Series:
        """Compatibility alias used by ecosystem post-estimation adapters."""

        return self.params.copy()

    @property
    def coefficients(self) -> pd.Series:
        return self.params.copy()

    @property
    def zstats(self) -> pd.Series:
        """Compatibility alias; clustered and unadjusted fits use t inference."""

        return self.test_statistics.copy()

    @property
    def tstats(self) -> pd.Series:
        return self.test_statistics.copy()

    @property
    def causal_interpretation(self) -> str:
        return (
            "The coefficient on an endogenous regressor has a causal interpretation only "
            "when the stated instrument relevance, independence, exclusion, and estimand-specific "
            "assumptions are substantively credible."
        )

    def vcov(self) -> pd.DataFrame:
        """Return a defensive copy of the covariance matrix."""

        return self.covariance.copy()

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        """Return coefficient confidence intervals using the fitted reference law."""

        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between zero and one.")
        probability = 0.5 + level / 2.0
        if self.inference_distribution == "normal":
            critical = float(norm.ppf(probability))
        else:
            if self.inference_df is None or self.inference_df <= 0:
                raise ValueError("Positive inference degrees of freedom are required.")
            critical = float(t.ppf(probability, self.inference_df))
        return pd.DataFrame(
            {
                "lower": self.params - critical * self.standard_errors,
                "upper": self.params + critical * self.standard_errors,
            }
        )

    def confint(self, alpha: float = 0.05) -> pd.DataFrame:
        """Statsmodels-style alias accepting a tail probability."""

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one.")
        return self.conf_int(level=1.0 - alpha)

    def summary_frame(self, level: float = 0.95) -> pd.DataFrame:
        """Return the common labeled coefficient-table representation."""

        intervals = self.conf_int(level=level)
        frame = pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.standard_errors,
                "stat": self.test_statistics,
                "p_value": self.pvalues,
                "ci_lower": intervals["lower"],
                "ci_upper": intervals["upper"],
            }
        )
        frame.attrs["statistic_distribution"] = self.inference_distribution
        frame.attrs["statistic_df"] = self.inference_df
        return frame

    def predict(self, endogenous: Any, exogenous: Any | None = None) -> pd.Series:
        """Predict the structural outcome with the fitted schema and parameter order."""

        design, index = prepare_prediction_data(
            endogenous,
            exogenous,
            endogenous_names=self.endogenous_names,
            exogenous_names=self.exogenous_names,
            add_constant=self.add_constant,
        )
        prediction = design @ self.params.to_numpy(dtype=float)
        return pd.Series(prediction, index=index, name="predicted")

    def to_markdown(self, digits: int = 4) -> str:
        """Render a dependency-free Markdown model summary."""

        if digits < 0:
            raise ValueError("digits must be non-negative.")
        lines = ["# IV/2SLS result", ""]
        lines.extend(
            [
                f"- Outcome: `{self.y_name}`",
                f"- Observations: `{self.nobs}`",
                f"- Residual df: `{self.df_resid}`",
                f"- Covariance: `{self.covariance_type}`",
                f"- Inference: `{self.inference_distribution}`",
                "",
                "| term | coef | std_err | stat | p_value |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for term, row in self.summary_frame().iterrows():
            lines.append(
                f"| {term} | {row['coef']:.{digits}f} | {row['std_err']:.{digits}f} | "
                f"{row['stat']:.{digits}f} | {row['p_value']:.{digits}f} |"
            )
        if self.first_stage:
            lines.extend(["", "## First-stage relevance diagnostics", ""])
            for name, diagnostic in self.first_stage.items():
                lines.append(
                    f"- `{name}`: partial R² `{diagnostic.partial_r_squared:.{digits}f}`, "
                    f"classical F `{diagnostic.classical_f_statistic:.{digits}f}`"
                )
        if self.overidentification is not None:
            lines.extend(
                [
                    "",
                    "## Overidentification",
                    "",
                    f"- Sargan χ²({self.overidentification.df}): "
                    f"`{self.overidentification.statistic:.{digits}f}` "
                    f"(p=`{self.overidentification.p_value:.{digits}f}`)",
                ]
            )
        lines.extend(["", f"> {self.causal_interpretation}"])
        return "\n".join(lines)


class IV2SLS:
    """Linear two-stage least squares with explicit identification diagnostics.

    Parameters
    ----------
    covariance:
        ``"unadjusted"`` for homoskedastic inference, ``"robust"`` for HC1,
        or ``"clustered"`` for one-way CR1 inference.
    add_constant:
        Add an intercept to both the structural and instrument designs.
    missing:
        ``"raise"`` rejects any incomplete row. ``"drop"`` removes incomplete
        rows jointly from every input and records how many were removed.

    Notes
    -----
    ``instruments`` passed to :meth:`fit` are excluded instruments only.
    Included exogenous regressors are automatically included in the instrument
    matrix, which avoids ambiguous or duplicated instrument specifications.
    """

    def __init__(
        self,
        *,
        covariance: CovarianceType = "robust",
        add_constant: bool = True,
        missing: MissingPolicy = "raise",
    ) -> None:
        if covariance not in {"unadjusted", "robust", "clustered"}:
            raise ValueError("covariance must be 'unadjusted', 'robust', or 'clustered'.")
        if missing not in {"raise", "drop"}:
            raise ValueError("missing must be 'raise' or 'drop'.")
        if not isinstance(add_constant, bool):
            raise TypeError("add_constant must be a boolean.")
        self.covariance = covariance
        self.add_constant = add_constant
        self.missing = missing

    def fit(
        self,
        y: Any,
        *,
        endogenous: Any,
        instruments: Any,
        exogenous: Any | None = None,
        clusters: Any | None = None,
    ) -> IV2SLSResult:
        """Fit a linear IV model from an outcome, regressors, and excluded instruments."""

        if self.covariance == "clustered" and clusters is None:
            raise ValueError("clusters are required when covariance='clustered'.")
        if self.covariance != "clustered" and clusters is not None:
            raise ValueError("clusters may be supplied only when covariance='clustered'.")
        prepared = prepare_iv_data(
            y,
            endogenous,
            instruments,
            exogenous,
            clusters=clusters,
            missing=self.missing,
        )

        nobs = len(prepared.y)
        included_pieces: list[np.ndarray] = []
        included_names: list[str] = []
        if self.add_constant:
            if "const" in {
                *prepared.endogenous_names,
                *prepared.exogenous_names,
                *prepared.instrument_names,
            }:
                raise ValueError(
                    "The reserved name 'const' is present while add_constant=True; "
                    "remove it or set add_constant=False."
                )
            included_pieces.append(np.ones((nobs, 1), dtype=float))
            included_names.append("const")
        if prepared.exogenous.shape[1]:
            included_pieces.append(prepared.exogenous)
            included_names.extend(prepared.exogenous_names)
        included_exogenous = (
            np.column_stack(included_pieces)
            if included_pieces
            else np.empty((nobs, 0), dtype=float)
        )

        structural = np.column_stack([included_exogenous, prepared.endogenous])
        full_instruments = np.column_stack([included_exogenous, prepared.instruments])
        structural_names = (*included_names, *prepared.endogenous_names)
        n_parameters = structural.shape[1]
        n_instruments = full_instruments.shape[1]
        n_endogenous = prepared.endogenous.shape[1]
        n_excluded = prepared.instruments.shape[1]

        if n_excluded < n_endogenous:
            raise ValueError(
                "The model is underidentified: the number of excluded instruments "
                "must be at least the number of endogenous regressors."
            )
        if nobs <= n_parameters:
            raise ValueError("The number of observations must exceed the structural rank.")
        if nobs <= n_instruments:
            raise ValueError(
                "The number of observations must exceed the number of instrument columns."
            )
        structural_rank = int(np.linalg.matrix_rank(structural))
        if structural_rank < n_parameters:
            raise ValueError(
                "The structural design is rank deficient; remove collinear regressors."
            )
        instrument_rank = int(np.linalg.matrix_rank(full_instruments))
        if instrument_rank < n_instruments:
            raise ValueError(
                "The instrument design is rank deficient; remove collinear instruments."
            )
        cross_rank = int(np.linalg.matrix_rank(full_instruments.T @ structural))
        if cross_rank < n_parameters:
            raise ValueError(
                "The model is not identified: the instruments do not span every "
                "structural regressor."
            )

        ztz_inverse = np.linalg.pinv(full_instruments.T @ full_instruments)
        xz = structural.T @ full_instruments
        normal_matrix = xz @ ztz_inverse @ xz.T
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
        right_hand_side = xz @ ztz_inverse @ full_instruments.T @ prepared.y
        try:
            coefficients = np.linalg.solve(normal_matrix, right_hand_side)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "The IV normal equations are singular; the model is not numerically identified."
            ) from error

        fitted = structural @ coefficients
        residual = prepared.y - fitted
        covariance_estimate = iv_covariance(
            structural,
            full_instruments,
            residual,
            covariance=self.covariance,
            clusters=prepared.clusters,
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance_estimate.matrix), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = coefficients / standard_errors
        if covariance_estimate.distribution == "normal":
            pvalues = 2.0 * norm.sf(np.abs(statistics))
        else:
            assert covariance_estimate.df is not None
            pvalues = 2.0 * t.sf(np.abs(statistics), covariance_estimate.df)

        first_stage = first_stage_diagnostics(
            prepared.endogenous,
            full_instruments,
            included_exogenous,
            endogenous_names=prepared.endogenous_names,
            excluded_count=n_excluded,
            covariance=self.covariance,
            clusters=prepared.clusters,
        )
        overidentification_df = n_instruments - n_parameters
        overidentification = (
            sargan_test(
                residual,
                full_instruments,
                overidentification_df=overidentification_df,
            )
            if overidentification_df > 0 and self.covariance == "unadjusted"
            else None
        )

        notes: list[str] = []
        if prepared.dropped_rows:
            notes.append(f"Dropped {prepared.dropped_rows} row(s) jointly because missing='drop'.")
        if overidentification_df == 0:
            notes.append(
                "The model is exactly identified; overidentifying restrictions cannot be tested."
            )
        elif self.covariance != "unadjusted":
            notes.append(
                "A Sargan test is not reported with heteroskedasticity-robust or clustered "
                "inference because its homoskedastic reference law would be invalid."
            )
        weak_names = [
            name for name, diagnostic in first_stage.items() if diagnostic.weak_instrument_warning
        ]
        if weak_names:
            notes.append("Classical first-stage F < 10 for: " + ", ".join(weak_names) + ".")
        condition_number = float(np.linalg.cond(normal_matrix))
        if not np.isfinite(condition_number) or condition_number > 1e12:
            notes.append(
                "The IV normal matrix is ill-conditioned; estimates may be numerically unstable."
            )

        parameter_index = pd.Index(structural_names, dtype="object")
        params = pd.Series(coefficients, index=parameter_index, name="coef")
        covariance_frame = pd.DataFrame(
            covariance_estimate.matrix,
            index=parameter_index,
            columns=parameter_index,
        )
        return IV2SLSResult(
            params=params,
            covariance=covariance_frame,
            standard_errors=pd.Series(standard_errors, index=parameter_index, name="std_err"),
            test_statistics=pd.Series(statistics, index=parameter_index, name="stat"),
            pvalues=pd.Series(pvalues, index=parameter_index, name="p_value"),
            residuals=pd.Series(residual, index=prepared.index, name="residual"),
            fitted_values=pd.Series(fitted, index=prepared.index, name="fitted"),
            first_stage=first_stage,
            overidentification=overidentification,
            nobs=int(nobs),
            rank=structural_rank,
            df_resid=int(nobs - structural_rank),
            covariance_type=self.covariance,
            inference_distribution=covariance_estimate.distribution,
            inference_df=covariance_estimate.df,
            n_clusters=covariance_estimate.n_clusters,
            y_name=prepared.y_name,
            endogenous_names=prepared.endogenous_names,
            exogenous_names=prepared.exogenous_names,
            instrument_names=prepared.instrument_names,
            estimation_index=prepared.index.copy(),
            dropped_rows=prepared.dropped_rows,
            add_constant=self.add_constant,
            notes=tuple(notes),
            assumptions=(
                "Instrument relevance",
                "Instrument independence/exogeneity",
                "Exclusion restriction",
                "Correctly specified linear structural equation",
                "Treatment consistency and no interference as required by the design",
                "Estimand-specific assumptions such as monotonicity when interpreting a LATE",
            ),
        )


__all__ = ["IV2SLS", "IV2SLSResult"]
