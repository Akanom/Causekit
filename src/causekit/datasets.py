"""Opt-in, hash-verified real datasets used by examples and parity benchmarks.

CauseKit does not redistribute these datasets. Downloads are HTTPS-only, cached outside
the repository by default, and accepted only when their SHA-256 digest matches the
release-pinned registry below.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import pandas as pd


@dataclass(frozen=True)
class RealDataset:
    """Immutable provenance for a downloadable real-data example."""

    name: str
    filename: str
    url: str
    sha256: str
    description: str
    provenance_url: str


REAL_DATASETS: dict[str, RealDataset] = {
    "hsng": RealDataset(
        name="hsng",
        filename="hsng.dta",
        url="https://www.stata-press.com/data/r19/hsng.dta",
        sha256="d19cd25299af57569d93d8f16b4f72d5ffc7f897c9247d8a8e5fddddef43ad11",
        description="1980 U.S. Census state housing data used in Stata's IV examples.",
        provenance_url="https://www.stata-press.com/data/r19/r.html",
    ),
    "nsw_mixtape": RealDataset(
        name="nsw_mixtape",
        filename="nsw_mixtape.dta",
        url=("https://raw.githubusercontent.com/scunning1975/mixtape/master/nsw_mixtape.dta"),
        sha256="fc424cfc9d7861f4b95a6612f27c7e842671fea5a8612edcfe0273ee62e6f0a4",
        description="National Supported Work job-training experiment data.",
        provenance_url="https://mixtape.scunning.com/05-matching_and_subclassification",
    ),
    "cattaneo2": RealDataset(
        name="cattaneo2",
        filename="cattaneo2.dta",
        url="https://www.stata-press.com/data/r19/cattaneo2.dta",
        sha256="631e926eb9981828ba2e542b32c16ae08f336b9efa10621651a8a185405e0577",
        description="Maternal smoking and birthweight treatment-effects data.",
        provenance_url="https://www.stata-press.com/data/r19/causal.html",
    ),
    "hospdd": RealDataset(
        name="hospdd",
        filename="hospdd.dta",
        url="https://www.stata-press.com/data/r19/hospdd.dta",
        sha256="e3ae6451e89cb915c546ab772410046726f280ad7d117611376beb4f46a521bb",
        description="Hospital procedure and patient-satisfaction difference-in-differences data.",
        provenance_url="https://www.stata-press.com/data/r19/causal.html",
    ),
}


class _HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        resolved = urljoin(req.full_url, newurl)
        if urlsplit(resolved).scheme.lower() != "https":
            raise RuntimeError("Dataset download redirected away from HTTPS.")
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_real_data_directory() -> Path:
    """Return the platform-local cache used by the real-data examples."""

    local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return local_root / "causekit" / "datasets"


def _download(specification: RealDataset, destination: Path) -> None:
    if urlsplit(specification.url).scheme.lower() != "https":  # pragma: no cover - registry guard
        raise RuntimeError("Dataset registry URLs must use HTTPS.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = Request(
        specification.url,
        headers={"User-Agent": "causekit-real-data-example/0.6.0a4"},
    )
    opener = build_opener(
        HTTPSHandler(context=ssl.create_default_context()),
        _HTTPSOnlyRedirectHandler(),
    )
    try:
        with opener.open(request, timeout=90) as response:
            if urlsplit(response.geturl()).scheme.lower() != "https":
                raise RuntimeError("Dataset download redirected away from HTTPS.")
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = _sha256(temporary)
        if actual != specification.sha256:
            raise RuntimeError(
                f"Dataset SHA-256 mismatch for {specification.name}: "
                f"expected {specification.sha256}, received {actual}."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verified_real_data_path(
    name: str,
    *,
    data_directory: str | Path | None = None,
    download: bool = False,
) -> Path:
    """Return one verified source path, downloading only with explicit authorization."""

    try:
        specification = REAL_DATASETS[name]
    except KeyError as error:
        available = ", ".join(sorted(REAL_DATASETS))
        raise ValueError(f"Unknown real dataset {name!r}; choose one of: {available}.") from error
    root = default_real_data_directory() if data_directory is None else Path(data_directory)
    path = root / specification.filename
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Pinned dataset {name!r} is absent at {path}. "
                "Re-run with explicit download authorization."
            )
        _download(specification, path)
    actual = _sha256(path)
    if actual != specification.sha256:
        raise RuntimeError(
            f"Refusing unverified dataset {name!r} at {path}: "
            f"expected {specification.sha256}, received {actual}."
        )
    return path


def load_real_dataset(
    name: str,
    *,
    data_directory: str | Path | None = None,
    download: bool = False,
) -> pd.DataFrame:
    """Load one verified Stata-format dataset without converting categorical labels."""

    path = verified_real_data_path(
        name,
        data_directory=data_directory,
        download=download,
    )
    return pd.read_stata(path, convert_categoricals=False)


__all__ = [
    "REAL_DATASETS",
    "RealDataset",
    "default_real_data_directory",
    "load_real_dataset",
    "verified_real_data_path",
]
