"""Prepare a hash-verified local Stata site for the official rdrobust comparator."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "5d7b5cdb7a304b87981736f4d6fa63002a470ee1"
SOURCE_BASE = f"https://raw.githubusercontent.com/rdpackages/rdrobust/{SOURCE_COMMIT}/stata"
RDROBUST_STATA_VERSION = "11.1.0"
FILES = {
    "stata.toc": "772f23e47c0b1283616ea4cb3c40556e6a4e019faf6b6009404007261c79ab7f",
    "rdrobust.pkg": "a5152d98c378708522433a6fc1cfd6535459b254d0f0a6b86dd846a8fc25bd09",
    "rdbwselect.ado": "789ff489d79cb92f74da34f82b40de1c5ed2b780ca9e0d90ad296e8c57ec683a",
    "rdplot.ado": "1ab28b4d1e4e5b6a0e2e0563d9b9b399ffb86346467513054977567093f204b5",
    "rdplot.sthlp": "41aa4f40af4d7f3d9de6cd78e449cf9c2b0e519ebfd20e629e87d85ba669fa74",
    "rdplot_illustration.do": ("0b72b5ef5d2f3382546997f9ef815e22f6cad3792c8ffcc6871354b06af96ad6"),
    "rdbwselect.sthlp": ("8584942234dd5b723881d0557fdfbf476e515faa2d0799caf99eac12a38cc745"),
    "rdrobust.ado": "c82916177517c9b9bf63b432678ef94fb5fe0883c2b936188903b4875eb5a863",
    "rdrobust.sthlp": "6bd246cc9c9d3395862c6584bba24e86f355bef012560ac9a3ebbeb17008e7b3",
    "rdrobustplot.ado": ("aaa3e2642c2aeebfa74d638429f9663ef783e6565aa4b2595f2f5edd883cea17"),
    "rdrobustplot.sthlp": ("f2e97a693b83ed0be8f0c9b33cbd2ef054a9d4701ce429c972084d7a8db717e3"),
    "lrdrobust.mlib": "8326368068fab41e1e002c0d3c1c8251ec5b830dbecc912926cc880fd2a2b663",
    "rdrobust_illustration.do": (
        "87d54aa7527ca7b7b2381030c838a7afac49a2004550c1ae85bf8d13dd476cfc"
    ),
    "rdrobust_illustration_new.do": (
        "53d6611bbae055d334572f660eff248a624395e2e85e9a65c579d57d3c5873e4"
    ),
    "rdrobust_senate.dta": ("cbf8b6fcf832fd7cde817a0332a8151ecec255aa7081afdf2e98c53bd197a60a"),
    "rdrobust_functions.do": ("62977ca8a134004e6a31e4b7764fa39ce1159cf7fa791fd4d4ebb28caba7a9ac"),
}


def _default_output() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path(tempfile.gettempdir())
    return base / "causekit" / "parity" / f"rdrobust-stata-{RDROBUST_STATA_VERSION}"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verified_payload(name: str, expected: str) -> bytes:
    url = f"{SOURCE_BASE}/{name}"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        payload = response.read()
    observed = _digest(payload)
    if observed != expected:
        raise RuntimeError(
            f"Official rdrobust file hash changed for {name}: "
            f"expected {expected}, observed {observed}."
        )
    return payload


def prepare(output: Path) -> Path:
    """Materialize the pinned local installation site and return its absolute path."""

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "compiled").mkdir(exist_ok=True)
    for name, expected in FILES.items():
        target = output / name
        if target.exists() and _digest(target.read_bytes()) == expected:
            continue
        payload = _verified_payload(name, expected)
        temporary = target.with_suffix(target.suffix + ".download")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output(),
        help="Local Stata-site directory (defaults outside the OneDrive workspace).",
    )
    arguments = parser.parse_args()
    output = prepare(arguments.output)
    print(f"source_commit={SOURCE_COMMIT}")
    print(f"rdrobust_stata_version={RDROBUST_STATA_VERSION}")
    print(f"local_stata_site={output}")
    print(f'net install rdrobust, from("{output}") replace')
    print(f'do "benchmarks/validate_rd_stata.do" "{output}"')


if __name__ == "__main__":
    main()
