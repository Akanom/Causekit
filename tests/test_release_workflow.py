from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"


def test_publish_workflow_uses_release_only_trusted_publishing() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "release:\n    types: [published]" in source
    assert "environment:\n      name: pypi" in source
    assert "https://pypi.org/p/causekit" in source
    assert source.count("id-token: write") == 1
    assert "needs: build" in source
    assert "password:" not in source
    assert "PYPI_TOKEN" not in source
    assert "secrets." not in source


def test_publish_workflow_pins_actions_and_checks_release_identity() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in source
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in source
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in source
    assert "actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0" in source
    assert "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33" in source
    assert "github.event.release.tag_name" in source
    assert 'expected_tag = f"v{version}"' in source
    assert "python -m twine check --strict dist/*" in source
    assert "attestations: true" in source
    assert "print-hash: true" in source
