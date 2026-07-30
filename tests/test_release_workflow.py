from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


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

    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in source
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in source
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in source
    assert "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c" in source
    assert "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33" in source
    assert "github.event.release.tag_name" in source
    assert 'expected_tag = f"v{version}"' in source
    assert "python -m twine check --strict dist/*" in source
    assert "attestations: true" in source
    assert "print-hash: true" in source


def test_ci_workflow_pins_node_24_actions() -> None:
    source = CI_WORKFLOW.read_text(encoding="utf-8")

    assert source.count("actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803") == 4
    assert source.count("actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1") == 4
    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" not in source
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" not in source
