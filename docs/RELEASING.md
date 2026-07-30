# Releasing CauseKit

CauseKit publishes immutable distributions to PyPI from GitHub Actions. The release
workflow uses PyPI Trusted Publishing; no long-lived PyPI token belongs in GitHub,
developer configuration, or repository files.

## Trusted Publisher identity

The PyPI publisher and GitHub environment must use these exact values:

- PyPI project: `causekit`
- GitHub owner: `Akanom`
- GitHub repository: `Causekit`
- Workflow: `publish.yml`
- Environment: `pypi`

The GitHub environment permits only version tags matching `v*`. The workflow grants
`id-token: write` only to the publish job. A separate unprivileged job checks that the
release tag is `v{project.version}`, builds the source and wheel distributions, validates
their metadata, and transfers them as a GitHub artifact. The official PyPA publisher then
uploads those files and their PEP 740 attestations.

## Release checklist

1. Confirm `pyproject.toml`, `src/causekit/__init__.py`, `CITATION.cff`, documentation,
   notebook pins, validation evidence, and `CHANGELOG.md` all name the same version.
2. Replace `Unreleased` in that version's changelog heading with the release date.
3. Run the complete Python-version CI matrix, minimum-dependency tests, dependency audit,
   build, strict Twine metadata check, and clean-wheel smoke test.
4. Merge the reviewed release commit to `main` only after CI is green.
5. Create and push the annotated tag `v{version}` from the merged commit.
6. Publish a GitHub prerelease for alpha or beta versions. The `published` release event
   invokes `.github/workflows/publish.yml`.
7. Require a successful publish deployment, then verify the release and hashes at
   `https://pypi.org/project/causekit/{version}/` and install the exact version from the
   public index in a clean environment.

PyPI release files and version numbers cannot be replaced. If publication is incorrect,
yank the affected release, document why, and publish a new version; never reuse a tag or
overwrite release history.
