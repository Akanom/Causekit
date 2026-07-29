# Security policy

## Supported versions

`causekit` has not made a stable release. Security fixes are provided for the latest
prerelease and the current `main` branch on the Python versions declared in
`pyproject.toml` (currently Python 3.10 through 3.13). Older prereleases may not receive
backported fixes.

## Reporting a vulnerability

Use the repository's GitHub private security-advisory form. Do not open a public issue
containing exploit details, credentials, private datasets, unpublished research material,
or output that could identify research participants.

Include, when available:

- the affected version or commit;
- operating system and Python version;
- dependency versions;
- impact and realistic attack scenario;
- minimal reproduction steps using synthetic data; and
- a proposed mitigation or disclosure constraint.

The maintainer aims to acknowledge a report within five business days and provide an
initial status update within ten business days. Remediation timing depends on severity,
exploitability, and coordinated-disclosure needs.

Statistical disagreement, numerical instability, or an incorrect methodological claim is
normally a correctness issue rather than a security vulnerability. Report it through the
ordinary issue tracker unless exploitation could expose data, cross a trust boundary,
cause unsafe code execution, or materially corrupt high-stakes results without detection.

## Data and model safety

The core estimator operates on in-memory numeric data and does not require network access.
Users remain responsible for protecting source data, fitted outputs, cluster identifiers,
and exported reports. Do not attach confidential data to bug reports; provide a synthetic
reproducer instead.

Treat files from untrusted sources as untrusted before loading them with third-party data
tools. `causekit` does not make Python pickle or arbitrary serialized objects safe.
Optional reporting integrations can move results beyond the core process; review their
destination, access controls, and redaction policy before sending sensitive output.

Large or adversarial arrays can cause excessive memory use or expensive matrix
factorizations. Validate dimensions and resource limits before exposing fitting through a
multi-user service. Do not log raw rows or identifiers in application error handlers.

The optional real-data registry is the package-owned path that may access the network.
Network access is disabled unless the caller explicitly requests a download. Downloads
must use HTTPS, follow only HTTPS redirects, write through a temporary file, match the
release-pinned SHA-256 digest exactly, and fail closed on any mismatch. Verified files are
cached outside the repository by default; they are never imported as package data.

## Dependency and release policy

Runtime and optional dependencies use bounded compatibility ranges. Confirmed critical
and high-severity vulnerabilities block release unless the maintainer records a justified,
time-bounded exception. Scanner alerts require package-, version-, artifact-, and
file-level review; an alert alone is not evidence that the vulnerable path is reachable.

Before release, resolve and review the dependency graph, run the test and build gates,
inspect wheel and source-distribution contents, and verify clean installation from both
artifacts. Publication credentials or tokens must never be stored in source, tests,
documentation, logs, or generated examples. Prefer short-lived trusted-publishing
credentials and protected release environments.

## Research-use warning

This package is research software, not a substitute for design review or independent
replication. Instrument validity cannot be established by software alone. Independently
validate estimates and assumptions before using results for policy, financial, legal,
medical, or other high-stakes decisions.
