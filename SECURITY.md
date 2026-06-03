# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Dataset Generator, please report it responsibly.

**Do not open a public issue.** Instead, please send a detailed report to the project maintainers.

We will acknowledge your report within 48 hours and provide a timeline for resolution within one week.

## Scope

Security concerns include but are not limited to:

- Arbitrary code execution via malicious schemas or input files
- Path traversal vulnerabilities in file I/O
- Insecure handling of user-supplied configuration
- Dependency vulnerabilities in critical paths

## Best Practices for Users

- Always review schema files from untrusted sources before using `from-schema`
- Use virtual environments to isolate dependencies
- Keep `dataset-generator` and its dependencies up to date
- Do not expose generated datasets containing sensitive patterns to public repositories
