# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for security-sensitive bugs.
2. Open a private security advisory on GitHub: [Advance RAG Security](https://github.com/RaghavOG/advance-rag/security/advisories/new) (or email the maintainer).
3. Include:
   - A description of the vulnerability
   - Steps to reproduce
   - Possible impact (e.g. data exposure, privilege escalation)
   - Any suggested fix, if you have one

We will acknowledge your report and work on a fix. After a patch is released, we can credit you in the release notes (unless you prefer to remain anonymous).

## Best Practices for This Repo

- **Never commit** `.env` or any file containing API keys, passwords, or connection strings. Use `.env.example` for documentation only.
- Keep dependencies up to date (`pip install -U -r requirements.txt`, `npm update`). Consider enabling Dependabot for this repository.
- Run the application with minimal required permissions; avoid storing secrets in code or config that is committed.

Thank you for helping keep this project and its users safe.
