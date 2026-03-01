# Contributing to Advance RAG

Thank you for your interest in contributing. This document gives a short guide to how to contribute.

## Getting Started

1. **Fork and clone** the repository.
2. **Set up the environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   copy .env.example .env
   # Edit .env with at least OPENAI_API_KEY; use VECTOR_STORE=chroma for local dev.
   ```
3. **Run the health check** to confirm everything works:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   # Visit http://127.0.0.1:8000/
   ```

4. **Optional — Pre-commit (lint before every commit):**
   ```bash
   pip install pre-commit ruff
   npm install
   pre-commit install
   ```
   Commits will then run lint on staged files (Ruff for Python, ESLint for frontend) and block if hooks fail. See [.pre-commit-config.yaml](.pre-commit-config.yaml) and [.lintstagedrc](.lintstagedrc).

**Node / Python versions:** Use Node from [.nvmrc](.nvmrc) (e.g. `nvm use`) and Python from [.python-version](.python-version) (e.g. `pyenv install` if you use pyenv) so CI and local match.

## How to Contribute

- **Bug reports and feature requests:** Open a [GitHub Issue](https://github.com/RaghavOG/advance-rag/issues). Use the issue templates if available.
- **Code changes:** Open a Pull Request (PR) against the default branch. Keep PRs focused and reasonably sized.

## Branching and Pull Requests

- Create a branch from the default branch (e.g. `main` or `master`): `git checkout -b fix/short-description` or `feature/short-description`.
- Make your changes. Follow existing code style (see below).
- Run any existing tests and lint (e.g. `pytest`, `ruff check`).
- Push your branch and open a PR. Fill in the PR description and link related issues if applicable.
- Address review feedback. Once approved, a maintainer will merge.

## Code Style

- **Python:** 4-space indent, UTF-8, max line length 100–120 if you use a linter. This repo may use `ruff` or `black`; check for config in the repo.
- **EditorConfig:** This repo has an [.editorconfig](.editorconfig) file. Using an EditorConfig plugin in your editor helps keep indentation and line endings consistent.
- **Imports:** Prefer absolute imports for project modules. Keep `__future__` and standard library imports at the top, then third-party, then local.

## Project Structure

- Core RAG logic lives under `config/`, `embeddings/`, `vectorstores/`, `ingestion/`, `query/`, `retrieval/`, `compression/`, `generation/`, `graph/`, `pipeline/`.
- API and health checks: `backend/`.
- Persistence: `database/`.
- Scripts: `scripts/`. Document new scripts in `Scripts.md` and in docstrings.

## Commit Messages

- Use a short summary line; optional body for detail.
- Examples: `Fix Pinecone 404 message in health check`, `Add CONTRIBUTING.md`.

## Questions

If something is unclear, open an issue with the question so others can benefit from the answer.

Thank you for contributing.
