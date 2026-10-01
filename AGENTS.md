# Repository Guidelines

## Project Structure & Module Organization

This repository is at its initial setup stage. It currently contains `.git/` and a local `.venv/` configured for Python 3.10. There are no tracked application modules, tests, assets, or dependency manifests yet.

As implementation begins, place application code in `src/`, automated tests in `tests/`, and supporting documentation in `docs/`. Keep module boundaries aligned with responsibilities. Add asset or data directories only when needed; document their purpose and avoid committing large generated datasets. Treat `.venv/` as local tooling, never application source.

## Build, Test, and Development Commands

Run commands from the repository root using PowerShell:

- `& .\.venv\Scripts\python.exe --version` checks the local interpreter.
- `.\.venv\Scripts\Activate.ps1` activates the existing virtual environment.
- `python -m unittest discover -s tests -p "test_*.py"` runs standard-library tests after a `tests/` directory is introduced.

No application entry point, build command, dependency installation workflow, or test runner is currently configured. Document these commands when adding the corresponding files. Record new dependencies in a committed manifest rather than relying on the local environment.

## Coding Style & Naming Conventions

For new Python code, use four-space indentation and PEP 8 conventions. Name modules, functions, and variables with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Add type hints to public interfaces and keep functions focused. No formatter or linter is configured; commit configuration alongside any tooling introduced.

## Testing Guidelines

No testing framework or coverage threshold has been established. Prefer standard-library `unittest` until the project selects another framework. Name test files `test_*.py` and test methods `test_*`. Cover normal behavior, boundary cases, and failure paths; add regression tests for bug fixes.

## Commit & Pull Request Guidelines

There are no commits yet, so no existing message convention can be inferred. Use concise imperative subjects, such as `Add input validation`, and keep changes focused. Pull requests should explain the purpose, summarize behavior changes, link relevant issues, and report validation performed. Include screenshots only for visual changes.

## Security & Configuration

Keep credentials, local configuration, `.venv/`, and generated outputs out of commits. Add appropriate `.gitignore` rules as files are introduced. Provide sanitized configuration examples and document required environment variables.

## Project-Specific Constraints

This repository is a strict 10-hour portfolio MVP for semiconductor manufacturing AI quality analytics.

- Prefer completing the end-to-end workflow over adding features.
- PR-AUC is the primary ML evaluation metric.
- Treat manufacturing failure as the positive class.
- Do not describe SHAP associations as causal relationships or root causes.
- Never report predicted risk as measured manufacturing yield.
- Keep preprocessing fitted only on training data.
- Do not tune models or thresholds using the held-out test set.
- Do not introduce Kubernetes, GKE, Airflow, Terraform, CI/CD, Streamlit, React, LLMs, deep learning, Vertex AI, or GPUs.
- Use BigQuery Connected Sheets as the reporting interface.
- Deployment scope ends at Docker → Artifact Registry → Cloud Run.
- Stop optional refinement before expanding scope beyond the 10-hour budget.
