
# Contributing to rag-bencher

Thanks for considering a contribution!

## Setup
- Use Python 3.12–3.14 (recommended: `uv`, install with `make setup`).
- `make sync` creates/refreshes a local venv with all extras for development.
- Alternatively: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`.
- Before sending a PR run: `make dev` (lint + typecheck + tests).

## Tests and checks
- `make lint` (flake8 + isort check + black check) and `make typecheck` (mypy).
- `make test` runs CPU/offline tests via tox; `make test-all` runs the full Python matrix.
- `make test-all-gpu` covers GPU-tagged tests; only run on a GPU host.
- `make help` for an extensive list of targets.
- Keep default runs offline-friendly; mark GPU or network-heavy tests explicitly.

## Coding style
- Follow the formatting enforced by the Make targets; do not mix style guides.
- Prefer type hints on public functions and keep docstrings concise.
- Small, focused PRs are easier to review than sweeping changes.

## Adding a new pipeline or backend
1. Add pipeline logic under `src/rag_bencher/pipelines/` or a provider/vector adapter under `src/rag_bencher/providers` / `src/rag_bencher/vector`.
2. Supply a config in `configs/` (or `configs/providers/`) showing how to enable it.
3. Cover it with tests in `tests/` (prefer offline fixtures; keep GPU-tagged tests separate).
4. Update documentation: README quickstart, `docs/architecture.md`, and example references as needed.
5. Keep CHANGELOG entries user-facing and brief.

## Documentation
- Update README and the docs/ pages when adding features or changing workflows.
- Include sample configs or datasets in `examples/` if they improve discoverability.

## Release process (maintainers)
- Create a Git tag: `git tag X.Y.Z && git push origin X.Y.Z`.
- Create a GitHub Release from the tag; the publish workflow (`.github/workflows/release.yml`) uploads to PyPI.
- Creating a release triggers a workflow that creates a PR for updating the `CHANGELOG.md`. Merge said PR asap.
