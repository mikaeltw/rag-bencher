
# Contributing to rag-bench

Thanks for considering a contribution!

## Setup
- Use Python 3.11 (we recommend `uv`)
- `uv venv && source .venv/bin/activate && uv sync`
- `make fmt && make lint && make test`

## Dev workflow
- Branch from `main`.
- Add or update tests in `tests/`.
- Run `make fmt && make lint && make test` before pushing.
- Open a PR with a clear description and screenshots/reports when relevant.

## Adding a pipeline
- Put code in `rag_bench/pipelines/`.
- Provide a config in `configs/` and docs in README.
- Add a unit test (offline-friendly when possible).

## Release process (maintainers)
- Update `CHANGELOG.md`.
- Create a Git tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
- Create a GitHub Release from the tag; the **publish** workflow uploads to PyPI.
