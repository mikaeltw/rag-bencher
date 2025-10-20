
# Release guide

This project uses **setuptools-scm** to derive versions from Git tags.

## One-time setup
- In the rag-bench GitHub repo, add a secret **PYPI_API_TOKEN** with the corresponding PyPI token.

## Cutting a release
1. Ensure `main` is green (CI passing).
2. Update `CHANGELOG.md`.
3. Tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. Create a GitHub Release from the tag. The **publish** action will:
   - Build sdist/wheel
   - Upload artifacts to **PyPI** using `PYPI_API_TOKEN`

## TestPyPI (optional)
- Duplicate `publish.yml` into `publish-testpypi.yml` and point to `TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/` and a `TEST_PYPI_API_TOKEN` secret.


## TestPyPI release

Two options:

1. **Pre-release tag**: Create a pre-release on GitHub (e.g., `v0.1.0-rc1`) and mark it as a **pre-release**.
   The `publish-testpypi` workflow triggers on `release: prereleased` and uploads to TestPyPI.

2. **Manual dispatch**: Run the workflow **publish-testpypi** from the Actions tab.
   Ensure you have set the repository secret **TEST_PYPI_API_TOKEN**.

Install from TestPyPI:
```bash
pip install -i https://test.pypi.org/simple/ rag-bench
```
