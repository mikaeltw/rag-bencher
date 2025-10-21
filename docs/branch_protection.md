
# Branch protection (enable in GitHub UI)

Protect your default branch (`main` or `master`):
- Require pull request before merging (+1 approval)
- Require status checks to pass (select `ci` only)
- Require branches to be up to date
- Include administrators
- (Optional) Linear history, signed commits

Publishing is configured to trigger **only on Releases** via:
- `.github/workflows/publish-testpypi.yml` (pre-releases)
- `.github/workflows/publish-pypi.yml` (releases)
