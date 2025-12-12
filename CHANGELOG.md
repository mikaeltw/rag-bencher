
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.0.3] - 2025-12-12
### What’s Changed
### 🚀 Features

- Tweak changelog release handling (#7) by @mikaeltw

### 🐛 Fixes

- add fromJSON to treat as boolean (#23) by @mikaeltw
- Add commitish to release drafter configuration (#19) by @mikaeltw
- Use always to force scheduling of workflow_call (#17) by @mikaeltw
- Explicit logic to force coveralls to be scheduled (#15) by @mikaeltw
- yet another try at understanding GA dep chains (#13) by @mikaeltw
- Adjusting logic to mark skipped checks and schedule needs jobs (#11) by @mikaeltw
- Fix changelog logic (#9) by @mikaeltw
- fix syntax error in GA yml (#5) by @mikaeltw

### 🧰 Maintenance

- Move commitish (#22) by @mikaeltw
- Update and make documentation reflect the latest workflows and intent… (#21) by @mikaeltw
- Added gates for GPU suite on external pushes and PRs (#4) by @mikaeltw
- Changelog updater (#3) by @mikaeltw
- PR template update (#2) by @mikaeltw
- Standardise GH naming (#1) by @mikaeltw

### Contributors
@mikaeltw

**Full Changelog**: https://github.com/mikaeltw/rag-bencher/compare/0.0.1...0.0.3


## [0.0.1] - 2025-12-07
### Initial release of rag-bencher
- First public release with RAG baselines (naive, multi-query, HyDE, rerank).
- Strict config validation, dataset registry, eval harness, and multi-run reports.
- Caching, usage tracking, CI, linting, and PyPI publish workflow.
