## Why

After M6 the rewrite does its job end-to-end, but only on this machine: there is no CI, nothing enforces the lint/format/type/coverage gates between commits, 23 source files aren't `ruff format`-clean, two tests silently download a tokenizer from the Hugging Face Hub, the package has no `LICENSE` file or changelog (though `pyproject.toml` links to one), and its version, `0.1.0`, is *lower* than the `0.1.2` of the old tool already published to PyPI under the same name. M7 closes the gap between "works here" and "green CI, publishable" (plan milestone M7).

## What Changes

- Add GitHub Actions CI on pushes to `main` and pull requests: a lint job (`ruff check`, `ruff format --check`, `ty check`), a test job over Python 3.11, 3.12 and 3.13 enforcing ≥95% coverage against the locked dependencies, and a build job that builds the sdist and wheel, checks what the wheel contains, and smoke-tests the installed wheel.
- Add a publish workflow triggered by publishing a GitHub release: build and smoke-test once, check the tag matches the package version, then publish with PyPI trusted publishing — to TestPyPI for a prerelease, to PyPI otherwise.
- Add `.pre-commit-config.yaml`: file hygiene checks, `uv lock` freshness, GitHub workflow schema validation, and ruff, ty and pytest run through `uv run` so hooks use the same locked tool versions as CI.
- Extend the `justfile`: `lint`, `fmt`, `test` (all three Pythons), `build`, `check`, `release` and `push-release`, alongside the existing `eval`. `release` bumps with `uv version --bump`, so it works on macOS and Linux alike (the old recipe's GNU `sed -i` doesn't).
- Make the test suite hermetic: the two tokenizer tests that currently download from the Hugging Face Hub load a small tokenizer file built inside the test instead, the real-model test skips unless the *model* (not just its tokenizer) is cached, and CI runs tests with the Hub in offline mode so a new download fails instead of silently succeeding.
- Apply `ruff format` once across the source tree.
- Add `indexter --version`.
- Make the package publishable: a `LICENSE` file declared via `license-files`, classifiers that describe a CLI tool, a `CHANGELOG.md` whose entry for the rewrite says what changes for someone upgrading from 0.1.x, and a release path whose first release is `0.2.0` — above the old tool's `0.1.2` — reached through `just release` rather than a hand edit.
- **BREAKING** (for users of the published 0.1.x): `0.2.0` replaces the Qdrant-based tool wholesale — different commands, MCP tools, configuration keys and storage. The changelog and README carry an "Upgrading from 0.1" note; no code migrates old configuration or data.
- Update the README: install from PyPI with a uv-managed interpreter, upgrading from 0.1, and a short development section (setup, hooks, recipes, releasing).

## Capabilities

### New Capabilities
- `continuous-integration`: The CI workflow — triggers, the lint, test-matrix and build jobs, locked dependencies, the coverage floor, the wheel-content and smoke checks, and a test suite that needs no network or model download.
- `release-process`: Versioning, the `just release`/`push-release` recipes, the changelog, package metadata required to publish, and the publish workflow (tag/version check, TestPyPI for prereleases, PyPI trusted publishing).
- `developer-workflow`: The pre-commit hooks and the `justfile` recipes a contributor runs locally, and their agreement with what CI runs.

### Modified Capabilities
- `repo-management-cli`: the single CLI entry point gains a `--version` option that prints the installed package version.

## Impact

- **New files**: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `.pre-commit-config.yaml`, `scripts/check_dist.py` (the wheel/sdist content check and smoke test shared by `just build` and both workflows), `LICENSE`, `CHANGELOG.md`.
- **Changed files**: `justfile`; `pyproject.toml` (`license-files`, classifiers, `scripts/` excluded from the sdist); `cli.py` (`--version`) and its tests; `index/tests/test_embed.py` (hermetic tokenizer tests, corrected model-cache gate); `README.md`; 23 source files reformatted by `ruff format` with no behavior change.
- **Schema, databases, search, MCP tools**: unchanged.
- **Dependencies**: none added. `pre-commit` is a contributor tool installed outside the project, as in the old repository.
- **Outside the repository** (performed by the maintainer, not by this change): pointing a GitHub remote at this history, configuring PyPI and TestPyPI trusted publishers for `publish.yml`, and creating the release. CI goes green on the first push; the package is published by the first release.
