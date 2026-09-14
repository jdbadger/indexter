# developer-workflow Specification

## Purpose

CI enforces the lint, format, type and coverage gates on push, but a contributor should not have
to wait for a failed CI run to learn a commit is unformatted or a workflow file is malformed. This
capability is the local half of that enforcement: pre-commit hooks that catch file hygiene,
lockfile drift and code-quality issues before a commit lands, running the same tool versions CI
uses, plus the `justfile` recipes — `lint`, `test`, `build`, `check`, `fmt`, `release` and
`push-release` — that let a contributor reproduce CI locally and cut a release.

## Requirements

### Requirement: Pre-commit hooks

The repository SHALL have a pre-commit configuration that, on commit, checks TOML, YAML and JSON files parse; rejects large added files; fixes trailing whitespace and missing final newlines outside test fixture directories; checks `uv.lock` is up to date; validates GitHub workflow files against their schema; runs `ruff check --fix` and `ruff format`; runs `ty check` over `src/indexter`; and runs the tests affected by the change.

#### Scenario: Unformatted file committed

- **WHEN** a contributor commits a Python file that `ruff format` would change
- **THEN** the hook reformats it and the commit is stopped so the change can be staged

#### Scenario: Invalid workflow file

- **WHEN** a contributor commits a `.github/workflows` file that does not match the GitHub Actions workflow schema
- **THEN** the commit is stopped with the schema error

#### Scenario: Fixture files left alone

- **WHEN** a contributor commits a change under a `tests/fixtures` directory containing trailing whitespace
- **THEN** the whitespace hooks do not modify it

### Requirement: Hooks and CI use the same tool versions

The ruff, ty and pytest hooks SHALL run through `uv run` against the project's dependency groups, so the tool versions they use are those in `uv.lock` — the same versions CI uses. The configuration SHALL NOT pin a separate ruff or ty version.

#### Scenario: Upgrading ruff

- **WHEN** ruff is upgraded in `uv.lock`
- **THEN** the pre-commit ruff hooks and the CI lint job both use the upgraded version with no change to the pre-commit configuration

### Requirement: The per-commit test hook runs only affected tests

The pytest hook SHALL select tests affected by the change rather than the full suite, and SHALL NOT enforce the total coverage floor, which only a full run can meet. The full suite and the coverage floor SHALL remain enforced by CI and by the local full-test recipe.

#### Scenario: Small change

- **WHEN** a contributor commits a change to one module
- **THEN** the hook runs the tests that depend on that module and does not fail on total coverage

### Requirement: Justfile recipes mirror CI

The `justfile` SHALL provide recipes that run the same commands as CI: `lint` (the lint job's checks), `test` (the full suite with the coverage floor under Python 3.11, 3.12 and 3.13), `build` (build the distributions, check their contents and smoke-test the wheel), and `check` (all three). It SHALL also provide `fmt` (apply ruff formatting and auto-fixable lint fixes), `release` and `push-release`, keep `eval`, and list the recipes when run with no arguments.

#### Scenario: Local pre-push check

- **WHEN** a contributor runs `just check` on a tree that passes CI
- **THEN** lint, the three-version test run and the build checks all pass locally

#### Scenario: Listing recipes

- **WHEN** `just` is run with no arguments
- **THEN** the available recipes are listed with their descriptions

### Requirement: Pushing a release

`just push-release` SHALL push `main` and the tag for the current project version to `origin`, and SHALL fail without pushing anything if that tag does not exist locally.

#### Scenario: No tag for the current version

- **WHEN** `just push-release` is run and no `v<version>` tag exists for the project's current version
- **THEN** it exits non-zero, tells the maintainer to run `just release` first, and pushes nothing

### Requirement: Contributor documentation

`CONTRIBUTING.md` SHALL describe how to fork the repository, set up a development environment, install the pre-commit hooks, run the recipes, open a pull request, and — for maintainers — cut a release from release candidate to final. The README SHALL point to it rather than duplicating it.

#### Scenario: New contributor

- **WHEN** a contributor follows `CONTRIBUTING.md` on a fresh fork
- **THEN** they can install dependencies and hooks, run `just check`, and open a pull request against `main`
