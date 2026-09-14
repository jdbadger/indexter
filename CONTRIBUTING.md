# Contributing

## Setting up a fork

1. Fork the repository on GitHub (the "Fork" button on
   [jdbadger/indexter](https://github.com/jdbadger/indexter)).
2. Clone your fork and set up the development environment:

   ```bash
   git clone https://github.com/<you>/indexter
   cd indexter
   uv sync --group test
   pre-commit install
   ```

   `uv sync --group test` creates `.venv` with the test dependencies (and,
   transitively, the dev tools — lint, format, types); `pre-commit install`
   wires up the hooks in `.pre-commit-config.yaml` so formatting, linting,
   type-checking and an affected-tests pass run on every commit.
3. Add the upstream repository so you can keep your fork's `main` in sync:

   ```bash
   git remote add upstream https://github.com/jdbadger/indexter
   git fetch upstream
   ```

## Making a change

1. Branch from an up-to-date `main`:

   ```bash
   git fetch upstream
   git checkout -b my-change upstream/main
   ```
2. Make your change, with tests. `just -l` lists the available recipes:

   | Recipe | Runs |
   |---|---|
   | `just lint` | `ruff check`, `ruff format --check`, `ty check` — what CI's lint job runs |
   | `just fmt` | `ruff format` and `ruff check --fix` |
   | `just test` | The full suite under Python 3.11, 3.12 and 3.13, offline like CI |
   | `just build` | `uv build`, then checks the distribution contents and smoke-tests the wheel |
   | `just check` | `lint`, `test` and `build`, in order — the full local pre-push check |
   | `just release [bump] [rc]` | Bumps the version, commits and tags a release (maintainers only) |
   | `just push-release` | Pushes the release commit and tag created by `just release` (maintainers only) |

   Run `just check` before pushing — it's the same lint, test matrix and
   build checks CI runs.
3. Push to your fork and open a pull request against `jdbadger/indexter`'s
   `main` branch:

   ```bash
   git push -u origin my-change
   gh pr create --fill --repo jdbadger/indexter
   ```

   (Or open the PR from the compare view GitHub shows after pushing.) CI
   runs automatically on the PR; make sure it's green before asking for
   review.

## Releasing (maintainers)

1. `just release minor 1` — bumps to the first release candidate of the next
   minor version (for example `0.2.0rc1`), commits `pyproject.toml` and
   `uv.lock`, and tags it.
2. `just push-release` — pushes `main` and the tag.
3. Publish a **prerelease** GitHub release from that tag. The `publish`
   workflow uploads it to TestPyPI via trusted publishing.
4. Install it from TestPyPI and verify (`indexter --version`, a quick
   `indexter init` against a real repository).
5. Move the changelog's `## [Unreleased]` heading to `## [<version>] -
   <date>` (`just release` refuses to cut a final version without this) and
   commit it.
6. `just release` (no arguments) — promotes the current release candidate to
   the final version, commits and tags it.
7. `just push-release`, then publish a non-prerelease GitHub release from
   that tag — the `publish` workflow uploads it to PyPI.
