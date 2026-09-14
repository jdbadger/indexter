## 1. Hermetic tests and formatting

- [x] 1.1 In `index/tests/test_embed.py`, add a fixture that builds a minimal `tokenizers` WordLevel tokenizer, saves it as `tokenizer.json` under `tmp_path`, and monkeypatches `huggingface_hub.hf_hub_download` to return it; use it in `test_tokenizer_without_model` and `test_fastembed_tokenizer_without_model`
- [x] 1.2 Add `_model_cached_locally(model_name)` (weights present with `local_files_only=True`) and gate `TestRealModel` on it instead of `TOKENIZER_CACHED`
- [x] 1.3 Confirm the suite passes (failures: 0; real-model/real-tokenizer tests skip) with `HF_HUB_OFFLINE=1` and an empty `HF_HOME` in the scratchpad
- [x] 1.4 Run `ruff format src eval` once; confirm `ruff format --check`, `ruff check`, `ty check src/indexter` are clean and the full suite still passes with 100% coverage (no snapshot changes)

## 2. `indexter --version`

- [x] 2.1 Add an eager `--version` callback option to the Typer app printing `indexter <importlib.metadata.version("indexter")>` and exiting zero
- [x] 2.2 Tests in `tests/test_cli.py`: `--version` output matches the installed version and exits zero; `--version list` prints the version without running `list`; `--help` and no-args still list commands

## 3. Package metadata, license and changelog

- [x] 3.1 Add `LICENSE` (MIT, lifted from `~/dev/indexter`) and `license-files = ["LICENSE"]` in `pyproject.toml`
- [x] 3.2 Replace the `Typing :: Typed` and `Libraries :: Python Modules` classifiers with `Topic :: Software Development` and `Topic :: Text Processing :: Indexing`; add `scripts/**` to `source-exclude`
- [x] 3.3 Write `CHANGELOG.md` (Keep a Changelog): an `## [Unreleased]` entry describing the rewrite (Added / Changed / Removed) with an "Upgrading from 0.1" subsection per design decision 12, above the old 0.1.2 and 0.1.1 entries lifted verbatim
- [x] 3.4 Confirm `uv lock --check` still passes (metadata-only changes)

## 4. Distribution checks (`scripts/check_dist.py`)

- [x] 4.1 Content check: exactly one wheel and one sdist in `dist/`; wheel contains `indexter/db/schema.sql`, `indexter/skill/SKILL.md` and `*.dist-info/licenses/LICENSE`, no `tests/` path or `conftest.py`; sdist contains no `eval/` or `scripts/`; each failure names the offending path and exits non-zero
- [x] 4.2 Smoke test: `uv run --isolated --no-project --managed-python --with <wheel>` runs `indexter --version` (must equal `uv version --short`) and a snippet creating a database via `indexter.db.connection.open_db` in a temporary directory, with `XDG_DATA_HOME`/`XDG_CONFIG_HOME` pointed at temporary directories
- [x] 4.3 Verify locally: `uv build` then the script passes; a deliberately broken wheel (a copy with a `tests/` entry added, in the scratchpad) makes it fail naming the path

## 5. Justfile

- [x] 5.1 Add `lint`, `fmt`, `test` (3.11/3.12/3.13 with the coverage floor and `-n auto`), `build` (clean `dist/`, `uv build`, `scripts/check_dist.py`), and `check` recipes, each with a description comment; keep `list` as the default and `eval` unchanged
- [x] 5.2 Add `release bump="" rc=""` per design decision 7: clean-tree check, target version via `uv version --dry-run --short`, prerelease check when promoting, `## [<version>]` changelog check for final versions before writing, then `uv version --bump …`, commit `pyproject.toml` + `uv.lock` as `release: v<version>`, annotated tag
- [x] 5.3 Add `push-release`: fail if `v<version>` tag is missing, else `git push origin main` and the tag
- [x] 5.4 Verify `release` in a throwaway clone in the scratchpad (never in this repository): `minor 1` → `0.2.0rc1` commit and tag; `"" 2` → `0.2.0rc2`; `just release` without a `[0.2.0]` changelog heading fails and changes nothing; with the heading → `0.2.0`; `just release` at `0.2.0` fails; a dirty tree fails; `push-release` without a tag fails (no remote configured, so nothing can be pushed)

## 6. CI and publish workflows

- [x] 6.1 Write `.github/workflows/ci.yml` per design decision 2: triggers, `concurrency` with cancel-in-progress, `permissions: contents: read`; `lint`, `test` (matrix 3.11/3.12/3.13, `fail-fast: false`, `HF_HUB_OFFLINE=1`, `--cov-report=term-missing`) and `build` jobs, each using `astral-sh/setup-uv` with caching, `uv python install`, and `uv sync --locked`
- [x] 6.2 Write `.github/workflows/publish.yml` per design decision 6: `release: published`; build job with `scripts/check_dist.py` and a tag-equals-`v<version>` check naming both on mismatch, uploading `dist/`; `publish-testpypi` (prerelease, `testpypi` environment) and `publish-pypi` (otherwise, `pypi` environment), each `id-token: write`, uv cache disabled, `uv publish`
- [x] 6.3 Validate both workflow files with `uvx check-jsonschema --builtin-schema vendor.github-workflows`
- [x] 6.4 Run every CI command locally exactly as written in `ci.yml`: the lint commands, `uv sync --locked` + pytest under each of 3.11, 3.12 and 3.13 with `HF_HUB_OFFLINE=1`, and the build job's steps

## 7. Pre-commit

- [x] 7.1 Write `.pre-commit-config.yaml` per design decision 8: `pre-commit-hooks` (`check-toml`, `check-yaml`, `check-json`, `check-added-large-files`, `end-of-file-fixer` and `trailing-whitespace` excluding `tests/fixtures/`), `uv-pre-commit` `uv-lock`, `check-jsonschema` `check-github-workflows`, and local `uv run` hooks for `ruff check --fix`, `ruff format`, `ty check src/indexter`, and `pytest --testmon -p no:xdist --no-cov`; pin hook repository `rev`s to current releases
- [x] 7.2 Run `pre-commit run --all-files`; commit-worthy fixes (whitespace/EOF outside fixtures) are applied and the suite still passes; a second run is clean
- [x] 7.3 Confirm the whitespace hooks leave `tests/fixtures/` untouched (`git status` shows no fixture changes)

## 8. README

- [x] 8.1 Install section: `uv tool install --managed-python indexter` from PyPI as the primary path, install-from-checkout as secondary, and why a uv-managed interpreter matters
- [x] 8.2 "Upgrading from 0.1" section matching the changelog: what's gone, deleting 0.1 config keys such as `default_root`, removing the old Qdrant container/data, re-running `indexter init` and re-registering the MCP server
- [x] 8.3 "Development" section: `uv sync --group test`, `pre-commit install`, the recipes table, and the release steps (rc to TestPyPI, verify, changelog heading, final to PyPI); add the `indexter --version` row's mention in CLI docs and a CI status badge

## 9. Verification

- [x] 9.1 `just lint` and `just test` pass (all three Pythons, ≥95% coverage — expected 100%)
- [x] 9.2 `just build` passes: distribution contents and the isolated-wheel smoke test (version printed, database created with sqlite-vec)
- [x] 9.3 `uv publish --dry-run` against the built distributions reports no metadata problems (no upload; stop and report if it would require credentials to validate)
- [x] 9.4 Every scenario in `continuous-integration`, `release-process`, `developer-workflow` and the modified `repo-management-cli` requirement is covered by a test, a local run recorded above, or — for the GitHub-only behaviors (triggers, cancellation, trusted publishing, environments) — the validated workflow file; note in design.md's Open Questions which scenarios can only be observed after the maintainer pushes and releases
