## Context

M1–M6 built and verified the rewrite locally. Every gate so far has been run by hand: `uv run --group test pytest` (1,052 tests, 100% coverage, `--cov-fail-under=95` already in `addopts`), `ruff check`, `ty check src/indexter`, and a `uv build` wheel inspection in M6. The `justfile` has one real recipe, `eval`. There is no `.github/`, no pre-commit configuration, no `LICENSE` and no `CHANGELOG.md`, and the history has no Git remote.

The old tool (`~/dev/indexter`, published to PyPI as `indexter` 0.1.1 and 0.1.2 from `github.com/jdbadger/indexter`) had a working release setup worth reusing in shape: a CI matrix over 3.11–3.13, a publish workflow on GitHub releases with trusted publishing (TestPyPI for prereleases), a pre-commit config, and `just release`/`push-release` recipes.

Checked before writing this:

| Fact | Finding |
|---|---|
| Package version | `0.1.0` in `pyproject.toml`; PyPI already has `indexter` 0.1.2 (old tool) |
| `ruff format --check src eval` | 23 files would be reformatted; `ruff check` clean |
| `ty check src/indexter` | clean (`eval/` has 15 diagnostics; it isn't part of the package) |
| Tests offline (`HF_HUB_OFFLINE=1`, empty `HF_HOME`) | 2 failures: `test_tokenizer_without_model`, `test_fastembed_tokenizer_without_model` download `tokenizer.json`; 2 real-model tests skip |
| Real-model test gate | `TestRealModel` skips on `TOKENIZER_CACHED`, so a cached tokenizer with an uncached model would download the model |
| Suite time | ~4 s with `-n auto`, locally |
| Locked tool versions | ruff 0.16.7, ty 0.0.80, torch 2.14.0 (Linux resolution pulls 15 `nvidia-*` CUDA wheels) |
| `uv` | 0.11.7; `uv version --bump {major,minor,patch,stable,alpha,beta,rc,post,dev}`, `uv publish --dry-run`, `uv tool install --managed-python` all available |
| Old `just release` | uses GNU `sed -i "s/…/"`, which fails on macOS BSD `sed` |
| Wheel exclusions | `wheel-exclude`/`source-exclude` already drop tests, `conftest.py`, `__pycache__`, and `eval/` from the sdist |
| Classifiers | include `Typing :: Typed` (no `py.typed` shipped) and `Libraries :: Python Modules` (there is no library API) |
| CLI | no `--version` |
| `pre-commit` | installed as a global tool on the maintainer's machine, not a project dependency |

Constraints: settled decision 12 (uv, ruff, ty, pytest, ≥95% coverage, co-located tests); sqlite-vec needs a Python whose `sqlite3` can load extensions, which uv-managed interpreters provide; the eval stays out of CI (retrieval-eval spec).

## Goals / Non-Goals

**Goals:**
- A push or pull request gets lint, a three-version test matrix and a packaging check, all against `uv.lock`.
- What CI runs and what a contributor runs locally (hooks, recipes) are the same commands at the same tool versions.
- Publishing is one GitHub release away, with nothing about the package blocking upload: a version above what PyPI has, license file, accurate metadata, a changelog.
- The test suite passes with no network.
- Someone with 0.1.x installed learns, before and after upgrading, that 0.2.0 is a different tool.

**Non-Goals:**
- Actually publishing, creating the GitHub remote, or configuring trusted publishers — maintainer steps outside the repository.
- Migrating 0.1.x configuration or data (Qdrant collections, `repos.json`).
- An OS matrix beyond Linux, or Windows support.
- CPU-only torch resolution, dependency changes, or trimming the install footprint.
- Running the retrieval eval or any real-model test in CI.
- Documentation sites, badges beyond CI status, or code-signing.

## Decisions

### 1. The first release is `0.2.0`, reached through the release recipe

PyPI rejects a version at or below one already published, and `0.1.x` belongs to the old tool. The rewrite is incompatible with 0.1 in every user-facing surface, so its first release is the next minor under 0.x (still pre-1.0, Alpha classifier kept) rather than `1.0.0`, which would promise a stability the MCP surface hasn't earned yet.

This change does not edit `version` by hand. Checked with `uv version --dry-run`: from `0.1.0`, `--bump minor --bump rc` gives `0.2.0rc1`; from `0.2.0rc1`, `--bump rc` gives `0.2.0rc2` and `--bump stable` gives `0.2.0`; a bare `--bump rc` on a final version is refused. So `just release minor 1` produces the first candidate for TestPyPI, and `just release` the final `0.2.0` — the same path every later release takes. Until then the tree's `0.1.0` is a development version that is never published: the publish workflow's tag check and PyPI's own ordering both stop it.

*Alternative:* a new distribution name. Rejected — the old name, repository, and `uv tool upgrade indexter` path are what existing users have.

### 2. CI: three jobs, one workflow

`ci.yml`, on `push` to `main` and `pull_request`, with `concurrency` cancelling superseded runs of the same ref, `permissions: contents: read`, on `ubuntu-latest`:

- **lint** — `uv sync --locked --group dev`, then `ruff check src eval`, `ruff format --check src eval`, `ty check src/indexter`.
- **test** — matrix `python-version: ["3.11", "3.12", "3.13"]`, `fail-fast: false`; `uv sync --locked --python <v> --group test`, then `uv run --python <v> --group test pytest -n auto`, with `HF_HUB_OFFLINE=1` set on the job. Coverage floor and report come from `addopts` plus `--cov-report=term-missing`.
- **build** — `uv build`, then a wheel-content check and a smoke test (decision 5).

`astral-sh/setup-uv` with its cache enabled (keyed on `uv.lock`) and `uv python install <v>` so every job runs under a uv-managed interpreter — the same condition sqlite-vec needs in production. `uv sync --locked` fails the job if `uv.lock` is stale rather than quietly re-resolving.

*Why `HF_HUB_OFFLINE=1`:* GitHub runners have network, so a test that downloads would pass in CI and flake under Hub rate limits later. Offline mode turns an accidental download into an immediate failure.

*Why `-n auto`:* already a test dependency; parallelism keeps the matrix quick. Coverage combines across workers via pytest-cov.

*Alternative:* invoking `just` recipes from CI. Rejected — adds an install step and a layer of indirection; recipes and CI instead run literally the same commands, and the developer-workflow spec holds them to it.

### 3. Hermetic tokenizer tests

`test_tokenizer_without_model` and `test_fastembed_tokenizer_without_model` exist to prove that asking for a tokenizer never loads the model. They keep exercising the real `_load_tokenizer` path: a test fixture builds a tiny `tokenizers` WordLevel tokenizer, saves it as `tokenizer.json` under `tmp_path`, and monkeypatches `huggingface_hub.hf_hub_download` to return that path. `TestRealModel` gets its own gate, `_model_cached_locally`, checking the model's weights are in the local cache (`local_files_only=True`), instead of reusing the tokenizer gate.

*Alternative:* skip them when uncached. Rejected — they'd never run in CI, and they cover the laziness contract, not the real tokenizer (`TestRealTokenizer` does that).

### 4. One-time formatting, then enforced

`ruff format src eval` in its own task, verified behavior-neutral by the full suite (inline snapshots use `ruff format` as their format command, so they already agree with it). From then on `ruff format --check` gates CI and the pre-commit hook formats on commit. Fixture sources remain excluded by the existing `extend-exclude`.

### 5. Packaging checks and smoke test

After `uv build`, a short Python check (stdlib `zipfile`/`tarfile`, run by `just build` and CI alike) asserts the wheel contains `indexter/db/schema.sql`, `indexter/skill/SKILL.md` and `LICENSE` metadata, and no `tests/` directories or `conftest.py`; and the sdist contains no `eval/`.

The smoke test installs only the built wheel into an isolated environment on a uv-managed interpreter — `uv run --isolated --no-project --managed-python --with dist/*.whl` — and runs:

1. `indexter --version`, which must print the `pyproject.toml` version;
2. a Python snippet that opens a new database through `indexter.db.connection.open_db` in a temporary directory, which loads sqlite-vec, applies the schema and inserts into `vectors`.

That exercises the one dependency most likely to break on a clean install (the loadable extension) without downloading a model.

### 6. Publish workflow

`publish.yml`, on `release: published`:

- **build** — checkout, uv, `uv build`, the decision-5 checks, then fail unless the release tag equals `v` + `uv version --short`; upload `dist/` as an artifact.
- **publish-testpypi** — if `github.event.release.prerelease`: environment `testpypi`, `permissions: id-token: write`, `uv publish --publish-url https://test.pypi.org/legacy/`.
- **publish-pypi** — otherwise: environment `pypi`, `id-token: write`, `uv publish`.

Trusted publishing, no stored tokens; lifted in shape from the old repository, with the tag check added (the old workflow would publish whatever version `pyproject.toml` held, whatever the tag said). Publish jobs disable the uv cache, as the old workflow learned to.

### 7. Release recipes

`just release [bump] [rc]`:

- refuses a dirty working tree;
- `bump` given → `uv version --bump <bump>`, adding `--bump rc` when `rc` is given (`0.1.0` + `minor 1` → `0.2.0rc1`);
- only `rc` given → `uv version --bump rc` (`0.2.0rc1` → `0.2.0rc2`);
- neither → `uv version --bump stable` (`0.2.0rc2` → `0.2.0`), failing with a usage message if the current version isn't a prerelease;
- for a final (non-rc) version, refuses unless `CHANGELOG.md` has a `## [<version>]` heading, so the entry is dated before the tag exists — checked *before* the version is written, by computing the target with `uv version --dry-run --short`;
- commits `pyproject.toml` and `uv.lock` as `release: v<version>` and creates an annotated tag `v<version>`.

`just push-release` pushes `main` and the current version's tag, failing if the tag doesn't exist. Neither recipe publishes; publishing is creating the GitHub release. `uv version` edits `pyproject.toml` and the lockfile itself, so no `sed`, and the recipes work on macOS.

### 8. Pre-commit hooks use the locked tools

`.pre-commit-config.yaml`:

- `pre-commit/pre-commit-hooks`: `check-toml`, `check-yaml`, `check-json`, `check-added-large-files`, `end-of-file-fixer`, `trailing-whitespace` (fixture directories excluded, since their odd formatting is deliberate);
- `astral-sh/uv-pre-commit`: `uv-lock`;
- `python-jsonschema/check-jsonschema`: `check-github-workflows`;
- `local` hooks via `uv run --group dev`: `ruff check --fix`, `ruff format`, `ty check src/indexter`; and `uv run --group test pytest --testmon -p no:xdist --no-cov` so a commit reruns only tests affected by the change.

*Why local ruff/ty hooks:* the `ruff-pre-commit` mirror pins its own ruff version in the YAML, which drifts from `uv.lock` and from CI; `uv run` uses exactly the locked version. The coverage floor isn't checked per commit (testmon runs a subset, which would always fail a total-coverage floor); CI enforces it.

### 9. Justfile recipes

| Recipe | Runs |
|---|---|
| `list` | `just -l` (kept) |
| `lint` | the CI lint job's three commands |
| `fmt` | `ruff format src eval` and `ruff check --fix src eval` |
| `test` | pytest under 3.11, 3.12 and 3.13, as CI does |
| `build` | `uv build` plus the decision-5 checks and smoke test |
| `check` | `lint`, `test`, `build` |
| `release`, `push-release` | decision 7 |
| `eval` | unchanged |

The decision-5 checks live in one script, `scripts/check_dist.py`, so `just build` and both workflows call the same file. It's a development script, excluded from the sdist like `eval/` (`source-exclude` gains `scripts/**`), and outside the package, so outside coverage.

### 10. `indexter --version`

A Typer callback option on the app, `--version`, eager, printing `indexter <version>` from `importlib.metadata.version("indexter")` and exiting zero. It's what the smoke test and bug reports need, and it's the conventional way to tell 0.1 from 0.2 on a machine that has both.

### 11. Package metadata

- `version` untouched (decision 1); `license = "MIT"` kept; `license-files = ["LICENSE"]`; `LICENSE` lifted from the old repository.
- Classifiers: drop `Typing :: Typed` and `Topic :: Software Development :: Libraries :: Python Modules`; add `Topic :: Software Development` and `Topic :: Text Processing :: Indexing`. Keep Alpha, Console, 3.11–3.13.
- `CHANGELOG.md` in Keep a Changelog format, with an `## [Unreleased]` entry describing the rewrite above the old 0.1.1 and 0.1.2 entries (the project's history continues; the links in `pyproject.toml` already point at it). Releasing renames it to `## [0.2.0] - <date>` (decision 7).

### 12. Upgrading from 0.1 is documented, not automated

The rewrite's changelog entry and a README section say: the old commands, MCP tools and `repos.json` are gone; Qdrant is no longer used (its container and data can be removed); `~/.config/indexter/config.toml` keys from 0.1 (for example `default_root`) are rejected by 0.2 and must be deleted — the error already names the key and the file; re-register the MCP server and run `indexter init` per repository.

*Alternative:* tolerate or auto-strip known 0.1 keys. Rejected — it would carve an exception into the configuration spec's reject-unknown-keys rule for a one-time transition, and silently editing a user's config file is worse than a clear error plus a note.

### 13. README install path

Install becomes `uv tool install --managed-python indexter`, so the tool always gets an interpreter that can load sqlite-vec even when a system Python 3.11–3.13 is present; from-checkout install stays as a secondary option. A "Development" section covers `uv sync --group test`, `pre-commit install`, the recipes, and the release steps (`just release minor 1` → `just push-release` → publish a prerelease GitHub release → verify on TestPyPI → `just release` → `just push-release` → publish the release).

## Risks / Trade-offs

- [Linux CI resolves CUDA torch — ~3 GB of wheels installed per matrix job, though no test imports torch] → setup-uv's cache keyed on `uv.lock` makes it a first-run cost; if runs are still slow, a CPU-only torch index for Linux is a follow-up that changes resolution for source installs and deserves its own decision.
- [Green CI can't be observed until the maintainer pushes to GitHub] → every CI command is run locally under 3.11, 3.12 and 3.13 before the change is done, and `check-github-workflows` validates the workflow files against GitHub's schema.
- [Publishing 0.2.0 under the old name breaks 0.1.x users on `uv tool upgrade`] → changelog and README upgrade notes, a minor-version bump, and the config error that already names the offending key and file.
- [`--testmon` pre-commit hook can miss a test affected through a non-Python file (schema, SKILL.md, fixtures)] → CI runs the full suite; `just test` is the local full run.
- [Trusted publishers misconfigured, or the environment names don't match] → a prerelease to TestPyPI is the first publish, and the release steps say to verify it before the final release.
- [Hook repositories' `rev`s and GitHub action versions age] → pinned to current releases at implementation; `pre-commit autoupdate` is the documented refresh.

## Migration Plan

In this repository the change lands like earlier milestones — local and reversible by reverting its commits. Everything after it is the maintainer's, in order:

1. Add the GitHub remote and push; CI runs on the push.
2. Create `testpypi` and `pypi` environments in the GitHub repository and register `publish.yml` as a trusted publisher for each on TestPyPI and PyPI.
3. `just release minor 1` → `just push-release` → publish a *prerelease* GitHub release for `v0.2.0rc1` → install it from TestPyPI and run `indexter --version`.
4. Rename the changelog's `[Unreleased]` heading to `[0.2.0] - <date>` and commit → `just release` → `just push-release` → publish the GitHub release for `v0.2.0`.

Rollback: a bad release can't be re-uploaded under the same version; yank it on PyPI and release the next patch.

## Open Questions

- **Where this history goes on GitHub** — replacing `jdbadger/indexter`'s `main` (the old tool stays reachable through its `v0.1.x` tags) or pushing to a new branch first. The maintainer's call; nothing in this change depends on it.

- **Scenarios only observable after the maintainer pushes and releases** — everything else in the four specs was covered by a test, a local run recorded in `tasks.md`, or direct inspection of the validated workflow files (`.github/workflows/ci.yml`, `.github/workflows/publish.yml`); the following need a real GitHub Actions run and can't be exercised from this repository:
  - CI actually triggering on a pushed pull request or a push to `main` (`continuous-integration`: "Pull request triggers CI").
  - A superseded run actually being cancelled by `concurrency` (`continuous-integration`: "Superseded run is cancelled") — the config is in `ci.yml` and validated against the schema, but cancellation is GitHub's own behavior.
  - The test matrix's `fail-fast: false` actually letting 3.12/3.13 finish when 3.11 fails (`continuous-integration`: "One version fails") — each version passes locally; independent-failure behavior is a property of the GitHub Actions matrix runner.
  - Trusted publishing actually exchanging an OIDC token for a PyPI/TestPyPI upload, and the `testpypi`/`pypi` deployment environments actually gating which job runs (`release-process`: "Prerelease", "Final release") — `uv publish --dry-run` confirmed the package metadata locally (task 9.3), but trusted publishing only works inside a real Actions run with the environments and publishers registered (Migration Plan steps 2–4).
  - The publish workflow's tag-equality step actually stopping a real mismatched release before any publish job starts (`release-process`: "Tag does not match the version") — the shell logic itself was verified standalone against both a matching and a mismatched tag/version pair, but only a real `release: published` event exercises it end to end, including that no publish job runs afterward.
