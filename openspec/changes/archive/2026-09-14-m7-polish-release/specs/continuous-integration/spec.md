## ADDED Requirements

### Requirement: CI runs on every push to main and every pull request

The repository SHALL have a GitHub Actions workflow that runs on pushes to `main` and on pull requests. A newer run for the same branch or pull request SHALL cancel an older run still in progress. The workflow SHALL request no permissions beyond reading repository contents.

#### Scenario: Pull request triggers CI

- **WHEN** a pull request is opened or updated
- **THEN** the CI workflow runs its lint, test and build jobs against the pull request's head

#### Scenario: Superseded run is cancelled

- **WHEN** a second commit is pushed to the same pull request while the first commit's run is still in progress
- **THEN** the first run is cancelled and the second runs

### Requirement: Jobs run against the locked dependencies on a uv-managed interpreter

Every CI job SHALL install dependencies with `uv sync --locked` and run under a uv-managed Python interpreter. A `uv.lock` that no longer matches `pyproject.toml` SHALL fail the job rather than being re-resolved.

#### Scenario: Stale lockfile

- **WHEN** a commit changes a dependency in `pyproject.toml` without updating `uv.lock`
- **THEN** CI fails at dependency installation with uv's lockfile-out-of-date error

#### Scenario: Extension-capable interpreter

- **WHEN** the test job runs the database tests
- **THEN** they run on a uv-managed interpreter whose `sqlite3` module can load sqlite-vec

### Requirement: Lint job

A lint job SHALL run `ruff check` and `ruff format --check` over `src` and `eval`, and `ty check` over `src/indexter`, using the tool versions in `uv.lock`, and SHALL fail if any of them reports a problem.

#### Scenario: Unformatted file

- **WHEN** a commit contains a Python file under `src` that `ruff format` would change
- **THEN** the lint job fails and names the file

#### Scenario: Type error

- **WHEN** a commit introduces a `ty` diagnostic in `src/indexter`
- **THEN** the lint job fails

### Requirement: Test matrix over Python 3.11, 3.12 and 3.13 with a coverage floor

A test job SHALL run the full test suite once per Python version 3.11, 3.12 and 3.13, each run independent of the others' outcome, and SHALL fail any run whose total coverage of the `indexter` package is below 95%. The coverage report SHALL list missing lines.

#### Scenario: One version fails

- **WHEN** a test fails only under Python 3.11
- **THEN** the 3.11 run fails and the 3.12 and 3.13 runs still complete and report their own results

#### Scenario: Coverage drops below the floor

- **WHEN** a commit adds untested code that brings total coverage below 95%
- **THEN** each test run fails with the coverage shortfall and the uncovered lines

### Requirement: The test suite needs no network and downloads no model

The test suite SHALL pass with no network access and an empty Hugging Face cache. Tests that need a real embedding model or its real tokenizer SHALL skip unless that model or tokenizer is already in the local cache, and each such skip SHALL be gated on the specific files it needs. CI SHALL run tests with the Hugging Face Hub in offline mode.

#### Scenario: Empty cache, no network

- **WHEN** the suite runs with `HF_HUB_OFFLINE=1` and an empty `HF_HOME`
- **THEN** every test passes or skips, and none fails for a missing download

#### Scenario: Tokenizer laziness tested without a download

- **WHEN** the tests asserting that loading a tokenizer does not load the model run with an empty cache
- **THEN** they run (not skip) against a tokenizer file created by the test

#### Scenario: Tokenizer cached but model not

- **WHEN** the default model's tokenizer is cached locally but its weights are not
- **THEN** the real-model test skips rather than downloading the model

### Requirement: Build job checks the distributions and smoke-tests the wheel

A build job SHALL build the sdist and wheel, then fail unless the wheel contains `indexter/db/schema.sql`, `indexter/skill/SKILL.md` and the license file, and contains no `tests` directory or `conftest.py`, and the sdist contains no `eval/` or `scripts/` directory. It SHALL then install only the built wheel into an isolated environment on a uv-managed interpreter and fail unless `indexter --version` prints the project's version and a new database can be created through the installed package, loading sqlite-vec. The same checks SHALL be runnable locally with one command.

#### Scenario: Test file leaks into the wheel

- **WHEN** a packaging change causes a `tests` directory to be included in the wheel
- **THEN** the build job fails and names the offending path

#### Scenario: Installed wheel cannot load sqlite-vec

- **WHEN** the installed wheel's database creation fails to load the sqlite-vec extension
- **THEN** the build job fails with the database error

#### Scenario: Successful smoke test

- **WHEN** the wheel builds with the expected contents
- **THEN** the isolated install prints `indexter <version>` matching `pyproject.toml`, creates a database in a temporary directory, and the job passes

### Requirement: The retrieval eval is not run in CI

CI SHALL NOT run the retrieval eval or any test that requires a real embedding model to be downloaded.

#### Scenario: CI workflow contents

- **WHEN** the CI workflow is inspected
- **THEN** no job invokes `just eval` or `eval/run_eval.py`
