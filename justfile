# List all the commands in this file
list:
    just -l

# Check formatting and lint and types -- what the CI lint job runs.
lint:
    uv run --group dev ruff check src scripts
    uv run --group dev ruff format --check src scripts
    uv run --group dev ty check src/indexter scripts

# Apply formatting and auto-fixable lint fixes.
fmt:
    uv run --group dev ruff format src scripts
    uv run --group dev ruff check --fix src scripts

# Run the full test suite under 3.11, 3.12 and 3.13, offline like CI --
# enforces the 95% coverage floor already set in pyproject.toml.
test:
    HF_HUB_OFFLINE=1 uv run --python 3.11 --group test pytest -n auto --cov-report=term-missing
    HF_HUB_OFFLINE=1 uv run --python 3.12 --group test pytest -n auto --cov-report=term-missing
    HF_HUB_OFFLINE=1 uv run --python 3.13 --group test pytest -n auto --cov-report=term-missing

# Build the sdist and wheel, then check their contents and smoke-test the
# installed wheel -- what the CI build job and the publish workflow run.
build:
    rm -rf dist
    uv build
    uv run python scripts/check_dist.py

# Regenerate the `indexter init` banner art (src/indexter/_logo.py) from indexter.png.
logo:
    uv run python scripts/render_logo.py
    uv run --group dev ruff format src/indexter/_logo.py

# lint, test and build, in that order -- the full local pre-push check.
check: lint test build

# Bump the version, commit and tag a release. Usage: just release [bump] [rc]
#   just release minor 1   - first release candidate of the next minor (0.2.0rc1)
#   just release "" 2      - next release candidate (0.2.0rc2)
#   just release           - promote the current release candidate to final (0.2.0)
#   just release patch     - bump straight to a final version, no candidate
release bump="" rc="":
    #!/usr/bin/env bash
    set -euo pipefail

    if ! git diff --quiet || ! git diff --staged --quiet; then
        echo "Error: working tree is not clean. Commit or stash changes first." >&2
        exit 1
    fi

    bump_args=()
    if [ -n "{{bump}}" ]; then
        bump_args+=(--bump "{{bump}}")
    fi
    if [ -n "{{rc}}" ]; then
        bump_args+=(--bump rc)
    elif [ -z "{{bump}}" ]; then
        bump_args+=(--bump stable)
    fi

    new_version=$(uv version --dry-run --short "${bump_args[@]}")

    if [ -z "{{rc}}" ] && ! grep -qF "## [${new_version}]" CHANGELOG.md; then
        echo "Error: CHANGELOG.md has no '## [${new_version}]' heading. Add one before releasing a final version." >&2
        exit 1
    fi

    uv version "${bump_args[@]}"

    git add pyproject.toml uv.lock
    git commit -m "release: v${new_version}"
    git tag -a "v${new_version}" -m "Release v${new_version}"

    echo "✓ Released v${new_version} (committed and tagged locally)"
    echo "✓ Run 'just push-release' to push to origin"

# Push the release commit and tag created by `just release`.
push-release:
    #!/usr/bin/env bash
    set -euo pipefail

    version=$(uv version --short)

    if ! git rev-parse "v${version}" >/dev/null 2>&1; then
        echo "Error: no tag found for version v${version}. Run 'just release' first." >&2
        exit 1
    fi

    git push origin main "v${version}"
    echo "✓ Pushed main and v${version} to origin"
    echo "✓ Next: publish a Release from tag v${version} to trigger the publish workflow."

