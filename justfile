# List all the commands in this file
list:
    just -l

# Run all the tests against multiple Python versions. Usage: just test
test:
    uv run --python 3.11 --group dev pytest
    uv run --python 3.12 --group dev pytest
    uv run --python 3.13 --group dev pytest

# Bump version, commit, and tag. Usage: just release <major|minor|patch>
release bump:
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Check for clean working directory
    if ! git diff --quiet || ! git diff --staged --quiet; then
        echo "Error: Working directory not clean. Commit or stash changes first."
        exit 1
    fi
    
    # Bump version using uv (updates pyproject.toml and uv.lock)
    uv version --bump {{bump}}
    
    # Get the new version
    NEW_VERSION=$(uv version --short)
    
    # Commit and tag
    git add pyproject.toml uv.lock
    git commit -m "Bump version to ${NEW_VERSION}"
    git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
    
    echo "✓ Version bumped to ${NEW_VERSION}"
    echo "✓ Committed and tagged locally"
    echo "✓ Run 'just push-release' to push to origin"

# Push release commit and tag to origin. Usage: just push-release
push-release:
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Get current version
    NEW_VERSION=$(uv version --short)
    
    # Check if tag exists
    if ! git rev-parse "v${NEW_VERSION}" >/dev/null 2>&1; then
        echo "Error: No tag found for version v${NEW_VERSION}"
        echo "Run 'just release <major|minor|patch>' first to create a release"
        exit 1
    fi
    
    git push origin main --tags
    echo "✓ Pushed to origin"
    echo "✓ Next: Create a Release from tag v${NEW_VERSION} to trigger publish workflow."
