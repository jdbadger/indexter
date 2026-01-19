# List all the commands in this file
list:
    just -l

# Run all the tests against multiple Python versions. Usage: just test
test:
    uv run --python 3.11 --group test pytest
    uv run --python 3.12 --group test pytest
    uv run --python 3.13 --group test pytest

# Bump version, commit, and tag. Usage: just release [bump] [rc_number]
# Examples: 
#   just release patch 1    - Bump patch version and add rc1 suffix
#   just release            - Remove rc suffix from current version (for final release)
#   just release minor      - Bump minor version (no rc)
release bump="" rc="":
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Check for clean working directory
    if ! git diff --quiet || ! git diff --staged --quiet; then
        echo "Error: Working directory not clean. Commit or stash changes first."
        exit 1
    fi
    
    # Get current version
    CURRENT_VERSION=$(uv version --short)
    
    # If bump argument provided, bump the version
    if [ -n "{{bump}}" ]; then
        uv version --bump {{bump}}
        BASE_VERSION=$(uv version --short)
    else
        # No bump - strip rc suffix if present
        BASE_VERSION="${CURRENT_VERSION%%rc*}"
        if [ "$BASE_VERSION" = "$CURRENT_VERSION" ]; then
            echo "Error: Current version ($CURRENT_VERSION) has no rc suffix to remove."
            echo "Use 'just release <major|minor|patch>' to bump version."
            exit 1
        fi
    fi
    
    # If rc number provided, append rc suffix
    if [ -n "{{rc}}" ]; then
        NEW_VERSION="${BASE_VERSION}rc{{rc}}"
        # Manually set the RC version in pyproject.toml
        sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
        # Regenerate lock file with new version
        uv lock
        echo "✓ Created release candidate version ${NEW_VERSION}"
    else
        NEW_VERSION="${BASE_VERSION}"
        # Ensure version is set correctly (in case we stripped rc)
        sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
        uv lock
    fi
    
    # Commit and tag
    git add pyproject.toml uv.lock
    git commit -m "release: v${NEW_VERSION}"
    git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
    
    echo "✓ Version set to ${NEW_VERSION}"
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
