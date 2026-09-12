"""Filesystem locations: XDG-aware app directories and per-repo database paths.

Nothing here creates directories as a side effect of computing a path --
only `ensure_dir` does that, and only writers call it.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

_APP_NAME = "indexter"

# Runs of anything outside lowercase alphanumerics collapse to a single hyphen;
# leading/trailing hyphens are trimmed.
_SLUG_DISALLOWED_RE = re.compile(r"[^a-z0-9]+")


def _xdg_dir(env_var: str, default_relative: str) -> Path:
    """Resolve an XDG base directory, treating an unset or empty value as absent."""
    value = os.environ.get(env_var)
    base = Path(value) if value else Path.home() / default_relative
    return base / _APP_NAME


def data_dir() -> Path:
    """Return `$XDG_DATA_HOME/indexter`, defaulting to `~/.local/share/indexter`.

    Does not create the directory.
    """
    return _xdg_dir("XDG_DATA_HOME", ".local/share")


def config_dir() -> Path:
    """Return `$XDG_CONFIG_HOME/indexter`, defaulting to `~/.config/indexter`.

    Does not create the directory.
    """
    return _xdg_dir("XDG_CONFIG_HOME", ".config")


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if it doesn't exist yet, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_repo_path(repo: str | Path) -> Path:
    """Resolve a repo path to its canonical form.

    Handles relative paths, `.`/`..` segments, trailing slashes, and symlinks --
    two different spellings of the same directory always resolve identically.
    """
    return Path(repo).resolve()


def slugify(name: str) -> str:
    """Reduce a name to lowercase alphanumerics and hyphens.

    Falls back to "repo" if nothing alphanumeric survives, so the database
    filename is never empty.
    """
    slug = _SLUG_DISALLOWED_RE.sub("-", name.lower()).strip("-")
    return slug or "repo"


def db_path(repo: str | Path) -> Path:
    """Return the deterministic database path for a repository.

    A pure function of the canonical repo path: the same repo always maps to
    the same file, and there is no registry recording the mapping -- the
    reverse lookup lives in the database's own `project_metadata`.
    """
    canonical = canonical_repo_path(repo)
    slug = slugify(canonical.name)
    digest = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:12]
    return data_dir() / f"{slug}-{digest}.db"
