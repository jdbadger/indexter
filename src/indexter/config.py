"""Layered configuration: packaged defaults -> global user config -> per-repo
config -> explicit overrides. Unknown keys and wrong types are rejected loudly.
"""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from indexter.paths import config_dir

GLOBAL_CONFIG_FILENAME = "config.toml"
REPO_CONFIG_FILENAME = "indexter.toml"


class ConfigError(Exception):
    """A configuration file could not be parsed or validated."""


class Settings(BaseModel):
    """Resolved, validated configuration. Frozen: consumers get a fixed snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embed_batch_size: int = 32
    ignore_patterns: tuple[str, ...] = ()
    max_file_size_bytes: int = 1_000_000
    search_limit: int = 10
    snippet_max_lines: int = 40


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e


def _load_global_layer() -> dict[str, Any]:
    path = config_dir() / GLOBAL_CONFIG_FILENAME
    if not path.is_file():
        return {}
    return _read_toml(path)


def _load_repo_layer(repo: Path) -> dict[str, Any]:
    dedicated = repo / REPO_CONFIG_FILENAME
    pyproject = repo / "pyproject.toml"

    dedicated_exists = dedicated.is_file()
    pyproject_table = None
    if pyproject.is_file():
        pyproject_data = _read_toml(pyproject)
        pyproject_table = pyproject_data.get("tool", {}).get("indexter")

    if dedicated_exists and pyproject_table is not None:
        warnings.warn(
            f"Both {dedicated} and [tool.indexter] in {pyproject} are present; "
            f"using {dedicated} and ignoring the pyproject table.",
            stacklevel=2,
        )

    if dedicated_exists:
        return _read_toml(dedicated)
    if pyproject_table is not None:
        return dict(pyproject_table)
    return {}


def _source_for_key(key: str, global_layer: dict, repo_layer: dict, repo: Path | None) -> str:
    """Best-effort attribution of which file a bad key/value came from, for error messages."""
    if key in repo_layer and repo is not None:
        dedicated = repo / REPO_CONFIG_FILENAME
        if dedicated.is_file():
            return str(dedicated)
        return str(repo / "pyproject.toml")
    if key in global_layer:
        return str(config_dir() / GLOBAL_CONFIG_FILENAME)
    return "<explicit override>"


def load_settings(repo: str | Path | None = None, **overrides: Any) -> Settings:
    """Resolve settings: defaults <- global config <- repo config <- overrides.

    `repo`, when given, is the repository root to look for `indexter.toml` /
    `[tool.indexter]` in. Unknown keys or wrong-typed values raise ConfigError
    naming the offending key and its source file.
    """
    global_layer = _load_global_layer()
    repo_layer = _load_repo_layer(Path(repo)) if repo is not None else {}

    merged: dict[str, Any] = {**global_layer, **repo_layer, **overrides}

    try:
        return Settings(**merged)
    except ValidationError as e:
        repo_path = Path(repo) if repo is not None else None
        messages = []
        for error in e.errors():
            key = ".".join(str(loc) for loc in error["loc"]) or "<root>"
            if error["type"] == "extra_forbidden":
                source = _source_for_key(key, global_layer, repo_layer, repo_path)
                messages.append(f"unknown setting {key!r} in {source}")
            else:
                source = _source_for_key(key, global_layer, repo_layer, repo_path)
                expected = Settings.model_fields[key].annotation if key in Settings.model_fields else "?"
                messages.append(f"invalid value for {key!r} in {source}: expected {expected}, {error['msg']}")
        raise ConfigError("; ".join(messages)) from e
