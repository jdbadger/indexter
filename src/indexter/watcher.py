"""Background file watcher for automatic re-indexing.

Uses ``watchfiles`` (Rust-backed ``awatch``) to monitor registered
repositories and trigger incremental re-indexing when source files change.
WSL environments are auto-detected and fall back to polling.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re

from qdrant_client import QdrantClient
from watchfiles import Change, DefaultFilter, awatch

from .config.config import WatchSettings
from .repo import Repo

logger = logging.getLogger(__name__)

# How long to sleep when no repos are registered before rechecking.
_NO_REPOS_SLEEP = 10


class IndexterFilter(DefaultFilter):
    """Watchfiles filter built from indexter ignore patterns.

    Converts indexter-style ignore patterns into watchfiles filter rules:
    - Patterns ending in ``/`` (e.g. ``__pycache__/``) become ignored directories.
    - Glob patterns (e.g. ``*.pyc``) become regex entity patterns.
    - Bare names without wildcards (e.g. ``.DS_Store``) become exact-match entity patterns.

    The ``DefaultFilter`` base already ignores common dirs like ``.git`` and
    ``__pycache__``; this class *extends* those defaults.
    """

    def __init__(self, ignore_patterns: list[str]) -> None:
        extra_dirs: list[str] = []
        extra_entity_patterns: list[str] = []

        for pattern in ignore_patterns:
            if pattern.endswith("/"):
                # Directory pattern — strip trailing slash
                extra_dirs.append(pattern.rstrip("/"))
            elif "*" in pattern or "?" in pattern or "[" in pattern:
                # Glob → regex
                extra_entity_patterns.append(fnmatch.translate(pattern))
            else:
                # Bare filename — exact match
                extra_entity_patterns.append(re.escape(pattern) + "$")

        # Grab defaults from a temporary base instance
        base = DefaultFilter()
        all_dirs = tuple(set(base.ignore_dirs) | set(extra_dirs))
        all_entity = tuple(list(base.ignore_entity_patterns) + extra_entity_patterns)

        super().__init__(ignore_dirs=all_dirs, ignore_entity_patterns=all_entity)


def _group_changes_by_repo(
    changes: set[tuple[Change, str]],
    repos: list[Repo],
) -> dict[str, list[tuple[Change, str]]]:
    """Map each changed file to the repo whose path is a prefix."""
    grouped: dict[str, list[tuple[Change, str]]] = {}
    for change_type, path in changes:
        for repo in repos:
            if path.startswith(repo.path):
                grouped.setdefault(repo.name, []).append((change_type, path))
                break
    return grouped


async def watch_repos(
    client: QdrantClient,
    stop_event: asyncio.Event,
    watch_settings: WatchSettings,
) -> None:
    """Watch registered repos and re-index on file changes.

    Outer loop re-fetches the repo list so newly registered (or removed)
    repos are picked up without restarting the server.
    """
    while not stop_event.is_set():
        repos = Repo.get_all()

        if not repos:
            logger.debug("No repos registered — sleeping %ds", _NO_REPOS_SLEEP)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=_NO_REPOS_SLEEP)
            except TimeoutError:
                pass
            continue

        paths = [repo.path for repo in repos]

        # Build a combined filter from all repos' ignore patterns
        all_patterns: list[str] = []
        for repo in repos:
            all_patterns.extend(repo.settings.ignore_patterns)
        watch_filter = IndexterFilter(list(set(all_patterns)))

        logger.info("Watching %d repo(s): %s", len(repos), ", ".join(r.name for r in repos))

        try:
            async for changes in awatch(
                *paths,
                watch_filter=watch_filter,
                stop_event=stop_event,
                debounce=watch_settings.debounce_ms,
                poll_delay_ms=watch_settings.poll_delay_ms,
            ):
                grouped = _group_changes_by_repo(changes, repos)
                for repo_name, repo_changes in grouped.items():
                    try:
                        repo = next(r for r in repos if r.name == repo_name)
                        if not repo.is_stale:
                            logger.debug("Repo %s not stale — skipping", repo_name)
                            continue
                        logger.info(
                            "Re-indexing %s (%d file change(s))",
                            repo_name,
                            len(repo_changes),
                        )
                        repo.index(client, full=False)
                    except Exception:
                        logger.exception("Error re-indexing %s", repo_name)
        except asyncio.CancelledError:
            logger.info("Watcher cancelled")
            return
        except Exception:
            logger.exception("Watcher error — restarting loop")
            await asyncio.sleep(5)
