"""Synchronous repository walker: decides which files are candidates for
indexing, cheaply.

Lifted from `~/dev/indexter`'s async walker (dropping `anyio` -- SQLite is a
local file, embedding is CPU/GPU-bound, walking is syscall-bound; nothing
here is I/O-bound in the way that justifies an event loop). Also split in
two: `Walker.walk()` yields stat-level candidates without reading file
contents, and `read_file()` is a separate step. M3's sync-on-search compares
stat data against the `files` table and only reads files that look changed --
an eager walker would cost a full repository read on every no-op search.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

from indexter.config import Settings

logger = logging.getLogger(__name__)

# Extensions that are almost never source code worth indexing.
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".sqlite",
    ".db",
    ".pickle",
    ".pkl",
}


# Always ignored regardless of configured patterns or .gitignore contents --
# a repo's own .gitignore has no reason to list its VCS directory, since git
# itself never walks into it, but our walker isn't git and needs telling.
_ALWAYS_IGNORED = [".git/"]


def compute_hash(relpath: str, content: str) -> str:
    """SHA-256 over the relative path and content combined, so a move/rename
    changes the hash even when the content doesn't.
    """
    return hashlib.sha256(f"{relpath}:{content}".encode()).hexdigest()


class IgnorePatternMatcher:
    """Matches paths against gitignore-style patterns."""

    def __init__(self, patterns: list[str] | None = None) -> None:
        self._patterns = list(patterns or [])
        self._spec = pathspec.GitIgnoreSpec.from_lines(self._patterns)

    def add_patterns(self, patterns: list[str]) -> None:
        self._patterns.extend(patterns)
        self._spec = pathspec.GitIgnoreSpec.from_lines(self._patterns)

    def add_patterns_from_file(self, path: Path) -> None:
        """Add patterns from a gitignore-style file. Missing file is not an error."""
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as e:
            logger.warning("Failed to read ignore file %s: %s", path, e)
            return
        self.add_patterns(lines)

    def should_ignore(self, path: str) -> bool:
        return self._spec.match_file(path)


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def _is_minified(path: Path) -> bool:
    name = path.name.lower()
    return ".min." in name or name.endswith(".min")


@dataclass(frozen=True, slots=True)
class WalkedFile:
    """A file the walker considers a candidate. No content -- see module docstring."""

    path: str  # repo-relative, forward-slash separated
    size: int
    mtime: float
    extension: str  # lowercased, including the leading dot; "" if none


class Walker:
    """Walks a repository, applying every filter that doesn't require reading
    file contents: ignore patterns, binary/minified detection, size, emptiness.
    """

    def __init__(self, repo_path: str | Path, settings: Settings) -> None:
        self.repo_path = Path(repo_path)
        self.settings = settings
        self._matcher = self._build_matcher()

    def _build_matcher(self) -> IgnorePatternMatcher:
        matcher = IgnorePatternMatcher([*_ALWAYS_IGNORED, *self.settings.ignore_patterns])
        matcher.add_patterns_from_file(self.repo_path / ".gitignore")
        return matcher

    def walk(self) -> Iterator[WalkedFile]:
        """Yield one `WalkedFile` per candidate, lazily."""
        try:
            repo_resolved = self.repo_path.resolve()
        except OSError as e:
            logger.warning("Cannot resolve repo path %s: %s", self.repo_path, e)
            return
        yield from self._walk_dir(self.repo_path, repo_resolved)

    def _walk_dir(self, directory: Path, repo_resolved: Path) -> Iterator[WalkedFile]:
        try:
            entries = sorted(directory.iterdir())
        except PermissionError as e:
            logger.warning("Permission denied: %s: %s", directory, e)
            return
        except OSError as e:
            logger.warning("Error reading directory %s: %s", directory, e)
            return

        for entry in entries:
            try:
                relative_str = entry.relative_to(self.repo_path).as_posix()
            except ValueError:
                continue

            try:
                is_dir = entry.is_dir()
            except OSError as e:
                logger.warning("Error accessing %s: %s", entry, e)
                continue

            if is_dir:
                if self._matcher.should_ignore(relative_str + "/"):
                    logger.debug("Pruning directory: %s", relative_str)
                    continue
                if entry.is_symlink() and not self._symlink_dir_within_repo(entry, repo_resolved):
                    logger.debug("Skipping symlinked directory: %s", relative_str)
                    continue
                yield from self._walk_dir(entry, repo_resolved)
                continue

            try:
                if not entry.is_file():
                    continue
            except OSError as e:
                logger.warning("Error accessing %s: %s", entry, e)
                continue

            walked = self._consider_file(entry, relative_str)
            if walked is not None:
                yield walked

    @staticmethod
    def _symlink_dir_within_repo(entry: Path, repo_resolved: Path) -> bool:
        try:
            resolved = entry.resolve()
            resolved.relative_to(repo_resolved)
        except (ValueError, OSError):
            return False
        return True

    def _consider_file(self, entry: Path, relative_str: str) -> WalkedFile | None:
        if self._matcher.should_ignore(relative_str):
            logger.debug("Ignoring (pattern match): %s", relative_str)
            return None
        if _is_binary(entry):
            logger.debug("Ignoring (binary): %s", relative_str)
            return None
        if _is_minified(entry):
            logger.debug("Ignoring (minified): %s", relative_str)
            return None

        try:
            stat = entry.stat()
        except OSError as e:
            logger.warning("Cannot stat %s: %s", relative_str, e)
            return None

        if stat.st_size > self.settings.max_file_size_bytes:
            logger.debug("Ignoring (too large): %s", relative_str)
            return None
        if stat.st_size == 0:
            logger.debug("Ignoring (empty): %s", relative_str)
            return None

        return WalkedFile(
            path=relative_str,
            size=stat.st_size,
            mtime=stat.st_mtime,
            extension=entry.suffix.lower(),
        )


def read_file(repo_path: str | Path, relpath: str) -> tuple[str, str] | None:
    """Read and hash one file. Returns `(content, hash)`, or `None` if it
    can't be decoded as UTF-8 or Latin-1.
    """
    full_path = Path(repo_path) / relpath
    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = full_path.read_text(encoding="latin-1")
        except (UnicodeDecodeError, OSError):
            return None
    except OSError:
        return None
    return content, compute_hash(relpath, content)
