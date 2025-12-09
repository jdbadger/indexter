"""File system walker that respects .gitignore and indexter config."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pathspec

from .config import settings
from .config.repo import RepoFileConfig
from .exceptions import validate_git_repository
from .utils import compute_hash

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class IgnorePatternMatcher:
    """Matches files against gitignore-style patterns."""

    def __init__(self, patterns: list[str] | None = None):
        """Initialize with optional patterns."""
        self._patterns = patterns or []
        self._spec = pathspec.PathSpec.from_lines("gitwildmatch", self._patterns)

    async def add_patterns_from_file(self, file_path: Path) -> None:
        """Add patterns from a gitignore-style file asynchronously."""
        apath = anyio.Path(file_path)
        if await apath.exists():
            try:
                content = await apath.read_text()
                lines = content.splitlines()
                self._patterns.extend(lines)
                self._spec = pathspec.PathSpec.from_lines("gitwildmatch", self._patterns)
                logger.debug(f"Loaded {len(lines)} patterns from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to read ignore file {file_path}: {e}")

    def add_patterns(self, patterns: list[str]) -> None:
        """Add additional patterns."""
        self._patterns.extend(patterns)
        self._spec = pathspec.PathSpec.from_lines("gitwildmatch", self._patterns)

    def should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        return self._spec.match_file(path)


class Walker:
    """Walks a git repository respecting ignore patterns."""

    # Binary file extensions to skip
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
        ".min.js",
        ".min.css",  # Minified files
    }

    def __init__(self, repo_path: Path, config: RepoFileConfig | None = None):
        """Initialize the walker for a repository.

        NOTE: This constructor performs sync validation only. Use the async
        factory method `RepoWalker.create()` for full async initialization.

        Args:
            repo_path: Path to the repository root.
            config: Optional per-repo configuration from .indexter.conf.
        """
        self.repo_path = repo_path.resolve()
        self._config = config
        self._validate_repo()
        # These will be set by _async_init or create()
        self._max_file_size = 10 * 1024 * 1024  # Default 10 MB
        self._extra_ignore_patterns: list[str] = []
        self._matcher = IgnorePatternMatcher(settings.default_ignore_patterns.copy())
        self._initialized = False

    @classmethod
    async def create(cls, repo_path: Path) -> Walker:
        """Async factory method to create and initialize a RepoWalker.

        Args:
            repo_path: Path to the repository root.

        Returns:
            Fully initialized RepoWalker instance.
        """
        walker = cls(repo_path)
        await walker._async_init()
        return walker

    async def _async_init(self) -> None:
        """Async initialization - load config and build matcher."""
        await self._load_config()
        self._matcher = await self._build_matcher()
        self._initialized = True

    async def _load_config(self) -> None:
        """Load per-repo configuration asynchronously."""
        if self._config is None:
            # Load from .indexter.conf if it exists
            self._config = await RepoFileConfig.from_repo(self.repo_path)

        # Apply config overrides
        self._max_file_size = self._config.max_file_size
        self._extra_ignore_patterns = self._config.ignore_patterns

        if self._extra_ignore_patterns:
            logger.debug(f"Extra ignore patterns: {self._extra_ignore_patterns}")

    @staticmethod
    async def _read_document_content(file_path: Path) -> str | None:
        """Read file content with encoding fallback asynchronously."""
        try:
            return await anyio.Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return await anyio.Path(file_path).read_text(encoding="latin-1")
            except Exception:
                return None
        except Exception:
            return None

    def _validate_repo(self) -> None:
        """Validate that the path is a git repository."""
        if not self.repo_path.is_dir():
            raise ValueError(f"{self.repo_path} is not a directory")
        validate_git_repository(self.repo_path)

    async def _build_matcher(self) -> IgnorePatternMatcher:
        """Build the ignore pattern matcher asynchronously."""
        matcher = IgnorePatternMatcher(settings.default_ignore_patterns.copy())

        # Add .gitignore patterns
        gitignore_path = self.repo_path / ".gitignore"
        await matcher.add_patterns_from_file(gitignore_path)

        # Add extra patterns from indexter.toml or pyproject.toml [tool.indexter]
        if self._extra_ignore_patterns:
            matcher.add_patterns(self._extra_ignore_patterns)

        return matcher

    def _is_binary_file(self, path: Path) -> bool:
        """Check if a file is likely binary based on extension."""
        return path.suffix.lower() in self.BINARY_EXTENSIONS

    def _is_minified(self, path: Path) -> bool:
        """Check if a file is likely minified."""
        name = path.name.lower()
        return ".min." in name or name.endswith(".min")

    async def _walk_recursive(self, directory: anyio.Path) -> AsyncIterator[anyio.Path]:
        """Recursively walk a directory yielding files asynchronously.

        Args:
            directory: Directory to walk.

        Yields:
            Path to each file found.
        """
        try:
            entries = [entry async for entry in directory.iterdir()]
        except PermissionError as e:
            logger.warning(f"Permission denied: {directory}: {e}")
            return
        except OSError as e:
            logger.warning(f"Error reading directory {directory}: {e}")
            return

        for entry in entries:
            try:
                relative = entry.relative_to(self.repo_path)
                relative_str = str(relative)

                # Check if directory should be ignored (early pruning)
                if await entry.is_dir():
                    # Add trailing slash for directory matching
                    if self._matcher.should_ignore(relative_str + "/"):
                        logger.debug(f"Pruning directory: {relative_str}")
                        continue
                    async for sub_entry in self._walk_recursive(entry):
                        yield sub_entry
                elif await entry.is_file():
                    yield entry
            except OSError as e:
                # Handle stat errors (permissions, broken symlinks, etc.)
                logger.warning(f"Error accessing {entry}: {e}")
                continue

    async def walk(self) -> AsyncIterator[dict]:
        """Walk the repository and yield file info for each relevant file.

        Yields:
            Dict with path, size_bytes, mtime, content, and content_hash.
        """
        if not self._initialized:
            await self._async_init()

        async for path in self._walk_recursive(anyio.Path(self.repo_path)):
            relative_path = str(path.relative_to(self.repo_path))

            # Check ignore patterns
            if self._matcher.should_ignore(relative_path):
                logger.debug(f"Ignoring (pattern match): {relative_path}")
                continue

            # Skip binary files
            if self._is_binary_file(Path(path)):
                logger.debug(f"Ignoring (binary): {relative_path}")
                continue

            # Skip minified files
            if self._is_minified(Path(path)):
                logger.debug(f"Ignoring (minified): {relative_path}")
                continue

            # Get file stats
            try:
                stat = await path.stat()
            except OSError as e:
                logger.warning(f"Cannot stat {relative_path}: {e}")
                continue

            # Skip large files (use config value)
            if stat.st_size > self._max_file_size:
                logger.debug(f"Ignoring (too large): {relative_path}")
                continue

            # Skip empty files
            if stat.st_size == 0:
                logger.debug(f"Ignoring (empty): {relative_path}")
                continue

            # Compute document content_bytes and content_hash
            content = await self._read_document_content(Path(path))
            if content is None:
                logger.debug(f"Ignoring (cannot read): {relative_path}")
                continue
            hash = compute_hash(f"{relative_path}:{content}")

            yield {
                "path": relative_path,
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "content": content,
                "hash": hash,
            }
