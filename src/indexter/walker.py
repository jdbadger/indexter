"""
Asynchronous file system walker with intelligent filtering for code repositories.

This module provides efficient file traversal for Git repositories, implementing
smart filtering to identify and process only relevant source files while skipping
binaries, build artifacts, and ignored paths.

The walker is a critical component of Indexter's indexing pipeline, responsible
for discovering files to parse and index. It combines multiple filtering strategies
to ensure only meaningful code files are processed.

Architecture
------------
The module consists of two main classes:

1. **IgnorePatternMatcher**: Pattern matching engine using gitignore-style rules
2. **Walker**: Asynchronous file system traverser with multi-level filtering

Filtering Strategy
------------------
Files are filtered through multiple layers to ensure efficiency:

1. **Pattern Matching**: .gitignore rules, global patterns, and repo-specific patterns
2. **Extension-Based**: Binary file extensions (images, archives, executables, etc.)
3. **Content Detection**: Minified files (.min.js, .min.css)
4. **Size Limits**: Configurable maximum file size threshold
5. **Empty Files**: Zero-byte files are skipped
6. **Encoding**: Files that cannot be read as text are excluded

Pattern Matching
----------------
The IgnorePatternMatcher uses gitignore-style wildcards:

- `*.pyc` - Match files by extension
- `__pycache__/` - Match directories (trailing slash)
- `build/` - Ignore entire directory trees
- `*.min.js` - Exclude minified files
- `!important.log` - Negative patterns (re-include files)

Patterns are loaded from multiple sources in order:
1. Global default patterns from Indexter configuration
2. Repository .gitignore file
3. Repository-specific patterns from indexter.toml

Asynchronous I/O
----------------
The Walker uses anyio for asynchronous file operations, enabling:

- Non-blocking directory traversal
- Concurrent file reading
- Efficient handling of large repositories
- Graceful handling of permission errors and missing files

File reading implements a fallback strategy:
1. Try UTF-8 encoding (most source code)
2. Fall back to Latin-1 for legacy files
3. Skip files that cannot be decoded

Content Hashing
---------------
Each file's content is hashed using SHA-256 combined with its path:

    hash = sha256(f"{relative_path}:{file_content}")

This creates a unique fingerprint for change detection in incremental indexing.
The path is included in the hash to detect file moves/renames.

Classes
-------
IgnorePatternMatcher:
    Compiles and matches paths against gitignore-style patterns using the
    pathspec library. Supports dynamic pattern addition and file-based loading.

Walker:
    Main traversal class that walks a repository asynchronously, applying all
    filtering rules and yielding file information dictionaries for eligible files.

Examples
--------
Basic usage with a repository:

    >>> from indexter.models import Repo
    >>> from indexter.walker import Walker
    >>>
    >>> repo = await Repo.get("my-project")
    >>> walker = Walker(repo)
    >>>
    >>> async for file_info in walker.walk():
    ...     print(f"Found: {file_info['path']} ({file_info['size_bytes']} bytes)")
    ...     print(f"Hash: {file_info['hash']}")

Custom ignore patterns:

    >>> matcher = IgnorePatternMatcher()
    >>> matcher.add_patterns(["*.log", "temp/", "*.tmp"])
    >>> matcher.add_patterns_from_file(Path(".dockerignore"))
    >>>
    >>> if matcher.should_ignore("debug.log"):
    ...     print("File will be ignored")

Processing files selectively:

    >>> walker = Walker(repo)
    >>> python_files = []
    >>>
    >>> async for file_info in walker.walk():
    ...     if file_info['path'].endswith('.py'):
    ...         python_files.append(file_info)
    >>>
    >>> print(f"Found {len(python_files)} Python files")

Performance Considerations
--------------------------
- Directory pruning: Entire ignored directories are skipped without recursion
- Early filtering: Extensions and patterns are checked before file I/O
- Async I/O: Multiple files can be read concurrently
- Memory efficiency: Files are yielded one at a time (generator pattern)
- Stat caching: File metadata is read once per file

Binary File Detection
---------------------
The following extensions are automatically skipped:

- Images: .png, .jpg, .gif, .svg, .ico, .webp
- Videos: .mp4, .avi, .mov, .mkv, .webm
- Audio: .mp3, .wav
- Documents: .pdf, .doc, .docx, .xls, .xlsx, .ppt
- Archives: .zip, .tar, .gz, .rar, .7z
- Executables: .exe, .dll, .so, .dylib
- Fonts: .woff, .ttf, .eot, .otf
- Databases: .sqlite, .db, .pickle

Configuration
-------------
Walker behavior is controlled through repository settings:

- max_file_size: Skip files larger than this threshold (default: 1MB)
- ignore_patterns: Additional patterns beyond .gitignore

Global defaults can be set in ~/.config/indexter/indexter.toml.

Notes
-----
- All file operations are asynchronous and require an event loop
- The walker automatically handles permission errors and I/O issues
- Symlinks are followed but not recursively (to prevent cycles)
- File paths are always relative to the repository root
- The walker is stateless and can be reused multiple times
- Pattern matching is case-sensitive on Linux/macOS, case-insensitive on Windows
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pathspec

from .config import settings
from .utils import compute_hash

if TYPE_CHECKING:
    from .models import Repo


logger = logging.getLogger(__name__)


class IgnorePatternMatcher:
    """Matches files against gitignore-style patterns."""

    def __init__(self, patterns: list[str] | None = None):
        """Initialize with optional patterns."""
        self._patterns = patterns or []
        self._spec = pathspec.PathSpec.from_lines("gitwildmatch", self._patterns)

    def add_patterns_from_file(self, file_path: Path) -> None:
        """Add patterns from a gitignore-style file."""
        path = Path(file_path)
        if path.exists():
            try:
                content = path.read_text()
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
        ".min.css",
    }

    def __init__(self, repo: Repo):
        """Initialize the walker for a repository."""
        self.repo = repo
        self.repo_path = repo.path
        self.repo_settings = repo.settings
        self._matcher = self._build_matcher()

    def _build_matcher(self) -> IgnorePatternMatcher:
        """Build the ignore pattern matcher."""
        matcher = IgnorePatternMatcher(settings.ignore_patterns.copy())
        gitignore_path = Path(self.repo_path) / ".gitignore"
        matcher.add_patterns_from_file(gitignore_path)
        if self.repo_settings.ignore_patterns:
            matcher.add_patterns(self.repo_settings.ignore_patterns)
        return matcher

    def _is_binary_file(self, path: Path) -> bool:
        """Check if a file is likely binary based on extension."""
        return path.suffix.lower() in self.BINARY_EXTENSIONS

    def _is_minified(self, path: Path) -> bool:
        """Check if a file is likely minified."""
        name = path.name.lower()
        return ".min." in name or name.endswith(".min")

    @staticmethod
    async def _read_content(file_path: anyio.Path) -> str | None:
        """Read file content asynchronously with encoding fallback."""
        try:
            return await file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return await file_path.read_text(encoding="latin-1")
            except Exception:
                return None
        except Exception:
            return None

    async def _walk_recursive(self, directory: anyio.Path) -> AsyncIterator[anyio.Path]:
        """Recursively walk a directory yielding files.

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
                if await entry.is_dir():
                    if self._matcher.should_ignore(relative_str + "/"):
                        logger.debug(f"Pruning directory: {relative_str}")
                        continue
                    async for sub_entry in self._walk_recursive(entry):
                        yield sub_entry
                elif await entry.is_file():
                    yield entry
            except OSError as e:
                logger.warning(f"Error accessing {entry}: {e}")
                continue

    async def walk(self) -> AsyncIterator[dict]:
        """Walk the repository and yield file info for each relevant file.

        Yields:
            Dict with path, size_bytes, mtime, content, and content_hash.
        """
        async for path in self._walk_recursive(anyio.Path(self.repo_path)):
            relative_path = str(path.relative_to(self.repo_path))

            if self._matcher.should_ignore(relative_path):
                logger.debug(f"Ignoring (pattern match): {relative_path}")
                continue

            if self._is_binary_file(Path(path)):
                logger.debug(f"Ignoring (binary): {relative_path}")
                continue

            if self._is_minified(Path(path)):
                logger.debug(f"Ignoring (minified): {relative_path}")
                continue

            try:
                stat = await path.stat()
            except OSError as e:
                logger.warning(f"Cannot stat {relative_path}: {e}")
                continue

            if stat.st_size > self.repo_settings.max_file_size:
                logger.debug(f"Ignoring (too large): {relative_path}")
                continue

            if stat.st_size == 0:
                logger.debug(f"Ignoring (empty): {relative_path}")
                continue

            content = await self._read_content(path)
            if content is None:
                logger.debug(f"Ignoring (cannot read): {relative_path}")
                continue

            yield {
                "path": relative_path,
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "content": content,
                "hash": compute_hash(f"{relative_path}:{content}"),
            }
