"""Unit tests for indexter.walker."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from indexter.config.repo import RepoFileConfig
from indexter.exceptions import InvalidGitRepositoryError
from indexter.walker import IgnorePatternMatcher, Walker

# ============================================================================
# IgnorePatternMatcher Tests
# ============================================================================


def test_ignore_pattern_matcher_initialization():
    """Test IgnorePatternMatcher initializes with empty patterns."""
    matcher = IgnorePatternMatcher()
    assert matcher._patterns == []
    assert not matcher.should_ignore("test.py")


def test_ignore_pattern_matcher_with_patterns():
    """Test IgnorePatternMatcher with initial patterns."""
    patterns = ["*.pyc", "__pycache__/", "*.log"]
    matcher = IgnorePatternMatcher(patterns)

    assert matcher.should_ignore("test.pyc")
    assert matcher.should_ignore("__pycache__/")
    assert matcher.should_ignore("debug.log")
    assert not matcher.should_ignore("test.py")


def test_ignore_pattern_matcher_add_patterns():
    """Test adding patterns dynamically."""
    matcher = IgnorePatternMatcher(["*.pyc"])
    assert matcher.should_ignore("test.pyc")
    assert not matcher.should_ignore("test.log")

    matcher.add_patterns(["*.log"])
    assert matcher.should_ignore("test.log")


def test_ignore_pattern_matcher_directory_patterns():
    """Test directory-specific patterns."""
    matcher = IgnorePatternMatcher(["build/", "dist/"])

    assert matcher.should_ignore("build/")
    assert matcher.should_ignore("build/file.txt")
    assert matcher.should_ignore("dist/")
    assert not matcher.should_ignore("builder.py")


def test_ignore_pattern_matcher_wildcard_patterns():
    """Test wildcard patterns."""
    matcher = IgnorePatternMatcher(["*.min.*", "test_*.py"])

    assert matcher.should_ignore("script.min.js")
    assert matcher.should_ignore("style.min.css")
    assert matcher.should_ignore("test_utils.py")
    assert not matcher.should_ignore("utils.py")


@pytest.mark.asyncio
async def test_ignore_pattern_matcher_add_from_file(tmp_path: Path):
    """Test loading patterns from a file."""
    gitignore = tmp_path / ".gitignore"
    await anyio.Path(gitignore).write_text("*.pyc\n__pycache__/\n*.log\n")

    matcher = IgnorePatternMatcher()
    await matcher.add_patterns_from_file(gitignore)

    assert matcher.should_ignore("test.pyc")
    assert matcher.should_ignore("__pycache__/")
    assert matcher.should_ignore("debug.log")


@pytest.mark.asyncio
async def test_ignore_pattern_matcher_nonexistent_file(tmp_path: Path):
    """Test loading from nonexistent file doesn't error."""
    matcher = IgnorePatternMatcher()
    await matcher.add_patterns_from_file(tmp_path / "nonexistent")

    assert matcher._patterns == []


@pytest.mark.asyncio
async def test_ignore_pattern_matcher_invalid_file(tmp_path: Path):
    """Test handling of unreadable file."""
    bad_file = tmp_path / "bad"
    bad_file.mkdir()

    matcher = IgnorePatternMatcher()
    await matcher.add_patterns_from_file(bad_file)  # Should not raise

    assert matcher._patterns == []


# ============================================================================
# Walker Tests - Initialization
# ============================================================================


@pytest.mark.asyncio
async def test_walker_create_factory(git_repo_path: Path):
    """Test Walker.create() factory method."""
    walker = await Walker.create(git_repo_path)

    assert walker.repo_path == git_repo_path.resolve()
    assert walker._initialized is True
    assert walker._matcher is not None


def test_walker_init_validates_directory(tmp_path: Path):
    """Test Walker validates directory exists."""
    with pytest.raises(ValueError, match="is not a directory"):
        Walker(tmp_path / "nonexistent")


@patch("indexter.walker.validate_git_repository")
def test_walker_init_validates_git_repo(mock_validate: MagicMock, tmp_path: Path):
    """Test Walker validates git repository."""
    mock_validate.side_effect = InvalidGitRepositoryError()

    with pytest.raises(InvalidGitRepositoryError):
        Walker(tmp_path)


@pytest.mark.asyncio
async def test_walker_default_ignore_patterns(git_repo_path: Path):
    """Test Walker has default ignore patterns."""
    walker = await Walker.create(git_repo_path)

    # Test default patterns are applied
    assert walker._matcher.should_ignore(".git/")
    assert walker._matcher.should_ignore("__pycache__/")
    assert walker._matcher.should_ignore("node_modules/")
    assert walker._matcher.should_ignore(".venv/")


@pytest.mark.asyncio
async def test_walker_loads_gitignore(git_repo_with_gitignore: Path):
    """Test Walker loads .gitignore patterns."""
    walker = await Walker.create(git_repo_with_gitignore)

    # Patterns from .gitignore fixture
    assert walker._matcher.should_ignore("*.log")
    assert walker._matcher.should_ignore("temp/")


@pytest.mark.asyncio
@patch("indexter.walker.RepoFileConfig.from_repo")
async def test_walker_loads_config(mock_from_repo: AsyncMock, git_repo_path: Path):
    """Test Walker loads repo configuration."""
    config = RepoFileConfig(ignore_patterns=["custom_pattern/"], max_file_size=5000)
    mock_from_repo.return_value = config

    walker = await Walker.create(git_repo_path)

    assert walker._max_file_size == 5000
    assert walker._matcher.should_ignore("custom_pattern/")


# ============================================================================
# Walker Tests - Binary Detection
# ============================================================================


@pytest.mark.asyncio
async def test_walker_is_binary_file(git_repo_path: Path):
    """Test binary file detection."""
    walker = await Walker.create(git_repo_path)

    assert walker._is_binary_file(Path("image.png"))
    assert walker._is_binary_file(Path("video.mp4"))
    assert walker._is_binary_file(Path("doc.pdf"))
    assert walker._is_binary_file(Path("archive.zip"))
    assert not walker._is_binary_file(Path("script.py"))
    assert not walker._is_binary_file(Path("readme.md"))


@pytest.mark.asyncio
async def test_walker_is_minified(git_repo_path: Path):
    """Test minified file detection."""
    walker = await Walker.create(git_repo_path)

    assert walker._is_minified(Path("script.min.js"))
    assert walker._is_minified(Path("style.min.css"))
    assert walker._is_minified(Path("bundle.min"))
    assert not walker._is_minified(Path("script.js"))
    assert not walker._is_minified(Path("minimal.py"))


# ============================================================================
# Walker Tests - File Walking
# ============================================================================


@pytest.mark.asyncio
async def test_walker_walk_basic(git_repo_with_files: Path):
    """Test basic file walking."""
    walker = await Walker.create(git_repo_with_files)

    files = []
    async for file_info in walker.walk():
        files.append(file_info)

    assert len(files) > 0
    for file_info in files:
        assert "path" in file_info
        assert "size_bytes" in file_info
        assert "mtime" in file_info
        assert "content" in file_info
        assert "hash" in file_info


@pytest.mark.asyncio
async def test_walker_skips_ignored_patterns(git_repo_with_files: Path):
    """Test walker skips files matching ignore patterns."""
    walker = await Walker.create(git_repo_with_files)

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should not include ignored files
    assert not any(".pyc" in f for f in files)
    assert not any("__pycache__" in f for f in files)
    assert not any(".git" in f for f in files)


@pytest.mark.asyncio
async def test_walker_skips_binary_files(git_repo_with_binary: Path):
    """Test walker skips binary files."""
    walker = await Walker.create(git_repo_with_binary)

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should not include binary files
    assert not any(f.endswith(".png") for f in files)
    assert not any(f.endswith(".pdf") for f in files)


@pytest.mark.asyncio
async def test_walker_skips_minified_files(git_repo_with_minified: Path):
    """Test walker skips minified files."""
    walker = await Walker.create(git_repo_with_minified)

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should not include minified files
    assert not any(".min.js" in f for f in files)
    assert not any(".min.css" in f for f in files)


@pytest.mark.asyncio
async def test_walker_skips_large_files(tmp_path: Path):
    """Test walker skips files exceeding max size."""
    # Create a git repo
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True
    )

    # Create small file
    small_file = repo_path / "small.py"
    await anyio.Path(small_file).write_text("print('small')")
    subprocess.run(["git", "add", "small.py"], cwd=repo_path, capture_output=True, check=True)

    # Create large file (> 10MB default limit)
    large_file = repo_path / "large_file.txt"
    # Create 11MB of data
    large_content = "x" * (11 * 1024 * 1024)
    await anyio.Path(large_file).write_text(large_content)

    # Verify the file is actually large
    stat = await anyio.Path(large_file).stat()
    assert stat.st_size > (10 * 1024 * 1024), f"Large file is only {stat.st_size} bytes"

    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True, check=True
    )

    walker = await Walker.create(repo_path)

    files = []
    sizes = {}
    async for file_info in walker.walk():
        files.append(file_info["path"])
        sizes[file_info["path"]] = file_info["size_bytes"]

    # Should not include large file (11MB > 10MB default limit)
    assert "large_file.txt" not in files
    # But should include the small file
    assert "small.py" in files


@pytest.mark.asyncio
async def test_walker_skips_empty_files(git_repo_with_empty_file: Path):
    """Test walker skips empty files."""
    walker = await Walker.create(git_repo_with_empty_file)

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should not include empty file
    assert "empty.txt" not in files


@pytest.mark.asyncio
async def test_walker_handles_encoding_fallback(git_repo_with_latin1: Path):
    """Test walker handles non-UTF8 files with encoding fallback."""
    walker = await Walker.create(git_repo_with_latin1)

    files = []
    async for file_info in walker.walk():
        if file_info["path"] == "latin1.txt":
            files.append(file_info)

    assert len(files) == 1
    assert files[0]["content"] is not None


@pytest.mark.asyncio
async def test_walker_handles_unreadable_files(git_repo_with_unreadable: Path):
    """Test walker skips unreadable files."""
    walker = await Walker.create(git_repo_with_unreadable)

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should not include unreadable file
    assert "unreadable.bin" not in files


@pytest.mark.asyncio
async def test_walker_computes_hash(git_repo_with_files: Path):
    """Test walker computes hash for files."""
    walker = await Walker.create(git_repo_with_files)

    files = []
    async for file_info in walker.walk():
        files.append(file_info)

    for file_info in files:
        assert file_info["hash"] is not None
        assert len(file_info["hash"]) == 64  # SHA256 hex length


@pytest.mark.asyncio
async def test_walker_nested_directories(git_repo_nested: Path):
    """Test walker handles nested directory structures."""
    walker = await Walker.create(git_repo_nested)

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should find files in nested dirs
    assert any("src/" in f for f in files)
    assert any("tests/" in f for f in files)


@pytest.mark.asyncio
async def test_walker_permission_error_handling(git_repo_path: Path):
    """Test walker handles permission errors gracefully."""
    walker = await Walker.create(git_repo_path)

    with patch.object(anyio.Path, "iterdir", side_effect=PermissionError("No access")):
        files = []
        # Should not raise, just skip the directory
        async for file_info in walker.walk():
            files.append(file_info)


@pytest.mark.asyncio
async def test_walker_os_error_handling(git_repo_path: Path):
    """Test walker handles OS errors gracefully."""
    walker = await Walker.create(git_repo_path)

    with patch.object(anyio.Path, "iterdir", side_effect=OSError("Disk error")):
        files = []
        # Should not raise, just skip the directory
        async for file_info in walker.walk():
            files.append(file_info)


@pytest.mark.asyncio
async def test_walker_stat_error_handling(git_repo_with_files: Path):
    """Test walker handles stat errors gracefully."""
    walker = await Walker.create(git_repo_with_files)

    original_stat = anyio.Path.stat

    async def mock_stat(self):
        # Fail for specific file
        if "test.py" in str(self):
            raise OSError("Cannot stat")
        return await original_stat(self)

    with patch.object(anyio.Path, "stat", mock_stat):
        files = []
        # Should not raise, just skip the problematic file
        async for file_info in walker.walk():
            files.append(file_info["path"])

        assert "test.py" not in files


@pytest.mark.asyncio
async def test_walker_prunes_ignored_directories(git_repo_with_nested_ignored: Path):
    """Test walker prunes entire ignored directories."""
    walker = await Walker.create(git_repo_with_nested_ignored)

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should not traverse into ignored directories
    assert not any("node_modules/" in f for f in files)
    assert not any(".venv/" in f for f in files)


@pytest.mark.asyncio
async def test_walker_relative_paths(git_repo_with_files: Path):
    """Test walker returns relative paths."""
    walker = await Walker.create(git_repo_with_files)

    async for file_info in walker.walk():
        path = file_info["path"]
        # Should be relative, not absolute
        assert not path.startswith("/")
        assert not Path(path).is_absolute()


@pytest.mark.asyncio
async def test_walker_file_metadata(git_repo_with_files: Path):
    """Test walker includes correct file metadata."""
    walker = await Walker.create(git_repo_with_files)

    async for file_info in walker.walk():
        # Check metadata fields
        assert isinstance(file_info["size_bytes"], int)
        assert file_info["size_bytes"] > 0
        assert isinstance(file_info["mtime"], float)
        assert file_info["mtime"] > 0
        assert isinstance(file_info["content"], str)
        assert len(file_info["content"]) > 0


@pytest.mark.asyncio
async def test_walker_respects_max_file_size_config(tmp_path: Path, git_repo_path: Path):
    """Test walker respects max_file_size from config."""
    config = RepoFileConfig(
        max_file_size=100  # Very small
    )

    # Create a file larger than 100 bytes
    test_file = git_repo_path / "large.txt"
    await anyio.Path(test_file).write_text("x" * 200)

    walker = Walker(git_repo_path, config)
    await walker._async_init()

    files = []
    async for file_info in walker.walk():
        files.append(file_info["path"])

    # Should not include the file due to size limit
    assert "large.txt" not in files


@pytest.mark.asyncio
async def test_walker_content_includes_path_in_hash(git_repo_with_files: Path):
    """Test that hash includes both path and content."""
    walker = await Walker.create(git_repo_with_files)

    # Create two files with same content but different paths
    file1 = git_repo_with_files / "file1.txt"
    file2 = git_repo_with_files / "file2.txt"

    content = "Same content"
    await anyio.Path(file1).write_text(content)
    await anyio.Path(file2).write_text(content)

    files = {}
    async for file_info in walker.walk():
        files[file_info["path"]] = file_info["hash"]

    # Hashes should be different because paths are different
    if "file1.txt" in files and "file2.txt" in files:
        assert files["file1.txt"] != files["file2.txt"]


@pytest.mark.asyncio
async def test_walker_read_document_content_utf8(tmp_path: Path):
    """Test reading UTF-8 encoded files."""
    test_file = tmp_path / "utf8.txt"
    content = "Hello, 世界! 🌍"
    await anyio.Path(test_file).write_text(content, encoding="utf-8")

    result = await Walker._read_document_content(test_file)

    assert result == content


@pytest.mark.asyncio
async def test_walker_read_document_content_latin1_fallback(tmp_path: Path):
    """Test fallback to latin-1 for non-UTF8 files."""
    test_file = tmp_path / "latin1.txt"
    # Write latin-1 encoded content
    await anyio.Path(test_file).write_bytes(b"Test \xe9 content")

    result = await Walker._read_document_content(test_file)

    assert result is not None


@pytest.mark.asyncio
async def test_walker_read_document_content_failure(tmp_path: Path):
    """Test handling of unreadable files."""
    test_file = tmp_path / "nonexistent.txt"

    result = await Walker._read_document_content(test_file)

    assert result is None


@pytest.mark.asyncio
async def test_walker_initialization_without_create(git_repo_path: Path):
    """Test that walker auto-initializes if walk() called before create()."""
    walker = Walker(git_repo_path)

    assert walker._initialized is False

    # Calling walk() should trigger initialization
    files = []
    async for file_info in walker.walk():
        files.append(file_info)
        break  # Just check one file

    assert walker._initialized is True
