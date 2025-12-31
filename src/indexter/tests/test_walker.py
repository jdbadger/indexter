"""Tests for walker.py module."""

import logging
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest

from indexter.walker import IgnorePatternMatcher, Walker


class TestIgnorePatternMatcher:
    """Tests for IgnorePatternMatcher class."""

    def test_init_with_no_patterns(self):
        """Test initialization without any patterns."""
        matcher = IgnorePatternMatcher()
        assert matcher._patterns == []
        assert not matcher.should_ignore("any_file.py")

    def test_init_with_patterns(self):
        """Test initialization with patterns."""
        patterns = ["*.pyc", "__pycache__/", "*.log"]
        matcher = IgnorePatternMatcher(patterns)
        assert matcher._patterns == patterns
        assert matcher.should_ignore("test.pyc")
        assert matcher.should_ignore("__pycache__/file.py")
        assert matcher.should_ignore("debug.log")

    def test_add_patterns_from_file_success(self, tmp_path):
        """Test adding patterns from a valid gitignore file."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n*.log\n")

        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(gitignore)

        assert "*.pyc" in matcher._patterns
        assert "__pycache__/" in matcher._patterns
        assert "*.log" in matcher._patterns
        assert matcher.should_ignore("test.pyc")

    def test_add_patterns_from_file_nonexistent(self, caplog):
        """Test adding patterns from nonexistent file."""
        matcher = IgnorePatternMatcher()
        nonexistent = Path("/nonexistent/.gitignore")

        with caplog.at_level(logging.DEBUG):
            matcher.add_patterns_from_file(nonexistent)

        assert matcher._patterns == []

    def test_add_patterns_from_file_read_error(self, tmp_path, caplog):
        """Test handling of file read errors."""
        bad_file = tmp_path / "bad_file"
        bad_file.touch()
        bad_file.chmod(0o000)  # Remove all permissions

        matcher = IgnorePatternMatcher()

        with caplog.at_level(logging.WARNING):
            matcher.add_patterns_from_file(bad_file)

        assert any("Failed to read ignore file" in record.message for record in caplog.records)

    def test_add_patterns(self):
        """Test adding patterns programmatically."""
        matcher = IgnorePatternMatcher(["*.pyc"])
        matcher.add_patterns(["*.log", "dist/"])

        assert "*.pyc" in matcher._patterns
        assert "*.log" in matcher._patterns
        assert "dist/" in matcher._patterns
        assert matcher.should_ignore("test.log")
        assert matcher.should_ignore("dist/file.js")

    def test_should_ignore_basic_patterns(self):
        """Test basic pattern matching."""
        patterns = ["*.pyc", "*.log", "node_modules/", ".git/"]
        matcher = IgnorePatternMatcher(patterns)

        assert matcher.should_ignore("test.pyc")
        assert matcher.should_ignore("debug.log")
        assert matcher.should_ignore("node_modules/package.json")
        assert matcher.should_ignore(".git/config")
        assert not matcher.should_ignore("test.py")
        assert not matcher.should_ignore("README.md")

    def test_should_ignore_directory_patterns(self):
        """Test directory-specific patterns."""
        patterns = ["__pycache__/", "dist/", "build/"]
        matcher = IgnorePatternMatcher(patterns)

        assert matcher.should_ignore("__pycache__/test.pyc")
        assert matcher.should_ignore("dist/bundle.js")
        assert matcher.should_ignore("build/output.o")
        assert not matcher.should_ignore("src/main.py")

    def test_should_ignore_glob_patterns(self):
        """Test wildcard glob patterns."""
        patterns = ["test_*.py", "*.tmp", "temp*"]
        matcher = IgnorePatternMatcher(patterns)

        assert matcher.should_ignore("test_utils.py")
        assert matcher.should_ignore("data.tmp")
        assert matcher.should_ignore("tempfile")
        assert not matcher.should_ignore("utils.py")


class TestWalker:
    """Tests for Walker class."""

    def test_init(self, mock_repo):
        """Test Walker initialization."""
        walker = Walker(mock_repo)

        assert walker.repo == mock_repo
        assert walker.repo_path == mock_repo.path
        assert walker.repo_settings == mock_repo.settings
        assert walker._matcher is not None

    def test_binary_extensions(self):
        """Test BINARY_EXTENSIONS class attribute."""
        assert ".png" in Walker.BINARY_EXTENSIONS
        assert ".jpg" in Walker.BINARY_EXTENSIONS
        assert ".pdf" in Walker.BINARY_EXTENSIONS
        assert ".zip" in Walker.BINARY_EXTENSIONS
        assert ".min.js" in Walker.BINARY_EXTENSIONS
        assert ".py" not in Walker.BINARY_EXTENSIONS

    def test_is_binary_file(self, mock_repo):
        """Test binary file detection."""
        walker = Walker(mock_repo)

        assert walker._is_binary_file(Path("image.png"))
        assert walker._is_binary_file(Path("doc.pdf"))
        assert walker._is_binary_file(Path("archive.zip"))
        # Note: script.min.js has suffix .js which is NOT in BINARY_EXTENSIONS
        # Minified files are handled separately by _is_minified()
        assert not walker._is_binary_file(Path("script.min.js"))
        assert not walker._is_binary_file(Path("script.py"))
        assert not walker._is_binary_file(Path("README.md"))
        assert not walker._is_binary_file(Path("app.js"))

    def test_is_minified(self, mock_repo):
        """Test minified file detection."""
        walker = Walker(mock_repo)

        assert walker._is_minified(Path("bundle.min.js"))
        assert walker._is_minified(Path("styles.min.css"))
        assert walker._is_minified(Path("app.MIN.js"))  # case insensitive
        assert walker._is_minified(Path("vendor.min"))
        assert not walker._is_minified(Path("bundle.js"))
        assert not walker._is_minified(Path("styles.css"))

    @pytest.mark.anyio
    async def test_read_content_utf8(self, tmp_path):
        """Test reading UTF-8 encoded files."""
        test_file = tmp_path / "test.txt"
        content = "Hello, World! 你好世界"
        test_file.write_text(content, encoding="utf-8")

        result = await Walker._read_content(anyio.Path(test_file))
        assert result == content

    @pytest.mark.anyio
    async def test_read_content_latin1_fallback(self, tmp_path):
        """Test fallback to latin-1 encoding."""
        test_file = tmp_path / "test.txt"
        content = "Café"
        test_file.write_bytes(content.encode("latin-1"))

        result = await Walker._read_content(anyio.Path(test_file))
        assert result is not None

    @pytest.mark.anyio
    async def test_read_content_unreadable(self, tmp_path):
        """Test handling of unreadable files."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        # Mock to force all encodings to fail
        with patch.object(anyio.Path, "read_text", side_effect=Exception("Read error")):
            result = await Walker._read_content(anyio.Path(test_file))
            assert result is None

    def test_build_matcher(self, mock_repo, tmp_path):
        """Test matcher building with gitignore and repo settings."""
        # Create a mock gitignore
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\ntemp/\n")
        mock_repo.path = str(tmp_path)
        mock_repo.settings.path = tmp_path
        mock_repo.settings.ignore_patterns = ["custom/", "*.custom"]

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = ["*.pyc", "__pycache__/"]
            walker = Walker(mock_repo)

        # Check that patterns from all sources are included
        assert "*.pyc" in walker._matcher._patterns
        assert "*.log" in walker._matcher._patterns
        assert "custom/" in walker._matcher._patterns

    def test_build_matcher_no_gitignore(self, mock_repo, tmp_path):
        """Test matcher building when .gitignore doesn't exist."""
        mock_repo.path = str(tmp_path)
        mock_repo.settings.path = tmp_path
        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = ["*.pyc"]
            walker = Walker(mock_repo)

        assert walker._matcher is not None
        assert "*.pyc" in walker._matcher._patterns

    @pytest.mark.anyio
    async def test_walk_recursive_permission_error(self, mock_repo, tmp_path, caplog):
        """Test handling of permission errors during directory walk."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            walker = Walker(mock_repo)

        # Mock iterdir to raise PermissionError
        async def mock_iterdir_error():
            raise PermissionError("Access denied")
            yield  # Make it an async generator

        with patch.object(anyio.Path, "iterdir", side_effect=lambda: mock_iterdir_error()):
            with caplog.at_level(logging.WARNING):
                [item async for item in walker._walk_recursive(anyio.Path(repo_path))]

        assert any("Permission denied" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_recursive_os_error(self, mock_repo, tmp_path, caplog):
        """Test handling of OS errors during directory walk."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            walker = Walker(mock_repo)

        # Mock iterdir to raise OSError
        async def mock_iterdir_error():
            raise OSError("Disk error")
            yield  # Make it an async generator

        with patch.object(anyio.Path, "iterdir", side_effect=lambda: mock_iterdir_error()):
            with caplog.at_level(logging.WARNING):
                [item async for item in walker._walk_recursive(anyio.Path(repo_path))]

        assert any("Error reading directory" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_recursive_ignores_directories(self, mock_repo, tmp_path):
        """Test that ignored directories are pruned from traversal."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path

        # Create directory structure
        (repo_path / "src").mkdir()
        (repo_path / "node_modules").mkdir()
        (repo_path / "src" / "main.py").touch()
        (repo_path / "node_modules" / "package.json").touch()

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = ["node_modules/"]
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        files = [f async for f in walker._walk_recursive(anyio.Path(repo_path))]
        file_names = [f.name for f in files]

        assert "main.py" in file_names
        assert "package.json" not in file_names

    @pytest.mark.anyio
    async def test_walk_recursive_yields_files(self, mock_repo, tmp_path):
        """Test that walk_recursive yields all non-ignored files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path

        # Create test files
        (repo_path / "file1.py").touch()
        (repo_path / "file2.txt").touch()
        sub_dir = repo_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file3.py").touch()

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        files = [f async for f in walker._walk_recursive(anyio.Path(repo_path))]
        file_names = [f.name for f in files]

        assert len(files) == 3
        assert "file1.py" in file_names
        assert "file2.txt" in file_names
        assert "file3.py" in file_names

    @pytest.mark.anyio
    async def test_walk_ignores_binary_files(self, mock_repo, tmp_path, caplog):
        """Test that binary files are skipped."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024 * 1024

        # Create files
        (repo_path / "script.py").write_text("print('hello')")
        (repo_path / "image.png").write_bytes(b"fake image data")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        with caplog.at_level(logging.DEBUG):
            results = [r async for r in walker.walk()]

        assert len(results) == 1
        assert results[0]["path"] == "script.py"
        assert any("Ignoring (binary)" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_ignores_minified_files(self, mock_repo, tmp_path, caplog):
        """Test that minified files are skipped."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024 * 1024

        # Create files
        (repo_path / "app.js").write_text("console.log('hello');")
        (repo_path / "app.min.js").write_text("console.log('hello');")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        with caplog.at_level(logging.DEBUG):
            results = [r async for r in walker.walk()]

        assert len(results) == 1
        assert results[0]["path"] == "app.js"
        assert any("Ignoring (minified)" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_ignores_pattern_matched_files(self, mock_repo, tmp_path, caplog):
        """Test that pattern-matched files are skipped."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024 * 1024

        # Create files
        (repo_path / "main.py").write_text("print('main')")
        (repo_path / "test.pyc").write_bytes(b"compiled")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = ["*.pyc"]
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        with caplog.at_level(logging.DEBUG):
            results = [r async for r in walker.walk()]

        assert len(results) == 1
        assert results[0]["path"] == "main.py"
        assert any("Ignoring (pattern match)" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_ignores_large_files(self, mock_repo, tmp_path, caplog):
        """Test that files exceeding max_file_size are skipped."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 100  # 100 bytes

        # Create files
        (repo_path / "small.py").write_text("# small")
        (repo_path / "large.py").write_text("x" * 200)

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        with caplog.at_level(logging.DEBUG):
            results = [r async for r in walker.walk()]

        assert len(results) == 1
        assert results[0]["path"] == "small.py"
        assert any("Ignoring (too large)" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_ignores_empty_files(self, mock_repo, tmp_path, caplog):
        """Test that empty files are skipped."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024

        # Create files
        (repo_path / "empty.py").touch()
        (repo_path / "nonempty.py").write_text("print('hello')")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        with caplog.at_level(logging.DEBUG):
            results = [r async for r in walker.walk()]

        assert len(results) == 1
        assert results[0]["path"] == "nonempty.py"
        assert any("Ignoring (empty)" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_handles_stat_error(self, mock_repo, tmp_path, caplog):
        """Test handling of stat errors during walk."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024

        (repo_path / "test.py").write_text("print('hello')")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = [".git/"]
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        # Mock stat to raise an error in the walk method
        original_stat = anyio.Path.stat

        async def mock_stat(self, **kwargs):
            # Fail on test.py during the walk phase (when getting file size)
            if "test.py" in str(self):
                raise OSError("Stat error")
            return await original_stat(self, **kwargs)

        with patch.object(anyio.Path, "stat", mock_stat):
            with caplog.at_level(logging.WARNING):
                results = [r async for r in walker.walk()]

        assert len(results) == 0
        assert any("Cannot stat" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_handles_unreadable_files(self, mock_repo, tmp_path, caplog):
        """Test handling of files that cannot be read."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024

        (repo_path / "test.py").write_text("print('hello')")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        # Mock _read_content to return None
        async def mock_read(*args, **kwargs):
            return None

        with patch.object(Walker, "_read_content", side_effect=mock_read):
            with caplog.at_level(logging.DEBUG):
                results = [r async for r in walker.walk()]

        assert len(results) == 0
        assert any("Ignoring (cannot read)" in record.message for record in caplog.records)

    @pytest.mark.anyio
    async def test_walk_returns_complete_file_info(self, mock_repo, tmp_path):
        """Test that walk returns complete file information."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024 * 1024

        test_file = repo_path / "test.py"
        content = "print('hello world')"
        test_file.write_text(content)

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        results = [r async for r in walker.walk()]

        assert len(results) == 1
        result = results[0]

        assert result["path"] == "test.py"
        assert result["content"] == content
        assert result["size_bytes"] == len(content)
        assert "mtime" in result
        assert "hash" in result
        assert isinstance(result["hash"], str)
        assert len(result["hash"]) == 64  # SHA256 hex digest

    @pytest.mark.anyio
    async def test_walk_hash_includes_path_and_content(self, mock_repo, tmp_path):
        """Test that hash is computed from path:content."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024 * 1024

        test_file = repo_path / "test.py"
        content = "print('hello')"
        test_file.write_text(content)

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        with patch("indexter.walker.compute_hash") as mock_hash:
            mock_hash.return_value = "fake_hash"
            results = [r async for r in walker.walk()]

        # Verify compute_hash was called with path:content
        mock_hash.assert_called_once_with("test.py:" + content)
        assert results[0]["hash"] == "fake_hash"

    @pytest.mark.anyio
    async def test_walk_complex_directory_structure(self, mock_repo, tmp_path):
        """Test walking a complex directory structure."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024 * 1024

        # Create complex structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("main")
        (repo_path / "src" / "utils").mkdir()
        (repo_path / "src" / "utils" / "helper.py").write_text("helper")
        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "test_main.py").write_text("test")
        (repo_path / "README.md").write_text("readme")
        (repo_path / ".git").mkdir()
        (repo_path / ".git" / "config").write_text("config")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = [".git/"]
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        results = [r async for r in walker.walk()]
        paths = [r["path"] for r in results]

        assert "src/main.py" in paths
        assert "src/utils/helper.py" in paths
        assert "tests/test_main.py" in paths
        assert "README.md" in paths
        assert ".git/config" not in paths  # Should be ignored

    @pytest.mark.anyio
    async def test_walk_handles_symlink_errors(self, mock_repo, tmp_path, caplog):
        """Test handling of errors when accessing symlinks."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        mock_repo.path = str(repo_path)
        mock_repo.settings.path = repo_path
        mock_repo.settings.max_file_size = 1024

        (repo_path / "regular.py").write_text("print('hello')")

        with patch("indexter.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = []
            mock_repo.settings.ignore_patterns = []
            walker = Walker(mock_repo)

        # Mock to simulate broken symlink or access error
        async def mock_walk_with_error(directory):
            async for entry in directory.iterdir():
                if "regular.py" in str(entry):
                    raise OSError("Broken symlink")
                yield entry

        with patch.object(walker, "_walk_recursive", side_effect=mock_walk_with_error):
            with caplog.at_level(logging.WARNING):
                try:
                    [r async for r in walker.walk()]
                except OSError:
                    pass  # Expected
