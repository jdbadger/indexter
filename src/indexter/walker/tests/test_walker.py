from pathlib import Path
from unittest.mock import MagicMock, patch

import anyio
import pytest

from indexter.walker.models import DocumentMetadata
from indexter.walker.walker import IgnorePatternMatcher, Walker, compute_hash


class TestComputeHash:
    """Tests for compute_hash function."""

    def test_should_compute_sha256_hash(self):
        """Test compute_hash returns a valid SHA256 hash."""
        content = "test content"
        result = compute_hash(content)

        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64 hex characters

    def test_should_produce_consistent_hash_for_same_content(self):
        """Test compute_hash produces the same hash for identical content."""
        content = "consistent content"
        hash1 = compute_hash(content)
        hash2 = compute_hash(content)

        assert hash1 == hash2

    def test_should_produce_different_hash_for_different_content(self):
        """Test compute_hash produces different hashes for different content."""
        hash1 = compute_hash("content1")
        hash2 = compute_hash("content2")

        assert hash1 != hash2

    @pytest.mark.parametrize(
        "content",
        ["", "a", "Hello, World!", "multi\nline\ncontent", "unicode: café 日本語 🚀"],
    )
    def test_should_handle_various_content_types(self, content):
        """Test compute_hash handles various content types."""
        result = compute_hash(content)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_should_handle_large_content(self):
        """Test compute_hash handles large content."""
        content = "x" * 1000000  # 1 MB of data
        result = compute_hash(content)

        assert isinstance(result, str)
        assert len(result) == 64


class TestIgnorePatternMatcher:
    """Tests for IgnorePatternMatcher class."""

    def test_should_initialize_with_no_patterns(self):
        """Test IgnorePatternMatcher initializes with empty patterns."""
        matcher = IgnorePatternMatcher()

        assert not matcher.should_ignore("test.py")

    def test_should_initialize_with_patterns(self):
        """Test IgnorePatternMatcher initializes with provided patterns."""
        patterns = ["*.pyc", "__pycache__/"]
        matcher = IgnorePatternMatcher(patterns)

        assert matcher.should_ignore("test.pyc")
        assert matcher.should_ignore("__pycache__/")

    @pytest.mark.parametrize(
        "pattern,path,should_match",
        [
            ("*.pyc", "test.pyc", True),
            ("*.pyc", "test.py", False),
            ("__pycache__/", "__pycache__/", True),
            ("*.log", "debug.log", True),
            ("*.log", "debug.txt", False),
            ("build/", "build/", True),
            ("*.min.js", "app.min.js", True),
            ("*.min.js", "app.js", False),
        ],
    )
    def test_should_match_patterns_correctly(self, pattern, path, should_match):
        """Test IgnorePatternMatcher matches patterns correctly."""
        matcher = IgnorePatternMatcher([pattern])

        assert matcher.should_ignore(path) == should_match

    def test_should_add_patterns(self):
        """Test add_patterns adds new patterns."""
        matcher = IgnorePatternMatcher(["*.pyc"])

        matcher.add_patterns(["*.log", "temp/"])

        assert matcher.should_ignore("test.pyc")
        assert matcher.should_ignore("debug.log")
        assert matcher.should_ignore("temp/")

    def test_should_load_patterns_from_file(self, tmp_path):
        """Test add_patterns_from_file loads patterns from a file."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n*.log\n")

        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(gitignore)

        assert matcher.should_ignore("test.pyc")
        assert matcher.should_ignore("__pycache__/")
        assert matcher.should_ignore("debug.log")

    def test_should_handle_missing_ignore_file(self, tmp_path):
        """Test add_patterns_from_file handles missing file gracefully."""
        missing_file = tmp_path / "nonexistent.gitignore"

        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(missing_file)  # Should not raise

        assert not matcher.should_ignore("test.py")

    def test_should_handle_empty_ignore_file(self, tmp_path):
        """Test add_patterns_from_file handles empty file."""
        empty_file = tmp_path / ".gitignore"
        empty_file.write_text("")

        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(empty_file)

        assert not matcher.should_ignore("test.py")

    def test_should_handle_comments_in_ignore_file(self, tmp_path):
        """Test add_patterns_from_file handles comments."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("# This is a comment\n*.pyc\n# Another comment\n*.log\n")

        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(gitignore)

        assert matcher.should_ignore("test.pyc")
        assert matcher.should_ignore("debug.log")

    def test_should_handle_blank_lines_in_ignore_file(self, tmp_path):
        """Test add_patterns_from_file handles blank lines."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n\n\n*.log\n")

        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(gitignore)

        assert matcher.should_ignore("test.pyc")
        assert matcher.should_ignore("debug.log")

    def test_should_handle_unreadable_file(self, tmp_path):
        """Test add_patterns_from_file handles unreadable file gracefully."""
        with patch("pathlib.Path.read_text", side_effect=PermissionError("Access denied")):
            gitignore = tmp_path / ".gitignore"
            gitignore.write_text("*.pyc\n")

            matcher = IgnorePatternMatcher()
            matcher.add_patterns_from_file(gitignore)  # Should not raise

    @pytest.mark.parametrize(
        "patterns,path,should_match",
        [
            (["*.py", "!test_*.py"], "main.py", True),
            (["*.py", "!test_*.py"], "test_main.py", False),
            (["build/", "dist/"], "build/", True),
            (["build/", "dist/"], "dist/", True),
            (["build/", "dist/"], "src/", False),
        ],
    )
    def test_should_handle_multiple_patterns(self, patterns, path, should_match):
        """Test matcher handles multiple patterns correctly."""
        matcher = IgnorePatternMatcher(patterns)

        assert matcher.should_ignore(path) == should_match


class TestWalker:
    """Tests for Walker class."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock Repo object for testing."""
        mock = MagicMock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        mock.name = "test_repo"
        mock.path = str(repo_path)
        mock.settings = MagicMock()
        mock.settings.max_file_size = 1024 * 1024  # 1 MB
        mock.settings.ignore_patterns = []
        return mock

    def test_should_initialize_walker(self, mock_repo):
        """Test Walker initializes correctly."""
        walker = Walker(mock_repo)

        assert walker.repo == mock_repo
        assert walker.repo_path == mock_repo.path
        assert walker.repo_settings == mock_repo.settings

    def test_should_build_matcher_with_default_patterns(self, mock_repo):
        """Test _build_matcher includes default patterns."""
        with patch("indexter.walker.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = ["*.pyc"]
            walker = Walker(mock_repo)

            assert walker._matcher.should_ignore("test.pyc")

    def test_should_build_matcher_with_gitignore(self, mock_repo):
        """Test _build_matcher loads .gitignore file."""
        gitignore = Path(mock_repo.path) / ".gitignore"
        gitignore.write_text("*.log\n")

        walker = Walker(mock_repo)

        assert walker._matcher.should_ignore("debug.log")

    def test_should_build_matcher_with_repo_patterns(self, mock_repo):
        """Test _build_matcher includes repo-specific patterns."""
        mock_repo.settings.ignore_patterns = ["*.tmp"]

        walker = Walker(mock_repo)

        assert walker._matcher.should_ignore("temp.tmp")

    @pytest.mark.parametrize(
        "filename,is_binary",
        [
            ("image.png", True),
            ("video.mp4", True),
            ("document.pdf", True),
            ("archive.zip", True),
            ("executable.exe", True),
            ("font.woff", True),
            ("database.sqlite", True),
            ("script.py", False),
            ("code.js", False),
            ("style.css", False),
            ("data.json", False),
            ("README.md", False),
        ],
    )
    def test_should_detect_binary_files(self, mock_repo, filename, is_binary):
        """Test _is_binary_file detects binary files correctly."""
        walker = Walker(mock_repo)

        assert walker._is_binary_file(Path(filename)) == is_binary

    @pytest.mark.parametrize(
        "filename,is_minified",
        [
            ("app.min.js", True),
            ("style.min.css", True),
            ("script.js", False),
            ("style.css", False),
            ("app.min.js.map", True),  # Contains .min. in the name
            ("app.js.map", False),
        ],
    )
    def test_should_detect_minified_files(self, mock_repo, filename, is_minified):
        """Test _is_minified detects minified files correctly."""
        walker = Walker(mock_repo)

        assert walker._is_minified(Path(filename)) == is_minified

    @pytest.mark.asyncio
    async def test_should_read_utf8_content(self, tmp_path):
        """Test _read_content reads UTF-8 encoded files."""
        test_file = anyio.Path(tmp_path / "test.txt")
        await test_file.write_text("Hello, World!", encoding="utf-8")

        content = await Walker._read_content(test_file)

        assert content == "Hello, World!"

    @pytest.mark.asyncio
    async def test_should_fallback_to_latin1(self, tmp_path):
        """Test _read_content falls back to Latin-1 encoding."""
        test_file = anyio.Path(tmp_path / "test.txt")
        # Write content that's valid Latin-1 but not UTF-8
        await test_file.write_bytes(b"\xe9\xe8\xe0")  # é è à in Latin-1

        content = await Walker._read_content(test_file)

        assert content is not None
        assert "é" in content or len(content) > 0

    @pytest.mark.asyncio
    async def test_should_handle_latin1_decode_error(self, tmp_path):
        """Test _read_content returns None when Latin-1 also fails."""
        test_file = anyio.Path(tmp_path / "test.txt")
        # Create a file and mock read_text to raise exceptions
        await test_file.write_bytes(b"\x00\x01\x02")

        # Mock to raise UnicodeDecodeError for UTF-8 and Exception for Latin-1
        call_count = [0]

        async def mock_read_text(encoding=None):
            call_count[0] += 1
            if call_count[0] == 1:  # First call (UTF-8)
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")
            else:  # Second call (Latin-1)
                raise OSError("Simulated error")

        with patch.object(anyio.Path, "read_text", mock_read_text):
            content = await Walker._read_content(test_file)

        assert content is None

    @pytest.mark.asyncio
    async def test_should_return_none_for_unreadable_file(self, tmp_path):
        """Test _read_content returns None for unreadable files."""
        test_file = anyio.Path(tmp_path / "nonexistent.txt")

        content = await Walker._read_content(test_file)

        assert content is None

    @pytest.mark.asyncio
    async def test_should_walk_empty_directory(self, mock_repo):
        """Test walk handles empty directory."""
        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_should_walk_single_file(self, mock_repo):
        """Test walk yields single file."""
        test_file = Path(mock_repo.path) / "test.py"
        test_file.write_text("print('hello')")

        walker = Walker(mock_repo)

        files = []
        async for relpath, content, metadata in walker.walk():
            files.append((relpath, content, metadata))

        assert len(files) == 1
        assert files[0][0] == "test.py"
        assert files[0][1] == "print('hello')"
        assert isinstance(files[0][2], DocumentMetadata)

    @pytest.mark.asyncio
    async def test_should_walk_multiple_files(self, mock_repo):
        """Test walk yields multiple files."""
        (Path(mock_repo.path) / "file1.py").write_text("# File 1")
        (Path(mock_repo.path) / "file2.js").write_text("// File 2")
        (Path(mock_repo.path) / "file3.md").write_text("# File 3")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert len(files) == 3
        assert set(files) == {"file1.py", "file2.js", "file3.md"}

    @pytest.mark.asyncio
    async def test_should_walk_nested_directories(self, mock_repo):
        """Test walk recurses into subdirectories."""
        src_dir = Path(mock_repo.path) / "src"
        src_dir.mkdir()
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        (Path(mock_repo.path) / "main.py").write_text("# Main")
        (src_dir / "app.py").write_text("# App")
        (utils_dir / "helper.py").write_text("# Helper")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert len(files) == 3
        assert "main.py" in files
        assert "src/app.py" in files or "src\\app.py" in files
        assert "src/utils/helper.py" in files or "src\\utils\\helper.py" in files

    @pytest.mark.asyncio
    async def test_should_skip_ignored_patterns(self, mock_repo):
        """Test walk skips files matching ignore patterns."""
        gitignore = Path(mock_repo.path) / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n")

        (Path(mock_repo.path) / "main.py").write_text("# Main")
        (Path(mock_repo.path) / "cache.pyc").write_text("compiled")

        pycache = Path(mock_repo.path) / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_text("cached")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert "main.py" in files
        assert "cache.pyc" not in files
        assert not any("__pycache__" in f for f in files)

    @pytest.mark.asyncio
    async def test_should_skip_binary_files(self, mock_repo):
        """Test walk skips binary files."""
        (Path(mock_repo.path) / "script.py").write_text("# Script")
        (Path(mock_repo.path) / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert "script.py" in files
        assert "image.png" not in files

    @pytest.mark.asyncio
    async def test_should_skip_minified_files(self, mock_repo):
        """Test walk skips minified files."""
        (Path(mock_repo.path) / "app.js").write_text("console.log('app');")
        (Path(mock_repo.path) / "app.min.js").write_text("console.log('minified');")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert "app.js" in files
        assert "app.min.js" not in files

    @pytest.mark.asyncio
    async def test_should_skip_large_files(self, mock_repo):
        """Test walk skips files larger than max_file_size."""
        mock_repo.settings.max_file_size = 100  # 100 bytes

        (Path(mock_repo.path) / "small.py").write_text("# Small")
        (Path(mock_repo.path) / "large.py").write_text("x" * 200)

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert "small.py" in files
        assert "large.py" not in files

    @pytest.mark.asyncio
    async def test_should_skip_empty_files(self, mock_repo):
        """Test walk skips empty files."""
        (Path(mock_repo.path) / "nonempty.py").write_text("# Not empty")
        (Path(mock_repo.path) / "empty.py").write_text("")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert "nonempty.py" in files
        assert "empty.py" not in files

    @pytest.mark.asyncio
    async def test_should_yield_correct_metadata(self, mock_repo):
        """Test walk yields correct DocumentMetadata."""
        test_file = Path(mock_repo.path) / "test.py"
        test_content = "print('test')"
        test_file.write_text(test_content)

        walker = Walker(mock_repo)

        async for relpath, content, metadata in walker.walk():
            assert relpath == "test.py"
            assert content == test_content
            assert metadata.repo == "test_repo"
            assert metadata.repo_path == mock_repo.path
            assert metadata.ext == ".py"
            assert metadata.size_bytes == len(test_content.encode())
            assert metadata.mtime > 0
            assert len(metadata.hash) == 64

    @pytest.mark.asyncio
    async def test_should_compute_hash_with_path_and_content(self, mock_repo):
        """Test walk computes hash from path and content."""
        test_file = Path(mock_repo.path) / "test.py"
        test_content = "print('test')"
        test_file.write_text(test_content)

        walker = Walker(mock_repo)

        async for relpath, content, metadata in walker.walk():
            expected_hash = compute_hash(f"{relpath}:{content}")
            assert metadata.hash == expected_hash

    @pytest.mark.asyncio
    async def test_should_handle_permission_errors(self, mock_repo):
        """Test walk handles permission errors gracefully."""
        test_file = Path(mock_repo.path) / "test.py"
        test_file.write_text("content")

        walker = Walker(mock_repo)

        # Mock stat to raise PermissionError
        with patch.object(anyio.Path, "stat", side_effect=PermissionError("Access denied")):
            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            # Should not raise, just skip the file
            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_should_handle_os_errors_during_stat(self, mock_repo):
        """Test walk handles OSError during stat."""
        test_file = Path(mock_repo.path) / "test.py"
        test_file.write_text("content")

        walker = Walker(mock_repo)

        # Mock stat to raise OSError
        with patch.object(anyio.Path, "stat", side_effect=OSError("I/O error")):
            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            # Should not raise, just skip the file
            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_should_handle_unreadable_files(self, mock_repo):
        """Test walk skips files that cannot be read."""
        test_file = Path(mock_repo.path) / "test.py"
        test_file.write_text("content")

        walker = Walker(mock_repo)

        # Mock _read_content to return None
        with patch.object(Walker, "_read_content", return_value=None):
            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_should_prune_ignored_directories(self, mock_repo):
        """Test walk prunes entire ignored directories."""
        gitignore = Path(mock_repo.path) / ".gitignore"
        gitignore.write_text("node_modules/\n")

        node_modules = Path(mock_repo.path) / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.json").write_text("{}")

        (Path(mock_repo.path) / "index.js").write_text("// Main")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert "index.js" in files
        assert not any("node_modules" in f for f in files)

    @pytest.mark.asyncio
    async def test_should_handle_deep_directory_nesting(self, mock_repo):
        """Test walk handles deeply nested directories."""
        # Create deeply nested directory structure
        current = Path(mock_repo.path)
        for i in range(10):
            current = current / f"level{i}"
            current.mkdir()

        deep_file = current / "deep.py"
        deep_file.write_text("# Deep file")

        walker = Walker(mock_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        assert len(files) == 1
        assert "deep.py" in files[0]

    @pytest.mark.asyncio
    async def test_should_handle_permission_error_on_directory(self, mock_repo):
        """Test walk handles PermissionError when listing directory."""
        test_dir = Path(mock_repo.path) / "restricted"
        test_dir.mkdir()
        (test_dir / "file.py").write_text("# File")

        walker = Walker(mock_repo)

        # Mock iterdir to raise PermissionError
        with patch.object(anyio.Path, "iterdir", side_effect=PermissionError("Access denied")):
            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            # Should handle the error and return empty
            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_should_handle_os_error_on_directory(self, mock_repo):
        """Test walk handles OSError when listing directory."""
        test_dir = Path(mock_repo.path) / "problematic"
        test_dir.mkdir()
        (test_dir / "file.py").write_text("# File")

        walker = Walker(mock_repo)

        # Mock iterdir to raise OSError
        with patch.object(anyio.Path, "iterdir", side_effect=OSError("I/O error")):
            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            # Should handle the error and return empty
            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_should_handle_symlink_outside_repo(self, tmp_path):
        """Test walk handles symlinks pointing outside the repository."""
        mock = MagicMock()
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock.name = "test_repo"
        mock.path = str(repo_path)
        mock.settings = MagicMock()
        mock.settings.max_file_size = 1024 * 1024
        mock.settings.ignore_patterns = []

        # Create target outside repo
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "file.py").write_text("# Outside")

        # Create symlink inside repo pointing outside
        try:
            link_dir = repo_path / "link"
            link_dir.symlink_to(outside_dir)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

        walker = Walker(mock)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        # Should not follow symlink outside repo
        assert not any("outside" in f or "link" in f for f in files)

    @pytest.mark.asyncio
    async def test_should_handle_broken_symlink(self, tmp_path):
        """Test walk handles broken symlinks gracefully."""
        mock = MagicMock()
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock.name = "test_repo"
        mock.path = str(repo_path)
        mock.settings = MagicMock()
        mock.settings.max_file_size = 1024 * 1024
        mock.settings.ignore_patterns = []

        # Create a broken symlink
        try:
            link_dir = repo_path / "broken_link"
            link_dir.symlink_to(tmp_path / "nonexistent")
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

        (repo_path / "good_file.py").write_text("# Good file")

        walker = Walker(mock)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        # Should skip broken symlink but find good file
        assert "good_file.py" in files
        assert not any("broken_link" in f for f in files)

    @pytest.mark.asyncio
    async def test_should_handle_value_error_on_entry(self, mock_repo):
        """Test walk handles ValueError when computing relative path."""
        (Path(mock_repo.path) / "file.py").write_text("# File")

        walker = Walker(mock_repo)

        # Create an async mock that raises ValueError
        async def mock_iterdir(self):
            # Create a mock entry that will cause ValueError
            mock_entry = MagicMock(spec=anyio.Path)
            mock_entry.relative_to = MagicMock(side_effect=ValueError("Not a relative path"))
            yield mock_entry

        with patch.object(anyio.Path, "iterdir", mock_iterdir):
            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            # Should handle the error and skip the file
            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_should_handle_os_error_on_entry(self, mock_repo):
        """Test walk handles OSError when accessing entry."""
        test_dir = Path(mock_repo.path) / "subdir"
        test_dir.mkdir()
        (test_dir / "file.py").write_text("# File")

        walker = Walker(mock_repo)

        # Mock is_dir to raise OSError
        with patch.object(anyio.Path, "is_dir", side_effect=OSError("Access error")):
            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            # Should handle the error
            assert len(files) == 0


class TestWalkerIntegration:
    """Integration tests for Walker."""

    @pytest.fixture
    def integration_repo(self, tmp_path):
        """Create a realistic repository structure for integration tests."""
        mock = MagicMock()
        repo_path = tmp_path / "integration_repo"
        repo_path.mkdir()

        mock.name = "integration_repo"
        mock.path = str(repo_path)
        mock.settings = MagicMock()
        mock.settings.max_file_size = 1024 * 1024
        mock.settings.ignore_patterns = []

        # Create directory structure
        (repo_path / "src").mkdir()
        (repo_path / "tests").mkdir()
        (repo_path / "docs").mkdir()
        (repo_path / "build").mkdir()

        # Create files
        (repo_path / "README.md").write_text("# Integration Repo")
        (repo_path / "src" / "main.py").write_text("def main(): pass")
        (repo_path / "src" / "utils.py").write_text("def helper(): pass")
        (repo_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        (repo_path / "docs" / "guide.md").write_text("# Guide")
        (repo_path / "build" / "output.log").write_text("Build output")

        # Create .gitignore
        (repo_path / ".gitignore").write_text("build/\n*.pyc\n__pycache__/\n")

        return mock

    @pytest.mark.asyncio
    async def test_should_walk_realistic_repository(self, integration_repo):
        """Test walk handles realistic repository structure."""
        walker = Walker(integration_repo)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        # Should include source files and docs, but not build directory
        assert "README.md" in files
        assert any("main.py" in f for f in files)
        assert any("utils.py" in f for f in files)
        assert any("test_main.py" in f for f in files)
        assert any("guide.md" in f for f in files)
        assert not any("build" in f for f in files)

    @pytest.mark.asyncio
    async def test_should_produce_consistent_results(self, integration_repo):
        """Test walk produces consistent results across multiple runs."""
        walker = Walker(integration_repo)

        # First walk
        files1 = []
        async for relpath, _content, metadata in walker.walk():
            files1.append((relpath, metadata.hash))

        # Second walk
        files2 = []
        async for relpath, _content, metadata in walker.walk():
            files2.append((relpath, metadata.hash))

        assert files1 == files2

    @pytest.mark.asyncio
    async def test_should_detect_file_changes(self, integration_repo):
        """Test walk detects file content changes through hash."""
        walker = Walker(integration_repo)

        # First walk
        hashes1 = {}
        async for relpath, _content, metadata in walker.walk():
            if relpath == "README.md":
                hashes1[relpath] = metadata.hash

        # Modify file
        readme = Path(integration_repo.path) / "README.md"
        readme.write_text("# Modified Integration Repo")

        # Second walk
        hashes2 = {}
        async for relpath, _content, metadata in walker.walk():
            if relpath == "README.md":
                hashes2[relpath] = metadata.hash

        assert hashes1["README.md"] != hashes2["README.md"]

    @pytest.mark.asyncio
    async def test_should_handle_mixed_file_types(self, tmp_path):
        """Test walk handles repository with mixed file types."""
        mock = MagicMock()
        repo_path = tmp_path / "mixed_repo"
        repo_path.mkdir()

        mock.name = "mixed_repo"
        mock.path = str(repo_path)
        mock.settings = MagicMock()
        mock.settings.max_file_size = 1024 * 1024
        mock.settings.ignore_patterns = []

        # Create various file types
        (repo_path / "script.py").write_text("# Python")
        (repo_path / "app.js").write_text("// JavaScript")
        (repo_path / "style.css").write_text("/* CSS */")
        (repo_path / "data.json").write_text('{"key": "value"}')
        (repo_path / "README.md").write_text("# Markdown")
        (repo_path / "config.toml").write_text("[section]")
        (repo_path / "image.png").write_bytes(b"\x89PNG")
        (repo_path / "binary.exe").write_bytes(b"MZ\x90\x00")

        walker = Walker(mock)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        # Should include text files but not binaries
        assert "script.py" in files
        assert "app.js" in files
        assert "style.css" in files
        assert "data.json" in files
        assert "README.md" in files
        assert "config.toml" in files
        assert "image.png" not in files
        assert "binary.exe" not in files

    @pytest.mark.asyncio
    async def test_should_respect_multiple_ignore_sources(self, tmp_path):
        """Test walk respects patterns from multiple sources."""
        mock = MagicMock()
        repo_path = tmp_path / "ignore_test_repo"
        repo_path.mkdir()

        mock.name = "ignore_test_repo"
        mock.path = str(repo_path)
        mock.settings = MagicMock()
        mock.settings.max_file_size = 1024 * 1024
        mock.settings.ignore_patterns = ["*.tmp"]  # Repo-specific pattern

        # Global pattern (mocked in settings)
        # .gitignore pattern
        (repo_path / ".gitignore").write_text("*.log\n")

        # Create files
        (repo_path / "main.py").write_text("# Main")
        (repo_path / "debug.log").write_text("log content")
        (repo_path / "temp.tmp").write_text("temp content")
        (repo_path / "cache.pyc").write_text("cached")

        with patch("indexter.walker.walker.settings") as mock_settings:
            mock_settings.ignore_patterns = ["*.pyc"]
            walker = Walker(mock)

            files = []
            async for relpath, _content, _metadata in walker.walk():
                files.append(relpath)

            assert "main.py" in files
            assert "debug.log" not in files  # .gitignore
            assert "temp.tmp" not in files  # repo-specific
            assert "cache.pyc" not in files  # global settings

    @pytest.mark.asyncio
    async def test_should_handle_symlinks_within_repo(self, tmp_path):
        """Test walk handles symlinks within the repository."""
        mock = MagicMock()
        repo_path = tmp_path / "symlink_repo"
        repo_path.mkdir()

        mock.name = "symlink_repo"
        mock.path = str(repo_path)
        mock.settings = MagicMock()
        mock.settings.max_file_size = 1024 * 1024
        mock.settings.ignore_patterns = []

        # Create target directory and file
        target_dir = repo_path / "target"
        target_dir.mkdir()
        (target_dir / "file.py").write_text("# Target file")

        # Create symlink within repo
        try:
            link_dir = repo_path / "link"
            link_dir.symlink_to(target_dir)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

        walker = Walker(mock)

        files = []
        async for relpath, _content, _metadata in walker.walk():
            files.append(relpath)

        # Should find the file through the symlink
        assert any("file.py" in f for f in files)

    @pytest.mark.asyncio
    async def test_should_validate_metadata_fields(self, integration_repo):
        """Test walk produces valid metadata for all files."""
        walker = Walker(integration_repo)

        async for _relpath, _content, metadata in walker.walk():
            # Validate all metadata fields
            assert isinstance(metadata.repo, str)
            assert metadata.repo == "integration_repo"
            assert isinstance(metadata.repo_path, str)
            assert metadata.repo_path == integration_repo.path
            assert isinstance(metadata.hash, str)
            assert len(metadata.hash) == 64
            assert isinstance(metadata.ext, str)
            assert isinstance(metadata.size_bytes, int)
            assert metadata.size_bytes > 0
            assert isinstance(metadata.mtime, float)
            assert metadata.mtime > 0
