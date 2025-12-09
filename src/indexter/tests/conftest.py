"""Pytest fixtures for indexter tests."""

import subprocess
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from indexter.config.repo import RepoConfig
from indexter.models import Document, IndexResult, Node, NodeMetadata


def init_git_repo(repo_path: Path) -> None:
    """Initialize a git repository using subprocess.

    This replaces gitpython's GitRepo.init() for test fixtures.
    """
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )


def git_add(repo_path: Path, files: list[str]) -> None:
    """Stage files for commit using subprocess."""
    subprocess.run(
        ["git", "add"] + files,
        cwd=repo_path,
        capture_output=True,
        check=True,
    )


def git_commit(repo_path: Path, message: str) -> None:
    """Create a commit using subprocess."""
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )


@pytest.fixture
def sample_repo_path(tmp_path: Path) -> Path:
    """Create a temporary repository path."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    return repo_path


@pytest.fixture
def sample_repo_config(sample_repo_path: Path) -> RepoConfig:
    """Create a sample RepoConfig."""
    return RepoConfig(
        path=sample_repo_path,
        name="test_repo",
        collection_name="indexter_test_repo",
    )


@pytest.fixture
def sample_document() -> Document:
    """Create a sample Document."""
    return Document(
        path="src/test.py",
        size_bytes=1024,
        mtime=1234567890.0,
        content="def hello():\n    print('Hello, world!')",
        hash="abc123def456",
    )


@pytest.fixture
def sample_node_metadata(sample_repo_path: Path) -> NodeMetadata:
    """Create sample NodeMetadata."""
    return NodeMetadata(
        hash="abc123def456",
        repo_path=str(sample_repo_path),
        document_path="src/test.py",
        language="python",
        node_type="function",
        node_name="hello",
        start_byte=0,
        end_byte=42,
        start_line=1,
        end_line=2,
        documentation="Test function",
        parent_scope=None,
        signature="def hello():",
        extra={},
    )


@pytest.fixture
def sample_node(sample_node_metadata: NodeMetadata) -> Node:
    """Create a sample Node."""
    return Node(
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        content="def hello():\n    print('Hello, world!')",
        metadata=sample_node_metadata,
    )


@pytest.fixture
def sample_index_result() -> IndexResult:
    """Create a sample IndexResult."""
    return IndexResult(
        files_synced=["file1.py", "file2.py"],
        files_deleted=["old_file.py"],
        files_checked=10,
        skipped_files=2,
        nodes_added=5,
        nodes_deleted=1,
        nodes_updated=3,
        errors=["Error parsing file3.py"],
    )


@pytest.fixture
def mock_store() -> MagicMock:
    """Create a mock store object."""
    store = MagicMock()
    store.ensure_collection = AsyncMock()
    store.delete_collection = AsyncMock()
    store.get_document_hashes = AsyncMock(return_value={})
    store.delete_by_document_paths = AsyncMock()
    store.upsert_nodes = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.count_nodes = AsyncMock(return_value=0)
    return store


@pytest.fixture
def mock_walker() -> MagicMock:
    """Create a mock Walker object."""
    walker = MagicMock()

    async def async_walk() -> AsyncIterator[dict]:
        """Mock async walk method."""
        yield {
            "path": "test.py",
            "size_bytes": 100,
            "mtime": 1234567890.0,
            "content": "print('test')",
            "hash": "test_hash",
        }

    walker.walk = async_walk
    return walker


@pytest.fixture
def mock_parser() -> MagicMock:
    """Create a mock parser object."""
    parser = MagicMock()
    parser.parse.return_value = [
        (
            "def test():\n    pass",
            {
                "language": "python",
                "node_type": "function",
                "node_name": "test",
                "start_byte": 0,
                "end_byte": 20,
                "start_line": 1,
                "end_line": 2,
                "documentation": None,
                "parent_scope": None,
                "signature": "def test():",
                "extra": {},
            },
        )
    ]
    return parser


# ============================================================================
# Walker-specific fixtures
# ============================================================================


@pytest.fixture
async def git_repo_path(tmp_path: Path) -> Path:
    """Create a git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    init_git_repo(repo_path)

    # Create initial commit
    test_file = repo_path / "README.md"
    test_file.write_text("# Test Repo")
    git_add(repo_path, ["README.md"])
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_gitignore(tmp_path: Path) -> Path:
    """Create a git repository with .gitignore."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create .gitignore
    gitignore = repo_path / ".gitignore"
    await anyio.Path(gitignore).write_text("*.log\ntemp/\n")

    # Create test file
    test_file = repo_path / "test.py"
    await anyio.Path(test_file).write_text("print('test')")

    git_add(repo_path, [".gitignore", "test.py"])
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_files(tmp_path: Path) -> Path:
    """Create a git repository with various files."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create some test files
    files = {
        "test.py": "print('test')",
        "README.md": "# Test",
        "config.json": '{"key": "value"}',
    }

    for filename, content in files.items():
        file_path = repo_path / filename
        await anyio.Path(file_path).write_text(content)

    git_add(repo_path, list(files.keys()))
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_binary(tmp_path: Path) -> Path:
    """Create a git repository with binary files."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create text file
    text_file = repo_path / "test.py"
    await anyio.Path(text_file).write_text("print('test')")

    # Create binary files (empty but with binary extensions)
    binary_files = ["image.png", "doc.pdf", "archive.zip"]
    for filename in binary_files:
        file_path = repo_path / filename
        await anyio.Path(file_path).write_bytes(b"\x00\x01\x02")

    git_add(repo_path, ["test.py"] + binary_files)
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_minified(tmp_path: Path) -> Path:
    """Create a git repository with minified files."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create normal file
    normal_file = repo_path / "script.js"
    await anyio.Path(normal_file).write_text("function test() {}")

    # Create minified files
    minified_files = {
        "script.min.js": "function test(){}",
        "style.min.css": ".test{color:red}",
    }

    for filename, content in minified_files.items():
        file_path = repo_path / filename
        await anyio.Path(file_path).write_text(content)

    git_add(repo_path, ["script.js"] + list(minified_files.keys()))
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_large_file(tmp_path: Path) -> Path:
    """Create a git repository with a large file."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create normal file
    normal_file = repo_path / "small.py"
    await anyio.Path(normal_file).write_text("print('small')")

    # Create large file (> 1MB)
    large_file = repo_path / "large_file.txt"
    await anyio.Path(large_file).write_text("x" * (2 * 1024 * 1024))

    git_add(repo_path, ["small.py", "large_file.txt"])
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_empty_file(tmp_path: Path) -> Path:
    """Create a git repository with an empty file."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create normal file
    normal_file = repo_path / "test.py"
    await anyio.Path(normal_file).write_text("print('test')")

    # Create empty file
    empty_file = repo_path / "empty.txt"
    await anyio.Path(empty_file).write_text("")

    git_add(repo_path, ["test.py", "empty.txt"])
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_latin1(tmp_path: Path) -> Path:
    """Create a git repository with latin-1 encoded file."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create latin-1 encoded file
    latin1_file = repo_path / "latin1.txt"
    await anyio.Path(latin1_file).write_bytes(b"Test \xe9 content")

    git_add(repo_path, ["latin1.txt"])
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_unreadable(tmp_path: Path) -> Path:
    """Create a git repository with an unreadable binary file."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create normal file
    normal_file = repo_path / "test.py"
    await anyio.Path(normal_file).write_text("print('test')")

    # Create binary file that can't be decoded
    unreadable_file = repo_path / "unreadable.bin"
    await anyio.Path(unreadable_file).write_bytes(b"\x00\xff\xfe\xfd")

    git_add(repo_path, ["test.py", "unreadable.bin"])
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_nested(tmp_path: Path) -> Path:
    """Create a git repository with nested directories."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create nested structure
    (repo_path / "src").mkdir()
    (repo_path / "tests").mkdir()
    (repo_path / "src" / "utils").mkdir()

    files = {
        "src/main.py": "print('main')",
        "src/utils/helper.py": "def helper(): pass",
        "tests/test_main.py": "def test_main(): pass",
    }

    for file_path, content in files.items():
        full_path = repo_path / file_path
        await anyio.Path(full_path).write_text(content)

    git_add(repo_path, list(files.keys()))
    git_commit(repo_path, "Initial commit")

    return repo_path


@pytest.fixture
async def git_repo_with_nested_ignored(tmp_path: Path) -> Path:
    """Create a git repository with nested ignored directories."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    init_git_repo(repo_path)

    # Create directories
    (repo_path / "src").mkdir()
    (repo_path / "node_modules").mkdir()
    (repo_path / ".venv").mkdir()

    # Create files in all directories
    files = {
        "src/main.py": "print('main')",
        "node_modules/package.js": "module.exports = {}",
        ".venv/lib.py": "# venv file",
    }

    for file_path, content in files.items():
        full_path = repo_path / file_path
        await anyio.Path(full_path).write_text(content)

    # Only add the src file to git
    git_add(repo_path, ["src/main.py"])
    git_commit(repo_path, "Initial commit")

    return repo_path
