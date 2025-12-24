"""Unit tests for indexter.models."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from indexter.config.repo import RepoConfig
from indexter.exceptions import InvalidGitRepositoryError, RepoNotFoundError
from indexter.models import Document, IndexResult, Node, NodeMetadata, Repo

# ============================================================================
# IndexResult Tests
# ============================================================================


def test_index_result_default_initialization():
    """Test IndexResult initializes with default empty values."""
    result = IndexResult()

    assert result.files_synced == []
    assert result.files_deleted == []
    assert result.files_checked == 0
    assert result.skipped_files == 0
    assert result.nodes_added == 0
    assert result.nodes_deleted == 0
    assert result.nodes_updated == 0
    assert result.errors == []
    assert isinstance(result.indexed_at, datetime)


def test_index_result_with_values(sample_index_result: IndexResult):
    """Test IndexResult with provided values."""
    assert sample_index_result.files_synced == ["file1.py", "file2.py"]
    assert sample_index_result.files_deleted == ["old_file.py"]
    assert sample_index_result.files_checked == 10
    assert sample_index_result.skipped_files == 2
    assert sample_index_result.nodes_added == 5
    assert sample_index_result.nodes_deleted == 1
    assert sample_index_result.nodes_updated == 3
    assert sample_index_result.errors == ["Error parsing file3.py"]


def test_index_result_serialization(sample_index_result: IndexResult):
    """Test IndexResult can be serialized."""
    data = sample_index_result.model_dump()

    assert data["files_synced"] == ["file1.py", "file2.py"]
    assert data["files_deleted"] == ["old_file.py"]
    assert data["files_checked"] == 10
    assert data["nodes_added"] == 5


# ============================================================================
# NodeMetadata Tests
# ============================================================================


def test_node_metadata_initialization(sample_node_metadata: NodeMetadata):
    """Test NodeMetadata initialization."""
    assert sample_node_metadata.hash == "abc123def456"
    assert sample_node_metadata.document_path == "src/test.py"
    assert sample_node_metadata.language == "python"
    assert sample_node_metadata.node_type == "function"
    assert sample_node_metadata.node_name == "hello"
    assert sample_node_metadata.start_byte == 0
    assert sample_node_metadata.end_byte == 42
    assert sample_node_metadata.start_line == 1
    assert sample_node_metadata.end_line == 2
    assert sample_node_metadata.documentation == "Test function"
    assert sample_node_metadata.parent_scope is None
    assert sample_node_metadata.signature == "def hello():"
    assert sample_node_metadata.extra == {}


def test_node_metadata_optional_fields():
    """Test NodeMetadata with optional fields set to None."""
    metadata = NodeMetadata(
        hash="test_hash",
        repo_path="/repo",
        document_path="test.py",
        language="python",
        node_type="function",
        node_name="test",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=2,
    )

    assert metadata.documentation is None
    assert metadata.parent_scope is None
    assert metadata.signature is None
    assert metadata.extra == {}


def test_node_metadata_with_parent_scope():
    """Test NodeMetadata with parent scope."""
    metadata = NodeMetadata(
        hash="test_hash",
        repo_path="/repo",
        document_path="test.py",
        language="python",
        node_type="method",
        node_name="test_method",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=2,
        parent_scope="TestClass",
    )

    assert metadata.parent_scope == "TestClass"


def test_node_metadata_with_extra():
    """Test NodeMetadata with extra data."""
    metadata = NodeMetadata(
        hash="test_hash",
        repo_path="/repo",
        document_path="test.py",
        language="python",
        node_type="function",
        node_name="test",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=2,
        extra={"custom_field": "value", "another": "data"},
    )

    assert metadata.extra["custom_field"] == "value"
    assert metadata.extra["another"] == "data"


# ============================================================================
# Node Tests
# ============================================================================


def test_node_initialization(sample_node: Node):
    """Test Node initialization."""
    assert sample_node.id == uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert sample_node.content == "def hello():\n    print('Hello, world!')"
    assert isinstance(sample_node.metadata, NodeMetadata)


def test_node_auto_generates_uuid():
    """Test Node generates UUID automatically."""
    metadata = NodeMetadata(
        hash="test_hash",
        repo_path="/repo",
        document_path="test.py",
        language="python",
        node_type="function",
        node_name="test",
        start_byte=0,
        end_byte=10,
        start_line=1,
        end_line=2,
    )
    node = Node(content="test content", metadata=metadata)

    assert isinstance(node.id, uuid.UUID)
    assert node.content == "test content"


def test_node_serialization(sample_node: Node):
    """Test Node can be serialized."""
    data = sample_node.model_dump()

    assert str(data["id"]) == "12345678-1234-5678-1234-567812345678"
    assert data["content"] == "def hello():\n    print('Hello, world!')"
    assert "metadata" in data
    assert data["metadata"]["node_name"] == "hello"


# ============================================================================
# Document Tests
# ============================================================================


def test_document_initialization(sample_document: Document):
    """Test Document initialization."""
    assert sample_document.path == "src/test.py"
    assert sample_document.size_bytes == 1024
    assert sample_document.mtime == 1234567890.0
    assert sample_document.content == "def hello():\n    print('Hello, world!')"
    assert sample_document.hash == "abc123def456"


def test_document_serialization(sample_document: Document):
    """Test Document can be serialized."""
    data = sample_document.model_dump()

    assert data["path"] == "src/test.py"
    assert data["size_bytes"] == 1024
    assert data["mtime"] == 1234567890.0
    assert data["hash"] == "abc123def456"


def test_document_validation():
    """Test Document validates required fields."""
    with pytest.raises(ValidationError):
        Document()


# ============================================================================
# Repo Tests - Computed Fields
# ============================================================================


def test_repo_computed_field_name(sample_repo_config: RepoConfig):
    """Test Repo.name computed field."""
    repo = Repo(repo_config=sample_repo_config)
    assert repo.name == "test_repo"


def test_repo_computed_field_path(sample_repo_config: RepoConfig, sample_repo_path: Path):
    """Test Repo.path computed field."""
    repo = Repo(repo_config=sample_repo_config)
    assert repo.path == str(sample_repo_path)


def test_repo_computed_field_collection_name(sample_repo_config: RepoConfig):
    """Test Repo.collection_name computed field."""
    repo = Repo(repo_config=sample_repo_config)
    assert repo.collection_name == "indexter_test_repo"


# ============================================================================
# Repo Tests - add() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
@patch("indexter.models.save_repos_config")
@patch("indexter.models.validate_git_repository")
async def test_repo_init_new_repository(
    mock_validate: MagicMock,
    mock_save: AsyncMock,
    mock_load: AsyncMock,
    sample_repo_path: Path,
):
    """Test adding a new repository."""
    mock_load.return_value = []
    mock_save.return_value = None

    repo = await Repo.init(sample_repo_path)

    assert repo.name == "test_repo"  # Derived from directory name
    assert repo.path == str(sample_repo_path)
    mock_validate.assert_called_once_with(sample_repo_path)
    mock_save.assert_called_once()


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
@patch("indexter.models.save_repos_config")
@patch("indexter.models.validate_git_repository")
async def test_repo_init_existing_repository(
    mock_validate: MagicMock,
    mock_save: AsyncMock,
    mock_load: AsyncMock,
    sample_repo_path: Path,
    sample_repo_config: RepoConfig,
):
    """Test adding a repository that already exists."""
    mock_load.return_value = [sample_repo_config]

    repo = await Repo.init(sample_repo_path)

    assert repo.name == "test_repo"
    assert repo.path == str(sample_repo_path)
    mock_save.assert_not_called()


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
@patch("indexter.models.validate_git_repository")
async def test_repo_init_invalid_git_repository(
    mock_validate: MagicMock,
    mock_load: AsyncMock,
    sample_repo_path: Path,
):
    """Test adding a path that is not a git repository."""
    mock_load.return_value = []
    mock_validate.side_effect = InvalidGitRepositoryError()

    with pytest.raises(ValueError, match="is not a git repository"):
        await Repo.init(sample_repo_path)


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
@patch("indexter.models.save_repos_config")
@patch("indexter.models.validate_git_repository")
async def test_repo_init_auto_generates_name(
    mock_validate: MagicMock,
    mock_save: AsyncMock,
    mock_load: AsyncMock,
    sample_repo_path: Path,
):
    """Test adding a repository without explicit name."""
    mock_load.return_value = []

    repo = await Repo.init(sample_repo_path)

    # Name should be auto-generated from path
    assert repo.name == sample_repo_path.name


# ============================================================================
# Repo Tests - get() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
async def test_repo_get_existing(
    mock_load: AsyncMock,
    sample_repo_path: Path,
    sample_repo_config: RepoConfig,
):
    """Test getting an existing repository."""
    mock_load.return_value = [sample_repo_config]

    repo = await Repo.get("test_repo")

    assert repo is not None
    assert repo.name == "test_repo"
    assert repo.path == str(sample_repo_path)


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
async def test_repo_get_nonexistent(
    mock_load: AsyncMock,
    tmp_path: Path,
):
    """Test getting a repository that doesn't exist."""
    mock_load.return_value = []

    with pytest.raises(RepoNotFoundError):
        await Repo.get("nonexistent")


# ============================================================================
# Repo Tests - list() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
async def test_repo_list_empty(mock_load: AsyncMock):
    """Test listing repositories when none are configured."""
    mock_load.return_value = []

    repos = await Repo.list()

    assert repos == []


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
async def test_repo_list_multiple(
    mock_load: AsyncMock,
    sample_repo_config: RepoConfig,
    tmp_path: Path,
):
    """Test listing multiple repositories."""
    config2 = RepoConfig(path=tmp_path / "repo2")
    mock_load.return_value = [sample_repo_config, config2]

    repos = await Repo.list()

    assert len(repos) == 2
    assert repos[0].name == "test_repo"
    assert repos[1].name == "repo2"


# ============================================================================
# Repo Tests - remove() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
@patch("indexter.models.save_repos_config")
@patch("indexter.models.store")
async def test_repo_remove_existing(
    mock_store: MagicMock,
    mock_save: AsyncMock,
    mock_load: AsyncMock,
    sample_repo_path: Path,
    sample_repo_config: RepoConfig,
):
    """Test removing an existing repository."""
    mock_load.return_value = [sample_repo_config]
    mock_store.delete_collection = AsyncMock()

    result = await Repo.remove("test_repo")

    assert result is True
    mock_store.delete_collection.assert_called_once_with("indexter_test_repo")
    mock_save.assert_called_once()

    # Check that the repo was removed from the list
    saved_repos = mock_save.call_args[0][0]
    assert len(saved_repos) == 0


@pytest.mark.asyncio
@patch("indexter.models.load_repos_config")
async def test_repo_remove_nonexistent(
    mock_load: AsyncMock,
    tmp_path: Path,
):
    """Test removing a repository that doesn't exist."""
    mock_load.return_value = []

    with pytest.raises(RepoNotFoundError):
        await Repo.remove("nonexistent")


# ============================================================================
# Repo Tests - get_document_hashes() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.Walker.create")
async def test_repo_get_document_hashes(
    mock_walker_create: AsyncMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
):
    """Test getting document hashes from repository."""
    mock_walker_create.return_value = mock_walker

    repo = Repo(repo_config=sample_repo_config)
    hashes = await repo.get_document_hashes()

    assert "test.py" in hashes
    assert hashes["test.py"] == "test_hash"


@pytest.mark.asyncio
@patch("indexter.models.Walker.create")
async def test_repo_get_document_hashes_empty(
    mock_walker_create: AsyncMock,
    sample_repo_config: RepoConfig,
):
    """Test getting document hashes when repository is empty."""

    async def empty_walk():
        return
        yield  # Make it an async generator

    mock_walker = MagicMock()
    mock_walker.walk = empty_walk
    mock_walker_create.return_value = mock_walker

    repo = Repo(repo_config=sample_repo_config)
    hashes = await repo.get_document_hashes()

    assert hashes == {}


# ============================================================================
# Repo Tests - search() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.store")
async def test_repo_search_basic(
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
):
    """Test basic search functionality."""
    mock_store.search = AsyncMock(return_value=[{"content": "test", "score": 0.9}])

    repo = Repo(repo_config=sample_repo_config)
    results = await repo.search("test query")

    assert len(results) == 1
    assert results[0]["content"] == "test"
    mock_store.search.assert_called_once_with(
        collection_name="indexter_test_repo",
        query="test query",
        limit=10,
        file_path=None,
        language=None,
        node_type=None,
        node_name=None,
        has_documentation=None,
    )


@pytest.mark.asyncio
@patch("indexter.models.store")
async def test_repo_search_with_filters(
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
):
    """Test search with various filters."""
    mock_store.search = AsyncMock(return_value=[])

    repo = Repo(repo_config=sample_repo_config)
    await repo.search(
        query="test",
        limit=5,
        file_path="src/test.py",
        language="python",
        node_type="function",
        node_name="test_func",
        has_documentation=True,
    )

    mock_store.search.assert_called_once_with(
        collection_name="indexter_test_repo",
        query="test",
        limit=5,
        file_path="src/test.py",
        language="python",
        node_type="function",
        node_name="test_func",
        has_documentation=True,
    )


# ============================================================================
# Repo Tests - status() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
async def test_repo_status(
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
):
    """Test getting repository status."""
    mock_walker_create.return_value = mock_walker
    mock_store.get_document_hashes = AsyncMock(
        return_value={
            "test.py": "test_hash",
            "old.py": "old_hash",
        }
    )
    mock_store.count_nodes = AsyncMock(return_value=42)

    repo = Repo(repo_config=sample_repo_config)
    status = await repo.status()

    assert status["repository"] == "test_repo"
    assert status["path"] == str(sample_repo_config.path)
    assert status["documents_indexed"] == 2
    assert status["nodes_indexed"] == 42


# ============================================================================
# Repo Tests - sync() method
# ============================================================================


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
async def test_repo_index_new_file(
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
    mock_parser: MagicMock,
):
    """Test syncing with a new file."""
    mock_walker_create.return_value = mock_walker
    mock_get_parser.return_value = mock_parser
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={})
    mock_store.upsert_nodes = AsyncMock()

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    assert len(result.files_synced) == 1
    assert "test.py" in result.files_synced
    assert result.nodes_added == 1
    assert result.files_checked == 1
    mock_store.ensure_collection.assert_called_once_with("indexter_test_repo")
    mock_store.upsert_nodes.assert_called_once()


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
async def test_repo_index_modified_file(
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
    mock_parser: MagicMock,
):
    """Test syncing with a modified file."""
    mock_walker_create.return_value = mock_walker
    mock_get_parser.return_value = mock_parser
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(
        return_value={
            "test.py": "old_hash"  # Different hash indicates modification
        }
    )
    mock_store.delete_by_document_paths = AsyncMock()
    mock_store.upsert_nodes = AsyncMock()

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    assert len(result.files_synced) == 1
    assert result.nodes_updated == 1
    assert result.nodes_added == 0
    mock_store.delete_by_document_paths.assert_called_once()


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
async def test_repo_index_unchanged_file(
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
):
    """Test syncing with an unchanged file."""
    mock_walker_create.return_value = mock_walker
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(
        return_value={
            "test.py": "test_hash"  # Same hash means unchanged
        }
    )
    mock_store.upsert_nodes = AsyncMock()

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    assert len(result.files_synced) == 0
    assert result.files_checked == 1
    assert result.nodes_added == 0
    assert result.nodes_updated == 0
    mock_get_parser.assert_not_called()


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
async def test_repo_index_deleted_file(
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
):
    """Test syncing when a file has been deleted."""

    async def empty_walk():
        return
        yield  # Make it an async generator

    mock_walker = MagicMock()
    mock_walker.walk = empty_walk
    mock_walker_create.return_value = mock_walker

    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={"deleted.py": "old_hash"})
    mock_store.delete_by_document_paths = AsyncMock()

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    assert len(result.files_deleted) == 1
    assert "deleted.py" in result.files_deleted
    mock_store.delete_by_document_paths.assert_called_once_with(
        "indexter_test_repo", ["deleted.py"]
    )


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
async def test_repo_index_parse_error(
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
):
    """Test syncing when parsing fails."""
    mock_walker_create.return_value = mock_walker
    mock_parser = MagicMock()
    mock_parser.parse.side_effect = Exception("Parse error")
    mock_get_parser.return_value = mock_parser
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={})

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    assert len(result.errors) == 1
    assert "Parse error" in result.errors[0]
    assert len(result.files_synced) == 0


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
async def test_repo_index_no_parser(
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
):
    """Test syncing when no parser is available for file type."""
    mock_walker_create.return_value = mock_walker
    mock_get_parser.return_value = None  # No parser available
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={})

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    assert len(result.files_synced) == 0
    assert result.files_checked == 1


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
async def test_repo_index_full_recreates_collection(
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
):
    """Test full sync recreates the collection."""

    async def empty_walk():
        return
        yield  # Make it an async generator

    mock_walker = MagicMock()
    mock_walker.walk = empty_walk
    mock_walker_create.return_value = mock_walker

    mock_store.delete_collection = AsyncMock()
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={})

    repo = Repo(repo_config=sample_repo_config)
    await repo.index(full=True)

    mock_store.delete_collection.assert_called_once_with("indexter_test_repo")
    mock_store.ensure_collection.assert_called_once()


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
@patch("indexter.models.RepoFileConfig.from_repo")
async def test_repo_index_respects_max_files_limit(
    mock_from_repo: AsyncMock,
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_parser: MagicMock,
):
    """Test that index respects max_sync_files config limit."""
    # Mock config with max_sync_files = 2
    mock_config = MagicMock()
    mock_config.max_sync_files = 2
    mock_config.upsert_batch_size = 100
    mock_from_repo.return_value = mock_config

    # Create walker that returns 3 files
    async def walk_multiple():
        for i in range(3):
            yield {
                "path": f"file{i}.py",
                "size_bytes": 100,
                "mtime": 1234567890.0,
                "content": "print('test')",
                "hash": f"hash{i}",
            }

    mock_walker = MagicMock()
    mock_walker.walk = walk_multiple
    mock_walker_create.return_value = mock_walker
    mock_get_parser.return_value = mock_parser
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={})
    mock_store.upsert_nodes = AsyncMock()

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    # Should only sync 2 files due to MAX_SYNC_FILES limit
    assert len(result.files_synced) == 2
    assert result.skipped_files == 1
    assert result.files_checked == 3


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
@patch("indexter.models.RepoFileConfig.from_repo")
async def test_repo_index_batches_upserts(
    mock_from_repo: AsyncMock,
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
):
    """Test that index batches node upserts according to upsert_batch_size config."""
    # Mock config with upsert_batch_size = 2
    mock_config = MagicMock()
    mock_config.max_sync_files = 500
    mock_config.upsert_batch_size = 2
    mock_from_repo.return_value = mock_config

    # Create walker that returns 3 files
    async def walk_multiple():
        for i in range(3):
            yield {
                "path": f"file{i}.py",
                "size_bytes": 100,
                "mtime": 1234567890.0,
                "content": "print('test')",
                "hash": f"hash{i}",
            }

    mock_walker = MagicMock()
    mock_walker.walk = walk_multiple
    mock_walker_create.return_value = mock_walker

    mock_parser = MagicMock()
    mock_parser.parse.return_value = [
        (
            "content",
            {
                "language": "python",
                "node_type": "function",
                "node_name": "test",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 2,
            },
        )
    ]
    mock_get_parser.return_value = mock_parser

    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={})
    mock_store.upsert_nodes = AsyncMock()

    repo = Repo(repo_config=sample_repo_config)
    await repo.index()

    # Should call upsert_nodes twice: once for batch of 2, once for remaining 1
    assert mock_store.upsert_nodes.call_count == 2


@pytest.mark.asyncio
@patch("indexter.models.store")
@patch("indexter.models.Walker.create")
@patch("indexter.models.get_parser")
async def test_repo_index_result_timestamps(
    mock_get_parser: MagicMock,
    mock_walker_create: AsyncMock,
    mock_store: MagicMock,
    sample_repo_config: RepoConfig,
    mock_walker: MagicMock,
    mock_parser: MagicMock,
):
    """Test that index result includes timestamp."""
    mock_walker_create.return_value = mock_walker
    mock_get_parser.return_value = mock_parser
    mock_store.ensure_collection = AsyncMock()
    mock_store.get_document_hashes = AsyncMock(return_value={})
    mock_store.upsert_nodes = AsyncMock()

    repo = Repo(repo_config=sample_repo_config)
    result = await repo.index()

    assert isinstance(result.indexed_at, datetime)
    # Should be very recent (within 1 second)
    now = datetime.now(UTC)
    assert (now - result.indexed_at).total_seconds() < 1
