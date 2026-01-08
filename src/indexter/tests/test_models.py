"""Tests for indexter.models module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from indexter.config import RepoSettings
from indexter.exceptions import RepoExistsError, RepoNotFoundError
from indexter.models import (
    Document,
    IndexResult,
    Node,
    NodeMetadata,
    Repo,
    RepoStatus,
    SearchResponse,
)

# ============================================================================
# IndexResult Tests
# ============================================================================


def test_index_result_defaults():
    """Test IndexResult with default values."""
    result = IndexResult()

    assert result.files_indexed == []
    assert result.files_deleted == []
    assert result.files_checked == 0
    assert result.skipped_files == 0
    assert result.nodes_added == 0
    assert result.nodes_deleted == 0
    assert result.nodes_updated == 0
    assert result.errors == []
    assert isinstance(result.indexed_at, datetime)
    assert result.indexed_at.tzinfo == UTC


def test_index_result_with_data():
    """Test IndexResult with custom data."""
    now = datetime.now(UTC)
    result = IndexResult(
        files_indexed=["file1.py", "file2.py"],
        files_deleted=["old.py"],
        files_checked=10,
        skipped_files=2,
        nodes_added=5,
        nodes_deleted=3,
        nodes_updated=7,
        indexed_at=now,
        errors=["Error 1", "Error 2"],
    )

    assert result.files_indexed == ["file1.py", "file2.py"]
    assert result.files_deleted == ["old.py"]
    assert result.files_checked == 10
    assert result.skipped_files == 2
    assert result.nodes_added == 5
    assert result.nodes_deleted == 3
    assert result.nodes_updated == 7
    assert result.indexed_at == now
    assert result.errors == ["Error 1", "Error 2"]


def test_index_result_summary():
    """Test IndexResult.summary property generates correct formatted string."""
    result = IndexResult(
        files_indexed=["file1.py", "file2.py", "file3.py"],
        nodes_added=10,
        nodes_updated=5,
        nodes_deleted=2,
        duration=1.5,
    )

    summary = result.summary
    assert "Indexed 3 files" in summary
    assert "+10 nodes" in summary
    assert "~5 updated" in summary
    assert "-2 deleted" in summary
    assert "in 1.50s" in summary
    assert isinstance(summary, str)


def test_index_result_summary_no_files():
    """Test IndexResult.summary with no files indexed."""
    result = IndexResult(
        files_indexed=[],
        nodes_added=0,
        nodes_updated=0,
        nodes_deleted=0,
        duration=0.25,
    )

    summary = result.summary
    assert "Indexed 0 files" in summary
    assert "+0 nodes" in summary
    assert "~0 updated" in summary
    assert "-0 deleted" in summary
    assert "in 0.25s" in summary


# ============================================================================
# NodeMetadata Tests
# ============================================================================


def test_node_metadata_required_fields():
    """Test NodeMetadata with required fields."""
    metadata = NodeMetadata(
        hash="abc123",
        repo_path="/path/to/repo",
        document_path="src/module.py",
        language="python",
        node_type="function",
        node_name="my_function",
        start_byte=100,
        end_byte=200,
        start_line=10,
        end_line=20,
    )

    assert metadata.hash == "abc123"
    assert metadata.repo_path == "/path/to/repo"
    assert metadata.document_path == "src/module.py"
    assert metadata.language == "python"
    assert metadata.node_type == "function"
    assert metadata.node_name == "my_function"
    assert metadata.start_byte == 100
    assert metadata.end_byte == 200
    assert metadata.start_line == 10
    assert metadata.end_line == 20
    assert metadata.documentation is None
    assert metadata.parent_scope is None
    assert metadata.signature is None
    assert metadata.extra == {}


def test_node_metadata_with_optional_fields():
    """Test NodeMetadata with all optional fields."""
    metadata = NodeMetadata(
        hash="def456",
        repo_path="/path/to/repo",
        document_path="src/class.py",
        language="python",
        node_type="method",
        node_name="process_data",
        start_byte=300,
        end_byte=500,
        start_line=30,
        end_line=50,
        documentation="Process the data efficiently.",
        parent_scope="DataProcessor",
        signature="def process_data(self, data: list) -> dict",
        extra={"decorator": "@staticmethod", "async": "true"},
    )

    assert metadata.documentation == "Process the data efficiently."
    assert metadata.parent_scope == "DataProcessor"
    assert metadata.signature == "def process_data(self, data: list) -> dict"
    assert metadata.extra == {"decorator": "@staticmethod", "async": "true"}


# ============================================================================
# Node Tests
# ============================================================================


def test_node_auto_generates_id():
    """Test that Node auto-generates UUID."""
    metadata = NodeMetadata(
        hash="xyz789",
        repo_path="/repo",
        document_path="file.py",
        language="python",
        node_type="class",
        node_name="MyClass",
        start_byte=0,
        end_byte=100,
        start_line=1,
        end_line=10,
    )
    node = Node(content="class MyClass:\n    pass", metadata=metadata)

    assert isinstance(node.id, UUID)
    assert node.content == "class MyClass:\n    pass"
    assert node.metadata == metadata


def test_node_with_custom_id():
    """Test Node with explicitly provided UUID."""
    custom_id = UUID("12345678-1234-5678-1234-567812345678")
    metadata = NodeMetadata(
        hash="xyz789",
        repo_path="/repo",
        document_path="file.py",
        language="python",
        node_type="class",
        node_name="MyClass",
        start_byte=0,
        end_byte=100,
        start_line=1,
        end_line=10,
    )
    node = Node(id=custom_id, content="class MyClass:\n    pass", metadata=metadata)

    assert node.id == custom_id


# ============================================================================
# Document Tests
# ============================================================================


def test_document_creation():
    """Test Document model creation."""
    doc = Document(
        path="src/main.py",
        size_bytes=1024,
        mtime=1234567890.0,
        content="print('hello')",
        hash="abc123def456",
    )

    assert doc.path == "src/main.py"
    assert doc.size_bytes == 1024
    assert doc.mtime == 1234567890.0
    assert doc.content == "print('hello')"
    assert doc.hash == "abc123def456"


# ============================================================================
# Repo Tests - Computed Fields
# ============================================================================


def test_repo_computed_fields(repo_settings):
    """Test Repo computed fields."""
    repo = Repo(settings=repo_settings)

    assert repo.collection_name == repo_settings.collection_name
    assert repo.name == repo_settings.name
    assert repo.path == str(repo_settings.path)


# ============================================================================
# Repo Tests - init()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_init_new_repo(tmp_path):
    """Test initializing a new repository."""
    repo_path = tmp_path / "new_repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[])):
        with patch.object(RepoSettings, "save", new=AsyncMock()) as mock_save:
            repo = await Repo.init(repo_path)

            assert repo.name == "new_repo"
            assert repo.path == str(repo_path.resolve())

            # Verify save was called with the new repo
            mock_save.assert_called_once()
            saved_repos = mock_save.call_args[0][0]
            assert len(saved_repos) == 1
            assert saved_repos[0].name == "new_repo"


@pytest.mark.asyncio
async def test_repo_init_already_configured(tmp_path):
    """Test initializing a repository that's already configured."""
    repo_path = tmp_path / "existing_repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    existing_settings = RepoSettings(path=repo_path.resolve())

    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[existing_settings])):
        with patch.object(RepoSettings, "save", new=AsyncMock()) as mock_save:
            repo = await Repo.init(repo_path)

            assert repo.name == "existing_repo"
            # Save should not be called since repo already exists
            mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_repo_init_name_conflict(tmp_path):
    """Test initializing a repo when another repo with same name exists."""
    repo_path1 = tmp_path / "my_repo"
    repo_path2 = tmp_path / "another_location" / "my_repo"
    repo_path1.mkdir()
    (repo_path1 / ".git").mkdir()  # Make it a valid git repo
    repo_path2.mkdir(parents=True)
    (repo_path2 / ".git").mkdir()  # Make it a valid git repo

    existing_settings = RepoSettings(path=repo_path1.resolve())

    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[existing_settings])):
        with pytest.raises(RepoExistsError) as exc_info:
            await Repo.init(repo_path2)

        assert "already exists" in str(exc_info.value)
        assert "my_repo" in str(exc_info.value)


# ============================================================================
# Repo Tests - get()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_get_existing(temp_git_repo):
    """Test getting an existing repository by name."""
    settings = RepoSettings(path=temp_git_repo)

    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[settings])):
        repo = await Repo.get(temp_git_repo.name)

        assert repo.name == temp_git_repo.name
        assert isinstance(repo, Repo)


@pytest.mark.asyncio
async def test_repo_get_not_found():
    """Test getting a non-existent repository."""
    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[])):
        with pytest.raises(RepoNotFoundError) as exc_info:
            await Repo.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)


# ============================================================================
# Repo Tests - list()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_list_empty():
    """Test listing repositories when none exist."""
    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[])):
        repos = await Repo.list()

        assert repos == []


@pytest.mark.asyncio
async def test_repo_list_multiple(tmp_path):
    """Test listing multiple repositories."""
    # Create three temp git repos
    repos_paths = []
    for i in range(1, 4):
        repo_path = tmp_path / f"repo{i}"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        repos_paths.append(repo_path)

    settings1 = RepoSettings(path=repos_paths[0])
    settings2 = RepoSettings(path=repos_paths[1])
    settings3 = RepoSettings(path=repos_paths[2])

    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[settings1, settings2, settings3])):
        repos = await Repo.list()

        assert len(repos) == 3
        assert all(isinstance(r, Repo) for r in repos)
        assert [r.name for r in repos] == ["repo1", "repo2", "repo3"]


# ============================================================================
# Repo Tests - remove()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_remove_existing(tmp_path):
    """Test removing an existing repository."""
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    (test_repo / ".git").mkdir()
    settings = RepoSettings(path=test_repo)

    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[settings])):
        with patch.object(RepoSettings, "save", new=AsyncMock()) as mock_save:
            with patch("indexter.models.store") as mock_store:
                mock_store.delete_collection = AsyncMock()

                result = await Repo.remove("test_repo")

                assert result is True
                mock_store.delete_collection.assert_called_once_with(settings.collection_name)
                mock_save.assert_called_once_with([])


@pytest.mark.asyncio
async def test_repo_remove_not_found():
    """Test removing a non-existent repository."""
    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[])):
        with pytest.raises(RepoNotFoundError):
            await Repo.remove("nonexistent")


@pytest.mark.asyncio
async def test_repo_remove_race_condition(tmp_path):
    """Test removing a repo that was deleted by another process (race condition)."""
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    (test_repo / ".git").mkdir()
    settings = RepoSettings(path=test_repo)

    # Simulate race condition: repo exists in first load (for get()),
    # but is gone in second load (for remove())
    load_call_count = 0

    async def mock_load():
        nonlocal load_call_count
        load_call_count += 1
        if load_call_count == 1:
            # First call from get() - repo exists
            return [settings]
        else:
            # Second call from remove() - repo already gone (deleted by another process)
            return []

    with patch.object(RepoSettings, "load", new=mock_load):
        with patch.object(RepoSettings, "save", new=AsyncMock()) as mock_save:
            with patch("indexter.models.store") as mock_store:
                mock_store.delete_collection = AsyncMock()

                result = await Repo.remove("test_repo")

                # Should return False since the repo wasn't in the list during save
                assert result is False
                # Save should still be called with empty list
                mock_save.assert_called_once_with([])


@pytest.mark.asyncio
async def test_repo_remove_keeps_other_repos(tmp_path):
    """Test that removing one repo doesn't affect others."""
    # Create three temp git repos
    repos_paths = []
    for i in range(1, 4):
        repo_path = tmp_path / f"repo{i}"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        repos_paths.append(repo_path)

    settings1 = RepoSettings(path=repos_paths[0])
    settings2 = RepoSettings(path=repos_paths[1])
    settings3 = RepoSettings(path=repos_paths[2])

    with patch.object(RepoSettings, "load", new=AsyncMock(return_value=[settings1, settings2, settings3])):
        with patch.object(RepoSettings, "save", new=AsyncMock()) as mock_save:
            with patch("indexter.models.store") as mock_store:
                mock_store.delete_collection = AsyncMock()

                await Repo.remove("repo2")

                # Verify remaining repos were saved
                saved_repos = mock_save.call_args[0][0]
                assert len(saved_repos) == 2
                assert [r.name for r in saved_repos] == ["repo1", "repo3"]


# ============================================================================
# Repo Tests - get_document_hashes()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_get_document_hashes(temp_git_repo):
    """Test getting document hashes from walker."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_docs = [
        {"path": "file1.py", "size_bytes": 100, "mtime": 123.0, "content": "a", "hash": "hash1"},
        {"path": "file2.py", "size_bytes": 200, "mtime": 456.0, "content": "b", "hash": "hash2"},
        {"path": "file3.py", "size_bytes": 300, "mtime": 789.0, "content": "c", "hash": "hash3"},
    ]

    async def mock_walk():
        for doc in mock_docs:
            yield doc

    with patch("indexter.models.Walker") as mock_walker_class:
        mock_walker = MagicMock()
        mock_walker.walk = mock_walk
        mock_walker_class.return_value = mock_walker

        hashes = await repo.get_document_hashes()

        assert hashes == {
            "file1.py": "hash1",
            "file2.py": "hash2",
            "file3.py": "hash3",
        }


# ============================================================================
# Repo Tests - search()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_search_basic(temp_git_repo):
    """Test basic search functionality."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    # Mock results need score and metadata fields for SearchResult model
    expected_results = [
        {"content": "result1", "score": 0.9, "metadata": {"file": "test.py"}},
        {"content": "result2", "score": 0.8, "metadata": {"file": "main.py"}},
    ]

    with patch("indexter.models.store") as mock_store:
        mock_store.search = AsyncMock(return_value=expected_results)

        response = await repo.search("test query", limit=5)

        # Check it's a SearchResponse model
        assert isinstance(response, SearchResponse)
        assert response.count == 2
        assert response.repository == repo.name
        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.results[0].content == "result1"
        assert response.results[0].score == 0.9
        assert response.results[1].content == "result2"
        assert response.results[1].score == 0.8

        mock_store.search.assert_called_once_with(
            collection_name=repo.collection_name,
            query="test query",
            limit=5,
            file_path=None,
            language=None,
            node_type=None,
            node_name=None,
            has_documentation=None,
        )


@pytest.mark.asyncio
async def test_repo_search_with_filters(temp_git_repo):
    """Test search with all filter parameters."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    with patch("indexter.models.store") as mock_store:
        mock_store.search = AsyncMock(return_value=[])

        response = await repo.search(
            query="find function",
            limit=20,
            file_path="src/module.py",
            language="python",
            node_type="function",
            node_name="my_func",
            has_documentation=True,
        )

        # Check it's a SearchResponse model
        assert isinstance(response, SearchResponse)
        assert response.count == 0
        assert response.repository == repo.name
        assert response.query == "find function"

        mock_store.search.assert_called_once_with(
            collection_name=repo.collection_name,
            query="find function",
            limit=20,
            file_path="src/module.py",
            language="python",
            node_type="function",
            node_name="my_func",
            has_documentation=True,
        )


# ============================================================================
# Repo Tests - status()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_status(temp_git_repo):
    """Test repository status reporting."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    # Mock local document hashes
    local_hashes = {
        "file1.py": "hash1",
        "file2.py": "hash2_new",  # Modified
        "file3.py": "hash3",
    }

    # Mock stored document hashes
    stored_hashes = {
        "file1.py": "hash1",
        "file2.py": "hash2_old",  # Stale
        "file4.py": "hash4",  # Deleted locally
    }

    # Need to mock the Walker to return local documents
    async def mock_walk():
        for path, hash in local_hashes.items():
            yield {
                "path": path,
                "size_bytes": 100,
                "mtime": 123.0,
                "content": "content",
                "hash": hash,
            }

    with patch("indexter.models.Walker") as mock_walker_class:
        mock_walker = MagicMock()
        mock_walker.walk = mock_walk
        mock_walker_class.return_value = mock_walker

        with patch("indexter.models.store") as mock_store:
            mock_store.get_document_hashes = AsyncMock(return_value=stored_hashes)
            mock_store.count_nodes = AsyncMock(return_value=42)

            status = await repo.status()

            # Check it's a RepoStatus model
            assert isinstance(status, RepoStatus)
            assert status.repository == temp_git_repo.name
            assert status.path == str(settings.path)
            assert status.nodes_indexed == 42
            assert status.documents_indexed == 3
            # hash2_old and hash4 are not in local_hashes
            assert status.documents_indexed_stale == 2


# ============================================================================
# Repo Tests - index()
# ============================================================================


@pytest.mark.asyncio
async def test_repo_index_full_sync(temp_git_repo):
    """Test full index sync (recreates collection)."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    with patch("indexter.models.store") as mock_store:
        mock_store.delete_collection = AsyncMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={})
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()

            async def mock_walk():
                return
                yield  # Make it an async generator

            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            result = await repo.index(full=True)

            mock_store.delete_collection.assert_called_once_with(repo.collection_name)
            mock_store.ensure_collection.assert_called_once_with(repo.collection_name)
            assert isinstance(result, IndexResult)
            # Verify summary is accessible
            assert isinstance(result.summary, str)
            assert "Indexed" in result.summary


@pytest.mark.asyncio
async def test_repo_index_new_file(temp_git_repo):
    """Test indexing a new file."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_doc = {
        "path": "new_file.py",
        "size_bytes": 100,
        "mtime": 123.0,
        "content": "def hello():\n    pass",
        "hash": "newhash",
    }

    async def mock_walk():
        yield mock_doc

    mock_parser = MagicMock()
    mock_parser.parse.return_value = [
        (
            "def hello():\n    pass",
            {
                "language": "python",
                "node_type": "function",
                "node_name": "hello",
                "start_byte": 0,
                "end_byte": 20,
                "start_line": 1,
                "end_line": 2,
            },
        )
    ]

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={})
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            with patch("indexter.models.get_parser", return_value=mock_parser):
                result = await repo.index()

                assert result.files_checked == 1
                assert result.files_indexed == ["new_file.py"]
                assert result.nodes_added == 1
                assert result.nodes_updated == 0
                assert result.files_deleted == []
                mock_store.upsert_nodes.assert_called_once()
                # Verify summary property
                assert "Indexed 1 files" in result.summary
                assert "+1 nodes" in result.summary
                assert "~0 updated" in result.summary


@pytest.mark.asyncio
async def test_repo_index_modified_file(temp_git_repo):
    """Test indexing a modified file."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_doc = {
        "path": "modified.py",
        "size_bytes": 150,
        "mtime": 456.0,
        "content": "def updated():\n    return 42",
        "hash": "newhash",
    }

    async def mock_walk():
        yield mock_doc

    mock_parser = MagicMock()
    mock_parser.parse.return_value = [
        (
            "def updated():\n    return 42",
            {
                "language": "python",
                "node_type": "function",
                "node_name": "updated",
                "start_byte": 0,
                "end_byte": 25,
                "start_line": 1,
                "end_line": 2,
            },
        )
    ]

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={"modified.py": "oldhash"})
        mock_store.delete_by_document_paths = AsyncMock()
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            with patch("indexter.models.get_parser", return_value=mock_parser):
                result = await repo.index()

                assert result.files_checked == 1
                assert result.files_indexed == ["modified.py"]
                assert result.nodes_added == 0
                assert result.nodes_updated == 1
                # Modified files should be deleted first
                mock_store.delete_by_document_paths.assert_called_once_with(repo.collection_name, ["modified.py"])
                # Verify summary property
                assert "Indexed 1 files" in result.summary
                assert "+0 nodes" in result.summary
                assert "~1 updated" in result.summary


@pytest.mark.asyncio
async def test_repo_index_deleted_file(temp_git_repo):
    """Test handling deleted files during indexing."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    async def mock_walk():
        return
        yield  # Empty generator - no files walked

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={"deleted.py": "oldhash"})
        mock_store.delete_by_document_paths = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            result = await repo.index()

            assert result.files_deleted == ["deleted.py"]
            assert result.nodes_deleted == 1
            mock_store.delete_by_document_paths.assert_called_once_with(repo.collection_name, ["deleted.py"])
            # Verify summary property
            assert "Indexed 0 files" in result.summary
            assert "-1 deleted" in result.summary


@pytest.mark.asyncio
async def test_repo_index_unchanged_file(temp_git_repo):
    """Test that unchanged files are skipped."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_doc = {
        "path": "unchanged.py",
        "size_bytes": 100,
        "mtime": 123.0,
        "content": "def same():\n    pass",
        "hash": "samehash",
    }

    async def mock_walk():
        yield mock_doc

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={"unchanged.py": "samehash"})
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            result = await repo.index()

            assert result.files_checked == 1
            assert result.files_indexed == []
            assert result.nodes_added == 0
            assert result.nodes_updated == 0
            # Should not upsert unchanged files
            mock_store.upsert_nodes.assert_not_called()
            # Verify summary property
            assert "Indexed 0 files" in result.summary
            assert "+0 nodes" in result.summary


@pytest.mark.asyncio
async def test_repo_index_respects_max_files(temp_git_repo):
    """Test that indexing respects max_files limit."""
    # Create indexter.toml with max_files setting
    (temp_git_repo / "indexter.toml").write_text("max_files = 2\n")

    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_docs = [
        {
            "path": f"file{i}.py",
            "size_bytes": 100,
            "mtime": 123.0,
            "content": f"# {i}",
            "hash": f"hash{i}",
        }
        for i in range(5)
    ]

    async def mock_walk():
        for doc in mock_docs:
            yield doc

    mock_parser = MagicMock()
    mock_parser.parse.return_value = [
        (
            "content",
            {
                "language": "python",
                "node_type": "comment",
                "node_name": "",
                "start_byte": 0,
                "end_byte": 10,
                "start_line": 1,
                "end_line": 1,
            },
        )
    ]

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={})
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            with patch("indexter.models.get_parser", return_value=mock_parser):
                result = await repo.index()

                assert result.files_checked == 5
                assert result.skipped_files == 3
                assert len(result.files_indexed) == 2
                # Verify summary property
                assert "Indexed 2 files" in result.summary


@pytest.mark.asyncio
async def test_repo_index_parser_error(temp_git_repo):
    """Test handling parser errors during indexing."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_doc = {
        "path": "broken.py",
        "size_bytes": 100,
        "mtime": 123.0,
        "content": "invalid syntax (",
        "hash": "brokenhash",
    }

    async def mock_walk():
        yield mock_doc

    mock_parser = MagicMock()
    mock_parser.parse.side_effect = Exception("Parse error!")

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={})
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            with patch("indexter.models.get_parser", return_value=mock_parser):
                result = await repo.index()

                assert result.files_checked == 1
                assert result.files_indexed == []
                assert len(result.errors) == 1
                assert "broken.py" in result.errors[0]
                assert "Parse error!" in result.errors[0]


@pytest.mark.asyncio
async def test_repo_index_no_parser_available(temp_git_repo):
    """Test handling files with no parser available."""
    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_doc = {
        "path": "unknown.xyz",
        "size_bytes": 100,
        "mtime": 123.0,
        "content": "some content",
        "hash": "unknownhash",
    }

    async def mock_walk():
        yield mock_doc

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={})
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            with patch("indexter.models.get_parser", return_value=None):
                result = await repo.index()

                assert result.files_checked == 1
                assert result.files_indexed == []
                # No parser, so file is silently skipped
                mock_store.upsert_nodes.assert_not_called()


@pytest.mark.asyncio
async def test_repo_index_batching(temp_git_repo):
    """Test that nodes are batched when upserting."""
    # Create indexter.toml with upsert_batch_size setting
    (temp_git_repo / "indexter.toml").write_text("upsert_batch_size = 2\n")

    settings = RepoSettings(path=temp_git_repo)
    repo = Repo(settings=settings)

    mock_docs = [
        {
            "path": f"file{i}.py",
            "size_bytes": 100,
            "mtime": 123.0,
            "content": f"def f{i}(): pass",
            "hash": f"hash{i}",
        }
        for i in range(3)
    ]

    async def mock_walk():
        for doc in mock_docs:
            yield doc

    mock_parser = MagicMock()
    mock_parser.parse.return_value = [
        (
            "def fn(): pass",
            {
                "language": "python",
                "node_type": "function",
                "node_name": "fn",
                "start_byte": 0,
                "end_byte": 15,
                "start_line": 1,
                "end_line": 1,
            },
        )
    ]

    with patch("indexter.models.store") as mock_store:
        mock_store.ensure_collection = AsyncMock()
        mock_store.get_document_hashes = AsyncMock(return_value={})
        mock_store.upsert_nodes = AsyncMock()

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker = MagicMock()
            mock_walker.walk = mock_walk
            mock_walker_class.return_value = mock_walker

            with patch("indexter.models.get_parser", return_value=mock_parser):
                result = await repo.index()

                # Should be called twice: once for batch of 2, once for remaining 1
                assert mock_store.upsert_nodes.call_count == 2
                assert result.files_indexed == ["file0.py", "file1.py", "file2.py"]
