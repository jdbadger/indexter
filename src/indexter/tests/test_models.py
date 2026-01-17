"""Comprehensive tests for the Repo and RepoMetadata models."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from indexter.config import RepoSettings
from indexter.exceptions import RepoExistsError, RepoNotFoundError
from indexter.models import Repo, RepoMetadata
from indexter.parser.models import Node, NodeMetadata
from indexter.store.models import IndexResult, SearchResults
from indexter.walker.models import Document, DocumentMetadata


# Helper function to create DocumentMetadata with all required fields
def create_doc_metadata(
    repo="test-repo",
    repo_path="/tmp/test",
    hash="hash",
    ext=".py",
    size_bytes=100,
    mtime=1234567890.0,
):
    """Create DocumentMetadata with all required fields."""
    return DocumentMetadata(
        repo=repo,
        repo_path=repo_path,
        hash=hash,
        ext=ext,
        size_bytes=size_bytes,
        mtime=mtime,
    )


class TestRepoMetadata:
    """Test RepoMetadata model."""

    def test_should_create_repo_metadata_with_required_fields(self):
        """Test RepoMetadata initializes with required fields."""
        metadata = RepoMetadata(
            document_paths=["src/main.py", "src/utils.py"],
            languages=["python"],
            node_types=["function", "class"],
            nodes_indexed=42,
            is_stale=False,
        )

        assert metadata.document_paths == ["src/main.py", "src/utils.py"]
        assert metadata.languages == ["python"]
        assert metadata.node_types == ["function", "class"]
        assert metadata.nodes_indexed == 42
        assert metadata.is_stale is False

    @pytest.mark.asyncio
    async def test_should_create_metadata_from_empty_repo(self, tmp_path):
        """Test from_repo with a repository containing no files."""
        git_repo = tmp_path / "empty-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "empty-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_empty-repo"

        repo = Repo(settings=mock_settings)

        # Empty async iterator
        async def empty_walk():
            return
            yield  # Make it a generator

        with patch("indexter.models.Walker") as mock_walker_class, patch("indexter.models.store") as mock_store:
            mock_walker_instance = Mock()
            mock_walker_instance.walk = empty_walk
            mock_walker_class.return_value = mock_walker_instance

            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.count_nodes = AsyncMock(return_value=0)

            metadata = await RepoMetadata.from_repo(repo)

            assert metadata.document_paths == []
            assert metadata.languages == []
            assert metadata.node_types == []
            assert metadata.nodes_indexed == 0
            assert metadata.is_stale is False

    @pytest.mark.asyncio
    async def test_should_create_metadata_from_repo_with_files(self, tmp_path):
        """Test from_repo extracts metadata from repository files."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "test-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_test-repo"

        repo = Repo(settings=mock_settings)

        # Create mock documents
        doc1 = Document(
            path="src/main.py",
            content="def hello(): pass",
            metadata=create_doc_metadata(repo="test-repo", repo_path=str(git_repo), hash="hash1"),
        )
        doc2 = Document(
            path="src/utils.py",
            content="class Utils: pass",
            metadata=create_doc_metadata(repo="test-repo", repo_path=str(git_repo), hash="hash2"),
        )

        async def mock_walk():
            yield doc1.path, doc1.content, doc1.metadata
            yield doc2.path, doc2.content, doc2.metadata

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch("indexter.models.store") as mock_store,
        ):
            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            # Mock parser to return nodes
            mock_parser1 = Mock()
            mock_parser1.language = "python"
            mock_parser1.parse.return_value = [
                (
                    "def hello(): pass",
                    NodeMetadata(
                        repo="test-repo",
                        repo_path=str(git_repo),
                        document_path="src/main.py",
                        language="python",
                        node_type="function",
                        node_name="hello",
                        start_byte=0,
                        end_byte=17,
                        start_line=1,
                        end_line=1,
                    ),
                )
            ]

            mock_parser2 = Mock()
            mock_parser2.language = "python"
            mock_parser2.parse.return_value = [
                (
                    "class Utils: pass",
                    NodeMetadata(
                        repo="test-repo",
                        repo_path=str(git_repo),
                        document_path="src/utils.py",
                        language="python",
                        node_type="class",
                        node_name="Utils",
                        start_byte=0,
                        end_byte=18,
                        start_line=1,
                        end_line=1,
                    ),
                )
            ]

            mock_parser_class.side_effect = [mock_parser1, mock_parser2]

            mock_store.get_document_hashes = AsyncMock(return_value={"src/main.py": "hash1", "src/utils.py": "hash2"})
            mock_store.count_nodes = AsyncMock(return_value=2)

            metadata = await RepoMetadata.from_repo(repo)

            assert set(metadata.document_paths) == {"src/main.py", "src/utils.py"}
            assert metadata.languages == ["python"]
            assert set(metadata.node_types) == {"function", "class"}
            assert metadata.nodes_indexed == 2
            assert metadata.is_stale is False

    @pytest.mark.asyncio
    async def test_should_detect_stale_repository(self, tmp_path):
        """Test from_repo detects when repository has changes."""
        git_repo = tmp_path / "stale-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "stale-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_stale-repo"

        repo = Repo(settings=mock_settings)

        doc = Document(
            path="src/changed.py",
            content="def new_code(): pass",
            metadata=create_doc_metadata(repo="stale-repo", repo_path=str(git_repo), hash="new_hash"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch("indexter.models.store") as mock_store,
        ):
            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            mock_parser = Mock()
            mock_parser.language = "python"
            mock_parser.parse.return_value = []
            mock_parser_class.return_value = mock_parser

            # Stored hash is different from local hash
            mock_store.get_document_hashes = AsyncMock(return_value={"src/changed.py": "old_hash"})
            mock_store.count_nodes = AsyncMock(return_value=1)

            metadata = await RepoMetadata.from_repo(repo)

            assert metadata.is_stale is True

    @pytest.mark.asyncio
    async def test_should_handle_parser_errors_gracefully(self, tmp_path, caplog):
        """Test from_repo handles parsing errors without failing."""
        git_repo = tmp_path / "error-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "error-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_error-repo"

        repo = Repo(settings=mock_settings)

        doc = Document(
            path="src/bad.py",
            content="invalid syntax!!!",
            metadata=create_doc_metadata(repo="error-repo", repo_path=str(git_repo), hash="hash1"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch("indexter.models.store") as mock_store,
        ):
            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            mock_parser = Mock()
            mock_parser.language = "python"
            mock_parser.parse.side_effect = Exception("Parse error")
            mock_parser_class.return_value = mock_parser

            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.count_nodes = AsyncMock(return_value=0)

            with caplog.at_level("WARNING"):
                metadata = await RepoMetadata.from_repo(repo)

            assert "Failed to parse" in caplog.text
            assert metadata.document_paths == ["src/bad.py"]
            assert metadata.nodes_indexed == 0

    @pytest.mark.asyncio
    async def test_should_skip_files_without_language(self, tmp_path):
        """Test from_repo skips files when parser has no language attribute."""
        git_repo = tmp_path / "nolang-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "nolang-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_nolang-repo"

        repo = Repo(settings=mock_settings)

        doc = Document(
            path="data.bin",
            content="binary data",
            metadata=create_doc_metadata(repo="nolang-repo", repo_path=str(git_repo), hash="hash1", ext=".bin"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch("indexter.models.store") as mock_store,
        ):
            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            # Parser without language attribute
            mock_parser = Mock(spec=[])  # Empty spec - no language attribute
            mock_parser_class.return_value = mock_parser

            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.count_nodes = AsyncMock(return_value=0)

            metadata = await RepoMetadata.from_repo(repo)

            assert metadata.languages == []
            assert metadata.node_types == []


class TestRepoInit:
    """Test Repo model initialization."""

    def test_should_create_repo_with_settings(self, tmp_path):
        """Test Repo initializes with RepoSettings."""
        git_repo = tmp_path / "test-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "test-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_test-repo"

        repo = Repo(settings=mock_settings)

        assert repo.settings == mock_settings
        assert repo.metadata is None

    def test_should_compute_collection_name_property(self, tmp_path):
        """Test collection_name computed property."""
        git_repo = tmp_path / "my-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.collection_name = "indexter_my-repo"

        repo = Repo(settings=mock_settings)

        assert repo.collection_name == "indexter_my-repo"

    def test_should_compute_name_property(self, tmp_path):
        """Test name computed property."""
        git_repo = tmp_path / "awesome-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "awesome-repo"

        repo = Repo(settings=mock_settings)

        assert repo.name == "awesome-repo"

    def test_should_compute_path_property(self, tmp_path):
        """Test path computed property."""
        git_repo = tmp_path / "path-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.path = git_repo

        repo = Repo(settings=mock_settings)

        assert repo.path == str(git_repo)


class TestRepoInitMethod:
    """Test Repo.init class method."""

    @pytest.mark.asyncio
    async def test_should_initialize_new_repository(self, tmp_path):
        """Test init creates and registers a new repository."""
        git_repo = tmp_path / "new-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[])
            mock_repo_settings_class.save = AsyncMock()

            # Mock the settings constructor
            mock_settings = Mock(spec=RepoSettings)
            mock_settings.name = "new-repo"
            mock_settings.path = git_repo
            mock_repo_settings_class.return_value = mock_settings

            repo = await Repo.init(git_repo)

            assert isinstance(repo, Repo)
            assert repo.settings == mock_settings
            mock_repo_settings_class.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_return_existing_repo_if_already_configured(self, tmp_path, caplog):
        """Test init returns existing Repo if same path already configured."""
        git_repo = tmp_path / "existing-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        existing_settings = Mock(spec=RepoSettings)
        existing_settings.name = "existing-repo"
        existing_settings.path = git_repo

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[existing_settings])

            # Mock the new settings constructor
            new_settings = Mock(spec=RepoSettings)
            new_settings.name = "existing-repo"
            new_settings.path = git_repo
            mock_repo_settings_class.return_value = new_settings

            with caplog.at_level("INFO"):
                repo = await Repo.init(git_repo)

            assert "already configured" in caplog.text
            assert repo.settings == existing_settings

    @pytest.mark.asyncio
    async def test_should_raise_error_for_duplicate_name_different_path(self, tmp_path):
        """Test init raises RepoExistsError for name conflict."""
        git_repo1 = tmp_path / "duplicate-name"
        git_repo1.mkdir()
        (git_repo1 / ".git").mkdir()

        git_repo2 = tmp_path / "other" / "duplicate-name"
        git_repo2.mkdir(parents=True)
        (git_repo2 / ".git").mkdir()

        existing_settings = Mock(spec=RepoSettings)
        existing_settings.name = "duplicate-name"
        existing_settings.path = git_repo1

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[existing_settings])

            new_settings = Mock(spec=RepoSettings)
            new_settings.name = "duplicate-name"
            new_settings.path = git_repo2
            mock_repo_settings_class.return_value = new_settings

            with pytest.raises(RepoExistsError, match="already exists"):
                await Repo.init(git_repo2)


class TestRepoGetMethods:
    """Test Repo.get_one and Repo.get_all class methods."""

    @pytest.mark.asyncio
    async def test_should_get_repo_by_name(self):
        """Test get_one retrieves repository by name."""
        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "target-repo"

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[mock_settings])

            repo = await Repo.get_one("target-repo")

            assert isinstance(repo, Repo)
            assert repo.settings == mock_settings
            assert repo.metadata is None

    @pytest.mark.asyncio
    async def test_should_get_repo_with_metadata(self):
        """Test get_one retrieves repository with metadata when requested."""
        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "meta-repo"
        mock_settings.collection_name = "indexter_meta-repo"

        with (
            patch("indexter.models.RepoSettings") as mock_repo_settings_class,
            patch("indexter.models.RepoMetadata") as mock_metadata_class,
        ):
            mock_repo_settings_class.load = AsyncMock(return_value=[mock_settings])

            mock_metadata = Mock(spec=RepoMetadata)
            mock_metadata_class.from_repo = AsyncMock(return_value=mock_metadata)

            repo = await Repo.get_one("meta-repo", with_metadata=True)

            assert repo.metadata == mock_metadata
            mock_metadata_class.from_repo.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_raise_error_when_repo_not_found(self):
        """Test get_one raises RepoNotFoundError for unknown name."""
        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[])

            with pytest.raises(RepoNotFoundError, match="not found"):
                await Repo.get_one("nonexistent-repo")

    @pytest.mark.asyncio
    async def test_should_get_all_repositories(self):
        """Test get_all retrieves all configured repositories."""
        mock_settings1 = Mock(spec=RepoSettings)
        mock_settings1.name = "repo1"

        mock_settings2 = Mock(spec=RepoSettings)
        mock_settings2.name = "repo2"

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[mock_settings1, mock_settings2])

            repos = await Repo.get_all()

            assert len(repos) == 2
            assert all(isinstance(r, Repo) for r in repos)
            assert repos[0].settings == mock_settings1
            assert repos[1].settings == mock_settings2

    @pytest.mark.asyncio
    async def test_should_get_all_repositories_with_metadata(self):
        """Test get_all retrieves all repositories with metadata."""
        mock_settings1 = Mock(spec=RepoSettings)
        mock_settings1.name = "repo1"
        mock_settings1.collection_name = "indexter_repo1"

        mock_settings2 = Mock(spec=RepoSettings)
        mock_settings2.name = "repo2"
        mock_settings2.collection_name = "indexter_repo2"

        with (
            patch("indexter.models.RepoSettings") as mock_repo_settings_class,
            patch("indexter.models.RepoMetadata") as mock_metadata_class,
        ):
            mock_repo_settings_class.load = AsyncMock(return_value=[mock_settings1, mock_settings2])

            mock_metadata1 = Mock(spec=RepoMetadata)
            mock_metadata2 = Mock(spec=RepoMetadata)
            mock_metadata_class.from_repo = AsyncMock(side_effect=[mock_metadata1, mock_metadata2])

            repos = await Repo.get_all(with_metadata=True)

            assert len(repos) == 2
            assert repos[0].metadata == mock_metadata1
            assert repos[1].metadata == mock_metadata2

    @pytest.mark.asyncio
    async def test_should_return_empty_list_when_no_repos(self):
        """Test get_all returns empty list when no repositories configured."""
        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[])

            repos = await Repo.get_all()

            assert repos == []


class TestRepoRemoveMethods:
    """Test Repo.remove_one and Repo.remove_all class methods."""

    @pytest.mark.asyncio
    async def test_should_remove_repository_and_collection(self, caplog):
        """Test remove_one deletes repository and its data."""
        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "remove-me"
        mock_settings.collection_name = "indexter_remove-me"

        with (
            patch("indexter.models.RepoSettings") as mock_repo_settings_class,
            patch("indexter.models.store") as mock_store,
        ):
            mock_repo_settings_class.load = AsyncMock(
                side_effect=[
                    [mock_settings],  # First call in get_one
                    [mock_settings],  # Second call in remove_one
                ]
            )
            mock_repo_settings_class.save = AsyncMock()
            mock_store.delete_collection = AsyncMock()

            with caplog.at_level("INFO"):
                result = await Repo.remove_one("remove-me")

            assert result is True
            assert "Removed repository" in caplog.text
            mock_store.delete_collection.assert_called_once_with("indexter_remove-me")
            mock_repo_settings_class.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_raise_error_when_removing_nonexistent_repo(self):
        """Test remove_one raises RepoNotFoundError."""
        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[])

            with pytest.raises(RepoNotFoundError):
                await Repo.remove_one("nonexistent")

    @pytest.mark.asyncio
    async def test_should_return_false_when_repo_already_removed(self):
        """Test remove_one returns False on race condition."""
        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "already-gone"
        mock_settings.collection_name = "indexter_already-gone"

        with (
            patch("indexter.models.RepoSettings") as mock_repo_settings_class,
            patch("indexter.models.store") as mock_store,
        ):
            # First call returns the repo, second call returns empty list (race condition)
            mock_repo_settings_class.load = AsyncMock(
                side_effect=[
                    [mock_settings],  # get_one succeeds
                    [],  # Already removed by another process
                ]
            )
            mock_repo_settings_class.save = AsyncMock()
            mock_store.delete_collection = AsyncMock()

            result = await Repo.remove_one("already-gone")

            assert result is False

    @pytest.mark.asyncio
    async def test_should_remove_all_repositories(self, caplog):
        """Test remove_all deletes all repositories and collections."""
        mock_settings1 = Mock(spec=RepoSettings)
        mock_settings1.collection_name = "indexter_repo1"

        mock_settings2 = Mock(spec=RepoSettings)
        mock_settings2.collection_name = "indexter_repo2"

        with (
            patch("indexter.models.RepoSettings") as mock_repo_settings_class,
            patch("indexter.models.store") as mock_store,
        ):
            mock_repo_settings_class.load = AsyncMock(return_value=[mock_settings1, mock_settings2])
            mock_repo_settings_class.save = AsyncMock()
            mock_store.delete_collection = AsyncMock()

            with caplog.at_level("INFO"):
                result = await Repo.remove_all()

            assert result is True
            assert "Removed all repositories" in caplog.text
            assert mock_store.delete_collection.call_count == 2
            mock_repo_settings_class.save.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_should_return_false_when_no_repos_to_remove(self):
        """Test remove_all returns False when no repositories exist."""
        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[])

            result = await Repo.remove_all()

            assert result is False


class TestRepoIndex:
    """Test Repo.index method."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock repository for testing."""
        git_repo = tmp_path / "index-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "index-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_index-repo"
        mock_settings.upsert_batch_size = 100
        mock_settings.max_files = 1000

        return Repo(settings=mock_settings)

    @pytest.mark.asyncio
    async def test_should_perform_full_index(self, mock_repo, caplog):
        """Test index with full=True recreates collection."""

        async def empty_walk():
            return
            yield  # Make it a generator

        with patch("indexter.models.store") as mock_store, patch("indexter.models.Walker") as mock_walker_class:
            mock_store.delete_collection = AsyncMock()
            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = empty_walk
            mock_walker_class.return_value = mock_walker_instance

            with caplog.at_level("INFO"):
                result = await mock_repo.index(full=True)

            assert "full index" in caplog.text.lower()
            mock_store.delete_collection.assert_called_once_with("indexter_index-repo")
            assert isinstance(result, IndexResult)

    @pytest.mark.asyncio
    async def test_should_perform_incremental_index_for_new_file(self, mock_repo):
        """Test incremental indexing adds new files."""
        doc = Document(
            path="src/new.py",
            content="def new_func(): pass",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="newhash"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})  # No existing files
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            # Mock parser
            node_metadata = NodeMetadata(
                repo="index-repo",
                repo_path=str(mock_repo.settings.path),
                document_path="src/new.py",
                language="python",
                node_type="function",
                node_name="new_func",
                start_byte=0,
                end_byte=20,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            mock_parser.parse.return_value = [("def new_func(): pass", node_metadata)]
            mock_parser_class.return_value = mock_parser

            result = await mock_repo.index()

            assert result.documents_checked == 1
            assert result.nodes_added == 1
            assert result.nodes_updated == 0
            assert "src/new.py" in result.documents_indexed
            mock_store.upsert_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_detect_and_update_modified_files(self, mock_repo):
        """Test incremental indexing updates modified files."""
        doc = Document(
            path="src/modified.py",
            content="def updated_func(): pass",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="newhash"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_store.ensure_collection = AsyncMock()
            # File exists with different hash
            mock_store.get_document_hashes = AsyncMock(return_value={"src/modified.py": "oldhash"})
            mock_store.delete_by_document_paths = AsyncMock()
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="index-repo",
                repo_path=str(mock_repo.settings.path),
                document_path="src/modified.py",
                language="python",
                node_type="function",
                node_name="updated_func",
                start_byte=0,
                end_byte=24,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            mock_parser.parse.return_value = [("def updated_func(): pass", node_metadata)]
            mock_parser_class.return_value = mock_parser

            result = await mock_repo.index()

            assert result.documents_checked == 1
            assert result.nodes_added == 0
            assert result.nodes_updated == 1
            mock_store.delete_by_document_paths.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_delete_removed_files(self, mock_repo):
        """Test incremental indexing deletes nodes for removed files."""

        async def mock_walk():
            return
            yield  # Make it a generator

        with patch("indexter.models.store") as mock_store, patch("indexter.models.Walker") as mock_walker_class:
            mock_store.ensure_collection = AsyncMock()
            # File exists in store but not on disk
            mock_store.get_document_hashes = AsyncMock(return_value={"src/deleted.py": "hash"})
            mock_store.delete_by_document_paths = AsyncMock()
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            result = await mock_repo.index()

            assert result.documents_deleted == ["src/deleted.py"]
            mock_store.delete_by_document_paths.assert_called_once_with("indexter_index-repo", ["src/deleted.py"])

    @pytest.mark.asyncio
    async def test_should_skip_unchanged_files(self, mock_repo):
        """Test incremental indexing skips unchanged files."""
        doc = Document(
            path="src/unchanged.py",
            content="def same_func(): pass",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="samehash"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_store.ensure_collection = AsyncMock()
            # Same hash - file unchanged
            mock_store.get_document_hashes = AsyncMock(return_value={"src/unchanged.py": "samehash"})
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            result = await mock_repo.index()

            assert result.documents_checked == 1
            assert result.nodes_added == 0
            assert result.nodes_updated == 0
            # Parser should not be called for unchanged files
            mock_parser_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_respect_max_files_limit(self, mock_repo, caplog):
        """Test index respects max_files setting."""
        mock_repo.settings.max_files = 2

        # Create 3 documents
        async def mock_walk():
            for i in range(3):
                doc = Document(
                    path=f"src/file{i}.py",
                    content=f"def func{i}(): pass",
                    metadata=create_doc_metadata(
                        repo="index-repo",
                        repo_path=str(mock_repo.settings.path),
                        hash=f"hash{i}",
                    ),
                )
                yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            # Mock parser
            mock_parser = Mock()
            node_metadata = NodeMetadata(
                repo="index-repo",
                repo_path=str(mock_repo.settings.path),
                document_path="file.py",
                language="python",
                node_type="function",
                node_name="func",
                start_byte=0,
                end_byte=20,
                start_line=1,
                end_line=1,
            )
            mock_parser.parse.return_value = [("def func(): pass", node_metadata)]
            mock_parser_class.return_value = mock_parser

            with caplog.at_level("WARNING"):
                result = await mock_repo.index()

            assert "limited to" in caplog.text.lower()
            assert result.skipped_documents == 1

    @pytest.mark.asyncio
    async def test_should_batch_upsert_operations(self, mock_repo):
        """Test index batches node upserts according to batch size."""
        mock_repo.settings.upsert_batch_size = 2

        # Create two files with 1 node each - first batch will have 2 nodes, second will have 1
        doc1 = Document(
            path="src/file1.py",
            content="def f1(): pass",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="hash1"),
        )
        doc2 = Document(
            path="src/file2.py",
            content="def f2(): pass",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="hash2"),
        )
        doc3 = Document(
            path="src/file3.py",
            content="def f3(): pass",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="hash3"),
        )

        async def mock_walk():
            yield doc1.path, doc1.content, doc1.metadata
            yield doc2.path, doc2.content, doc2.metadata
            yield doc3.path, doc3.content, doc3.metadata

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            # Each parser call returns 1 node
            def create_parser(doc):
                mock_parser = Mock()
                node_metadata = NodeMetadata(
                    repo="index-repo",
                    repo_path=str(mock_repo.settings.path),
                    document_path=doc.path,
                    language="python",
                    node_type="function",
                    node_name=doc.path.split("/")[-1].replace(".py", ""),
                    start_byte=0,
                    end_byte=20,
                    start_line=1,
                    end_line=1,
                )
                mock_parser.parse.return_value = [(doc.content, node_metadata)]
                return mock_parser

            parsers = [create_parser(doc1), create_parser(doc2), create_parser(doc3)]
            mock_parser_class.side_effect = parsers

            await mock_repo.index()

            # Should call upsert twice: once for first 2 nodes, once for remaining 1
            assert mock_store.upsert_nodes.call_count == 2

    @pytest.mark.asyncio
    async def test_should_handle_parsing_errors(self, mock_repo, caplog):
        """Test index handles parsing errors gracefully."""
        doc = Document(
            path="src/error.py",
            content="invalid syntax",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="hash"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            mock_parser = Mock()
            mock_parser.parse.side_effect = Exception("Parse failed")
            mock_parser_class.return_value = mock_parser

            with caplog.at_level("WARNING"):
                result = await mock_repo.index()

            assert len(result.errors) == 1
            assert "Failed to parse" in result.errors[0]
            assert "Parse failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_should_create_placeholder_for_empty_parse_result(self, mock_repo):
        """Test index creates placeholder node when parser returns no nodes."""
        doc = Document(
            path="src/empty.py",
            content="# Just a comment",
            metadata=create_doc_metadata(repo="index-repo", repo_path=str(mock_repo.settings.path), hash="hash"),
        )

        async def mock_walk():
            yield doc.path, doc.content, doc.metadata

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch("indexter.models.Node") as mock_node_class,
        ):
            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.upsert_nodes = AsyncMock()

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            # Parser returns no nodes
            mock_parser = Mock()
            mock_parser.parse.return_value = []
            mock_parser_class.return_value = mock_parser

            # Mock placeholder node
            placeholder_node = Mock(spec=Node)
            placeholder_node.metadata = Mock()
            placeholder_node.metadata.node_type = "__PLACEHOLDER__"
            mock_node_class.placeholder.return_value = placeholder_node

            result = await mock_repo.index()

            mock_node_class.placeholder.assert_called_once()
            mock_store.upsert_nodes.assert_called_once()
            # Placeholder should not be counted in documents_indexed
            assert "src/empty.py" not in result.documents_indexed

    @pytest.mark.asyncio
    async def test_should_calculate_duration_and_timestamp(self, mock_repo):
        """Test index result includes duration and timestamp."""

        async def empty_walk():
            return
            yield  # Make it a generator

        with (
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.datetime") as mock_datetime,
        ):
            start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            end_time = datetime(2024, 1, 1, 12, 0, 5, tzinfo=UTC)

            mock_datetime.now.side_effect = [start_time, end_time]
            mock_datetime.UTC = UTC

            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})

            mock_walker_instance = Mock()
            mock_walker_instance.walk = empty_walk
            mock_walker_class.return_value = mock_walker_instance

            result = await mock_repo.index()

            assert result.indexed_at == end_time
            assert result.duration == 5.0


class TestRepoSearch:
    """Test Repo.search method."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock repository for testing."""
        git_repo = tmp_path / "search-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "search-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_search-repo"
        mock_settings.top_k = 10

        return Repo(settings=mock_settings)

    @pytest.mark.asyncio
    async def test_should_search_with_default_limit(self, mock_repo):
        """Test search uses repository's top_k as default limit."""
        mock_results = Mock(spec=SearchResults)

        with patch("indexter.models.store") as mock_store:
            mock_store.search = AsyncMock(return_value=mock_results)

            await mock_repo.search("test query")

            mock_store.search.assert_called_once()
            call_kwargs = mock_store.search.call_args[1]
            assert call_kwargs["limit"] == 10
            assert call_kwargs["query"] == "test query"

    @pytest.mark.asyncio
    async def test_should_search_with_custom_limit(self, mock_repo):
        """Test search accepts custom limit parameter."""
        mock_results = Mock(spec=SearchResults)

        with patch("indexter.models.store") as mock_store:
            mock_store.search = AsyncMock(return_value=mock_results)

            await mock_repo.search("test query", limit=5)

            call_kwargs = mock_store.search.call_args[1]
            assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_should_search_with_all_filters(self, mock_repo):
        """Test search passes all filter parameters to store."""
        mock_results = Mock(spec=SearchResults)

        with patch("indexter.models.store") as mock_store:
            mock_store.search = AsyncMock(return_value=mock_results)

            await mock_repo.search(
                query="find functions",
                document_path="src/main.py",
                language="python",
                node_type="function",
                node_name="my_func",
                parent_scope="MyClass",
                has_documentation=True,
                limit=20,
            )

            call_kwargs = mock_store.search.call_args[1]
            assert call_kwargs["collection_name"] == "indexter_search-repo"
            assert call_kwargs["query"] == "find functions"
            assert call_kwargs["limit"] == 20
            assert call_kwargs["document_path"] == "src/main.py"
            assert call_kwargs["language"] == "python"
            assert call_kwargs["node_type"] == "function"
            assert call_kwargs["node_name"] == "my_func"
            assert call_kwargs["parent_scope"] == "MyClass"
            assert call_kwargs["has_documentation"] is True

    @pytest.mark.asyncio
    async def test_should_assign_repo_info_to_results(self, mock_repo):
        """Test search assigns repo name and path to results."""
        mock_results = Mock(spec=SearchResults)
        mock_results.repo = None
        mock_results.repo_path = None

        with patch("indexter.models.store") as mock_store:
            mock_store.search = AsyncMock(return_value=mock_results)

            result = await mock_repo.search("test")

            assert result.repo == "search-repo"
            assert result.repo_path == str(mock_repo.settings.path)


class TestRepoIntegration:
    """Integration tests for Repo workflows."""

    @pytest.mark.asyncio
    async def test_should_complete_full_workflow(self, tmp_path):
        """Test complete workflow: init -> index -> search."""
        # Create a test repository
        git_repo = tmp_path / "workflow-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        # Mock all dependencies
        with (
            patch("indexter.models.RepoSettings") as mock_repo_settings_class,
            patch("indexter.models.store") as mock_store,
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            # Setup RepoSettings
            mock_settings = Mock(spec=RepoSettings)
            mock_settings.name = "workflow-repo"
            mock_settings.path = git_repo
            mock_settings.collection_name = "indexter_workflow-repo"
            mock_settings.upsert_batch_size = 100
            mock_settings.max_files = 1000
            mock_settings.top_k = 10

            mock_repo_settings_class.load = AsyncMock(return_value=[])
            mock_repo_settings_class.save = AsyncMock()
            mock_repo_settings_class.return_value = mock_settings

            # Init
            repo = await Repo.init(git_repo)
            assert repo.name == "workflow-repo"

            # Setup for indexing
            doc = Document(
                path="src/app.py",
                content="def main(): pass",
                metadata=create_doc_metadata(repo="workflow-repo", repo_path=str(git_repo), hash="hash1"),
            )

            async def mock_walk():
                yield doc.path, doc.content, doc.metadata

            mock_walker_instance = Mock()
            mock_walker_instance.walk = mock_walk
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="workflow-repo",
                repo_path=str(git_repo),
                document_path="src/app.py",
                language="python",
                node_type="function",
                node_name="main",
                start_byte=0,
                end_byte=16,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            mock_parser.parse.return_value = [("def main(): pass", node_metadata)]
            mock_parser_class.return_value = mock_parser

            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})
            mock_store.upsert_nodes = AsyncMock()

            # Index
            index_result = await repo.index()
            assert index_result.nodes_added == 1

            # Search
            mock_search_results = Mock(spec=SearchResults)
            mock_store.search = AsyncMock(return_value=mock_search_results)

            search_results = await repo.search("main function")
            assert search_results.repo == "workflow-repo"

    @pytest.mark.asyncio
    async def test_should_handle_multiple_repositories(self, tmp_path):
        """Test managing multiple repositories simultaneously."""
        # Create two repositories
        repo1_path = tmp_path / "repo1"
        repo1_path.mkdir()
        (repo1_path / ".git").mkdir()

        repo2_path = tmp_path / "repo2"
        repo2_path.mkdir()
        (repo2_path / ".git").mkdir()

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            # Setup settings
            settings1 = Mock(spec=RepoSettings)
            settings1.name = "repo1"
            settings1.path = repo1_path
            settings1.collection_name = "indexter_repo1"

            settings2 = Mock(spec=RepoSettings)
            settings2.name = "repo2"
            settings2.path = repo2_path
            settings2.collection_name = "indexter_repo2"

            mock_repo_settings_class.load = AsyncMock(return_value=[settings1, settings2])

            # Get all repos
            repos = await Repo.get_all()

            assert len(repos) == 2
            assert repos[0].name == "repo1"
            assert repos[1].name == "repo2"

    @pytest.mark.asyncio
    async def test_should_handle_full_reindex(self, tmp_path):
        """Test full re-indexing workflow."""
        git_repo = tmp_path / "reindex-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "reindex-repo"
        mock_settings.path = git_repo
        mock_settings.collection_name = "indexter_reindex-repo"
        mock_settings.upsert_batch_size = 100
        mock_settings.max_files = 1000

        repo = Repo(settings=mock_settings)

        async def empty_walk():
            return
            yield  # Make it a generator

        with patch("indexter.models.store") as mock_store, patch("indexter.models.Walker") as mock_walker_class:
            mock_store.delete_collection = AsyncMock()
            mock_store.ensure_collection = AsyncMock()
            mock_store.get_document_hashes = AsyncMock(return_value={})

            mock_walker_instance = Mock()
            mock_walker_instance.walk = empty_walk
            mock_walker_class.return_value = mock_walker_instance

            # Full index should delete collection first
            await repo.index(full=True)

            mock_store.delete_collection.assert_called_once_with("indexter_reindex-repo")
            mock_store.ensure_collection.assert_called_once()
