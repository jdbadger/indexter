"""Comprehensive tests for the Repo and RepoMetadata models."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from indexter.config import RepoSettings
from indexter.exceptions import RepoExistsError, RepoNotFoundError
from indexter.models import Repo, RepoMetadata
from indexter.parser.models import NodeMetadata
from indexter.store.models import IndexResult, SearchResults
from indexter.walker.models import DocumentMetadata


def create_mock_store():
    """Create a mock VectorStore instance with common async methods."""
    mock_store = Mock()
    mock_store.count_nodes = AsyncMock(return_value=0)
    mock_store.ensure_collection = AsyncMock()
    mock_store.delete_collection = AsyncMock()
    mock_store.upsert_nodes = AsyncMock()
    mock_store.delete_by_hashes = AsyncMock()
    mock_store.search = AsyncMock()
    return mock_store


def create_doc_metadata(
    repo="test-repo",
    repo_path="/tmp/test",
    ext=".py",
    size_bytes=100,
    mtime=1234567890.0,
):
    """Create DocumentMetadata with all required fields."""
    return DocumentMetadata(
        repo=repo,
        repo_path=repo_path,
        ext=ext,
        size_bytes=size_bytes,
        mtime=mtime,
    )


class TestRepoMetadata:
    """Test RepoMetadata model."""

    def test_should_create_repo_metadata_with_defaults(self):
        """Test RepoMetadata initializes with default empty values."""
        metadata = RepoMetadata()

        assert metadata.document_paths == []
        assert metadata.documents == 0
        assert metadata.node_types == []
        assert metadata.nodes == 0
        assert metadata.languages == []

    def test_should_create_repo_metadata_with_values(self):
        """Test RepoMetadata initializes with provided values."""
        metadata = RepoMetadata(
            document_paths=["src/main.py", "src/utils.py"],
            documents=2,
            languages=["python"],
            node_types=["function", "class"],
            nodes=42,
        )

        assert metadata.document_paths == ["src/main.py", "src/utils.py"]
        assert metadata.documents == 2
        assert metadata.languages == ["python"]
        assert metadata.node_types == ["function", "class"]
        assert metadata.nodes == 42

    def test_should_return_empty_message_for_no_documents(self):
        """Test document_tree returns '(no documents)' when document_paths is empty."""
        metadata = RepoMetadata(document_paths=[])

        assert metadata.document_tree == "(no documents)"

    def test_should_render_single_file_at_root(self):
        """Test document_tree renders a single file with simple connector."""
        metadata = RepoMetadata(document_paths=["README.md"])

        expected = "└── README.md"
        assert metadata.document_tree == expected

    def test_should_render_multiple_files_at_root(self):
        """Test document_tree renders multiple files at root level."""
        metadata = RepoMetadata(
            document_paths=["README.md", "setup.py", "main.py"],
        )

        expected = "├── README.md\n├── main.py\n└── setup.py"
        assert metadata.document_tree == expected

    def test_should_render_nested_directory_structure(self):
        """Test document_tree renders nested directories with proper hierarchy."""
        metadata = RepoMetadata(
            document_paths=[
                "src/main.py",
                "src/utils.py",
                "tests/test_main.py",
            ],
        )

        expected = "├── src/\n│   ├── main.py\n│   └── utils.py\n└── tests/\n    └── test_main.py"
        assert metadata.document_tree == expected

    def test_should_add_trailing_slash_to_directories(self):
        """Test document_tree adds trailing / to directories."""
        metadata = RepoMetadata(
            document_paths=[
                "src/core/models.py",
                "src/core/utils.py",
            ],
        )

        expected = "└── src/\n    └── core/\n        ├── models.py\n        └── utils.py"
        assert metadata.document_tree == expected

    def test_should_render_complex_project_structure(self):
        """Test document_tree renders a complex project structure correctly."""
        metadata = RepoMetadata(
            document_paths=[
                "README.md",
                "pyproject.toml",
                "src/indexter/__init__.py",
                "src/indexter/config.py",
                "src/indexter/models.py",
                "src/indexter/cli/cli.py",
                "src/indexter/mcp/server.py",
                "tests/test_config.py",
                "tests/test_models.py",
            ],
        )

        expected = (
            "├── README.md\n"
            "├── pyproject.toml\n"
            "├── src/\n"
            "│   └── indexter/\n"
            "│       ├── __init__.py\n"
            "│       ├── cli/\n"
            "│       │   └── cli.py\n"
            "│       ├── config.py\n"
            "│       ├── mcp/\n"
            "│       │   └── server.py\n"
            "│       └── models.py\n"
            "└── tests/\n"
            "    ├── test_config.py\n"
            "    └── test_models.py"
        )
        assert metadata.document_tree == expected

    def test_should_handle_mixed_depth_paths(self):
        """Test document_tree handles paths at different nesting levels."""
        metadata = RepoMetadata(
            document_paths=[
                "root.py",
                "a/b.py",
                "a/c/d.py",
                "a/c/e/f.py",
            ],
        )

        expected = (
            "├── a/\n│   ├── b.py\n│   └── c/\n│       ├── d.py\n│       └── e/\n│           └── f.py\n└── root.py"
        )
        assert metadata.document_tree == expected

    def test_should_sort_paths_alphabetically(self):
        """Test document_tree sorts paths alphabetically."""
        metadata = RepoMetadata(
            document_paths=[
                "z.py",
                "a.py",
                "m.py",
            ],
        )

        expected = "├── a.py\n├── m.py\n└── z.py"
        assert metadata.document_tree == expected


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
        mock_settings.max_files = 1000

        repo = Repo(settings=mock_settings)

        assert repo.settings == mock_settings

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

    def test_should_compute_metadata_property(self, tmp_path):
        """Test metadata computed property aggregates file information."""
        git_repo = tmp_path / "meta-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "meta-repo"
        mock_settings.path = git_repo
        mock_settings.max_files = 1000

        repo = Repo(settings=mock_settings)

        doc_metadata = create_doc_metadata(
            repo="meta-repo",
            repo_path=str(git_repo),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([("src/main.py", "def hello(): pass", doc_metadata)])
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="meta-repo",
                repo_path=str(git_repo),
                document_path="src/main.py",
                document_hash="abc123",
                language="python",
                node_type="function",
                node_name="hello",
                start_byte=0,
                end_byte=17,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            mock_parser.language = "python"
            mock_parser.parse.return_value = [("def hello(): pass", node_metadata)]
            mock_parser_class.return_value = mock_parser

            metadata = repo.metadata

            assert metadata.documents == 1
            assert "src/main.py" in metadata.document_paths
            assert "python" in metadata.languages
            assert "function" in metadata.node_types
            assert metadata.nodes == 1

    def test_should_return_cache_manager(self, tmp_path):
        """Test cache property returns CacheManager instance."""
        git_repo = tmp_path / "cache-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "cache-repo"
        mock_settings.path = git_repo

        repo = Repo(settings=mock_settings)

        from indexter.cache import CacheManager

        assert isinstance(repo.cache, CacheManager)

    def test_should_respect_max_files_limit_in_metadata(self, tmp_path):
        """Test metadata property stops at max_files limit."""
        git_repo = tmp_path / "max-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "max-repo"
        mock_settings.path = git_repo
        mock_settings.max_files = 1  # Only process 1 file

        repo = Repo(settings=mock_settings)

        doc_metadata = create_doc_metadata(
            repo="max-repo",
            repo_path=str(git_repo),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_walker_instance = Mock()
            # Provide 3 files, but max_files is 1
            mock_walker_instance.walk.return_value = iter(
                [
                    ("file1.py", "def a(): pass", doc_metadata),
                    ("file2.py", "def b(): pass", doc_metadata),
                    ("file3.py", "def c(): pass", doc_metadata),
                ]
            )
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="max-repo",
                repo_path=str(git_repo),
                document_path="file1.py",
                document_hash="abc123",
                language="python",
                node_type="function",
                node_name="a",
                start_byte=0,
                end_byte=14,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            mock_parser.language = "python"
            mock_parser.parse.return_value = [("def a(): pass", node_metadata)]
            mock_parser_class.return_value = mock_parser

            metadata = repo.metadata

            # Should only have processed 1 file due to max_files limit
            assert metadata.documents == 1

    def test_should_continue_on_parse_error_in_metadata(self, tmp_path):
        """Test metadata property continues after parse errors."""
        git_repo = tmp_path / "error-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "error-repo"
        mock_settings.path = git_repo
        mock_settings.max_files = 10

        repo = Repo(settings=mock_settings)

        doc_metadata = create_doc_metadata(
            repo="error-repo",
            repo_path=str(git_repo),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
        ):
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter(
                [
                    ("bad.py", "invalid python {{{", doc_metadata),
                    ("good.py", "def ok(): pass", doc_metadata),
                ]
            )
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="error-repo",
                repo_path=str(git_repo),
                document_path="good.py",
                document_hash="abc123",
                language="python",
                node_type="function",
                node_name="ok",
                start_byte=0,
                end_byte=14,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            mock_parser.language = "python"
            # First call raises, second succeeds
            mock_parser.parse.side_effect = [
                Exception("Parse error"),
                [("def ok(): pass", node_metadata)],
            ]
            mock_parser_class.return_value = mock_parser

            metadata = repo.metadata

            # Should have processed 1 file successfully (skipped the bad one)
            assert metadata.documents == 1
            assert "good.py" in metadata.document_paths


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

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(
                side_effect=[
                    [mock_settings],  # First call in get_one
                    [mock_settings],  # Second call in remove_one
                ]
            )
            mock_repo_settings_class.save = AsyncMock()
            mock_store = create_mock_store()
            mock_store.delete_collection = AsyncMock()

            with caplog.at_level("INFO"):
                result = await Repo.remove_one("remove-me", mock_store)

            assert result is True
            assert "Removed repository" in caplog.text
            mock_store.delete_collection.assert_called_once_with("indexter_remove-me")
            mock_repo_settings_class.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_raise_error_when_removing_nonexistent_repo(self):
        """Test remove_one raises RepoNotFoundError."""
        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[])

            mock_store = create_mock_store()
            with pytest.raises(RepoNotFoundError):
                await Repo.remove_one("nonexistent", mock_store)

    @pytest.mark.asyncio
    async def test_should_return_false_when_repo_already_removed(self):
        """Test remove_one returns False on race condition."""
        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "already-gone"
        mock_settings.collection_name = "indexter_already-gone"

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(
                side_effect=[
                    [mock_settings],  # get_one succeeds
                    [],  # Already removed by another process
                ]
            )
            mock_repo_settings_class.save = AsyncMock()
            mock_store = create_mock_store()
            mock_store.delete_collection = AsyncMock()

            result = await Repo.remove_one("already-gone", mock_store)

            assert result is False

    @pytest.mark.asyncio
    async def test_should_remove_all_repositories(self, caplog, tmp_path):
        """Test remove_all deletes all repositories and collections."""
        # Create git repos
        repo1_path = tmp_path / "repo1"
        repo1_path.mkdir()
        (repo1_path / ".git").mkdir()

        repo2_path = tmp_path / "repo2"
        repo2_path.mkdir()
        (repo2_path / ".git").mkdir()

        mock_settings1 = Mock(spec=RepoSettings)
        mock_settings1.name = "repo1"
        mock_settings1.path = repo1_path
        mock_settings1.collection_name = "indexter_repo1"

        mock_settings2 = Mock(spec=RepoSettings)
        mock_settings2.name = "repo2"
        mock_settings2.path = repo2_path
        mock_settings2.collection_name = "indexter_repo2"

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[mock_settings1, mock_settings2])
            mock_repo_settings_class.save = AsyncMock()
            mock_store = create_mock_store()
            mock_store.delete_collection = AsyncMock()

            with caplog.at_level("INFO"):
                result = await Repo.remove_all(mock_store)

            assert result is True
            assert "Removed all repositories" in caplog.text
            assert mock_store.delete_collection.call_count == 2
            mock_repo_settings_class.save.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_should_return_false_when_no_repos_to_remove(self):
        """Test remove_all returns False when no repositories exist."""
        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            mock_repo_settings_class.load = AsyncMock(return_value=[])

            mock_store = create_mock_store()
            result = await Repo.remove_all(mock_store)

            assert result is False


class TestRepoBuildHashmap:
    """Test Repo._get_hashmap method."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock repository for testing."""
        git_repo = tmp_path / "hashmap-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "hashmap-repo"
        mock_settings.path = git_repo
        mock_settings.max_files = 1000

        return Repo(settings=mock_settings)

    def test_should_return_empty_dict_for_empty_repo(self, mock_repo):
        """Test _get_hashmap returns empty dict when no files."""
        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([])
            mock_walker_class.return_value = mock_walker_instance

            hashmap = mock_repo._get_hashmap()

            assert hashmap == {}

    def test_should_get_hashmap_with_single_file(self, mock_repo):
        """Test _get_hashmap creates correct structure for a single file."""
        doc_metadata = create_doc_metadata(
            repo="hashmap-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([("src/main.py", "def hello(): pass", doc_metadata)])
            mock_walker_class.return_value = mock_walker_instance

            hashmap = mock_repo._get_hashmap()

            assert "src/main.py" in hashmap
            # Value should be a single hash string (not a list)
            assert isinstance(hashmap["src/main.py"], str)
            assert len(hashmap["src/main.py"]) == 64

    def test_should_get_hashmap_with_multiple_files(self, mock_repo):
        """Test _get_hashmap creates a single hash per document."""
        doc_metadata = create_doc_metadata(
            repo="hashmap-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter(
                [
                    ("src/main.py", "def foo(): pass\ndef bar(): pass", doc_metadata),
                    ("src/utils.py", "def helper(): pass", doc_metadata),
                ]
            )
            mock_walker_class.return_value = mock_walker_instance

            hashmap = mock_repo._get_hashmap()

            assert len(hashmap) == 2
            assert "src/main.py" in hashmap
            assert "src/utils.py" in hashmap
            # Each value is a single hash string
            assert isinstance(hashmap["src/main.py"], str)
            assert isinstance(hashmap["src/utils.py"], str)
            # Different content should yield different hashes
            assert hashmap["src/main.py"] != hashmap["src/utils.py"]

    def test_should_respect_max_files_limit(self, mock_repo):
        """Test _get_hashmap respects max_files setting."""
        mock_repo.settings.max_files = 2

        doc_metadata = create_doc_metadata(
            repo="hashmap-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with patch("indexter.models.Walker") as mock_walker_class:
            # Walk returns 5 files, but max_files is 2
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter(
                [(f"src/file{i}.py", f"def f{i}(): pass", doc_metadata) for i in range(5)]
            )
            mock_walker_class.return_value = mock_walker_instance

            hashmap = mock_repo._get_hashmap()

            # Only 2 files should be processed
            assert len(hashmap) == 2

    def test_should_handle_errors_gracefully(self, mock_repo):
        """Test _get_hashmap continues when Document construction fails for a file."""
        doc_metadata = create_doc_metadata(
            repo="hashmap-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Document") as mock_document_class,
        ):
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter(
                [
                    ("src/bad.py", "invalid syntax", doc_metadata),
                    ("src/good.py", "def hello(): pass", doc_metadata),
                ]
            )
            mock_walker_class.return_value = mock_walker_instance

            # First Document construction raises, second succeeds
            mock_good_doc = Mock()
            mock_good_doc.path = "src/good.py"
            mock_good_doc.hash = "good_hash"
            mock_document_class.side_effect = [Exception("Construction error"), mock_good_doc]

            hashmap = mock_repo._get_hashmap()

            # Only the good file should be in the hashmap
            assert "src/bad.py" not in hashmap
            assert "src/good.py" in hashmap

    def test_should_hash_document_with_empty_parsed_content(self, mock_repo):
        """Test _get_hashmap hashes documents regardless of parse result content."""
        doc_metadata = create_doc_metadata(
            repo="hashmap-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with patch("indexter.models.Walker") as mock_walker_class:
            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([("src/empty.py", "# just a comment", doc_metadata)])
            mock_walker_class.return_value = mock_walker_instance

            hashmap = mock_repo._get_hashmap()

            assert "src/empty.py" in hashmap
            assert isinstance(hashmap["src/empty.py"], str)
            assert len(hashmap["src/empty.py"]) == 64


class TestRepoHashmapCache:
    """Test Repo._get_cached_hashmap and Repo._set_hashmap methods."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock repository with a mock cache manager."""
        git_repo = tmp_path / "cache-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "cache-repo"
        mock_settings.path = git_repo
        mock_settings.max_files = 1000

        repo = Repo(settings=mock_settings)
        return repo

    def test_get_cached_hashmap_should_return_empty_dict_when_no_cache(self, mock_repo):
        """Test _get_cached_hashmap returns empty dict when cache doesn't exist."""
        with patch.object(Repo, "cache") as mock_cache:
            mock_cache.get.return_value = None

            hashmap = mock_repo._get_cached_hashmap()

            assert hashmap == {}
            mock_cache.get.assert_called_once_with("hashmap")

    def test_get_cached_hashmap_should_return_parsed_json(self, mock_repo):
        """Test _get_cached_hashmap returns parsed JSON from cache."""
        cached_data = {"src/main.py": "hash1"}

        with patch.object(Repo, "cache") as mock_cache:
            mock_cache.get.return_value = json.dumps(cached_data)

            hashmap = mock_repo._get_cached_hashmap()

            assert hashmap == cached_data

    def test_set_hashmap_should_persist_to_cache(self, mock_repo):
        """Test _set_hashmap writes JSON to cache."""
        hashmap = {"src/main.py": "hash1"}

        with patch.object(Repo, "cache") as mock_cache:
            mock_repo._set_hashmap(hashmap)

            mock_cache.set.assert_called_once_with("hashmap", json.dumps(hashmap))

    def test_hashmap_roundtrip(self, mock_repo):
        """Test hashmap can be stored and retrieved correctly."""
        original_hashmap = {
            "src/main.py": "abc123",
            "src/utils.py": "ghi789",
        }

        # Simulate cache storage
        stored_data = None

        with patch.object(Repo, "cache") as mock_cache:

            def mock_set(key, value):
                nonlocal stored_data
                stored_data = value

            def mock_get(key):
                return stored_data

            mock_cache.set.side_effect = mock_set
            mock_cache.get.side_effect = mock_get

            mock_repo._set_hashmap(original_hashmap)
            retrieved_hashmap = mock_repo._get_cached_hashmap()

            assert retrieved_hashmap == original_hashmap


class TestRepoIsStale:
    """Test Repo.is_stale computed property."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock repository for testing."""
        git_repo = tmp_path / "stale-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        mock_settings = Mock(spec=RepoSettings)
        mock_settings.name = "stale-repo"
        mock_settings.path = git_repo
        mock_settings.max_files = 1000

        return Repo(settings=mock_settings)

    def test_should_return_true_when_no_cache_exists(self, mock_repo):
        """Test is_stale returns True when cached hashmap is empty."""
        with (
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
            patch.object(Repo, "_get_hashmap", return_value={"src/main.py": "hash1"}),
        ):
            assert mock_repo.is_stale is True

    def test_should_return_true_when_hashmaps_differ(self, mock_repo):
        """Test is_stale returns True when current and cached hashmaps differ."""
        cached = {"src/main.py": "old_hash"}
        current = {"src/main.py": "new_hash"}

        with (
            patch.object(Repo, "_get_cached_hashmap", return_value=cached),
            patch.object(Repo, "_get_hashmap", return_value=current),
        ):
            assert mock_repo.is_stale is True

    def test_should_return_false_when_hashmaps_match(self, mock_repo):
        """Test is_stale returns False when current and cached hashmaps are equal."""
        same_hashmap = {"src/main.py": "hash1"}

        with (
            patch.object(Repo, "_get_cached_hashmap", return_value=same_hashmap),
            patch.object(Repo, "_get_hashmap", return_value=same_hashmap.copy()),
        ):
            assert mock_repo.is_stale is False

    def test_should_detect_new_file(self, mock_repo):
        """Test is_stale returns True when a new file is added."""
        cached = {"src/main.py": "hash1"}
        current = {"src/main.py": "hash1", "src/new.py": "hash2"}

        with (
            patch.object(Repo, "_get_cached_hashmap", return_value=cached),
            patch.object(Repo, "_get_hashmap", return_value=current),
        ):
            assert mock_repo.is_stale is True

    def test_should_detect_deleted_file(self, mock_repo):
        """Test is_stale returns True when a file is deleted."""
        cached = {"src/main.py": "hash1", "src/old.py": "hash2"}
        current = {"src/main.py": "hash1"}

        with (
            patch.object(Repo, "_get_cached_hashmap", return_value=cached),
            patch.object(Repo, "_get_hashmap", return_value=current),
        ):
            assert mock_repo.is_stale is True


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
        """Test index with full=True recreates collection and clears cache."""
        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch.object(Repo, "_get_hashmap", return_value={}),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
            patch.object(Repo, "_set_hashmap"),
            patch.object(Repo, "cache") as mock_cache,
        ):
            mock_store = create_mock_store()

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([])
            mock_walker_class.return_value = mock_walker_instance

            with caplog.at_level("INFO"):
                result = await mock_repo.index(mock_store, full=True)

            assert "full index" in caplog.text.lower()
            mock_store.delete_collection.assert_called_once_with("indexter_index-repo")
            mock_cache.delete.assert_called_once_with("hashmap")
            assert isinstance(result, IndexResult)

    @pytest.mark.asyncio
    async def test_should_skip_indexing_when_hashmap_unchanged(self, mock_repo, caplog):
        """Test index returns early when hashmap matches cached hashmap."""
        same_hashmap = {"src/file.py": "hash1"}

        with (
            patch.object(Repo, "_get_hashmap", return_value=same_hashmap),
            patch.object(Repo, "_get_cached_hashmap", return_value=same_hashmap.copy()),
        ):
            mock_store = create_mock_store()

            with caplog.at_level("INFO"):
                result = await mock_repo.index(mock_store)

            assert "No changes detected" in caplog.text
            # ensure_collection is called, but upsert/delete are not
            mock_store.ensure_collection.assert_called_once()
            mock_store.upsert_nodes.assert_not_called()
            mock_store.delete_by_hashes.assert_not_called()
            assert result.nodes_added == 0

    @pytest.mark.asyncio
    async def test_should_delete_stale_nodes(self, mock_repo):
        """Test index deletes nodes that are no longer present."""
        cached_hashmap = {"src/main.py": "old_hash1", "src/old.py": "old_hash2"}
        current_hashmap = {"src/main.py": "new_hash"}

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch.object(Repo, "_get_hashmap", return_value=current_hashmap),
            patch.object(Repo, "_get_cached_hashmap", return_value=cached_hashmap),
            patch.object(Repo, "_set_hashmap"),
        ):
            mock_store = create_mock_store()
            mock_store.delete_by_hashes = AsyncMock(return_value=5)

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([])
            mock_walker_class.return_value = mock_walker_instance

            result = await mock_repo.index(mock_store)

            # Should delete 2 old hashes that are not in current
            mock_store.delete_by_hashes.assert_called_once()
            call_args = mock_store.delete_by_hashes.call_args
            assert call_args[0][0] == "indexter_index-repo"
            deleted_hashes = call_args[0][1]
            assert "old_hash1" in deleted_hashes
            assert "old_hash2" in deleted_hashes
            # nodes_deleted comes from store's return value
            assert result.nodes_deleted == 5
            # documents_deleted should contain the paths whose hashes were deleted
            assert "src/main.py" in result.documents_deleted
            assert "src/old.py" in result.documents_deleted

    @pytest.mark.asyncio
    async def test_should_index_new_files(self, mock_repo):
        """Test index processes and upserts new files."""
        current_hashmap = {"src/new.py": "new_hash"}
        cached_hashmap = {}

        doc_metadata = create_doc_metadata(
            repo="index-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch.object(Repo, "_get_hashmap", return_value=current_hashmap),
            patch.object(Repo, "_get_cached_hashmap", return_value=cached_hashmap),
            patch.object(Repo, "_set_hashmap") as mock_set_hashmap,
        ):
            mock_store = create_mock_store()

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([("src/new.py", "def new_func(): pass", doc_metadata)])
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="index-repo",
                repo_path=str(mock_repo.settings.path),
                document_path="src/new.py",
                document_hash="new_hash",
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

            result = await mock_repo.index(mock_store)

            mock_store.upsert_nodes.assert_called_once()
            assert result.nodes_added == 1
            mock_set_hashmap.assert_called_once_with(current_hashmap)

    @pytest.mark.asyncio
    async def test_should_batch_upsert_operations(self, mock_repo):
        """Test index batches node upserts according to batch size."""
        mock_repo.settings.upsert_batch_size = 2

        current_hashmap = {
            "src/file1.py": "hash1",
            "src/file2.py": "hash2",
            "src/file3.py": "hash3",
        }

        doc_metadata = create_doc_metadata(
            repo="index-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch.object(Repo, "_get_hashmap", return_value=current_hashmap),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
            patch.object(Repo, "_set_hashmap"),
        ):
            mock_store = create_mock_store()

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter(
                [(f"src/file{i}.py", f"def f{i}(): pass", doc_metadata) for i in range(1, 4)]
            )
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="index-repo",
                repo_path=str(mock_repo.settings.path),
                document_path="src/file.py",
                document_hash="hash1",
                language="python",
                node_type="function",
                node_name="f",
                start_byte=0,
                end_byte=15,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            mock_parser.parse.return_value = [("def f(): pass", node_metadata)]
            mock_parser_class.return_value = mock_parser

            await mock_repo.index(mock_store)

            # Should call upsert twice: once for first 2 nodes, once for remaining 1
            assert mock_store.upsert_nodes.call_count == 2

    @pytest.mark.asyncio
    async def test_should_handle_parsing_errors(self, mock_repo, caplog):
        """Test index handles parsing errors gracefully."""
        current_hashmap = {"src/error.py": "hash"}

        doc_metadata = create_doc_metadata(
            repo="index-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch.object(Repo, "_get_hashmap", return_value=current_hashmap),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
            patch.object(Repo, "_set_hashmap"),
        ):
            mock_store = create_mock_store()

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([("src/error.py", "invalid syntax", doc_metadata)])
            mock_walker_class.return_value = mock_walker_instance

            mock_parser = Mock()
            mock_parser.parse.side_effect = Exception("Parse failed")
            mock_parser_class.return_value = mock_parser

            with caplog.at_level("WARNING"):
                result = await mock_repo.index(mock_store)

            assert len(result.errors) == 1
            assert "Failed to parse" in result.errors[0]

    @pytest.mark.asyncio
    async def test_should_calculate_duration_and_timestamp(self, mock_repo):
        """Test index result includes duration and timestamp."""
        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.datetime") as mock_datetime,
            patch.object(Repo, "_get_hashmap", return_value={}),
            patch.object(Repo, "_get_cached_hashmap", return_value={"old": "hash"}),
            patch.object(Repo, "_set_hashmap"),
        ):
            start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            end_time = datetime(2024, 1, 1, 12, 0, 5, tzinfo=UTC)

            mock_datetime.now.side_effect = [start_time, end_time]
            mock_datetime.UTC = UTC

            mock_store = create_mock_store()

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([])
            mock_walker_class.return_value = mock_walker_instance

            result = await mock_repo.index(mock_store)

            assert result.indexed_at == end_time
            assert result.duration == 5.0

    @pytest.mark.asyncio
    async def test_should_create_placeholder_for_empty_parse_result(self, mock_repo):
        """Test index creates placeholder node when parser returns empty content."""
        current_hashmap = {"src/empty.py": "placeholder_hash"}

        doc_metadata = create_doc_metadata(
            repo="index-repo",
            repo_path=str(mock_repo.settings.path),
        )

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch("indexter.models.Parser") as mock_parser_class,
            patch.object(Repo, "_get_hashmap", return_value=current_hashmap),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
            patch.object(Repo, "_set_hashmap"),
        ):
            mock_store = create_mock_store()

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([("src/empty.py", "# Just a comment", doc_metadata)])
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="index-repo",
                repo_path=str(mock_repo.settings.path),
                document_path="src/empty.py",
                document_hash="placeholder_hash",
                language="python",
                node_type="N/A",
                node_name="empty",
                start_byte=0,
                end_byte=0,
                start_line=1,
                end_line=1,
            )
            mock_parser = Mock()
            # Parser returns empty string content
            mock_parser.parse.return_value = [("", node_metadata)]
            mock_parser_class.return_value = mock_parser

            await mock_repo.index(mock_store)

            # Node.from_parsed is called with the real class (not mocked)
            # The node should still be upserted
            mock_store.upsert_nodes.assert_called_once()


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

        mock_store = create_mock_store()
        mock_store.search = AsyncMock(return_value=mock_results)

        await mock_repo.search("test query", mock_store)

        mock_store.search.assert_called_once()
        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["query"] == "test query"

    @pytest.mark.asyncio
    async def test_should_search_with_custom_limit(self, mock_repo):
        """Test search accepts custom limit parameter."""
        mock_results = Mock(spec=SearchResults)

        mock_store = create_mock_store()
        mock_store.search = AsyncMock(return_value=mock_results)

        await mock_repo.search("test query", mock_store, limit=5)

        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_should_search_with_all_filters(self, mock_repo):
        """Test search passes all filter parameters to store."""
        mock_results = Mock(spec=SearchResults)

        mock_store = create_mock_store()
        mock_store.search = AsyncMock(return_value=mock_results)

        await mock_repo.search(
            query="find functions",
            store=mock_store,
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

        mock_store = create_mock_store()
        mock_store.search = AsyncMock(return_value=mock_results)

        result = await mock_repo.search("test", mock_store)

        assert result.repo == "search-repo"
        assert result.repo_path == str(mock_repo.settings.path)


class TestRepoIntegration:
    """Integration tests for Repo workflows."""

    @pytest.mark.asyncio
    async def test_should_complete_full_workflow(self, tmp_path):
        """Test complete workflow: init -> index -> search."""
        git_repo = tmp_path / "workflow-repo"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()

        with (
            patch("indexter.models.RepoSettings") as mock_repo_settings_class,
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
            doc_metadata = create_doc_metadata(
                repo="workflow-repo",
                repo_path=str(git_repo),
            )

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([("src/app.py", "def main(): pass", doc_metadata)])
            mock_walker_class.return_value = mock_walker_instance

            node_metadata = NodeMetadata(
                repo="workflow-repo",
                repo_path=str(git_repo),
                document_path="src/app.py",
                document_hash="hash1",
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

            mock_store = create_mock_store()

            # Mock hashmap methods
            with (
                patch.object(Repo, "_get_hashmap", return_value={"src/app.py": "hash1"}),
                patch.object(Repo, "_get_cached_hashmap", return_value={}),
                patch.object(Repo, "_set_hashmap"),
            ):
                # Index
                index_result = await repo.index(mock_store)
                assert index_result.nodes_added >= 1

            # Search
            mock_search_results = Mock(spec=SearchResults)
            mock_store.search = AsyncMock(return_value=mock_search_results)

            search_results = await repo.search("main function", mock_store)
            assert search_results.repo == "workflow-repo"

    @pytest.mark.asyncio
    async def test_should_handle_multiple_repositories(self, tmp_path):
        """Test managing multiple repositories simultaneously."""
        repo1_path = tmp_path / "repo1"
        repo1_path.mkdir()
        (repo1_path / ".git").mkdir()

        repo2_path = tmp_path / "repo2"
        repo2_path.mkdir()
        (repo2_path / ".git").mkdir()

        with patch("indexter.models.RepoSettings") as mock_repo_settings_class:
            settings1 = Mock(spec=RepoSettings)
            settings1.name = "repo1"
            settings1.path = repo1_path
            settings1.collection_name = "indexter_repo1"

            settings2 = Mock(spec=RepoSettings)
            settings2.name = "repo2"
            settings2.path = repo2_path
            settings2.collection_name = "indexter_repo2"

            mock_repo_settings_class.load = AsyncMock(return_value=[settings1, settings2])

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

        with (
            patch("indexter.models.Walker") as mock_walker_class,
            patch.object(Repo, "_get_hashmap", return_value={}),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
            patch.object(Repo, "_set_hashmap"),
            patch.object(Repo, "cache") as mock_cache,
        ):
            mock_store = create_mock_store()

            mock_walker_instance = Mock()
            mock_walker_instance.walk.return_value = iter([])
            mock_walker_class.return_value = mock_walker_instance

            # Full index should delete collection first
            await repo.index(mock_store, full=True)

            mock_store.delete_collection.assert_called_once_with("indexter_reindex-repo")
            mock_cache.delete.assert_called_once_with("hashmap")
            mock_store.ensure_collection.assert_called_once()
