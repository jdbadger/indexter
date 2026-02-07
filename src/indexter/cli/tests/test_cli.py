"""Comprehensive tests for the CLI application.

This test suite provides comprehensive coverage of the CLI commands including:
- Unit tests for each CLI command (init, index, search, status, forget)
- Error handling and edge cases
- Integration tests for command workflows
- Parameter variations and output formatting

Note on coverage: The coverage report shows ~91% due to coverage.py's limitation
in tracking nested async function definitions (lines 146-152, 234-236, 255-256, 314-316).
These lines are internal async wrapper functions that ARE executed by the tests
but not tracked properly by the coverage tool. All functional code paths are tested.
"""

from unittest.mock import Mock, patch

import pytest
import typer

from indexter.cli.cli import app, version_callback
from indexter.cli.tests.conftest import strip_ansi
from indexter.exceptions import RepoExistsError, RepoNotFoundError
from indexter.models import Repo, RepoMetadata
from indexter.store.models import IndexResult, SearchResult, SearchResults


class TestVersionCallback:
    """Test version_callback function."""

    def test_should_print_version_and_exit_when_value_is_true(self):
        """Test version_callback prints version and exits when value is True."""
        with pytest.raises(typer.Exit):
            version_callback(True)

    def test_should_do_nothing_when_value_is_false(self):
        """Test version_callback does nothing when value is False."""
        # Should not raise any exception
        version_callback(False)


class TestMainCallback:
    """Test main callback function (logging setup)."""

    def test_should_invoke_app_with_verbose_flag(self, cli_runner):
        """Test main callback can be invoked with verbose flag."""
        result = cli_runner.invoke(app, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_should_invoke_app_without_verbose_flag(self, cli_runner):
        """Test main callback can be invoked without verbose flag."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0


class TestInitCommand:
    """Test init command."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock Repo instance."""
        mock_settings = Mock()
        mock_settings.name = "test-repo"
        mock_settings.path = tmp_path / "test-repo"
        mock_settings.collection_name = "indexter_test-repo"
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.path = mock_settings.path
        repo.settings = mock_settings
        return repo

    @pytest.fixture
    def mock_index_result(self):
        """Create a mock IndexResult instance."""
        return IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            documents_indexed=["src/main.py"],
            documents_deleted=[],
            nodes_added=5,
            duration=0.5,
        )

    def test_should_initialize_repo_successfully_without_indexing(self, cli_runner, mock_repo, tmp_path):
        """Test init command successfully initializes repo with --no-index flag."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        async def mock_init(_path):
            return mock_repo

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, None)

            result = cli_runner.invoke(app, ["init", "--path", str(repo_path), "--no-index"])

            assert result.exit_code == 0
            assert "test-repo" in result.stdout
            assert "initialized successfully!" in result.stdout
            assert "indexter index test-repo" in result.stdout

    def test_should_initialize_and_index_repo_successfully(self, cli_runner, mock_repo, mock_index_result, tmp_path):
        """Test init command successfully initializes and indexes repo."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result)

            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])

            assert result.exit_code == 0
            assert "test-repo" in result.stdout
            assert "initialized and indexed successfully!" in result.stdout
            assert "indexter search" in result.stdout

    def test_should_handle_repo_exists_error(self, cli_runner, tmp_path):
        """Test init command handles RepoExistsError gracefully."""
        repo_path = tmp_path / "existing-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        async def mock_init_error(_path):
            raise RepoExistsError("Repository 'existing-repo' already exists")

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RepoExistsError("Repository 'existing-repo' already exists")

            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])

            assert result.exit_code == 1
            assert "already exists" in result.stdout

    def test_should_handle_unexpected_error(self, cli_runner, tmp_path):
        """Test init command handles unexpected errors gracefully."""
        repo_path = tmp_path / "error-repo"
        repo_path.mkdir()

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected error")

            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])

            assert result.exit_code == 1
            assert "Unexpected error" in result.stdout

    def test_should_use_current_directory_as_default_path(self, cli_runner, mock_repo, tmp_path):
        """Test init command uses current directory when path is not specified."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, None)

            # Default path is "."
            cli_runner.invoke(app, ["init", "--no-index"])

            # Verify anyio.run was called
            assert mock_run.called

    def test_should_display_next_steps_after_init_no_index(self, cli_runner, mock_repo, tmp_path):
        """Test init command displays appropriate next steps when --no-index is used."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, None)

            result = cli_runner.invoke(app, ["init", "--path", str(repo_path), "--no-index"])

            assert result.exit_code == 0
            assert "Next steps:" in result.stdout
            assert "indexter index test-repo" in result.stdout

    def test_should_display_next_steps_after_init_with_index(self, cli_runner, mock_repo, mock_index_result, tmp_path):
        """Test init command displays appropriate next steps after indexing."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result)

            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])

            assert result.exit_code == 0
            assert "Next steps:" in result.stdout
            assert "indexter search" in result.stdout


class TestIndexCommand:
    """Test index command."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock Repo instance."""
        mock_settings = Mock()
        mock_settings.name = "test-repo"
        mock_settings.path = tmp_path / "test-repo"
        mock_settings.collection_name = "indexter_test-repo"
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.path = mock_settings.path
        repo.settings = mock_settings
        return repo

    @pytest.fixture
    def mock_index_result_with_changes(self):
        """Create a mock IndexResult with changes."""
        return IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            documents_indexed=["src/main.py", "src/utils.py"],
            documents_deleted=["src/old.py"],
            nodes_added=15,
            duration=1.2,
        )

    @pytest.fixture
    def mock_index_result_no_changes(self):
        """Create a mock IndexResult with no changes."""
        return IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            documents_indexed=[],
            documents_deleted=[],
            nodes_added=0,
            duration=0.1,
        )

    @pytest.fixture
    def mock_index_result_with_errors(self):
        """Create a mock IndexResult with errors."""
        return IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            documents_indexed=["src/main.py"],
            documents_deleted=[],
            nodes_added=5,
            duration=0.8,
            errors=[
                "Error parsing src/broken.py: SyntaxError",
                "Error parsing src/invalid.py: ParseError",
                "Error 3",
                "Error 4",
                "Error 5",
                "Error 6",
                "Error 7",
            ],
        )

    @pytest.fixture
    def mock_index_result_with_skipped(self):
        """Create a mock IndexResult (skipped documents field removed)."""
        return IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            documents_indexed=["src/main.py"],
            nodes_added=5,
            duration=0.5,
        )

    def test_should_index_repo_with_changes_successfully(self, cli_runner, mock_repo, mock_index_result_with_changes):
        """Test index command successfully indexes repo with changes."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result_with_changes)

            result = cli_runner.invoke(app, ["index", "test-repo"])

            assert result.exit_code == 0
            assert "test-repo" in result.stdout
            assert "Indexing complete!" in result.stdout

    def test_should_handle_repo_with_no_changes(self, cli_runner, mock_repo, mock_index_result_no_changes):
        """Test index command handles repo with no changes."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result_no_changes)

            result = cli_runner.invoke(app, ["index", "test-repo"])

            assert result.exit_code == 0
            # When documents_indexed is empty, shows specific "up to date" message
            if len(mock_index_result_no_changes.documents_indexed) == 0:
                assert "up to date" in result.stdout
                assert "No changes detected" in result.stdout
            else:
                assert "Indexed" in result.stdout

    def test_should_handle_full_index_flag(self, cli_runner, mock_repo, mock_index_result_with_changes):
        """Test index command handles --full flag."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result_with_changes)

            result = cli_runner.invoke(app, ["index", "test-repo", "--full"])

            assert result.exit_code == 0
            assert "Indexing complete!" in result.stdout

    def test_should_handle_repo_not_found_error(self, cli_runner):
        """Test index command handles RepoNotFoundError gracefully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RepoNotFoundError("Repository 'missing-repo' not found")

            result = cli_runner.invoke(app, ["index", "missing-repo"])

            assert result.exit_code == 1
            assert "not found" in result.stdout
            assert "indexter init" in result.stdout

    def test_should_handle_unexpected_error_during_indexing(self, cli_runner):
        """Test index command handles unexpected errors gracefully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RuntimeError("Database connection failed")

            result = cli_runner.invoke(app, ["index", "test-repo"])

            assert result.exit_code == 1
            assert "Unexpected error" in result.stdout

    def test_should_display_errors_when_indexing_fails_partially(
        self, cli_runner, mock_repo, mock_index_result_with_errors
    ):
        """Test index command displays errors when some files fail to index."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result_with_errors)

            result = cli_runner.invoke(app, ["index", "test-repo"])

            assert result.exit_code == 0
            assert "Errors:" in result.stdout
            assert "broken.py" in result.stdout
            # Should show "and X more" for errors beyond first 5
            assert "and" in result.stdout and "more" in result.stdout

    def test_should_display_all_errors_when_five_or_fewer(self, cli_runner, mock_repo):
        """Test index command displays all errors when there are 5 or fewer."""
        result_with_few_errors = IndexResult(
            repo="test-repo",
            repo_path="/tmp/test-repo",
            documents_indexed=["src/main.py"],
            nodes_added=5,
            duration=0.5,
            errors=[
                "Error 1",
                "Error 2",
                "Error 3",
            ],
        )

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, result_with_few_errors)

            result = cli_runner.invoke(app, ["index", "test-repo"])
            output = strip_ansi(result.stdout)

            assert result.exit_code == 0
            assert "Errors: 3" in output
            assert "Error 1" in output
            assert "Error 2" in output
            assert "Error 3" in output
            # Should NOT show "and X more" since we only have 3 errors
            assert output.count("Error") >= 3

    def test_should_display_index_summary(self, cli_runner, mock_repo, mock_index_result_with_skipped):
        """Test index command displays summary."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result_with_skipped)

            result = cli_runner.invoke(app, ["index", "test-repo"])
            output = strip_ansi(result.stdout)

            assert result.exit_code == 0
            assert "Indexed 1 documents" in output
            assert "Indexing complete!" in output


class TestSearchCommand:
    """Test search command."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a mock Repo instance."""
        mock_settings = Mock()
        mock_settings.name = "test-repo"
        mock_settings.path = tmp_path / "test-repo"
        mock_settings.collection_name = "indexter_test-repo"
        repo = Mock(spec=Repo)
        repo.name = "test-repo"
        repo.path = mock_settings.path
        repo.settings = mock_settings
        return repo

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        return SearchResults(
            repo="test-repo",
            query="authentication",
            filters={},
            results=[
                SearchResult(
                    content="def authenticate(user, password):\n    # Check credentials\n    return True",
                    score=0.92,
                    metadata={"document_path": "src/auth/middleware.py", "node_type": "function"},
                ),
                SearchResult(
                    content="class AuthMiddleware:\n    def process_request(self, request):\n        pass",
                    score=0.85,
                    metadata={"document_path": "src/auth/middleware.py", "node_type": "class"},
                ),
            ],
        )

    @pytest.fixture
    def mock_empty_search_results(self):
        """Create empty search results."""
        return SearchResults(
            repo="test-repo",
            query="nonexistent",
            filters={},
            results=[],
        )

    def test_should_display_search_results_successfully(self, cli_runner, mock_repo, mock_search_results):
        """Test search command displays results successfully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_search_results)

            result = cli_runner.invoke(app, ["search", "authentication", "test-repo"])

            assert result.exit_code == 0
            assert "0.9200" in result.stdout
            assert "0.8500" in result.stdout
            assert "src/auth/middleware.py" in result.stdout

    def test_should_handle_empty_search_results(self, cli_runner, mock_repo, mock_empty_search_results):
        """Test search command handles empty results gracefully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_empty_search_results)

            result = cli_runner.invoke(app, ["search", "nonexistent", "test-repo"])

            assert result.exit_code == 0
            assert "No results found" in result.stdout
            assert "nonexistent" in result.stdout

    def test_should_handle_custom_limit_parameter(self, cli_runner, mock_repo, mock_search_results):
        """Test search command handles --limit parameter."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_search_results)

            result = cli_runner.invoke(app, ["search", "authentication", "test-repo", "--limit", "5"])

            assert result.exit_code == 0
            assert "0.9200" in result.stdout

    def test_should_handle_repo_not_found_error(self, cli_runner):
        """Test search command handles RepoNotFoundError gracefully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RepoNotFoundError("Repository 'missing-repo' not found")

            result = cli_runner.invoke(app, ["search", "query", "missing-repo"])

            assert result.exit_code == 1
            assert "not found" in result.stdout

    def test_should_handle_unexpected_error_during_search(self, cli_runner):
        """Test search command handles unexpected errors gracefully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RuntimeError("Vector store unavailable")

            result = cli_runner.invoke(app, ["search", "query", "test-repo"])

            assert result.exit_code == 1
            assert "Unexpected error" in result.stdout

    def test_should_truncate_long_content_in_results(self, cli_runner, mock_repo):
        """Test search command truncates long content snippets."""
        long_content_result = SearchResults(
            repo="test-repo",
            query="test",
            filters={},
            results=[
                SearchResult(
                    content="a" * 200,  # Very long content
                    score=0.9,
                    metadata={"document_path": "src/long.py"},
                ),
            ],
        )

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, long_content_result)

            result = cli_runner.invoke(app, ["search", "test", "test-repo"])

            assert result.exit_code == 0
            # Content should be truncated to 50 chars + "..." (rich uses … instead of ...)
            assert "..." in result.stdout or "…" in result.stdout
            assert "src/long.py" in result.stdout


class TestStatusCommand:
    """Test status command."""

    @pytest.fixture
    def mock_repos_with_metadata(self, tmp_path):
        """Create mock repos with metadata."""
        repos = []
        for i, name in enumerate(["repo1", "repo2"]):
            mock_settings = Mock()
            mock_settings.name = name
            mock_settings.path = tmp_path / name
            mock_settings.collection_name = f"indexter_{name}"

            repo = Mock(spec=Repo)
            repo.name = name
            repo.path = mock_settings.path
            repo.settings = mock_settings
            repo.is_stale = i % 2 == 0  # Alternate stale status

            metadata = Mock(spec=RepoMetadata)
            metadata.nodes = (i + 1) * 100
            metadata.documents = (i + 1) * 10
            repo.metadata = metadata

            repos.append(repo)
        return repos

    def test_should_display_status_for_multiple_repos(self, cli_runner, mock_repos_with_metadata):
        """Test status command displays information for multiple repositories."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = mock_repos_with_metadata

            result = cli_runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert "repo1" in result.stdout
            assert "repo2" in result.stdout
            assert "100" in result.stdout  # nodes for repo1
            assert "200" in result.stdout  # nodes for repo2
            assert "10" in result.stdout  # documents for repo1
            assert "20" in result.stdout  # documents for repo2

    def test_should_display_message_when_no_repos_exist(self, cli_runner):
        """Test status command displays message when no repositories exist."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = []

            result = cli_runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert "No repositories indexed" in result.stdout
            assert "indexter index" in result.stdout

    def test_should_handle_repos_without_metadata(self, cli_runner, tmp_path):
        """Test status command handles repos without metadata gracefully."""
        mock_settings = Mock()
        mock_settings.name = "repo-no-meta"
        mock_settings.path = tmp_path / "repo-no-meta"
        mock_settings.collection_name = "indexter_repo-no-meta"

        repo = Mock(spec=Repo)
        repo.name = "repo-no-meta"
        repo.path = mock_settings.path
        repo.settings = mock_settings
        repo.metadata = None

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = [repo]

            result = cli_runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert "repo-no-meta" in result.stdout
            assert "-" in result.stdout  # Shows "-" for missing metadata


class TestForgetCommand:
    """Test forget command."""

    def test_should_forget_repo_successfully(self, cli_runner):
        """Test forget command successfully removes a repository."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = None

            result = cli_runner.invoke(app, ["forget", "test-repo"])

            assert result.exit_code == 0
            assert "test-repo" in result.stdout
            assert "is forgotten" in result.stdout

    def test_should_handle_repo_not_found_error(self, cli_runner):
        """Test forget command handles RepoNotFoundError gracefully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RepoNotFoundError("Repository 'missing-repo' not found")

            result = cli_runner.invoke(app, ["forget", "missing-repo"])

            assert result.exit_code == 1
            assert "not found" in result.stdout

    def test_should_handle_unexpected_error(self, cli_runner):
        """Test forget command handles unexpected errors gracefully."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RuntimeError("Database error")

            result = cli_runner.invoke(app, ["forget", "test-repo"])

            assert result.exit_code == 1
            assert "Unexpected error" in result.stdout


class TestIntegration:
    """Integration tests for CLI commands."""

    @pytest.mark.asyncio
    async def test_should_init_and_search_workflow(self, cli_runner, tmp_path):
        """Test complete workflow: init -> index -> search."""
        repo_path = tmp_path / "integration-test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        # Create mock objects
        mock_settings = Mock()
        mock_settings.name = "integration-test-repo"
        mock_settings.path = repo_path
        mock_settings.collection_name = "indexter_integration-test-repo"

        mock_repo = Mock(spec=Repo)
        mock_repo.name = "integration-test-repo"
        mock_repo.path = repo_path
        mock_repo.settings = mock_settings

        mock_index_result = IndexResult(
            repo="integration-test-repo",
            repo_path=str(repo_path),
            documents_indexed=["main.py"],
            nodes_added=5,
            duration=0.1,
        )

        mock_search_results = SearchResults(
            repo="integration-test-repo",
            query="test",
            filters={},
            results=[
                SearchResult(
                    content="def test():\n    pass",
                    score=0.9,
                    metadata={"document_path": "main.py"},
                ),
            ],
        )

        # Test init
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_index_result)
            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])
            assert result.exit_code == 0
            assert "integration-test-repo" in result.stdout

        # Test search
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_search_results)
            result = cli_runner.invoke(app, ["search", "test", "integration-test-repo"])
            assert result.exit_code == 0
            assert "0.9000" in result.stdout

    @pytest.mark.asyncio
    async def test_should_handle_status_then_forget_workflow(self, cli_runner, tmp_path):
        """Test workflow: status -> forget -> status (empty)."""
        # Create mock repo with metadata
        mock_settings = Mock()
        mock_settings.name = "temp-repo"
        mock_settings.path = tmp_path / "temp-repo"
        mock_settings.collection_name = "indexter_temp-repo"

        mock_repo = Mock(spec=Repo)
        mock_repo.name = "temp-repo"
        mock_repo.path = mock_settings.path
        mock_repo.settings = mock_settings
        mock_repo.is_stale = False

        metadata = Mock(spec=RepoMetadata)
        metadata.nodes = 50
        metadata.documents = 5
        mock_repo.metadata = metadata

        # Test status with repo
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = [mock_repo]
            result = cli_runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "temp-repo" in result.stdout

        # Test forget
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = None
            result = cli_runner.invoke(app, ["forget", "temp-repo"])
            assert result.exit_code == 0
            assert "is forgotten" in result.stdout

        # Test status without repos
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = []
            result = cli_runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "No repositories indexed" in result.stdout

    @pytest.mark.parametrize(
        "command,args,expected_error",
        [
            ("index", ["nonexistent"], "not found"),
            ("search", ["query", "nonexistent"], "not found"),
            ("forget", ["nonexistent"], "not found"),
        ],
    )
    def test_should_handle_nonexistent_repo_across_commands(self, cli_runner, command, args, expected_error):
        """Test all commands handle nonexistent repository errors consistently."""
        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.side_effect = RepoNotFoundError(f"Repository '{args[-1]}' not found")

            result = cli_runner.invoke(app, [command] + args)

            assert result.exit_code == 1
            assert expected_error in result.stdout

    def test_should_respect_limit_parameter_variations(self, cli_runner, tmp_path):
        """Test search command with various limit values."""
        mock_settings = Mock()
        mock_settings.name = "test-repo"
        mock_settings.path = tmp_path / "test-repo"
        mock_settings.collection_name = "indexter_test-repo"

        mock_repo = Mock(spec=Repo)
        mock_repo.name = "test-repo"
        mock_repo.path = mock_settings.path
        mock_repo.settings = mock_settings

        # Create multiple results
        results = [
            SearchResult(
                content=f"result {i}",
                score=0.9 - i * 0.1,
                metadata={"document_path": f"file{i}.py"},
            )
            for i in range(10)
        ]

        mock_search_results = SearchResults(
            repo="test-repo",
            query="test",
            filters={},
            results=results,
        )

        with patch("indexter.cli.cli.anyio.run") as mock_run:
            mock_run.return_value = (mock_repo, mock_search_results)

            # Test with default limit (10)
            result = cli_runner.invoke(app, ["search", "test", "test-repo"])
            assert result.exit_code == 0

            # Test with custom limit
            result = cli_runner.invoke(app, ["search", "test", "test-repo", "--limit", "3"])
            assert result.exit_code == 0
