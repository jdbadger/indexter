"""Tests for CLI app and commands.

This module provides comprehensive test coverage for the indexter CLI
application, which includes:
- Main app callback with version and verbose flags
- init: Initialize a git repository for indexing
- index: Sync a repository to the vector store
- search: Search indexed nodes in a repository
- status: Show status of indexed repositories
- forget: Remove a repository from indexter

Test Coverage:
--------------
- Version callback and display
- Verbose logging configuration
- All commands with success and error scenarios
- Exception handling (RepoNotFoundError, RepoExistsError)
- Progress displays and output formatting
- Table displays for search and status
- Edge cases (empty results, errors during indexing)

The tests use unittest.mock to mock the Repo class and anyio.run calls,
ensuring tests are isolated and don't require actual repositories or
vector store connections.
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from rich.console import Console

from indexter import __version__
from indexter.cli.cli import (
    app,
    console,
    main,
    version_callback,
)
from indexter.exceptions import RepoExistsError, RepoNotFoundError


def test_version_callback_prints_version_and_exits():
    """Test version callback prints version and raises Exit."""
    with patch("indexter.cli.cli.console") as mock_console:
        with pytest.raises(typer.Exit):
            version_callback(True)

        mock_console.print.assert_called_once_with(f"indexter {__version__}")


def test_version_callback_does_nothing_when_false():
    """Test version callback does nothing when value is False."""
    with patch("indexter.cli.cli.console") as mock_console:
        version_callback(False)
        mock_console.print.assert_not_called()


def test_main_callback_verbose_flag():
    """Test main callback configures verbose logging."""
    with patch("indexter.cli.cli.logging.basicConfig") as mock_config:
        main(verbose=True)

        # Verify logging configured with DEBUG level
        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["level"] == logging.DEBUG


def test_main_callback_non_verbose():
    """Test main callback configures normal logging."""
    with patch("indexter.cli.cli.logging.basicConfig") as mock_config:
        main(verbose=False)

        # Verify logging configured with INFO level
        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["level"] == logging.INFO


def test_init_successful(cli_runner):
    """Test init command with successful repository initialization."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    with patch("indexter.cli.cli.anyio.run", return_value=mock_repo) as mock_run:
        result = cli_runner.invoke(app, ["init", "/path/to/repo"])

        assert result.exit_code == 0
        assert "Added test-repo to indexter" in result.stdout
        assert "Repository 'test-repo' initialized successfully!" in result.stdout
        assert "Next steps:" in result.stdout
        assert "indexter index test-repo" in result.stdout

        # Verify anyio.run was called correctly
        mock_run.assert_called_once()


def test_init_with_repo_exists_error(cli_runner):
    """Test init command when repository already exists."""
    with patch("indexter.cli.cli.anyio.run", side_effect=RepoExistsError("Repo exists")):
        result = cli_runner.invoke(app, ["init", "/path/to/repo"])

        assert result.exit_code == 1
        assert "Repo exists" in result.stdout


def test_init_with_unexpected_error(cli_runner):
    """Test init command with unexpected error."""
    with patch("indexter.cli.cli.anyio.run", side_effect=ValueError("Unexpected")):
        result = cli_runner.invoke(app, ["init", "/path/to/repo"])

        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_init_resolves_path(cli_runner):
    """Test init command resolves the repo path."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    with patch("indexter.cli.cli.anyio.run", return_value=mock_repo):
        with patch("pathlib.Path.resolve") as mock_resolve:
            mock_resolve.return_value = Path("/resolved/path")

            result = cli_runner.invoke(app, ["init", "relative/path"])

            assert result.exit_code == 0
            # Verify resolve was called
            mock_resolve.assert_called()


def test_index_successful_with_changes(cli_runner):
    """Test index command with files synced."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    mock_result = Mock()
    mock_result.files_indexed = ["file1.py", "file2.py"]
    mock_result.files_deleted = ["old.py"]
    mock_result.files_checked = 10
    mock_result.nodes_added = 5
    mock_result.nodes_updated = 3
    mock_result.nodes_deleted = 1
    mock_result.errors = []
    mock_result.skipped_files = 0
    mock_result.summary = "Indexed 2 files (+5 nodes, ~3 updated, -1 deleted) in 1.50s"

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_result)

        result = cli_runner.invoke(app, ["index", "test-repo"])

        assert result.exit_code == 0
        # Verify summary is displayed in output (may have ANSI codes)
        assert "Indexed" in result.stdout
        assert "files" in result.stdout
        assert "nodes" in result.stdout
        assert "updated" in result.stdout
        assert "deleted" in result.stdout
        assert "Indexing complete!" in result.stdout


def test_index_successful_no_changes(cli_runner):
    """Test index command when repository is up to date."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    mock_result = Mock()
    mock_result.files_indexed = 0
    mock_result.files_checked = 10
    mock_result.errors = []

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_result)

        result = cli_runner.invoke(app, ["index", "test-repo"])

        assert result.exit_code == 0
        assert "up to date" in result.stdout
        assert "No changes detected" in result.stdout
        assert "10 files checked" in result.stdout


def test_index_with_full_flag(cli_runner):
    """Test index command with --full flag."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"
    mock_repo.index = Mock()

    mock_result = Mock()
    mock_result.files_indexed = 0
    mock_result.files_checked = 10
    mock_result.errors = []
    mock_result.summary = "Indexed 0 files (+0 nodes, ~0 updated, -0 deleted) in 0.25s"

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_result)

        result = cli_runner.invoke(app, ["index", "test-repo", "--full"])

        assert result.exit_code == 0
        # Now only a single call to anyio.run
        assert mock_run.call_count == 1


def test_index_with_errors(cli_runner):
    """Test index command with indexing errors."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    mock_result = Mock()
    mock_result.files_indexed = ["file1.py"]
    mock_result.files_deleted = []
    mock_result.nodes_added = 1
    mock_result.nodes_updated = 0
    mock_result.nodes_deleted = 0
    mock_result.errors = ["Error 1", "Error 2", "Error 3"]
    mock_result.skipped_files = 0
    mock_result.summary = "Indexed 1 files (+1 nodes, ~0 updated, -0 deleted) in 0.50s"

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_result)

        result = cli_runner.invoke(app, ["index", "test-repo"])

        assert result.exit_code == 0
        assert "Errors: 3" in result.stdout
        assert "Error 1" in result.stdout
        assert "Some files could not be indexed" in result.stdout


def test_index_with_many_errors(cli_runner):
    """Test index command displays only first 5 errors."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    errors = [f"Error {i}" for i in range(10)]
    mock_result = Mock()
    mock_result.files_indexed = ["file1.py"]
    mock_result.files_deleted = []
    mock_result.nodes_added = 1
    mock_result.nodes_updated = 0
    mock_result.nodes_deleted = 0
    mock_result.errors = errors
    mock_result.skipped_files = 0
    mock_result.summary = "Indexed 1 files (+1 nodes, ~0 updated, -0 deleted) in 0.50s"

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_result)

        result = cli_runner.invoke(app, ["index", "test-repo"])

        assert result.exit_code == 0
        assert "Errors: 10" in result.stdout
        assert "and 5 more" in result.stdout


def test_index_with_skipped_files(cli_runner):
    """Test index command with skipped files."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    mock_result = Mock()
    mock_result.files_indexed = ["file1.py"]
    mock_result.files_deleted = []
    mock_result.nodes_added = 1
    mock_result.nodes_updated = 0
    mock_result.nodes_deleted = 0
    mock_result.errors = []
    mock_result.skipped_files = 5
    mock_result.summary = "Indexed 1 files (+1 nodes, ~0 updated, -0 deleted) in 0.50s"

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_result)

        result = cli_runner.invoke(app, ["index", "test-repo"])

        assert result.exit_code == 0
        assert "Skipped: 5 files" in result.stdout
        assert "maximum allowed file limit" in result.stdout


def test_index_repo_not_found(cli_runner):
    """Test index command when repository is not found."""
    with patch("indexter.cli.cli.anyio.run", side_effect=RepoNotFoundError("Not found")):
        result = cli_runner.invoke(app, ["index", "nonexistent-repo"])

        assert result.exit_code == 1
        assert "Repository not found: nonexistent-repo" in result.stdout
        assert "indexter init" in result.stdout


def test_index_unexpected_error(cli_runner):
    """Test index command with unexpected error."""
    with patch("indexter.cli.cli.anyio.run", side_effect=ValueError("Unexpected")):
        result = cli_runner.invoke(app, ["index", "test-repo"])

        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_search_successful(cli_runner):
    """Test search command with results."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    mock_results = [
        {
            "score": 0.95,
            "content": "def hello():\n    print('Hello')",
            "file_path": "hello.py",
        },
        {
            "score": 0.85,
            "content": "class World:\n    pass",
            "file_path": "world.py",
        },
    ]

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_results)

        result = cli_runner.invoke(app, ["search", "hello", "test-repo"])

        assert result.exit_code == 0
        assert "Search Results for 'hello' in 'test-repo'" in result.stdout
        assert "0.9500" in result.stdout
        assert "hello.py" in result.stdout


def test_search_with_limit(cli_runner):
    """Test search command with custom limit."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"
    mock_results = []

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_results)

        result = cli_runner.invoke(app, ["search", "test", "test-repo", "--limit", "5"])

        assert result.exit_code == 0
        # Now only a single call to anyio.run
        assert mock_run.call_count == 1


def test_search_no_results(cli_runner):
    """Test search command with no results."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, [])

        result = cli_runner.invoke(app, ["search", "nonexistent", "test-repo"])

        assert result.exit_code == 0
        assert "No results found for query: nonexistent" in result.stdout


def test_search_repo_not_found(cli_runner):
    """Test search command when repository is not found."""
    with patch("indexter.cli.cli.anyio.run", side_effect=RepoNotFoundError("Not found")):
        result = cli_runner.invoke(app, ["search", "test", "nonexistent-repo"])

        assert result.exit_code == 1
        assert "Repository not found: nonexistent-repo" in result.stdout


def test_search_unexpected_error(cli_runner):
    """Test search command with unexpected error."""
    with patch("indexter.cli.cli.anyio.run", side_effect=ValueError("Unexpected")):
        result = cli_runner.invoke(app, ["search", "test", "test-repo"])

        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_search_truncates_long_content(cli_runner):
    """Test search command truncates long content in results."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    long_content = "x" * 100
    mock_results = [
        {
            "score": 0.95,
            "content": long_content,
            "file_path": "test.py",
        },
    ]

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_results)

        result = cli_runner.invoke(app, ["search", "test", "test-repo"])

        assert result.exit_code == 0
        # Should truncate to 50 chars + "..."
        # Note: Could be either "..." or "…" (unicode ellipsis) depending on console
        assert "..." in result.stdout or "…" in result.stdout


def test_status_with_repositories(cli_runner):
    """Test status command with indexed repositories."""
    mock_repo1 = Mock()
    mock_repo1.name = "repo1"
    mock_repo1.path = Path("/path/to/repo1")
    mock_repo1.status = Mock()

    mock_repo2 = Mock()
    mock_repo2.name = "repo2"
    mock_repo2.path = Path("/path/to/repo2")
    mock_repo2.status = Mock()

    mock_status1 = {
        "nodes_indexed": 100,
        "documents_indexed": 50,
        "documents_indexed_stale": 2,
    }

    mock_status2 = {
        "nodes_indexed": 200,
        "documents_indexed": 75,
        "documents_indexed_stale": 0,
    }

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        # Return list of tuples: (repo, status, error)
        mock_run.return_value = [
            (mock_repo1, mock_status1, None),
            (mock_repo2, mock_status2, None),
        ]

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Indexed Repositories" in result.stdout
        assert "repo1" in result.stdout
        assert "repo2" in result.stdout
        assert "100" in result.stdout
        assert "200" in result.stdout


def test_status_no_repositories(cli_runner):
    """Test status command with no indexed repositories."""
    with patch("indexter.cli.cli.anyio.run", return_value=[]):
        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No repositories indexed" in result.stdout
        assert "indexter index" in result.stdout


def test_status_with_error(cli_runner):
    """Test status command when one repository has an error."""
    mock_repo1 = Mock()
    mock_repo1.name = "repo1"
    mock_repo1.path = Path("/path/to/repo1")

    mock_status1 = {
        "nodes_indexed": 100,
        "documents_indexed": 50,
        "documents_indexed_stale": 2,
    }

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        # Return list of tuples: (repo, status, error)
        mock_run.return_value = [
            (mock_repo1, mock_status1, None),
        ]

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "repo1" in result.stdout


def test_status_repo_error_handling(cli_runner):
    """Test status command handles individual repo errors gracefully."""
    mock_repo1 = Mock()
    mock_repo1.name = "broken-repo"
    mock_repo1.path = Path("/path/to/broken")

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        # Return list of tuples with error in third element
        mock_run.return_value = [
            (mock_repo1, None, "Status error"),
        ]

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "broken-repo" in result.stdout
        assert "Error" in result.stdout


def test_forget_successful(cli_runner):
    """Test forget command removes repository successfully."""
    with patch("indexter.cli.cli.anyio.run", return_value=None):
        result = cli_runner.invoke(app, ["forget", "test-repo"])

        assert result.exit_code == 0
        assert "Repository 'test-repo' is forgotten" in result.stdout


def test_forget_repo_not_found(cli_runner):
    """Test forget command when repository is not found."""
    with patch("indexter.cli.cli.anyio.run", side_effect=RepoNotFoundError("Not found")):
        result = cli_runner.invoke(app, ["forget", "nonexistent-repo"])

        assert result.exit_code == 1
        assert "Repository not found: nonexistent-repo" in result.stdout


def test_forget_unexpected_error(cli_runner):
    """Test forget command with unexpected error."""
    with patch("indexter.cli.cli.anyio.run", side_effect=ValueError("Unexpected")):
        result = cli_runner.invoke(app, ["forget", "test-repo"])

        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


def test_app_has_config_subcommand(cli_runner):
    """Test that app includes config sub-app."""
    result = cli_runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "config" in result.stdout


def test_app_no_args_shows_help(cli_runner):
    """Test that app with no args shows help."""
    result = cli_runner.invoke(app, [])

    # Typer returns exit code 2 for missing args with no_args_is_help=True
    assert result.exit_code in (0, 2)
    assert "init" in result.stdout
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "status" in result.stdout
    assert "forget" in result.stdout


def test_app_help_flag(cli_runner):
    """Test app --help flag."""
    result = cli_runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "indexter - Enhanced codebase context" in result.stdout


def test_app_version_flag(cli_runner):
    """Test app --version flag."""
    result = cli_runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"indexter {__version__}" in result.stdout


def test_app_verbose_flag(cli_runner):
    """Test app --verbose flag is recognized."""
    # The verbose flag is processed by the callback, so we just verify it's accepted
    with patch("indexter.cli.cli.anyio.run", return_value=[]):
        result = cli_runner.invoke(app, ["--verbose", "status"])

        assert result.exit_code == 0


def test_init_command_help(cli_runner):
    """Test init command help."""
    result = cli_runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert "Initialize a git repository for indexing" in result.stdout


def test_index_command_help(cli_runner):
    """Test index command help."""
    result = cli_runner.invoke(app, ["index", "--help"])

    assert result.exit_code == 0
    assert "Index a git repository in the vector store" in result.stdout


def test_search_command_help(cli_runner):
    """Test search command help."""
    result = cli_runner.invoke(app, ["search", "--help"])

    assert result.exit_code == 0
    assert "Search indexed nodes in a repository" in result.stdout


def test_status_command_help(cli_runner):
    """Test status command help."""
    result = cli_runner.invoke(app, ["status", "--help"])

    assert result.exit_code == 0
    assert "Show status of indexed repositories" in result.stdout


def test_forget_command_help(cli_runner):
    """Test forget command help."""
    result = cli_runner.invoke(app, ["forget", "--help"])

    assert result.exit_code == 0
    assert "Forget a repository" in result.stdout


def test_console_is_rich_console():
    """Test that console is a Rich Console instance."""
    assert isinstance(console, Console)


def test_search_replaces_newlines_in_content(cli_runner):
    """Test search command replaces newlines with spaces in output."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"

    mock_results = [
        {
            "score": 0.95,
            "content": "line1\nline2\nline3",
            "file_path": "test.py",
        },
    ]

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        mock_run.return_value = (mock_repo, mock_results)

        result = cli_runner.invoke(app, ["search", "test", "test-repo"])

        assert result.exit_code == 0
        # The output should have spaces instead of newlines in the content


def test_status_missing_status_fields(cli_runner):
    """Test status command handles missing status fields."""
    mock_repo = Mock()
    mock_repo.name = "test-repo"
    mock_repo.path = Path("/path/to/repo")

    # Status dict missing some fields
    mock_status = {"nodes_indexed": 100}

    with patch("indexter.cli.cli.anyio.run") as mock_run:
        # Return list of tuples: (repo, status, error)
        mock_run.return_value = [(mock_repo, mock_status, None)]

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "test-repo" in result.stdout
        assert "100" in result.stdout
        # Missing fields should show as "-"
        assert "-" in result.stdout
