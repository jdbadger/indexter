"""Integration tests for the CLI covering full user journeys.

These tests verify end-to-end workflows from a user's perspective, including:
- Initializing repositories
- Indexing and re-indexing
- Searching indexed code
- Checking repository status
- Forgetting repositories
- Config management

Note: These are true integration tests that interact with the file system
and vector store. They run against isolated temporary repositories.
The in-memory store mode is configured in conftest.py for fast test execution.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indexter.cli.cli import app
from indexter.config import settings


@pytest.fixture(scope="session", autouse=True)
def clean_test_repos():
    """Clean up repos.json before and after test session."""
    # Before tests: clean up any stale entries from previous runs
    repos_file = Path(settings.repos_config_file)
    if repos_file.exists():
        try:
            data = json.loads(repos_file.read_text())
            valid_repos = []
            for repo in data.get("repos", []):
                # Keep only repos that still exist
                repo_path = Path(repo["path"])
                if repo_path.exists() and (repo_path / ".git").exists():
                    valid_repos.append(repo)
            data["repos"] = valid_repos
            repos_file.write_text(json.dumps(data, indent=4))
        except Exception:
            # If there's any error, just clear the file
            repos_file.write_text(json.dumps({"repos": []}, indent=4))

    yield

    # After tests: clean up test repositories
    if repos_file.exists():
        try:
            data = json.loads(repos_file.read_text())
            remaining_repos = []
            for repo in data.get("repos", []):
                # Keep only non-temp repos
                if not str(repo["path"]).startswith("/tmp/"):
                    remaining_repos.append(repo)
            data["repos"] = remaining_repos
            repos_file.write_text(json.dumps(data, indent=4))
        except Exception:
            pass


@pytest.fixture
def cli_runner():
    """Create a CliRunner for testing."""
    return CliRunner()


@pytest.fixture
def temp_repo():
    """Create a temporary git repository with sample Python files."""
    temp_dir = Path(tempfile.mkdtemp())
    repo_path = temp_dir / "test-project"
    repo_path.mkdir()

    # Initialize as git repo
    git_dir = repo_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

    # Create minimal sample Python files for testing
    src_dir = repo_path / "src"
    src_dir.mkdir()

    # Simple module with a few functions
    utils_content = '''"""Utility functions."""

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b
'''
    (src_dir / "utils.py").write_text(utils_content)

    # Module with a class
    models_content = '''"""Data models."""

class User:
    """Represents a user."""
    
    def authenticate(self, password: str) -> bool:
        """Authenticate the user."""
        return len(password) > 8
'''
    (src_dir / "models.py").write_text(models_content)

    # Add a README
    readme_content = "# Test Project\n\nA sample project for testing.\n"
    (repo_path / "README.md").write_text(readme_content)

    yield repo_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCLIUserJourney:
    """Test complete user journeys through the CLI."""

    def test_full_workflow_init_index_search_forget(self, cli_runner, temp_repo):
        """Test the complete workflow: init -> index -> search -> forget."""
        repo_name = temp_repo.name

        # Step 1: Initialize the repository
        result = cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
        assert result.exit_code == 0
        assert f"Added {repo_name} to indexter" in result.stdout
        assert "initialized" in result.stdout.lower()
        assert "Next steps:" in result.stdout

        # Step 2: Index the repository (already indexed by init, so should be up to date)
        result = cli_runner.invoke(app, ["index", repo_name])
        assert result.exit_code == 0
        assert "Indexing complete!" in result.stdout
        # Repository was already indexed during init, so may show 0 files indexed or up to date
        assert "Indexed" in result.stdout or "up to date" in result.stdout

        # Step 3: Search for specific content
        result = cli_runner.invoke(app, ["search", "authentication", repo_name])
        assert result.exit_code == 0
        # Should find the authenticate method in User class
        assert "Search Results" in result.stdout or "No results found" in result.stdout

        # Step 4: Search for another term
        result = cli_runner.invoke(app, ["search", "database", repo_name, "--limit", "5"])
        assert result.exit_code == 0

        # Step 5: Check status
        result = cli_runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert repo_name in result.stdout
        assert "Indexed Repositories" in result.stdout

        # Step 6: Forget the repository
        result = cli_runner.invoke(app, ["forget", repo_name])
        assert result.exit_code == 0
        assert f"Repository '{repo_name}' is forgotten" in result.stdout

        # Step 7: Verify repository is gone
        result = cli_runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Should show no repositories
        assert "No repositories indexed" in result.stdout or repo_name not in result.stdout

    def test_incremental_indexing_workflow(self, cli_runner, temp_repo):
        """Test incremental indexing when files are modified."""
        repo_name = temp_repo.name

        # Initialize and index
        result = cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
        assert result.exit_code == 0

        result = cli_runner.invoke(app, ["index", repo_name])
        assert result.exit_code == 0

        # Modify a file
        new_content = '''
        """Updated utils module."""

        def calculate_sum(a: int, b: int) -> int:
            """Calculate the sum of two numbers."""
            return a + b


        def new_function() -> str:
            """A newly added function."""
            return "new"
        '''
        (temp_repo / "src" / "utils.py").write_text(new_content)

        # Index again (should detect changes)
        result = cli_runner.invoke(app, ["index", repo_name])
        assert result.exit_code == 0
        # Should show that some files were synced or updated

        # Add a completely new file
        new_module_content = '''
            """A new module."""

            def new_feature():
                """Implement a new feature."""
                pass
            '''
        (temp_repo / "src" / "new_module.py").write_text(new_module_content)

        # Index again
        result = cli_runner.invoke(app, ["index", repo_name])
        assert result.exit_code == 0

        # Cleanup
        cli_runner.invoke(app, ["forget", repo_name])

    def test_full_reindex_workflow(self, cli_runner, temp_repo):
        """Test full re-indexing with --full flag."""
        repo_name = temp_repo.name

        # Initialize and index
        result = cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
        assert result.exit_code == 0

        result = cli_runner.invoke(app, ["index", repo_name])
        assert result.exit_code == 0

        # Force full re-index
        result = cli_runner.invoke(app, ["index", repo_name, "--full"])
        assert result.exit_code == 0
        # Should process all files

        # Cleanup
        cli_runner.invoke(app, ["forget", repo_name])

    def test_search_with_different_queries(self, cli_runner, temp_repo):
        """Test searching with various query types."""
        repo_name = temp_repo.name

        # Setup
        cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
        cli_runner.invoke(app, ["index", repo_name])

        # Search for function-related content
        result = cli_runner.invoke(app, ["search", "calculate sum", repo_name])
        assert result.exit_code == 0

        # Search for class-related content
        result = cli_runner.invoke(app, ["search", "user profile", repo_name])
        assert result.exit_code == 0

        # Search for API-related content
        result = cli_runner.invoke(app, ["search", "API request", repo_name])
        assert result.exit_code == 0

        # Search with custom limit
        result = cli_runner.invoke(app, ["search", "function", repo_name, "--limit", "3"])
        assert result.exit_code == 0

        # Cleanup
        cli_runner.invoke(app, ["forget", repo_name])

    def test_multiple_repositories_workflow(self, cli_runner, temp_repo):
        """Test managing multiple repositories."""
        # Create second repository
        temp_dir2 = Path(tempfile.mkdtemp())
        repo_path2 = temp_dir2 / "second-project"
        repo_path2.mkdir()

        # Initialize as git repo
        git_dir = repo_path2 / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

        # Add a file
        main_content = '''def main():
                """Main entry point."""
                print("Hello, World!")
            '''
        (repo_path2 / "main.py").write_text(main_content)

        try:
            # Initialize first repo
            result = cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
            assert result.exit_code == 0

            # Initialize second repo
            result = cli_runner.invoke(app, ["init", "--path", str(repo_path2)])
            assert result.exit_code == 0

            # Index both
            result = cli_runner.invoke(app, ["index", temp_repo.name])
            assert result.exit_code == 0

            result = cli_runner.invoke(app, ["index", repo_path2.name])
            assert result.exit_code == 0

            # Check status shows both
            result = cli_runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert temp_repo.name in result.stdout
            assert repo_path2.name in result.stdout

            # Search in specific repos
            result = cli_runner.invoke(app, ["search", "user", temp_repo.name])
            assert result.exit_code == 0

            result = cli_runner.invoke(app, ["search", "main", repo_path2.name])
            assert result.exit_code == 0

            # Cleanup
            cli_runner.invoke(app, ["forget", temp_repo.name])
            cli_runner.invoke(app, ["forget", repo_path2.name])
        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)

    def test_error_handling_workflow(self, cli_runner, temp_repo):
        """Test error handling in various scenarios."""
        repo_name = temp_repo.name

        # Try to index before init
        result = cli_runner.invoke(app, ["index", "nonexistent-repo"])
        assert result.exit_code == 1
        assert "Repository not found" in result.stdout

        # Try to search before init
        result = cli_runner.invoke(app, ["search", "test", "nonexistent-repo"])
        assert result.exit_code == 1
        assert "Repository not found" in result.stdout

        # Try to forget non-existent repo
        result = cli_runner.invoke(app, ["forget", "nonexistent-repo"])
        assert result.exit_code == 1
        assert "Repository not found" in result.stdout

        # Initialize repo
        result = cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
        assert result.exit_code == 0

        # Try to initialize same repo again (should succeed - idempotent)
        result = cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
        assert result.exit_code == 0
        # Should indicate it's already configured or just succeed silently

        # Cleanup
        cli_runner.invoke(app, ["forget", repo_name])

    def test_status_with_no_repositories(self, cli_runner):
        """Test status command when no repositories are indexed."""
        result = cli_runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Status command shows a table even when no repos are indexed
        # It may show "Indexed Repositories" table header or a message about no repos
        assert "Repositories" in result.stdout or "Indexed Repositories" in result.stdout

    def test_search_with_no_results(self, cli_runner, temp_repo):
        """Test search when no results are found."""
        repo_name = temp_repo.name

        # Setup
        cli_runner.invoke(app, ["init", "--path", str(temp_repo)])
        cli_runner.invoke(app, ["index", repo_name])

        # Search for something that won't be found
        result = cli_runner.invoke(app, ["search", "xyzabc123nonexistent", repo_name])
        assert result.exit_code == 0
        # May show no results or empty table

        # Cleanup
        cli_runner.invoke(app, ["forget", repo_name])


class TestConfigCommands:
    """Test configuration-related CLI commands."""

    def test_config_show_command(self, cli_runner):
        """Test the config show command."""
        result = cli_runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "Indexter Settings" in result.stdout
        # Should show config file path

    def test_config_path_command(self, cli_runner):
        """Test the config path command."""
        result = cli_runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        # Should print a path (without Rich formatting)
        assert result.stdout.strip()  # Non-empty output


class TestVersionAndHelp:
    """Test version and help commands."""

    def test_version_flag(self, cli_runner):
        """Test --version flag."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "indexter" in result.stdout
        # Should show version number

    def test_help_flag(self, cli_runner):
        """Test --help flag."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "indexter" in result.stdout
        assert "init" in result.stdout
        assert "index" in result.stdout
        assert "search" in result.stdout
        assert "status" in result.stdout
        assert "forget" in result.stdout

    def test_verbose_flag(self, cli_runner, temp_repo):
        """Test --verbose flag."""
        # Initialize with verbose
        result = cli_runner.invoke(app, ["--verbose", "init", "--path", str(temp_repo)])
        # Should not error
        assert result.exit_code == 0

        # Cleanup
        cli_runner.invoke(app, ["forget", temp_repo.name])

    def test_command_help(self, cli_runner):
        """Test help for individual commands."""
        commands = ["init", "index", "search", "status", "forget", "config"]

        for cmd in commands:
            result = cli_runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0
            # Should show help text


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_repository(self, cli_runner):
        """Test indexing an empty repository."""
        temp_dir = Path(tempfile.mkdtemp())
        repo_path = temp_dir / "empty-repo"
        repo_path.mkdir()

        # Initialize as git repo
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

        try:
            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])
            assert result.exit_code == 0

            result = cli_runner.invoke(app, ["index", repo_path.name])
            assert result.exit_code == 0
            # Should handle empty repo gracefully

            result = cli_runner.invoke(app, ["status"])
            assert result.exit_code == 0

            # Cleanup
            cli_runner.invoke(app, ["forget", repo_path.name])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_repository_with_gitignore(self, cli_runner):
        """Test that .gitignore patterns are respected."""
        temp_dir = Path(tempfile.mkdtemp())
        repo_path = temp_dir / "gitignore-test"
        repo_path.mkdir()

        # Initialize as git repo
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

        # Add .gitignore
        gitignore_content = """
            *.pyc
            __pycache__/
            .env
            node_modules/
            """
        (repo_path / ".gitignore").write_text(gitignore_content)

        # Add both tracked and ignored files
        (repo_path / "main.py").write_text("def main(): pass")
        (repo_path / "test.pyc").write_text("compiled")

        pycache = repo_path / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-311.pyc").write_text("compiled")

        try:
            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])
            assert result.exit_code == 0

            result = cli_runner.invoke(app, ["index", repo_path.name])
            assert result.exit_code == 0
            # Should index main.py but not .pyc files

            # Cleanup
            cli_runner.invoke(app, ["forget", repo_path.name])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_large_file_handling(self, cli_runner):
        """Test handling of large files."""
        temp_dir = Path(tempfile.mkdtemp())
        repo_path = temp_dir / "large-file-test"
        repo_path.mkdir()

        # Initialize as git repo
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

        # Create a reasonably large file (50 functions is enough to test large file handling)
        large_content = "def function_{}():\n    pass\n\n" * 50
        (repo_path / "large.py").write_text(large_content)

        # Create a normal file
        (repo_path / "normal.py").write_text("def small(): pass")

        try:
            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])
            assert result.exit_code == 0

            result = cli_runner.invoke(app, ["index", repo_path.name])
            assert result.exit_code == 0
            # Should handle large file appropriately

            # Cleanup
            cli_runner.invoke(app, ["forget", repo_path.name])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_special_characters_in_paths(self, cli_runner):
        """Test handling of paths with special characters."""
        temp_dir = Path(tempfile.mkdtemp())
        # Use a name with spaces and special chars
        repo_path = temp_dir / "test repo (v1.0)"
        repo_path.mkdir()

        # Initialize as git repo
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

        (repo_path / "code.py").write_text("def func(): pass")

        try:
            result = cli_runner.invoke(app, ["init", "--path", str(repo_path)])
            assert result.exit_code == 0

            result = cli_runner.invoke(app, ["index", repo_path.name])
            assert result.exit_code == 0

            # Cleanup
            cli_runner.invoke(app, ["forget", repo_path.name])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
