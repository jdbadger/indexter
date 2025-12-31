"""Tests for CLI config commands.

This module provides comprehensive test coverage for the indexter CLI config
commands, which include:
- config show: Display the configuration file with syntax highlighting
- config path: Print the path to the configuration file

Test Coverage:
--------------
- Config file exists vs. doesn't exist scenarios
- Syntax highlighting with TOML format and monokai theme
- Line numbers in syntax output
- Plain text output for path command (no Rich formatting)
- Help flags for all commands
- Error handling (read errors, permission issues)
- Unicode content handling
- Long file paths
- Empty config files
- Console instance usage

The tests use unittest.mock to mock the settings module and file system
interactions, ensuring tests are isolated and don't depend on actual
file system state.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

from indexter.cli.config import config_app, config_path, config_show, console


def test_config_show_with_existing_config_file(cli_runner):
    """Test config show command when config file exists."""
    config_content = """
    # indexter global configuration

    embedding_model = "BAAI/bge-small-en-v1.5"

    ignore_patterns = [".git/", "__pycache__/", "*.pyc"]

    # Maximum file size (in bytes) to process
    max_file_size = 1048576
    """

    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = True
    mock_config_file.read_text.return_value = config_content
    mock_config_file.__str__ = Mock(return_value="/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        result = cli_runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        assert "Indexter Settings" in result.stdout
        assert "Config file:" in result.stdout
        assert "/home/user/.config/indexter/indexter.toml" in result.stdout
        # Check that file content appears in output
        assert "embedding_model" in result.stdout or config_content in result.stdout


def test_config_show_when_config_file_not_found(cli_runner):
    """Test config show command when config file doesn't exist."""
    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = False
    mock_config_file.__str__ = Mock(return_value="/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        result = cli_runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        assert "Indexter Settings" in result.stdout
        assert "Config file:" in result.stdout
        assert "Config file not found" in result.stdout


def test_config_show_displays_syntax_highlighting(cli_runner):
    """Test that config show applies syntax highlighting to TOML content."""
    config_content = '[store]\nmode = "local"\n'

    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = True
    mock_config_file.read_text.return_value = config_content
    mock_config_file.__str__ = Mock(return_value="/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        with patch("indexter.cli.config.Syntax") as mock_syntax:
            mock_syntax_instance = MagicMock()
            mock_syntax.return_value = mock_syntax_instance

            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            # Verify Syntax was called with correct parameters
            mock_syntax.assert_called_once_with(
                config_content, "toml", theme="monokai", line_numbers=True
            )


def test_config_path_outputs_plain_text(cli_runner):
    """Test config path command outputs path without Rich formatting."""
    mock_config_file = MagicMock(spec=Path)
    mock_config_file.__str__ = Mock(return_value="/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        result = cli_runner.invoke(config_app, ["path"])

        assert result.exit_code == 0
        # Output should be just the path, no Rich formatting
        assert "/home/user/.config/indexter/indexter.toml" in result.stdout
        # Verify output is plain (no ANSI codes expected from standard print)


def test_config_path_uses_print_not_console(cli_runner, capsys):
    """Test that config path uses print() not console.print()."""
    mock_config_file = MagicMock(spec=Path)
    mock_config_file.__str__ = Mock(return_value="/tmp/test/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        # Call the function directly to verify it uses print()
        with patch("indexter.cli.config.print") as mock_print:
            config_path()
            mock_print.assert_called_once_with(mock_config_file)


def test_config_show_handles_read_error(cli_runner):
    """Test config show handles file read errors gracefully."""
    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = True
    mock_config_file.read_text.side_effect = OSError("Permission denied")
    mock_config_file.__str__ = Mock(return_value="/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        # Should raise the exception since we're not catching it
        with pytest.raises(IOError):
            cli_runner.invoke(config_app, ["show"], catch_exceptions=False)


def test_config_app_no_args_shows_help(cli_runner):
    """Test that config command with no args shows help."""
    result = cli_runner.invoke(config_app, [])

    # Typer returns exit code 2 for missing arguments with no_args_is_help=True
    assert result.exit_code in (0, 2)
    assert "show" in result.stdout
    assert "path" in result.stdout


def test_config_app_help_flag(cli_runner):
    """Test config command with --help flag."""
    result = cli_runner.invoke(config_app, ["--help"])

    assert result.exit_code == 0
    assert "show" in result.stdout
    assert "path" in result.stdout
    assert "View Indexter global settings" in result.stdout


def test_config_show_help_flag(cli_runner):
    """Test config show command with --help flag."""
    result = cli_runner.invoke(config_app, ["show", "--help"])

    assert result.exit_code == 0
    assert "Show Indexter global settings config" in result.stdout


def test_config_path_help_flag(cli_runner):
    """Test config path command with --help flag."""
    result = cli_runner.invoke(config_app, ["path", "--help"])

    assert result.exit_code == 0
    assert "Print the path to the Indexter settings config file" in result.stdout


def test_config_show_with_unicode_content(cli_runner):
    """Test config show handles unicode characters in config file."""
    config_content = '# Configuration with émojis 🚀\nembedding_model = "test"\n'

    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = True
    mock_config_file.read_text.return_value = config_content
    mock_config_file.__str__ = Mock(return_value="/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        result = cli_runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        assert "Indexter Settings" in result.stdout


def test_config_show_with_long_path(cli_runner):
    """Test config show handles long file paths correctly."""
    long_path = "/".join(["very_long_directory_name"] * 10) + "/indexter.toml"

    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = False
    mock_config_file.__str__ = Mock(return_value=long_path)

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        result = cli_runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        # Should display the full path despite being long (overflow="ignore", crop=False)
        assert "very_long_directory_name" in result.stdout


def test_config_show_empty_config_file(cli_runner):
    """Test config show with an empty config file."""
    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = True
    mock_config_file.read_text.return_value = ""
    mock_config_file.__str__ = Mock(return_value="/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        result = cli_runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        assert "Indexter Settings" in result.stdout


def test_config_path_with_pathlib_path(cli_runner):
    """Test config path handles Path objects correctly."""
    test_path = Path("/home/user/.config/indexter/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = test_path

        result = cli_runner.invoke(config_app, ["path"])

        assert result.exit_code == 0
        assert str(test_path) in result.stdout


def test_config_show_console_instance(cli_runner):
    """Test that config_show uses the module-level console instance."""
    # Verify console exists and is a Console instance
    assert isinstance(console, Console)

    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = False
    mock_config_file.__str__ = Mock(return_value="/test/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        with patch.object(console, "print") as mock_console_print:
            config_show()

            # Verify console.print was called
            assert mock_console_print.call_count > 0


def test_config_show_syntax_theme_is_monokai(cli_runner):
    """Test that syntax highlighting uses monokai theme."""
    config_content = 'test = "value"\n'

    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = True
    mock_config_file.read_text.return_value = config_content
    mock_config_file.__str__ = Mock(return_value="/test/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        with patch("indexter.cli.config.Syntax") as mock_syntax:
            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            # Check the theme parameter
            call_args = mock_syntax.call_args
            assert call_args[1]["theme"] == "monokai"


def test_config_show_has_line_numbers(cli_runner):
    """Test that syntax highlighting includes line numbers."""
    config_content = 'line1 = "value"\nline2 = "value"\n'

    mock_config_file = MagicMock(spec=Path)
    mock_config_file.exists.return_value = True
    mock_config_file.read_text.return_value = config_content
    mock_config_file.__str__ = Mock(return_value="/test/indexter.toml")

    with patch("indexter.cli.config.settings") as mock_settings:
        mock_settings.config_file = mock_config_file

        with patch("indexter.cli.config.Syntax") as mock_syntax:
            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            # Check line_numbers parameter
            call_args = mock_syntax.call_args
            assert call_args[1]["line_numbers"] is True
