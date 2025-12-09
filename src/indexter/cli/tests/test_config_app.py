"""Unit tests for CLI config commands."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from indexter.cli.config import config_app


@pytest.fixture
def mock_settings():
    """Mock settings with global config file."""
    with patch("indexter.cli.config.settings") as mock:
        mock_path = MagicMock(spec=Path)
        mock_path.__str__ = Mock(return_value="/home/user/.config/indexter/config.toml")
        mock.global_config_file = mock_path
        mock.create_global_config.return_value = None
        yield mock


@pytest.fixture
def mock_console():
    """Mock the console object."""
    with patch("indexter.cli.config.console") as mock:
        yield mock


# --- config show command Tests ---


def test_config_show_file_exists(cli_runner, mock_settings, mock_console):
    """Test config show when config file exists."""
    mock_settings.global_config_file.exists.return_value = True
    mock_settings.global_config_file.read_text.return_value = "[store]\nhost = 'localhost'\n"

    result = cli_runner.invoke(config_app, ["show"])

    assert result.exit_code == 0
    mock_settings.global_config_file.read_text.assert_called_once()
    mock_console.print.assert_called()


def test_config_show_file_not_exists(cli_runner, mock_settings, mock_console):
    """Test config show when config file doesn't exist."""
    mock_settings.global_config_file.exists.return_value = False

    result = cli_runner.invoke(config_app, ["show"])

    assert result.exit_code == 0
    mock_settings.global_config_file.read_text.assert_not_called()
    # Check that message about no config is printed
    assert any("No global config" in str(call) for call in mock_console.print.call_args_list)


def test_config_show_displays_config_path(cli_runner, mock_settings, mock_console):
    """Test config show displays the config file path."""
    mock_settings.global_config_file.exists.return_value = False

    result = cli_runner.invoke(config_app, ["show"])

    assert result.exit_code == 0
    # Verify path is displayed
    assert any("Config file:" in str(call) for call in mock_console.print.call_args_list)


# --- config init command Tests ---


def test_config_init_creates_new_config(cli_runner, mock_settings, mock_console):
    """Test config init creates new config file."""
    mock_settings.global_config_file.exists.return_value = False

    result = cli_runner.invoke(config_app, ["init"])

    assert result.exit_code == 0
    mock_settings.create_global_config.assert_called_once()
    # Check success message
    assert any("Created global config" in str(call) for call in mock_console.print.call_args_list)


def test_config_init_file_exists_no_force(cli_runner, mock_settings, mock_console):
    """Test config init when file exists without --force flag."""
    mock_settings.global_config_file.exists.return_value = True

    result = cli_runner.invoke(config_app, ["init"])

    assert result.exit_code == 0
    mock_settings.create_global_config.assert_not_called()
    # Check warning message
    assert any("already exists" in str(call) for call in mock_console.print.call_args_list)


def test_config_init_file_exists_with_force(cli_runner, mock_settings, mock_console):
    """Test config init with --force flag overwrites existing file."""
    mock_settings.global_config_file.exists.return_value = True
    mock_settings.global_config_file.unlink = Mock()

    result = cli_runner.invoke(config_app, ["init", "--force"])

    assert result.exit_code == 0
    mock_settings.global_config_file.unlink.assert_called_once()
    mock_settings.create_global_config.assert_called_once()


def test_config_init_with_force_flag_short(cli_runner, mock_settings, mock_console):
    """Test config init with -f short flag."""
    mock_settings.global_config_file.exists.return_value = True
    mock_settings.global_config_file.unlink = Mock()

    result = cli_runner.invoke(config_app, ["init", "-f"])

    assert result.exit_code == 0
    mock_settings.global_config_file.unlink.assert_called_once()
    mock_settings.create_global_config.assert_called_once()


def test_config_init_displays_help_message(cli_runner, mock_settings, mock_console):
    """Test config init displays helpful instructions."""
    mock_settings.global_config_file.exists.return_value = False

    result = cli_runner.invoke(config_app, ["init"])

    assert result.exit_code == 0
    # Check for help messages about customization
    assert any(
        "customize" in str(call).lower() or "edit" in str(call).lower()
        for call in mock_console.print.call_args_list
    )


# --- config edit command Tests ---


def test_config_edit_file_exists(cli_runner, mock_settings):
    """Test config edit when file exists."""
    mock_settings.global_config_file.exists.return_value = True

    with (
        patch("indexter.cli.config.subprocess.run") as mock_run,
        patch.dict("os.environ", {"EDITOR": "vim"}),
    ):
        result = cli_runner.invoke(config_app, ["edit"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        # Verify vim was called with config path
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "vim"


def test_config_edit_file_not_exists_creates_it(cli_runner, mock_settings, mock_console):
    """Test config edit creates file if it doesn't exist."""
    mock_settings.global_config_file.exists.return_value = False

    with (
        patch("indexter.cli.config.subprocess.run") as mock_run,
        patch.dict("os.environ", {"EDITOR": "vim"}),
    ):
        result = cli_runner.invoke(config_app, ["edit"])

        assert result.exit_code == 0
        mock_settings.create_global_config.assert_called_once()
        mock_run.assert_called_once()


def test_config_edit_uses_custom_editor(cli_runner, mock_settings):
    """Test config edit uses $EDITOR environment variable."""
    mock_settings.global_config_file.exists.return_value = True

    with (
        patch("indexter.cli.config.subprocess.run") as mock_run,
        patch.dict("os.environ", {"EDITOR": "nano"}),
    ):
        result = cli_runner.invoke(config_app, ["edit"])

        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "nano"


def test_config_edit_defaults_to_vim(cli_runner, mock_settings):
    """Test config edit defaults to vim when $EDITOR not set."""
    mock_settings.global_config_file.exists.return_value = True

    with (
        patch("indexter.cli.config.subprocess.run") as mock_run,
        patch.dict("os.environ", {}, clear=True),
    ):  # No EDITOR set
        result = cli_runner.invoke(config_app, ["edit"])

        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "vim"


def test_config_edit_editor_not_found(cli_runner, mock_settings, mock_console):
    """Test config edit handles editor not found error."""
    mock_settings.global_config_file.exists.return_value = True

    with (
        patch("indexter.cli.config.subprocess.run", side_effect=FileNotFoundError()),
        patch.dict("os.environ", {"EDITOR": "nonexistent"}),
    ):
        result = cli_runner.invoke(config_app, ["edit"])

        assert result.exit_code == 1
        # Check error message was printed to console
        assert mock_console.print.called
        # Check that error messages contain relevant information
        call_args_str = " ".join(str(call) for call in mock_console.print.call_args_list)
        assert "not found" in call_args_str.lower() or "error" in call_args_str.lower()


def test_config_edit_passes_correct_path(cli_runner, mock_settings):
    """Test config edit passes correct path to editor."""
    mock_path = MagicMock(spec=Path)
    mock_path.__str__ = Mock(return_value="/test/path/config.toml")
    mock_path.exists.return_value = True
    mock_settings.global_config_file = mock_path

    with (
        patch("indexter.cli.config.subprocess.run") as mock_run,
        patch.dict("os.environ", {"EDITOR": "vim"}),
    ):
        result = cli_runner.invoke(config_app, ["edit"])

        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "/test/path/config.toml" in call_args[1]


# --- config path command Tests ---


def test_config_path_prints_path(cli_runner, mock_settings, capsys):
    """Test config path prints the config file path."""
    mock_path = MagicMock(spec=Path)
    mock_path.__str__ = Mock(return_value="/home/user/.config/indexter/config.toml")
    mock_settings.global_config_file = mock_path

    result = cli_runner.invoke(config_app, ["path"])

    assert result.exit_code == 0
    # The path command uses print() not console.print()
    assert "/home/user/.config/indexter/config.toml" in result.stdout


def test_config_path_output_format(cli_runner, mock_settings):
    """Test config path outputs plain text without Rich formatting."""
    mock_path = MagicMock(spec=Path)
    mock_path.__str__ = Mock(return_value="/test/config.toml")
    mock_settings.global_config_file = mock_path

    result = cli_runner.invoke(config_app, ["path"])

    assert result.exit_code == 0
    # Should be plain output (no Rich formatting markers)
    assert "/test/config.toml" in result.stdout
    assert result.stdout.strip() == "/test/config.toml"


def test_config_path_for_scripting(cli_runner, mock_settings):
    """Test config path is suitable for use in scripts."""
    mock_path = MagicMock(spec=Path)
    mock_path.__str__ = Mock(return_value="/path/to/config.toml")
    mock_settings.global_config_file = mock_path

    result = cli_runner.invoke(config_app, ["path"])

    assert result.exit_code == 0
    # Should be just the path with newline
    assert result.stdout.strip() == "/path/to/config.toml"
    # No extra formatting or messages
    assert result.stdout.count("\n") == 1


# --- Integration Tests ---


def test_config_app_has_all_commands(cli_runner):
    """Test config app has all expected commands."""
    result = cli_runner.invoke(config_app, ["--help"])

    assert result.exit_code == 0
    assert "show" in result.stdout
    assert "init" in result.stdout
    assert "edit" in result.stdout
    assert "path" in result.stdout


def test_config_app_requires_command(cli_runner):
    """Test config app requires a subcommand."""
    result = cli_runner.invoke(config_app, [])

    # Typer exits with code 2 when no command provided (shows help)
    assert result.exit_code in [0, 2]  # Either shows help (0) or error (2)
    # Should show available commands in output
    assert "show" in result.stdout or "Usage" in result.stdout


def test_config_show_command_help(cli_runner):
    """Test config show command has help text."""
    result = cli_runner.invoke(config_app, ["show", "--help"])

    assert result.exit_code == 0
    assert "Show global configuration" in result.stdout or "show" in result.stdout.lower()


def test_config_init_command_help(cli_runner):
    """Test config init command has help text."""
    result = cli_runner.invoke(config_app, ["init", "--help"])

    assert result.exit_code == 0
    assert "--force" in result.stdout or "-f" in result.stdout


def test_config_edit_command_help(cli_runner):
    """Test config edit command has help text."""
    result = cli_runner.invoke(config_app, ["edit", "--help"])

    assert result.exit_code == 0
    assert "edit" in result.stdout.lower() or "Open" in result.stdout


def test_config_path_command_help(cli_runner):
    """Test config path command has help text."""
    result = cli_runner.invoke(config_app, ["path", "--help"])

    assert result.exit_code == 0
    assert "path" in result.stdout.lower() or "Print" in result.stdout
