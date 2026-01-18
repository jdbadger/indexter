"""Comprehensive tests for the CLI config commands.

This test suite provides comprehensive coverage of the CLI config commands including:
- Unit tests for config show command (with and without config file)
- Unit tests for config path command
- Integration tests for config command workflows
- Output formatting and console display tests
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from indexter.cli.config import config_app, config_path
from indexter.cli.tests.conftest import strip_ansi


class TestConfigShow:
    """Test config show command."""

    def test_should_display_config_when_file_exists(self, cli_runner, tmp_path):
        """Test config show displays config file contents when file exists."""
        config_file = tmp_path / "config.toml"
        config_content = """# Indexter Configuration
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
max_file_size = 1048576

[store]
mode = "server"
"""
        config_file.write_text(config_content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])
            output = strip_ansi(result.stdout)

            assert result.exit_code == 0
            assert "Indexter Settings" in output
            assert str(config_file) in output
            assert "embedding_model" in output
            assert "sentence-transformers/all-MiniLM-L6-v2" in output

    def test_should_display_message_when_config_file_not_found(self, cli_runner, tmp_path):
        """Test config show displays appropriate message when config file doesn't exist."""
        config_file = tmp_path / "nonexistent.toml"

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])
            output = strip_ansi(result.stdout)

            assert result.exit_code == 0
            assert "Indexter Settings" in output
            assert str(config_file) in output
            assert "Config file not found" in output

    def test_should_display_config_with_syntax_highlighting(self, cli_runner, tmp_path):
        """Test config show uses syntax highlighting for TOML content."""
        config_file = tmp_path / "config.toml"
        config_content = """[store]
mode = "server"
url = "http://localhost:6333"
"""
        config_file.write_text(config_content)

        with patch("indexter.cli.config.settings") as mock_settings, patch("indexter.cli.config.Syntax") as mock_syntax:
            mock_settings.config_file = config_file
            mock_syntax_instance = Mock()
            mock_syntax.return_value = mock_syntax_instance

            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            # Verify Syntax was called with correct parameters
            mock_syntax.assert_called_once_with(config_content, "toml", theme="monokai", line_numbers=True)

    def test_should_display_empty_config_file(self, cli_runner, tmp_path):
        """Test config show handles empty config file gracefully."""
        config_file = tmp_path / "empty.toml"
        config_file.write_text("")

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            assert "Indexter Settings" in result.stdout

    def test_should_display_config_with_multiline_content(self, cli_runner, tmp_path):
        """Test config show displays multiline config properly."""
        config_file = tmp_path / "config.toml"
        config_content = """embedding_model = "test-model"
ignore_patterns = [
    ".git/",
    "__pycache__/",
    "*.pyc",
    "node_modules/"
]

[store]
mode = "server"

[mcp]
transport = "stdio"
"""
        config_file.write_text(config_content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])
            output = strip_ansi(result.stdout)

            assert result.exit_code == 0
            assert "ignore_patterns" in output
            assert "__pycache__/" in output
            assert "[store]" in output
            assert "[mcp]" in output

    def test_should_handle_config_with_special_characters(self, cli_runner, tmp_path):
        """Test config show handles special characters in config file."""
        config_file = tmp_path / "config.toml"
        config_content = """# Config with special chars: é, ñ, 中文
embedding_model = "model-with-special_chars-123"
api_key = "key!@#$%^&*()"
"""
        config_file.write_text(config_content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            assert "embedding_model" in result.stdout


class TestConfigPath:
    """Test config path command."""

    def test_should_print_config_file_path(self, cli_runner, tmp_path):
        """Test config path prints the config file path."""
        config_file = tmp_path / "config.toml"

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["path"])

            assert result.exit_code == 0
            assert str(config_file) in result.stdout.strip()

    def test_should_print_path_without_rich_formatting(self, cli_runner, tmp_path):
        """Test config path uses plain print (not Rich) for scripting compatibility."""
        config_file = tmp_path / "config.toml"

        with patch("indexter.cli.config.settings") as mock_settings, patch("builtins.print") as mock_print:
            mock_settings.config_file = config_file

            # Call the function directly to verify print is used
            config_path()

            # Verify plain print was called with the config file path
            mock_print.assert_called_once_with(config_file)

    def test_should_print_path_even_if_file_does_not_exist(self, cli_runner, tmp_path):
        """Test config path prints path even when config file doesn't exist."""
        config_file = tmp_path / "nonexistent.toml"

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["path"])

            assert result.exit_code == 0
            assert str(config_file) in result.stdout

    def test_should_print_absolute_path(self, cli_runner, tmp_path):
        """Test config path prints absolute path."""
        config_file = tmp_path / "subdir" / "config.toml"

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["path"])

            assert result.exit_code == 0
            # Should include the absolute path with subdir
            assert "subdir" in result.stdout
            assert str(config_file) in result.stdout

    def test_should_be_usable_in_shell_scripts(self, cli_runner, tmp_path):
        """Test config path output is suitable for command substitution."""
        config_file = tmp_path / "config.toml"

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["path"])

            assert result.exit_code == 0
            # Output should be clean (just the path, no formatting)
            output_lines = result.stdout.strip().split("\n")
            assert len(output_lines) == 1
            assert str(config_file) == output_lines[0]


class TestConfigApp:
    """Test config app (subcommand group)."""

    def test_should_display_help_when_no_args(self, cli_runner):
        """Test config command displays help when invoked without arguments."""
        result = cli_runner.invoke(config_app, [])

        # Typer returns exit code 2 for missing required arguments with no_args_is_help
        # but still displays help
        assert result.exit_code in (0, 2)
        assert "View Indexter global settings" in result.stdout or "config" in result.stdout.lower()

    def test_should_display_help_with_help_flag(self, cli_runner):
        """Test config command displays help with --help flag."""
        result = cli_runner.invoke(config_app, ["--help"])

        assert result.exit_code == 0
        assert "show" in result.stdout.lower()
        assert "path" in result.stdout.lower()

    def test_should_have_show_subcommand(self, cli_runner):
        """Test config app has show subcommand."""
        result = cli_runner.invoke(config_app, ["show", "--help"])

        assert result.exit_code == 0
        assert "show" in result.stdout.lower()

    def test_should_have_path_subcommand(self, cli_runner):
        """Test config app has path subcommand."""
        result = cli_runner.invoke(config_app, ["path", "--help"])

        assert result.exit_code == 0
        assert "path" in result.stdout.lower()


class TestIntegration:
    """Integration tests for config commands."""

    def test_should_show_then_get_path_workflow(self, cli_runner, tmp_path):
        """Test workflow: show config, then get path."""
        config_file = tmp_path / "config.toml"
        config_content = """embedding_model = "test-model"
"""
        config_file.write_text(config_content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            # First show the config
            result_show = cli_runner.invoke(config_app, ["show"])
            assert result_show.exit_code == 0
            assert "test-model" in result_show.stdout

            # Then get the path
            result_path = cli_runner.invoke(config_app, ["path"])
            assert result_path.exit_code == 0
            assert str(config_file) in result_path.stdout

    def test_should_handle_config_file_creation_workflow(self, cli_runner, tmp_path):
        """Test workflow: check path before/after config file creation."""
        config_file = tmp_path / "config.toml"

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            # Path exists even if file doesn't
            result_path = cli_runner.invoke(config_app, ["path"])
            assert result_path.exit_code == 0
            assert str(config_file) in result_path.stdout

            # Show indicates file not found
            result_show_before = cli_runner.invoke(config_app, ["show"])
            assert result_show_before.exit_code == 0
            assert "Config file not found" in result_show_before.stdout

            # Create the config file
            config_file.write_text("embedding_model = 'new-model'\n")

            # Show now displays content
            result_show_after = cli_runner.invoke(config_app, ["show"])
            assert result_show_after.exit_code == 0
            assert "new-model" in result_show_after.stdout
            assert "Config file not found" not in result_show_after.stdout

    def test_should_handle_different_config_paths(self, cli_runner):
        """Test commands work with different config file paths."""
        paths = [
            Path("/home/user/.config/indexter/config.toml"),
            Path("/tmp/custom/config.toml"),
            Path("relative/path/config.toml"),
        ]

        for cfg_path in paths:
            with patch("indexter.cli.config.settings") as mock_settings:
                mock_settings.config_file = cfg_path

                result = cli_runner.invoke(config_app, ["path"])
                assert result.exit_code == 0
                assert str(cfg_path) in result.stdout

    def test_should_display_config_with_all_sections(self, cli_runner, tmp_path):
        """Test show command with comprehensive config file."""
        config_file = tmp_path / "full-config.toml"
        config_content = """# Indexter Full Configuration
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
max_file_size = 1048576
max_files = 1000

ignore_patterns = [
    ".git/",
    "__pycache__/",
    "*.pyc",
    "node_modules/",
    "venv/",
    ".env"
]

[store]
mode = "server"
path = "~/.local/share/indexter"

[store.qdrant]
url = "http://localhost:6333"
api_key = ""
timeout = 30

[mcp]
transport = "stdio"

[mcp.http]
host = "localhost"
port = 8000
"""
        config_file.write_text(config_content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])
            output = strip_ansi(result.stdout)

            assert result.exit_code == 0
            assert "embedding_model" in output
            assert "ignore_patterns" in output
            assert "[store]" in output
            assert "[mcp]" in output
            assert "max_file_size" in output

    @pytest.mark.parametrize(
        "config_name,content",
        [
            ("minimal", "embedding_model = 'test'\n"),
            ("with-store", "[store]\nmode = 'memory'\n"),
            ("with-mcp", "[mcp]\ntransport = 'http'\n"),
            (
                "complex",
                """
embedding_model = "test"
[store]
mode = "server"
[mcp]
transport = "stdio"
""",
            ),
        ],
    )
    def test_should_handle_various_config_formats(self, cli_runner, tmp_path, config_name, content):
        """Test show command with various config file formats."""
        config_file = tmp_path / f"{config_name}.toml"
        config_file.write_text(content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            assert "Indexter Settings" in result.stdout


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_should_handle_very_large_config_file(self, cli_runner, tmp_path):
        """Test show command with very large config file."""
        config_file = tmp_path / "large.toml"
        # Create a large config with many entries
        large_content = "\n".join([f'key_{i} = "value_{i}"' for i in range(100)])
        config_file.write_text(large_content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            assert "key_0" in result.stdout
            assert "key_99" in result.stdout

    def test_should_handle_config_file_with_unicode(self, cli_runner, tmp_path):
        """Test show command with unicode content in config."""
        config_file = tmp_path / "unicode.toml"
        config_content = """# Configuration with unicode: 你好, مرحبا, שלום
model_name = "transformer-模型"
"""
        config_file.write_text(config_content)

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            assert "Indexter Settings" in result.stdout

    def test_should_handle_config_path_with_spaces(self, cli_runner, tmp_path):
        """Test path command with spaces in path."""
        config_dir = tmp_path / "config with spaces"
        config_dir.mkdir()
        config_file = config_dir / "config file.toml"

        with patch("indexter.cli.config.settings") as mock_settings:
            mock_settings.config_file = config_file

            result = cli_runner.invoke(config_app, ["path"])

            assert result.exit_code == 0
            assert str(config_file) in result.stdout
            assert "config with spaces" in result.stdout

    def test_should_handle_symlink_to_config(self, cli_runner, tmp_path):
        """Test commands handle symlinked config file."""
        real_config = tmp_path / "real_config.toml"
        real_config.write_text("model = 'test'\n")

        symlink_config = tmp_path / "config_link.toml"
        try:
            symlink_config.symlink_to(real_config)

            with patch("indexter.cli.config.settings") as mock_settings:
                mock_settings.config_file = symlink_config

                result = cli_runner.invoke(config_app, ["show"])

                assert result.exit_code == 0
                assert "model" in result.stdout
        except OSError:
            # Skip test if symlinks not supported
            pytest.skip("Symlinks not supported on this system")
