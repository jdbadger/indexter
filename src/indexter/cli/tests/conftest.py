"""Shared fixtures for CLI tests."""

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Create a CliRunner for testing."""
    return CliRunner()
