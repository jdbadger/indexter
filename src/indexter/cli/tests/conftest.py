"""Shared fixtures for CLI tests."""

import re

import pytest
from typer.testing import CliRunner

from indexter.config import StoreMode, settings


def pytest_configure(config):
    """Configure pytest to use in-memory store mode for faster tests."""
    settings.store.mode = StoreMode.memory


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text.

    Args:
        text: Text potentially containing ANSI escape codes

    Returns:
        Text with all ANSI codes removed
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


@pytest.fixture
def cli_runner():
    """Create a CliRunner for testing."""
    return CliRunner()
