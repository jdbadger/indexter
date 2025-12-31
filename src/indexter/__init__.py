"""Indexter: CLI tool and MCP server for enhanced codebase context via RAG."""

from importlib.metadata import version

from .models import Repo

__all__ = ["Repo"]

__version__ = version("indexter")
