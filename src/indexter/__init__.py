"""Indexter: CLI tool and MCP server for enhanced codebase context via RAG."""

from importlib.metadata import version

from .repo import Repo

__all__ = ["Repo"]

__version__ = version("indexter")
