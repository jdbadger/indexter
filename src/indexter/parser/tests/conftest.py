"""Shared fixtures for parser tests."""

import uuid

import pytest

from indexter.parser.models import Node, NodeMetadata


@pytest.fixture
def valid_node_metadata_data():
    """Fixture providing valid NodeMetadata data.

    Returns:
        dict: Valid data for creating a NodeMetadata instance.
    """
    return {
        "repo": "test-repo",
        "repo_path": "/home/user/repos/test-repo",
        "document_path": "src/main.py",
        "document_hash": "abc123def456",
        "language": "python",
        "node_type": "function",
        "node_name": "my_function",
        "start_byte": 0,
        "end_byte": 100,
        "start_line": 1,
        "end_line": 10,
        "documentation": "This is a docstring.",
        "parent_scope": None,
        "signature": "def my_function(arg1: str, arg2: int) -> bool",
        "extra": {},
    }


@pytest.fixture
def sample_node_metadata(valid_node_metadata_data):
    """Fixture providing a valid NodeMetadata instance.

    Args:
        valid_node_metadata_data: Fixture with valid metadata dictionary.

    Returns:
        NodeMetadata: A valid NodeMetadata instance.
    """
    return NodeMetadata(**valid_node_metadata_data)


@pytest.fixture
def sample_node(sample_node_metadata):
    """Fixture providing a valid Node instance.

    Args:
        sample_node_metadata: Fixture with valid NodeMetadata.

    Returns:
        Node: A valid Node instance with computed hash.
    """
    return Node(
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        content="def my_function(arg1: str, arg2: int) -> bool:\n    return True",
        metadata=sample_node_metadata,
    )


@pytest.fixture
def valid_node_data(sample_node_metadata):
    """Fixture providing valid Node data as a dictionary.

    Args:
        sample_node_metadata: Fixture with valid NodeMetadata.

    Returns:
        dict: Valid data for creating a Node instance.
    """
    return {
        "content": "def my_function(arg1: str, arg2: int) -> bool:\n    return True",
        "metadata": sample_node_metadata,
    }
