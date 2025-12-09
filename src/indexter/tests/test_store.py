"""Unit tests for indexter.store module.

These tests cover the VectorStore class which manages Qdrant vector database
operations including collection management, document hash tracking, node upserting,
deletion, and semantic search with filtering.

Note: Some tests may fail if there are bugs in store.py (e.g., incorrect parameter
names for Qdrant models). These failures help identify issues in the implementation.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indexter.config.store import StoreMode
from indexter.store import VectorStore, store


@pytest.fixture
def vector_store():
    """Create a VectorStore instance."""
    return VectorStore()


@pytest.fixture
def mock_qdrant_client():
    """Create a mock AsyncQdrantClient."""
    client = MagicMock()
    client.set_model = MagicMock()
    client.get_fastembed_vector_params = MagicMock(return_value={"test-vector": {}})
    client.create_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    client.get_collections = AsyncMock()
    client.scroll = AsyncMock()
    client.add = AsyncMock()
    client.delete = AsyncMock()
    client.query = AsyncMock()
    return client


@pytest.fixture
def sample_nodes(sample_node):
    """Create a list of sample nodes."""
    return [sample_node]


# Tests for __init__


def test_init():
    """Test VectorStore initialization."""
    store = VectorStore()
    assert store._client is None
    assert store._embedding_model_loaded is False
    assert store._initialized_collections == set()
    assert store._vector_name is None


# Tests for client property


@patch("indexter.store.AsyncQdrantClient")
@patch("indexter.store.settings")
def test_client_property_remote_mode(mock_settings, mock_client_class, vector_store):
    """Test client property creates remote client when mode is remote."""
    # Setup mock settings for remote mode
    mock_settings.store.mode = StoreMode.remote
    mock_settings.store.url = "http://localhost:6333"
    mock_settings.store.api_key = "test_key"
    mock_settings.store.use_grpc = False
    mock_settings.embedding.model_name = "test-model"

    # Setup mock client
    mock_client = MagicMock()
    mock_client.set_model = MagicMock()
    mock_client.get_fastembed_vector_params = MagicMock(return_value={"test-vector": {}})
    mock_client_class.return_value = mock_client

    # Access client
    client = vector_store.client

    # Verify client was created with remote settings
    assert client is not None
    mock_client_class.assert_called_once_with(
        url="http://localhost:6333",
        api_key="test_key",
        prefer_grpc=False,
    )
    mock_client.set_model.assert_called_once_with("test-model")
    assert vector_store._vector_name == "test-vector"


@patch("indexter.store.AsyncQdrantClient")
@patch("indexter.store.settings")
def test_client_property_local_mode(mock_settings, mock_client_class, vector_store, tmp_path):
    """Test client property creates local client when mode is local."""
    # Setup mock settings for local mode
    mock_settings.store.mode = StoreMode.local
    mock_settings.store.path = tmp_path / "store"
    mock_settings.embedding.model_name = "test-model"

    # Setup mock client
    mock_client = MagicMock()
    mock_client.set_model = MagicMock()
    mock_client.get_fastembed_vector_params = MagicMock(return_value={"test-vector": {}})
    mock_client_class.return_value = mock_client

    # Access client
    client = vector_store.client

    # Verify client was created with local path
    assert client is not None
    mock_client_class.assert_called_once_with(path=str(tmp_path / "store"))
    mock_client.set_model.assert_called_once_with("test-model")
    assert (tmp_path / "store").exists()  # Directory should be created


@patch("indexter.store.AsyncQdrantClient")
@patch("indexter.store.settings")
def test_client_property_local_mode_default_path(
    mock_settings, mock_client_class, vector_store, tmp_path
):
    """Test client property uses data_dir/store when path is None in local mode."""
    # Setup mock settings for local mode with no explicit path
    mock_settings.store.mode = StoreMode.local
    mock_settings.store.path = None
    mock_settings.data_dir = tmp_path / "data"
    mock_settings.embedding.model_name = "test-model"

    # Setup mock client
    mock_client = MagicMock()
    mock_client.set_model = MagicMock()
    mock_client.get_fastembed_vector_params = MagicMock(return_value={"test-vector": {}})
    mock_client_class.return_value = mock_client

    # Access client
    client = vector_store.client

    # Verify client was created with default path
    assert client is not None
    expected_path = str(tmp_path / "data" / "store")
    mock_client_class.assert_called_once_with(path=expected_path)
    assert (tmp_path / "data" / "store").exists()  # Directory should be created


@patch("indexter.store.AsyncQdrantClient")
@patch("indexter.store.settings")
def test_client_property_memory_mode(mock_settings, mock_client_class, vector_store):
    """Test client property creates in-memory client when mode is memory."""
    # Setup mock settings for memory mode
    mock_settings.store.mode = StoreMode.memory
    mock_settings.embedding.model_name = "test-model"

    # Setup mock client
    mock_client = MagicMock()
    mock_client.set_model = MagicMock()
    mock_client.get_fastembed_vector_params = MagicMock(return_value={"test-vector": {}})
    mock_client_class.return_value = mock_client

    # Access client
    client = vector_store.client

    # Verify client was created with in-memory location
    assert client is not None
    mock_client_class.assert_called_once_with(location=":memory:")
    mock_client.set_model.assert_called_once_with("test-model")


@patch("indexter.store.AsyncQdrantClient")
@patch("indexter.store.settings")
def test_client_property_cached(mock_settings, mock_client_class, vector_store):
    """Test client property returns cached client on subsequent accesses."""
    # Setup mock settings
    mock_settings.store.mode = StoreMode.remote
    mock_settings.store.url = "http://localhost:6333"
    mock_settings.store.api_key = "test_key"
    mock_settings.store.use_grpc = False
    mock_settings.embedding.model_name = "test-model"

    # Setup mock client
    mock_client = MagicMock()
    mock_client.set_model = MagicMock()
    mock_client.get_fastembed_vector_params = MagicMock(return_value={"test-vector": {}})
    mock_client_class.return_value = mock_client

    # Access client twice
    client1 = vector_store.client
    client2 = vector_store.client

    # Verify client was created only once
    assert client1 is client2
    assert mock_client_class.call_count == 1


# Tests for create_collection


@pytest.mark.asyncio
async def test_create_collection(vector_store, mock_qdrant_client):
    """Test creating a collection."""
    vector_store._client = mock_qdrant_client
    vector_store._vector_name = "test-vector"

    await vector_store.create_collection("test_collection")

    mock_qdrant_client.get_fastembed_vector_params.assert_called_once()
    mock_qdrant_client.create_collection.assert_called_once_with(
        collection_name="test_collection",
        vectors_config={"test-vector": {}},
    )


# Tests for delete_collection


@pytest.mark.asyncio
async def test_delete_collection(vector_store, mock_qdrant_client):
    """Test deleting a collection."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    await vector_store.delete_collection("test_collection")

    mock_qdrant_client.delete_collection.assert_called_once_with(collection_name="test_collection")
    assert "test_collection" not in vector_store._initialized_collections


@pytest.mark.asyncio
async def test_delete_collection_not_in_cache(vector_store, mock_qdrant_client):
    """Test deleting a collection that's not in the cache."""
    vector_store._client = mock_qdrant_client

    await vector_store.delete_collection("test_collection")

    mock_qdrant_client.delete_collection.assert_called_once_with(collection_name="test_collection")


# Tests for ensure_collection


@pytest.mark.asyncio
async def test_ensure_collection_cached(vector_store, mock_qdrant_client):
    """Test ensure_collection when collection is already cached."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    await vector_store.ensure_collection("test_collection")

    # Should not make any calls to client
    mock_qdrant_client.get_collections.assert_not_called()
    mock_qdrant_client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_collection_exists(vector_store, mock_qdrant_client):
    """Test ensure_collection when collection exists but not cached."""
    vector_store._client = mock_qdrant_client
    vector_store._vector_name = "test-vector"

    # Mock collections response
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collections_response = MagicMock()
    mock_collections_response.collections = [mock_collection]
    mock_qdrant_client.get_collections.return_value = mock_collections_response

    await vector_store.ensure_collection("test_collection")

    mock_qdrant_client.get_collections.assert_called_once()
    mock_qdrant_client.create_collection.assert_not_called()
    assert "test_collection" in vector_store._initialized_collections


@pytest.mark.asyncio
async def test_ensure_collection_not_exists(vector_store, mock_qdrant_client):
    """Test ensure_collection when collection does not exist."""
    vector_store._client = mock_qdrant_client
    vector_store._vector_name = "test-vector"

    # Mock empty collections response
    mock_collections_response = MagicMock()
    mock_collections_response.collections = []
    mock_qdrant_client.get_collections.return_value = mock_collections_response

    await vector_store.ensure_collection("test_collection")

    mock_qdrant_client.get_collections.assert_called_once()
    mock_qdrant_client.get_fastembed_vector_params.assert_called_once()
    mock_qdrant_client.create_collection.assert_called_once()
    assert "test_collection" in vector_store._initialized_collections


# Tests for get_document_hashes


@pytest.mark.asyncio
async def test_get_document_hashes_empty(vector_store, mock_qdrant_client):
    """Test getting document hashes when collection is empty."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    # Mock empty scroll response
    mock_qdrant_client.scroll.return_value = ([], None)

    result = await vector_store.get_document_hashes("test_collection")

    assert result == {}
    mock_qdrant_client.scroll.assert_called_once()


@pytest.mark.asyncio
async def test_get_document_hashes_with_documents(vector_store, mock_qdrant_client):
    """Test getting document hashes with documents."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    # Mock scroll response with points
    mock_point1 = MagicMock()
    mock_point1.payload = {"document_path": "file1.py", "hash": "hash1"}
    mock_point2 = MagicMock()
    mock_point2.payload = {"document_path": "file2.py", "hash": "hash2"}
    mock_point3 = MagicMock()
    mock_point3.payload = {"document_path": "file1.py", "hash": "hash1"}  # Duplicate

    mock_qdrant_client.scroll.return_value = ([mock_point1, mock_point2, mock_point3], None)

    result = await vector_store.get_document_hashes("test_collection")

    assert result == {"file1.py": "hash1", "file2.py": "hash2"}
    mock_qdrant_client.scroll.assert_called_once_with(
        collection_name="test_collection",
        limit=1000,
        offset=None,
        with_payload=["document_path", "hash"],
        with_vectors=False,
    )


@pytest.mark.asyncio
async def test_get_document_hashes_paginated(vector_store, mock_qdrant_client):
    """Test getting document hashes with pagination."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    # Mock paginated scroll responses
    mock_point1 = MagicMock()
    mock_point1.payload = {"document_path": "file1.py", "hash": "hash1"}
    mock_point2 = MagicMock()
    mock_point2.payload = {"document_path": "file2.py", "hash": "hash2"}

    mock_qdrant_client.scroll.side_effect = [
        ([mock_point1], "offset1"),
        ([mock_point2], None),
    ]

    result = await vector_store.get_document_hashes("test_collection")

    assert result == {"file1.py": "hash1", "file2.py": "hash2"}
    assert mock_qdrant_client.scroll.call_count == 2


@pytest.mark.asyncio
async def test_get_document_hashes_missing_payload(vector_store, mock_qdrant_client):
    """Test getting document hashes with missing payload."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    # Mock scroll response with incomplete payloads
    mock_point1 = MagicMock()
    mock_point1.payload = {"document_path": "file1.py", "hash": "hash1"}
    mock_point2 = MagicMock()
    mock_point2.payload = None
    mock_point3 = MagicMock()
    mock_point3.payload = {"document_path": "file2.py"}  # Missing hash
    mock_point4 = MagicMock()
    mock_point4.payload = {"hash": "hash3"}  # Missing document_path

    mock_qdrant_client.scroll.return_value = (
        [mock_point1, mock_point2, mock_point3, mock_point4],
        None,
    )

    result = await vector_store.get_document_hashes("test_collection")

    assert result == {"file1.py": "hash1"}


# Tests for upsert_nodes


@pytest.mark.asyncio
async def test_upsert_nodes_empty(vector_store, mock_qdrant_client):
    """Test upserting empty list of nodes."""
    vector_store._client = mock_qdrant_client

    result = await vector_store.upsert_nodes("test_collection", [])

    assert result == 0
    mock_qdrant_client.add.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_nodes_single(vector_store, mock_qdrant_client, sample_node):
    """Test upserting a single node."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    result = await vector_store.upsert_nodes("test_collection", [sample_node])

    assert result == 1
    mock_qdrant_client.add.assert_called_once()
    call_args = mock_qdrant_client.add.call_args
    assert call_args[1]["collection_name"] == "test_collection"
    assert call_args[1]["documents"] == [sample_node.content]
    assert call_args[1]["ids"] == [sample_node.id]
    assert len(call_args[1]["metadata"]) == 1


@pytest.mark.asyncio
async def test_upsert_nodes_multiple(vector_store, mock_qdrant_client, sample_node):
    """Test upserting multiple nodes."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    # Create multiple nodes
    node1 = sample_node
    node2 = sample_node.model_copy(update={"id": uuid.uuid4()})
    node3 = sample_node.model_copy(update={"id": uuid.uuid4()})

    result = await vector_store.upsert_nodes("test_collection", [node1, node2, node3])

    assert result == 3
    mock_qdrant_client.add.assert_called_once()
    call_args = mock_qdrant_client.add.call_args
    assert len(call_args[1]["documents"]) == 3
    assert len(call_args[1]["metadata"]) == 3
    assert len(call_args[1]["ids"]) == 3


@pytest.mark.asyncio
async def test_upsert_nodes_metadata_formatting(vector_store, mock_qdrant_client, sample_node):
    """Test that node metadata is correctly formatted."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    await vector_store.upsert_nodes("test_collection", [sample_node])

    call_args = mock_qdrant_client.add.call_args
    metadata = call_args[1]["metadata"][0]

    assert metadata["hash"] == sample_node.metadata.hash
    assert metadata["repo_path"] == sample_node.metadata.repo_path
    assert metadata["document_path"] == sample_node.metadata.document_path
    assert metadata["language"] == sample_node.metadata.language
    assert metadata["node_type"] == sample_node.metadata.node_type
    assert metadata["node_name"] == sample_node.metadata.node_name
    assert metadata["start_byte"] == sample_node.metadata.start_byte
    assert metadata["end_byte"] == sample_node.metadata.end_byte
    assert metadata["start_line"] == sample_node.metadata.start_line
    assert metadata["end_line"] == sample_node.metadata.end_line
    assert metadata["documentation"] == sample_node.metadata.documentation
    assert metadata["parent_scope"] == (sample_node.metadata.parent_scope or "")
    assert metadata["signature"] == sample_node.metadata.signature


# Tests for delete_by_document_paths


@pytest.mark.asyncio
async def test_delete_by_document_paths_empty(vector_store, mock_qdrant_client):
    """Test deleting with empty list."""
    vector_store._client = mock_qdrant_client

    result = await vector_store.delete_by_document_paths("test_collection", [])

    assert result == 0
    mock_qdrant_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_by_document_paths_single(vector_store, mock_qdrant_client):
    """Test deleting a single document path."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    result = await vector_store.delete_by_document_paths("test_collection", ["file1.py"])

    assert result == 1
    mock_qdrant_client.delete.assert_called_once()
    call_args = mock_qdrant_client.delete.call_args
    assert call_args[1]["collection_name"] == "test_collection"


@pytest.mark.asyncio
async def test_delete_by_document_paths_multiple(vector_store, mock_qdrant_client):
    """Test deleting multiple document paths."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    paths = ["file1.py", "file2.py", "file3.py"]
    result = await vector_store.delete_by_document_paths("test_collection", paths)

    assert result == 3
    mock_qdrant_client.delete.assert_called_once()


# Tests for search


@pytest.mark.asyncio
async def test_search_basic(vector_store, mock_qdrant_client):
    """Test basic search without filters."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    # Mock search results
    mock_result = MagicMock()
    mock_result.id = "test-id"
    mock_result.score = 0.95
    mock_result.content = "test content"
    mock_result.metadata = {
        "document_path": "test.py",
        "language": "python",
        "node_type": "function",
        "node_name": "test_func",
        "start_line": 1,
        "end_line": 10,
        "documentation": "Test docstring",
        "signature": "def test_func():",
        "parent_scope": "TestClass",
    }
    mock_qdrant_client.query.return_value = [mock_result]

    results = await vector_store.search("test_collection", "test query")

    assert len(results) == 1
    assert results[0]["id"] == "test-id"
    assert results[0]["score"] == 0.95
    assert results[0]["content"] == "test content"
    assert results[0]["file_path"] == "test.py"
    assert results[0]["language"] == "python"

    mock_qdrant_client.query.assert_called_once_with(
        collection_name="test_collection",
        query_text="test query",
        limit=10,
        query_filter=None,
    )


@pytest.mark.asyncio
async def test_search_with_limit(vector_store, mock_qdrant_client):
    """Test search with custom limit."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", limit=5)

    call_args = mock_qdrant_client.query.call_args
    assert call_args[1]["limit"] == 5


@pytest.mark.asyncio
async def test_search_with_file_path_exact(vector_store, mock_qdrant_client):
    """Test search with exact file path filter."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", file_path="test.py")

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None
    assert len(query_filter.must) == 1


@pytest.mark.asyncio
async def test_search_with_file_path_prefix(vector_store, mock_qdrant_client):
    """Test search with file path prefix filter."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", file_path="src/")

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None


@pytest.mark.asyncio
async def test_search_with_language_filter(vector_store, mock_qdrant_client):
    """Test search with language filter."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", language="python")

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None


@pytest.mark.asyncio
async def test_search_with_node_type_filter(vector_store, mock_qdrant_client):
    """Test search with node type filter."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", node_type="function")

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None


@pytest.mark.asyncio
async def test_search_with_node_name_filter(vector_store, mock_qdrant_client):
    """Test search with node name filter."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", node_name="test_func")

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None


@pytest.mark.asyncio
async def test_search_with_has_documentation_true(vector_store, mock_qdrant_client):
    """Test search filtering for nodes with documentation."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", has_documentation=True)

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None


@pytest.mark.asyncio
async def test_search_with_has_documentation_false(vector_store, mock_qdrant_client):
    """Test search filtering for nodes without documentation."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search("test_collection", "test query", has_documentation=False)

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None


@pytest.mark.asyncio
async def test_search_with_multiple_filters(vector_store, mock_qdrant_client):
    """Test search with multiple filters combined."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    await vector_store.search(
        "test_collection",
        "test query",
        file_path="test.py",
        language="python",
        node_type="function",
        node_name="test_func",
        has_documentation=True,
    )

    call_args = mock_qdrant_client.query.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None
    assert len(query_filter.must) == 5


@pytest.mark.asyncio
async def test_search_empty_results(vector_store, mock_qdrant_client):
    """Test search with no results."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")
    mock_qdrant_client.query.return_value = []

    results = await vector_store.search("test_collection", "test query")

    assert results == []


@pytest.mark.asyncio
async def test_search_result_formatting(vector_store, mock_qdrant_client):
    """Test that search results are correctly formatted."""
    vector_store._client = mock_qdrant_client
    vector_store._initialized_collections.add("test_collection")

    # Mock result with missing optional fields
    mock_result = MagicMock()
    mock_result.id = "test-id"
    mock_result.score = 0.95
    mock_result.content = "test content"
    mock_result.metadata = {}
    mock_qdrant_client.query.return_value = [mock_result]

    results = await vector_store.search("test_collection", "test query")

    assert len(results) == 1
    assert results[0]["id"] == "test-id"
    assert results[0]["score"] == 0.95
    assert results[0]["content"] == "test content"
    assert results[0]["file_path"] == ""
    assert results[0]["language"] == ""
    assert results[0]["node_type"] == ""
    assert results[0]["node_name"] == ""
    assert results[0]["start_line"] == 0
    assert results[0]["end_line"] == 0
    assert results[0]["documentation"] == ""
    assert results[0]["signature"] == ""
    assert results[0]["parent_scope"] == ""


# Test global store instance


def test_global_store_instance():
    """Test that global store instance exists."""
    assert isinstance(store, VectorStore)
