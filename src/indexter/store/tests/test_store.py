"""Comprehensive tests for the VectorStore class."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client import models

from indexter.config import StoreMode
from indexter.parser.models import Node, NodeMetadata
from indexter.store.models import SearchResults
from indexter.store.store import VectorStore


@pytest.fixture
def mock_qdrant_client():
    """Create a properly configured mock AsyncQdrantClient.

    This fixture ensures that synchronous methods like get_fastembed_vector_params
    and get_fastembed_sparse_vector_params are regular Mock objects (not AsyncMock),
    preventing "coroutine was never awaited" warnings.
    """
    mock_client = AsyncMock()
    # These methods are synchronous in AsyncQdrantClient, so use Mock
    mock_client.get_fastembed_vector_params = Mock(return_value={"v": {}})
    mock_client.get_fastembed_sparse_vector_params = Mock(return_value={"s": {}})
    mock_client.set_model = Mock()
    mock_client.set_sparse_model = Mock()
    # get_collections returns an async result with empty collections by default
    mock_client.get_collections.return_value = Mock(collections=[])
    return mock_client


class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_should_create_vector_store_with_default_values(self):
        """Test VectorStore initializes with None values."""
        store = VectorStore()

        assert store._client is None
        assert store._embedding_model_name is None
        assert store._sparse_embedding_model_name is None
        assert store._initialized_collections == set()
        assert store._vector_name is None
        assert store._sparse_vector_name is None


class TestVectorStoreClient:
    """Test VectorStore client property."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.store.mode = StoreMode.memory
        settings.embedding_model = "test-embedding-model"
        settings.sparse_embedding_model = "test-sparse-model"
        return settings

    @patch("indexter.store.store.settings")
    @patch("indexter.store.store.AsyncQdrantClient")
    def test_should_create_memory_client_on_first_access(self, mock_client_class, mock_settings):
        """Test client property creates in-memory client."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_fastembed_vector_params.return_value = {"dense-vector": {}}
        mock_client_instance.get_fastembed_sparse_vector_params.return_value = {"sparse-vector": {}}

        mock_settings.store.mode = StoreMode.memory
        mock_settings.store.timeout = 120
        mock_settings.embedding_model = "test-model"
        mock_settings.sparse_embedding_model = "sparse-model"

        store = VectorStore()
        client = store.client

        assert client is not None
        mock_client_class.assert_called_once_with(
            location=":memory:",
            timeout=120,
            local_inference_batch_size=mock_settings.upsert_batch_size,
        )
        mock_client_instance.set_model.assert_called_once_with("test-model")
        mock_client_instance.set_sparse_model.assert_called_once_with("sparse-model")
        assert store._embedding_model_name == "test-model"
        assert store._sparse_embedding_model_name == "sparse-model"
        assert store._vector_name == "dense-vector"
        assert store._sparse_vector_name == "sparse-vector"

    @patch("indexter.store.store.settings")
    @patch("indexter.store.store.AsyncQdrantClient")
    def test_should_create_server_client_with_connection_params(self, mock_client_class, mock_settings):
        """Test client property creates server client with gRPC options when prefer_grpc is True."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_fastembed_vector_params.return_value = {"vec": {}}
        mock_client_instance.get_fastembed_sparse_vector_params.return_value = {"sp": {}}

        mock_settings.store.mode = StoreMode.server
        mock_settings.store.host = "qdrant-host"
        mock_settings.store.port = 6333
        mock_settings.store.grpc_port = 6334
        mock_settings.store.prefer_grpc = True
        mock_settings.store.api_key = "secret-key"
        mock_settings.store.timeout = 120
        mock_settings.embedding_model = "embed"
        mock_settings.sparse_embedding_model = "sparse"

        store = VectorStore()
        client = store.client

        assert client is not None
        # Verify client was created with correct params including grpc_options
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["host"] == "qdrant-host"
        assert call_kwargs["port"] == 6333
        assert call_kwargs["grpc_port"] == 6334
        assert call_kwargs["prefer_grpc"] is True
        assert call_kwargs["api_key"] == "secret-key"
        assert call_kwargs["timeout"] == 120
        # gRPC options should be set when prefer_grpc is True
        assert call_kwargs["grpc_options"] is not None
        assert "grpc.keepalive_time_ms" in call_kwargs["grpc_options"]
        assert "grpc.enable_retries" in call_kwargs["grpc_options"]

    @patch("indexter.store.store.settings")
    @patch("indexter.store.store.AsyncQdrantClient")
    def test_should_create_server_client_without_grpc_options_when_http(self, mock_client_class, mock_settings):
        """Test client property creates server client without gRPC options when prefer_grpc is False."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_fastembed_vector_params.return_value = {"vec": {}}
        mock_client_instance.get_fastembed_sparse_vector_params.return_value = {"sp": {}}

        mock_settings.store.mode = StoreMode.server
        mock_settings.store.host = "qdrant-host"
        mock_settings.store.port = 6333
        mock_settings.store.grpc_port = 6334
        mock_settings.store.prefer_grpc = False  # HTTP mode
        mock_settings.store.api_key = None
        mock_settings.store.timeout = 120
        mock_settings.embedding_model = "embed"
        mock_settings.sparse_embedding_model = "sparse"

        store = VectorStore()
        client = store.client

        assert client is not None
        # Verify grpc_options is None when prefer_grpc is False
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["prefer_grpc"] is False
        assert call_kwargs["grpc_options"] is None

    @patch("indexter.store.store.settings")
    @patch("indexter.store.store.AsyncQdrantClient")
    def test_should_reuse_existing_client_on_subsequent_access(self, mock_client_class, mock_settings):
        """Test client property returns existing client."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_fastembed_vector_params.return_value = {"v": {}}
        mock_client_instance.get_fastembed_sparse_vector_params.return_value = {"s": {}}

        mock_settings.store.mode = StoreMode.memory
        mock_settings.embedding_model = "model"
        mock_settings.sparse_embedding_model = "sparse"

        store = VectorStore()
        client1 = store.client
        client2 = store.client

        assert client1 is client2
        assert mock_client_class.call_count == 1

    @patch("indexter.store.store.settings")
    @patch("indexter.store.store.AsyncQdrantClient")
    def test_should_handle_none_vector_params(self, mock_client_class, mock_settings):
        """Test client property handles None vector params."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.get_fastembed_vector_params.return_value = None
        mock_client_instance.get_fastembed_sparse_vector_params.return_value = None

        mock_settings.store.mode = StoreMode.memory
        mock_settings.embedding_model = "model"
        mock_settings.sparse_embedding_model = "sparse"

        store = VectorStore()
        client = store.client

        assert client is not None
        assert store._vector_name is None
        assert store._sparse_vector_name is None


class TestVectorStoreCreateCollection:
    """Test VectorStore create_collection method."""

    @pytest.mark.asyncio
    async def test_should_create_collection_with_vector_params(self):
        """Test create_collection creates collection with fastembed params."""
        store = VectorStore()
        mock_client = AsyncMock()
        dense_params = {"dense": {}}
        sparse_params = {"sparse": {}}
        # Override methods to return values synchronously
        mock_client.get_fastembed_vector_params = Mock(return_value=dense_params)
        mock_client.get_fastembed_sparse_vector_params = Mock(return_value=sparse_params)
        store._client = mock_client

        await store.create_collection("test-collection")

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "test-collection"
        assert call_args[1]["vectors_config"] == dense_params
        assert call_args[1]["sparse_vectors_config"] == sparse_params


class TestVectorStoreDeleteCollection:
    """Test VectorStore delete_collection method."""

    @pytest.mark.asyncio
    async def test_should_delete_collection(self):
        """Test delete_collection removes collection."""
        store = VectorStore()
        mock_client = AsyncMock()
        store._client = mock_client
        store._initialized_collections.add("test-collection")

        await store.delete_collection("test-collection")

        mock_client.delete_collection.assert_called_once_with(collection_name="test-collection")
        assert "test-collection" not in store._initialized_collections

    @pytest.mark.asyncio
    async def test_should_delete_collection_not_in_cache(self):
        """Test delete_collection handles collection not in cache."""
        store = VectorStore()
        mock_client = AsyncMock()
        store._client = mock_client

        await store.delete_collection("unknown-collection")

        mock_client.delete_collection.assert_called_once_with(collection_name="unknown-collection")
        assert "unknown-collection" not in store._initialized_collections


class TestVectorStoreEnsureCollection:
    """Test VectorStore ensure_collection method."""

    @pytest.mark.asyncio
    async def test_should_skip_if_collection_cached(self):
        """Test ensure_collection returns early if collection cached."""
        store = VectorStore()
        mock_client = AsyncMock()
        store._client = mock_client
        store._initialized_collections.add("cached-collection")

        await store.ensure_collection("cached-collection")

        mock_client.get_collections.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_create_collection_if_not_exists(self, mock_qdrant_client):
        """Test ensure_collection creates collection if it doesn't exist."""
        store = VectorStore()
        store._client = mock_qdrant_client

        await store.ensure_collection("new-collection")

        mock_qdrant_client.get_collections.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()
        assert "new-collection" in store._initialized_collections

    @pytest.mark.asyncio
    async def test_should_not_create_if_collection_exists(self, mock_qdrant_client):
        """Test ensure_collection doesn't create if collection exists."""
        store = VectorStore()
        existing = Mock()
        existing.name = "existing-collection"
        mock_qdrant_client.get_collections.return_value = Mock(collections=[existing])
        store._client = mock_qdrant_client

        await store.ensure_collection("existing-collection")

        mock_qdrant_client.get_collections.assert_called_once()
        mock_qdrant_client.create_collection.assert_not_called()
        assert "existing-collection" in store._initialized_collections


class TestVectorStoreUpsertNodes:
    """Test VectorStore upsert_nodes method."""

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes for testing."""
        return [
            Node(
                id=uuid.uuid4(),
                content="def test(): pass",
                metadata=NodeMetadata(
                    repo="test-repo",
                    repo_path="/path/to/repo",
                    document_path="test.py",
                    document_hash="doc_hash_1",
                    language="python",
                    node_type="function",
                    node_name="test",
                    start_byte=0,
                    end_byte=17,
                    start_line=1,
                    end_line=1,
                ),
            ),
            Node(
                id=uuid.uuid4(),
                content="class Test: pass",
                metadata=NodeMetadata(
                    repo="test-repo",
                    repo_path="/path/to/repo",
                    document_path="test.py",
                    document_hash="doc_hash_1",
                    language="python",
                    node_type="class",
                    node_name="Test",
                    start_byte=18,
                    end_byte=34,
                    start_line=2,
                    end_line=2,
                ),
            ),
        ]

    @pytest.mark.asyncio
    async def test_should_return_zero_for_empty_nodes_list(self):
        """Test upsert_nodes returns 0 for empty list."""
        store = VectorStore()

        count = await store.upsert_nodes("collection", [])

        assert count == 0

    @pytest.mark.asyncio
    async def test_should_upsert_nodes_to_collection(self, sample_nodes, mock_qdrant_client):
        """Test upsert_nodes inserts nodes into collection."""
        store = VectorStore()
        store._client = mock_qdrant_client
        store._vector_name = "dense-vector"
        store._embedding_model_name = "embed-model"
        store._sparse_vector_name = "sparse-vector"
        store._sparse_embedding_model_name = "sparse-model"

        count = await store.upsert_nodes("test-collection", sample_nodes)

        assert count == 2
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test-collection"
        assert len(call_args[1]["points"]) == 2

    @pytest.mark.asyncio
    async def test_should_raise_error_if_not_initialized(self, sample_nodes, mock_qdrant_client):
        """Test upsert_nodes raises error if vector store not initialized."""
        store = VectorStore()
        store._client = mock_qdrant_client
        # Don't set vector names

        with pytest.raises(RuntimeError, match="Vector store not properly initialized"):
            await store.upsert_nodes("test-collection", sample_nodes)

    @pytest.mark.asyncio
    async def test_should_create_points_with_correct_structure(self, sample_nodes, mock_qdrant_client):
        """Test upsert_nodes creates PointStruct objects correctly."""
        store = VectorStore()
        store._client = mock_qdrant_client
        store._vector_name = "vec"
        store._embedding_model_name = "embed"
        store._sparse_vector_name = "sparse"
        store._sparse_embedding_model_name = "sparse-embed"

        await store.upsert_nodes("collection", sample_nodes)

        points = mock_qdrant_client.upsert.call_args[1]["points"]
        assert all(isinstance(p, models.PointStruct) for p in points)
        assert all(p.id == node.id for p, node in zip(points, sample_nodes, strict=True))
        assert all("content" in p.payload for p in points)


class TestVectorStoreDeleteByHashes:
    """Test VectorStore delete_by_hashes method."""

    @pytest.mark.asyncio
    async def test_should_return_zero_for_empty_hashes_list(self):
        """Test delete_by_hashes returns 0 for empty list."""
        store = VectorStore()

        count = await store.delete_by_hashes("collection", [])

        assert count == 0

    @pytest.mark.asyncio
    async def test_should_delete_nodes_by_hashes(self, mock_qdrant_client):
        """Test delete_by_hashes deletes nodes matching document hashes."""
        store = VectorStore()
        store._client = mock_qdrant_client
        store._initialized_collections.add("test-collection")

        # Mock the count response before deletion
        mock_qdrant_client.count = AsyncMock(return_value=Mock(count=2))

        hashes = ["hash1", "hash2"]
        count = await store.delete_by_hashes("test-collection", hashes)

        assert count == 2
        mock_qdrant_client.count.assert_called_once()
        mock_qdrant_client.delete.assert_called_once()
        call_args = mock_qdrant_client.delete.call_args
        assert call_args[1]["collection_name"] == "test-collection"
        assert isinstance(call_args[1]["points_selector"], models.FilterSelector)

    @pytest.mark.asyncio
    async def test_should_create_filter_with_should_conditions(self, mock_qdrant_client):
        """Test delete_by_hashes creates filter with should conditions on document_hash key."""
        store = VectorStore()
        store._client = mock_qdrant_client
        store._initialized_collections.add("collection")

        mock_qdrant_client.count = AsyncMock(return_value=Mock(count=3))

        hashes = ["a", "b", "c"]
        await store.delete_by_hashes("collection", hashes)

        # Verify both count and delete use the same filter
        count_filter = mock_qdrant_client.count.call_args[1]["count_filter"]
        assert hasattr(count_filter, "should")
        assert len(count_filter.should) == 3
        for condition in count_filter.should:
            assert condition.key == "document_hash"

        points_selector = mock_qdrant_client.delete.call_args[1]["points_selector"]
        assert hasattr(points_selector.filter, "should")
        assert len(points_selector.filter.should) == 3
        for condition in points_selector.filter.should:
            assert condition.key == "document_hash"

    @pytest.mark.asyncio
    async def test_should_ensure_collection_exists(self, mock_qdrant_client):
        """Test delete_by_hashes ensures collection exists before delete."""
        store = VectorStore()
        store._client = mock_qdrant_client
        # Collection not in initialized set, so ensure_collection will be called

        mock_qdrant_client.count = AsyncMock(return_value=Mock(count=1))

        await store.delete_by_hashes("new-collection", ["hash1"])

        mock_qdrant_client.get_collections.assert_called_once()
        mock_qdrant_client.count.assert_called_once()
        mock_qdrant_client.delete.assert_called_once()


class TestVectorStoreSearch:
    """Test VectorStore search method."""

    @pytest.mark.asyncio
    async def test_should_perform_basic_search(self, mock_qdrant_client):
        """Test search performs basic query without filters."""
        store = VectorStore()

        point1 = Mock()
        point1.payload = {"content": "test content", "language": "python"}
        point1.score = 0.95
        mock_qdrant_client.query_points.return_value = Mock(points=[point1])

        store._client = mock_qdrant_client
        store._vector_name = "vec"
        store._embedding_model_name = "embed"
        store._sparse_vector_name = "sparse"
        store._sparse_embedding_model_name = "sparse-embed"

        results = await store.search("collection", "test query", limit=10)

        assert isinstance(results, SearchResults)
        assert len(results.results) == 1
        assert results.results[0].content == "test content"
        assert results.results[0].score == 0.95
        assert results.query == "test query"

    @pytest.mark.asyncio
    async def test_should_raise_error_if_not_initialized(self, mock_qdrant_client):
        """Test search raises error if vector store not initialized."""
        store = VectorStore()
        store._client = mock_qdrant_client

        with pytest.raises(RuntimeError, match="Vector store not properly initialized"):
            await store.search("collection", "query")

    @pytest.mark.asyncio
    async def test_should_filter_by_document_path_exact_match(self, mock_qdrant_client):
        """Test search filters by exact document path."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        await store.search("collection", "query", document_path="test.py")

        query_filter = mock_qdrant_client.query_points.call_args[1]["query_filter"]
        assert query_filter is not None
        assert len(query_filter.must) == 1

    @pytest.mark.asyncio
    async def test_should_filter_by_document_path_prefix_match(self, mock_qdrant_client):
        """Test search filters by document path prefix."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        await store.search("collection", "query", document_path="src/")

        query_filter = mock_qdrant_client.query_points.call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filter_param,filter_value",
        [
            ("language", "python"),
            ("node_type", "function"),
            ("node_name", "test_func"),
            ("parent_scope", "MyClass"),
        ],
    )
    async def test_should_filter_by_various_metadata(self, filter_param, filter_value, mock_qdrant_client):
        """Test search filters by various metadata fields."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        kwargs = {filter_param: filter_value}
        await store.search("collection", "query", **kwargs)

        query_filter = mock_qdrant_client.query_points.call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_should_filter_by_has_documentation_true(self, mock_qdrant_client):
        """Test search filters for nodes with documentation."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        await store.search("collection", "query", has_documentation=True)

        query_filter = mock_qdrant_client.query_points.call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_should_filter_by_has_documentation_false(self, mock_qdrant_client):
        """Test search filters for nodes without documentation."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        await store.search("collection", "query", has_documentation=False)

        query_filter = mock_qdrant_client.query_points.call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_should_combine_multiple_filters(self, mock_qdrant_client):
        """Test search combines multiple filter conditions."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        await store.search(
            "collection",
            "query",
            language="python",
            node_type="function",
            has_documentation=True,
        )

        query_filter = mock_qdrant_client.query_points.call_args[1]["query_filter"]
        assert query_filter is not None
        assert len(query_filter.must) == 3

    @pytest.mark.asyncio
    async def test_should_extract_content_from_payload(self, mock_qdrant_client):
        """Test search extracts content from point payload."""
        store = VectorStore()

        point1 = Mock()
        point1.payload = {"content": "extracted content", "other": "data"}
        point1.score = 0.8
        mock_qdrant_client.query_points.return_value = Mock(points=[point1])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        results = await store.search("collection", "query")

        assert results.results[0].content == "extracted content"
        assert "content" not in results.results[0].metadata
        assert results.results[0].metadata["other"] == "data"

    @pytest.mark.asyncio
    async def test_should_handle_missing_payload(self, mock_qdrant_client):
        """Test search handles points with None payload."""
        store = VectorStore()

        point1 = Mock()
        point1.payload = None
        point1.score = 0.5
        mock_qdrant_client.query_points.return_value = Mock(points=[point1])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        results = await store.search("collection", "query")

        assert results.results[0].content == ""
        assert results.results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_should_handle_missing_score(self, mock_qdrant_client):
        """Test search handles points with None score."""
        store = VectorStore()

        point1 = Mock()
        point1.payload = {"content": "test"}
        point1.score = None
        mock_qdrant_client.query_points.return_value = Mock(points=[point1])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        results = await store.search("collection", "query")

        assert results.results[0].score == 0.0

    @pytest.mark.asyncio
    async def test_should_include_filters_in_search_results(self, mock_qdrant_client):
        """Test search includes all filters in SearchResults."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        results = await store.search(
            "collection",
            "test query",
            document_path="test.py",
            language="python",
            node_type="function",
            node_name="test",
            parent_scope="MyClass",
            has_documentation=True,
            limit=5,
        )

        assert results.filters["document_path"] == "test.py"
        assert results.filters["language"] == "python"
        assert results.filters["node_type"] == "function"
        assert results.filters["node_name"] == "test"
        assert results.filters["parent_scope"] == "MyClass"
        assert results.filters["has_documentation"] is True

    @pytest.mark.asyncio
    async def test_should_respect_limit_parameter(self, mock_qdrant_client):
        """Test search respects the limit parameter."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        await store.search("collection", "query", limit=42)

        assert mock_qdrant_client.query_points.call_args[1]["limit"] == 42


class TestVectorStoreIntegration:
    """Integration tests for VectorStore."""

    @pytest.mark.asyncio
    async def test_should_handle_complete_workflow(self, mock_qdrant_client):
        """Test complete workflow: create, upsert, search, delete."""
        store = VectorStore()
        mock_qdrant_client.get_fastembed_vector_params.return_value = {"vec": {}}
        mock_qdrant_client.get_fastembed_sparse_vector_params.return_value = {"sparse": {}}
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        collection_info = Mock()
        collection_info.points_count = 2
        mock_qdrant_client.get_collection.return_value = collection_info

        store._client = mock_qdrant_client
        store._vector_name = "vec"
        store._embedding_model_name = "embed"
        store._sparse_vector_name = "sparse"
        store._sparse_embedding_model_name = "sparse-embed"

        # Create collection
        await store.create_collection("test-repo")

        # Upsert nodes
        nodes = [
            Node(
                content="test",
                metadata=NodeMetadata(
                    repo="test-repo",
                    repo_path="/path",
                    document_path="file.py",
                    document_hash="doc_hash_1",
                    language="python",
                    node_type="function",
                    start_byte=0,
                    end_byte=4,
                    start_line=1,
                    end_line=1,
                ),
            ),
        ]
        count = await store.upsert_nodes("test-repo", nodes)
        assert count == 1

        # Search
        results = await store.search("test-repo", "test query")
        assert isinstance(results, SearchResults)

        # Delete by hash
        mock_qdrant_client.count = AsyncMock(return_value=Mock(count=1))
        deleted = await store.delete_by_hashes("test-repo", ["some-hash"])
        assert deleted == 1

        # Delete collection
        await store.delete_collection("test-repo")

    @pytest.mark.asyncio
    async def test_should_handle_empty_search_results(self, mock_qdrant_client):
        """Test search with no results."""
        store = VectorStore()
        mock_qdrant_client.query_points.return_value = Mock(points=[])

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        results = await store.search("collection", "nonexistent query")

        assert len(results.results) == 0
        assert results.count == 0

    @pytest.mark.asyncio
    async def test_should_handle_large_batch_upsert(self, mock_qdrant_client):
        """Test upserting large batch of nodes with internal sub-batching."""
        store = VectorStore()

        store._client = mock_qdrant_client
        store._vector_name = "v"
        store._embedding_model_name = "e"
        store._sparse_vector_name = "s"
        store._sparse_embedding_model_name = "se"

        nodes = [
            Node(
                content=f"content {i}",
                metadata=NodeMetadata(
                    repo="repo",
                    repo_path="/path",
                    document_path=f"file{i}.py",
                    document_hash=f"doc_hash_{i}",
                    language="python",
                    node_type="function",
                    start_byte=0,
                    end_byte=10,
                    start_line=1,
                    end_line=1,
                ),
            )
            for i in range(100)
        ]

        count = await store.upsert_nodes("collection", nodes)

        assert count == 100
        # With default batch_size=10, we expect 10 upsert calls for 100 nodes
        assert mock_qdrant_client.upsert.call_count == 10
        # Each call should have at most 10 points
        for call in mock_qdrant_client.upsert.call_args_list:
            points = call[1]["points"]
            assert len(points) <= 10
