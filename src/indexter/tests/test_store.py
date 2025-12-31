from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import pytest

from indexter.config import StoreMode
from indexter.models import Node, NodeMetadata
from indexter.store import VectorStore


@pytest.fixture
def vector_store():
    """Create a fresh VectorStore instance."""
    return VectorStore()


@pytest.fixture
def sample_node():
    """Create a sample node for testing."""
    return Node(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        content="def hello_world():\n    print('Hello, World!')",
        metadata=NodeMetadata(
            hash="abc123",
            repo_path="/path/to/repo",
            document_path="src/main.py",
            language="python",
            node_type="function",
            node_name="hello_world",
            start_byte=0,
            end_byte=50,
            start_line=1,
            end_line=2,
            documentation="A simple hello world function",
            parent_scope=None,
            signature="def hello_world()",
            extra={},
        ),
    )


@pytest.fixture
def sample_nodes(sample_node):
    """Create a list of sample nodes."""
    node2 = Node(
        id=UUID("87654321-4321-8765-4321-876543218765"),
        content="class MyClass:\n    pass",
        metadata=NodeMetadata(
            hash="def456",
            repo_path="/path/to/repo",
            document_path="src/classes.py",
            language="python",
            node_type="class",
            node_name="MyClass",
            start_byte=0,
            end_byte=30,
            start_line=1,
            end_line=2,
            documentation=None,
            parent_scope=None,
            signature="class MyClass",
            extra={},
        ),
    )
    return [sample_node, node2]


class TestVectorStoreInit:
    """Tests for VectorStore initialization."""

    def test_init_creates_empty_store(self):
        """Test that initialization creates a store with None client."""
        store = VectorStore()
        assert store._client is None
        assert store._embedding_model_name is None
        assert store._initialized_collections == set()
        assert store._vector_name is None


class TestVectorStoreClient:
    """Tests for VectorStore.client property."""

    @patch("indexter.store.AsyncQdrantClient")
    @patch("indexter.store.settings")
    def test_client_local_mode(self, mock_settings, mock_qdrant_client):
        """Test client creation in local mode."""
        # Setup
        mock_settings.store.mode = StoreMode.local

        # Create a proper mock Path object for data_dir
        mock_store_path = MagicMock()
        mock_store_path.mkdir = Mock()
        mock_store_path.__str__ = Mock(return_value="/test/data/store")

        mock_data_dir = MagicMock()
        mock_data_dir.__truediv__ = Mock(return_value=mock_store_path)
        mock_settings.data_dir = mock_data_dir

        mock_settings.embedding_model = "test-model"

        mock_client_instance = MagicMock()
        mock_client_instance.get_fastembed_vector_params.return_value = {"test-vector": {}}
        mock_qdrant_client.return_value = mock_client_instance

        # Execute
        store = VectorStore()
        client = store.client

        # Assert
        assert client is mock_client_instance
        mock_qdrant_client.assert_called_once_with(path="/test/data/store")
        mock_client_instance.set_model.assert_called_once_with("test-model")
        assert store._embedding_model_name == "test-model"
        assert store._vector_name == "test-vector"

    @patch("indexter.store.AsyncQdrantClient")
    @patch("indexter.store.settings")
    def test_client_memory_mode(self, mock_settings, mock_qdrant_client):
        """Test client creation in memory mode."""
        # Setup
        mock_settings.store.mode = StoreMode.memory
        mock_settings.embedding_model = "test-model"

        mock_client_instance = MagicMock()
        mock_client_instance.get_fastembed_vector_params.return_value = {"test-vector": {}}
        mock_qdrant_client.return_value = mock_client_instance

        # Execute
        store = VectorStore()
        store.client  # noqa: B018

        # Assert
        mock_qdrant_client.assert_called_once_with(location=":memory:")
        mock_client_instance.set_model.assert_called_once_with("test-model")

    @patch("indexter.store.AsyncQdrantClient")
    @patch("indexter.store.settings")
    def test_client_remote_mode(self, mock_settings, mock_qdrant_client):
        """Test client creation in remote mode."""
        # Setup
        mock_settings.store.mode = StoreMode.remote
        mock_settings.store.host = "localhost"
        mock_settings.store.port = 6333
        mock_settings.store.grpc_port = 6334
        mock_settings.store.prefer_grpc = True
        mock_settings.store.api_key = "test-key"
        mock_settings.store.url = "http://localhost:6333"
        mock_settings.embedding_model = "test-model"

        mock_client_instance = MagicMock()
        mock_client_instance.get_fastembed_vector_params.return_value = {"test-vector": {}}
        mock_qdrant_client.return_value = mock_client_instance

        # Execute
        store = VectorStore()
        store.client  # noqa: B018

        # Assert
        mock_qdrant_client.assert_called_once_with(
            host="localhost",
            port=6333,
            grpc_port=6334,
            prefer_grpc=True,
            api_key="test-key",
        )
        mock_client_instance.set_model.assert_called_once_with("test-model")

    @patch("indexter.store.AsyncQdrantClient")
    @patch("indexter.store.settings")
    def test_client_cached_on_second_access(self, mock_settings, mock_qdrant_client):
        """Test that client is cached and not recreated."""
        # Setup
        mock_settings.store.mode = StoreMode.memory
        mock_settings.embedding_model = "test-model"

        mock_client_instance = MagicMock()
        mock_client_instance.get_fastembed_vector_params.return_value = {"test-vector": {}}
        mock_qdrant_client.return_value = mock_client_instance

        # Execute
        store = VectorStore()
        client1 = store.client
        client2 = store.client

        # Assert
        assert client1 is client2
        assert mock_qdrant_client.call_count == 1


class TestVectorStoreCreateCollection:
    """Tests for VectorStore.create_collection."""

    @pytest.mark.asyncio
    async def test_create_collection(self, vector_store):
        """Test creating a collection."""
        # Setup
        mock_client = AsyncMock()
        vector_params = {"test-vector": {"size": 384}}
        # get_fastembed_vector_params is a sync method
        mock_client.get_fastembed_vector_params = MagicMock(return_value=vector_params)
        vector_store._client = mock_client

        # Execute
        await vector_store.create_collection("test_collection")

        # Assert
        mock_client.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config=vector_params,
        )


class TestVectorStoreDeleteCollection:
    """Tests for VectorStore.delete_collection."""

    @pytest.mark.asyncio
    async def test_delete_collection(self, vector_store):
        """Test deleting a collection."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        await vector_store.delete_collection("test_collection")

        # Assert
        mock_client.delete_collection.assert_called_once_with(collection_name="test_collection")
        assert "test_collection" not in vector_store._initialized_collections

    @pytest.mark.asyncio
    async def test_delete_non_cached_collection(self, vector_store):
        """Test deleting a collection not in cache."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client

        # Execute
        await vector_store.delete_collection("test_collection")

        # Assert
        mock_client.delete_collection.assert_called_once_with(collection_name="test_collection")


class TestVectorStoreEnsureCollection:
    """Tests for VectorStore.ensure_collection."""

    @pytest.mark.asyncio
    async def test_ensure_collection_cached(self, vector_store):
        """Test that cached collections are not checked again."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        await vector_store.ensure_collection("test_collection")

        # Assert - should not call get_collections since it's cached
        mock_client.get_collections.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists(self, vector_store):
        """Test ensuring a collection that already exists."""
        # Setup
        mock_client = AsyncMock()

        # Create a mock with proper name attribute
        test_collection = MagicMock()
        test_collection.name = "test_collection"
        other_collection = MagicMock()
        other_collection.name = "other_collection"

        mock_collections = MagicMock()
        mock_collections.collections = [test_collection, other_collection]
        mock_client.get_collections.return_value = mock_collections
        vector_store._client = mock_client

        # Execute
        await vector_store.ensure_collection("test_collection")

        # Assert
        mock_client.get_collections.assert_called_once()
        mock_client.create_collection.assert_not_called()
        assert "test_collection" in vector_store._initialized_collections

    @pytest.mark.asyncio
    async def test_ensure_collection_does_not_exist(self, vector_store):
        """Test ensuring a collection that doesn't exist creates it."""
        # Setup
        mock_client = AsyncMock()

        # Create a mock with proper name attribute
        other_collection = MagicMock()
        other_collection.name = "other_collection"

        mock_collections = MagicMock()
        mock_collections.collections = [other_collection]
        mock_client.get_collections.return_value = mock_collections
        vector_params = {"test-vector": {"size": 384}}
        # get_fastembed_vector_params is a sync method
        mock_client.get_fastembed_vector_params = MagicMock(return_value=vector_params)
        vector_store._client = mock_client

        # Execute
        await vector_store.ensure_collection("test_collection")

        # Assert
        mock_client.get_collections.assert_called_once()
        mock_client.create_collection.assert_called_once()
        assert "test_collection" in vector_store._initialized_collections


class TestVectorStoreGetDocumentHashes:
    """Tests for VectorStore.get_document_hashes."""

    @pytest.mark.asyncio
    async def test_get_document_hashes_empty(self, vector_store):
        """Test getting document hashes from empty collection."""
        # Setup
        mock_client = AsyncMock()
        mock_client.scroll.return_value = ([], None)
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.get_document_hashes("test_collection")

        # Assert
        assert result == {}
        mock_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_hashes_single_page(self, vector_store):
        """Test getting document hashes with single page."""
        # Setup
        mock_client = AsyncMock()
        point1 = MagicMock()
        point1.payload = {"document_path": "file1.py", "hash": "hash1"}
        point2 = MagicMock()
        point2.payload = {"document_path": "file2.py", "hash": "hash2"}
        mock_client.scroll.return_value = ([point1, point2], None)
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.get_document_hashes("test_collection")

        # Assert
        assert result == {"file1.py": "hash1", "file2.py": "hash2"}
        mock_client.scroll.assert_called_once_with(
            collection_name="test_collection",
            limit=1000,
            offset=None,
            with_payload=["document_path", "hash"],
            with_vectors=False,
        )

    @pytest.mark.asyncio
    async def test_get_document_hashes_multiple_pages(self, vector_store):
        """Test getting document hashes with pagination."""
        # Setup
        mock_client = AsyncMock()
        point1 = MagicMock()
        point1.payload = {"document_path": "file1.py", "hash": "hash1"}
        point2 = MagicMock()
        point2.payload = {"document_path": "file2.py", "hash": "hash2"}

        # First page
        mock_client.scroll.side_effect = [
            ([point1], "offset1"),
            ([point2], None),
        ]
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.get_document_hashes("test_collection")

        # Assert
        assert result == {"file1.py": "hash1", "file2.py": "hash2"}
        assert mock_client.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_get_document_hashes_duplicate_paths(self, vector_store):
        """Test that only first occurrence of document_path is stored."""
        # Setup
        mock_client = AsyncMock()
        point1 = MagicMock()
        point1.payload = {"document_path": "file1.py", "hash": "hash1"}
        point2 = MagicMock()
        point2.payload = {"document_path": "file1.py", "hash": "hash2"}  # duplicate path
        mock_client.scroll.return_value = ([point1, point2], None)
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.get_document_hashes("test_collection")

        # Assert - should keep first hash
        assert result == {"file1.py": "hash1"}

    @pytest.mark.asyncio
    async def test_get_document_hashes_missing_payload(self, vector_store):
        """Test handling points with missing payload."""
        # Setup
        mock_client = AsyncMock()
        point1 = MagicMock()
        point1.payload = {"document_path": "file1.py", "hash": "hash1"}
        point2 = MagicMock()
        point2.payload = None  # missing payload
        mock_client.scroll.return_value = ([point1, point2], None)
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.get_document_hashes("test_collection")

        # Assert - should skip point2
        assert result == {"file1.py": "hash1"}


class TestVectorStoreCountNodes:
    """Tests for VectorStore.count_nodes."""

    @pytest.mark.asyncio
    async def test_count_nodes(self, vector_store):
        """Test counting nodes in a collection."""
        # Setup
        mock_client = AsyncMock()
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 42
        mock_client.get_collection.return_value = mock_collection_info
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.count_nodes("test_collection")

        # Assert
        assert result == 42
        mock_client.get_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_count_nodes_none_count(self, vector_store):
        """Test counting when points_count is None."""
        # Setup
        mock_client = AsyncMock()
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = None
        mock_client.get_collection.return_value = mock_collection_info
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.count_nodes("test_collection")

        # Assert
        assert result == 0


class TestVectorStoreUpsertNodes:
    """Tests for VectorStore.upsert_nodes."""

    @pytest.mark.asyncio
    async def test_upsert_nodes_empty_list(self, vector_store):
        """Test upserting empty list returns 0."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client

        # Execute
        result = await vector_store.upsert_nodes("test_collection", [])

        # Assert
        assert result == 0
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_nodes_single_node(self, vector_store, sample_node):
        """Test upserting a single node."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        result = await vector_store.upsert_nodes("test_collection", [sample_node])

        # Assert
        assert result == 1
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        points = call_args[1]["points"]
        assert len(points) == 1
        assert points[0].id == sample_node.id

    @pytest.mark.asyncio
    async def test_upsert_nodes_multiple_nodes(self, vector_store, sample_nodes):
        """Test upserting multiple nodes."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        result = await vector_store.upsert_nodes("test_collection", sample_nodes)

        # Assert
        assert result == 2
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_nodes_not_initialized_raises(self, vector_store, sample_node):
        """Test upserting when vector store not initialized raises error."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        # Don't set _vector_name or _embedding_model_name

        # Execute & Assert
        with pytest.raises(RuntimeError, match="Vector store not properly initialized"):
            await vector_store.upsert_nodes("test_collection", [sample_node])

    @pytest.mark.asyncio
    async def test_upsert_nodes_metadata_mapping(self, vector_store, sample_node):
        """Test that node metadata is correctly mapped to payload."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.upsert_nodes("test_collection", [sample_node])

        # Assert
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        payload = points[0].payload

        assert payload["document"] == sample_node.content
        assert payload["hash"] == "abc123"
        assert payload["repo_path"] == "/path/to/repo"
        assert payload["document_path"] == "src/main.py"
        assert payload["language"] == "python"
        assert payload["node_type"] == "function"
        assert payload["node_name"] == "hello_world"
        assert payload["start_byte"] == 0
        assert payload["end_byte"] == 50
        assert payload["start_line"] == 1
        assert payload["end_line"] == 2
        assert payload["documentation"] == "A simple hello world function"
        assert payload["parent_scope"] == ""
        assert payload["signature"] == "def hello_world()"


class TestVectorStoreDeleteByDocumentPaths:
    """Tests for VectorStore.delete_by_document_paths."""

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, vector_store):
        """Test deleting empty list returns 0."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client

        # Execute
        result = await vector_store.delete_by_document_paths("test_collection", [])

        # Assert
        assert result == 0
        mock_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_single_path(self, vector_store):
        """Test deleting by single document path."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.delete_by_document_paths("test_collection", ["file1.py"])

        # Assert
        assert result == 1
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_delete_multiple_paths(self, vector_store):
        """Test deleting by multiple document paths."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")

        # Execute
        result = await vector_store.delete_by_document_paths(
            "test_collection", ["file1.py", "file2.py", "file3.py"]
        )

        # Assert
        assert result == 3
        mock_client.delete.assert_called_once()


class TestVectorStoreSearch:
    """Tests for VectorStore.search."""

    @pytest.mark.asyncio
    async def test_search_basic(self, vector_store):
        """Test basic search without filters."""
        # Setup
        mock_client = AsyncMock()
        mock_point = MagicMock()
        mock_point.id = "point1"
        mock_point.score = 0.95
        mock_point.payload = {
            "document": "test content",
            "document_path": "file1.py",
            "language": "python",
            "node_type": "function",
            "node_name": "test_func",
            "start_line": 1,
            "end_line": 10,
            "documentation": "Test function",
            "signature": "def test_func()",
            "parent_scope": "",
        }
        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        results = await vector_store.search("test_collection", "test query")

        # Assert
        assert len(results) == 1
        assert results[0]["id"] == "point1"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "test content"
        assert results[0]["file_path"] == "file1.py"
        mock_client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_file_path_filter(self, vector_store):
        """Test search with file path filter."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", file_path="src/main.py")

        # Assert
        call_args = mock_client.query_points.call_args
        query_filter = call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_directory_filter(self, vector_store):
        """Test search with directory path (prefix match)."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", file_path="src/")

        # Assert
        call_args = mock_client.query_points.call_args
        query_filter = call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_language_filter(self, vector_store):
        """Test search with language filter."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", language="python")

        # Assert
        call_args = mock_client.query_points.call_args
        query_filter = call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_node_type_filter(self, vector_store):
        """Test search with node_type filter."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", node_type="function")

        # Assert
        call_args = mock_client.query_points.call_args
        query_filter = call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_node_name_filter(self, vector_store):
        """Test search with node_name filter."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", node_name="my_function")

        # Assert
        call_args = mock_client.query_points.call_args
        query_filter = call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_has_documentation_true(self, vector_store):
        """Test search filtering for nodes with documentation."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", has_documentation=True)

        # Assert
        call_args = mock_client.query_points.call_args
        query_filter = call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_has_documentation_false(self, vector_store):
        """Test search filtering for nodes without documentation."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", has_documentation=False)

        # Assert
        call_args = mock_client.query_points.call_args
        query_filter = call_args[1]["query_filter"]
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_limit(self, vector_store):
        """Test search with custom limit."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        await vector_store.search("test_collection", "test query", limit=50)

        # Assert
        call_args = mock_client.query_points.call_args
        assert call_args[1]["limit"] == 50

    @pytest.mark.asyncio
    async def test_search_not_initialized_raises(self, vector_store):
        """Test search when vector store not initialized raises error."""
        # Setup
        mock_client = AsyncMock()
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        # Don't set _vector_name or _embedding_model_name

        # Execute & Assert
        with pytest.raises(RuntimeError, match="Vector store not properly initialized"):
            await vector_store.search("test_collection", "test query")

    @pytest.mark.asyncio
    async def test_search_empty_results(self, vector_store):
        """Test search with no results."""
        # Setup
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        results = await vector_store.search("test_collection", "test query")

        # Assert
        assert results == []

    @pytest.mark.asyncio
    async def test_search_missing_payload_fields(self, vector_store):
        """Test search result formatting with missing payload fields."""
        # Setup
        mock_client = AsyncMock()
        mock_point = MagicMock()
        mock_point.id = "point1"
        mock_point.score = 0.95
        mock_point.payload = {}  # Empty payload
        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        results = await vector_store.search("test_collection", "test query")

        # Assert
        assert len(results) == 1
        assert results[0]["content"] == ""
        assert results[0]["file_path"] == ""
        assert results[0]["language"] == ""
        assert results[0]["node_type"] == ""
        assert results[0]["node_name"] == ""
        assert results[0]["start_line"] == 0
        assert results[0]["end_line"] == 0

    @pytest.mark.asyncio
    async def test_search_null_payload(self, vector_store):
        """Test search result formatting with null payload."""
        # Setup
        mock_client = AsyncMock()
        mock_point = MagicMock()
        mock_point.id = "point1"
        mock_point.score = 0.95
        mock_point.payload = None
        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results
        vector_store._client = mock_client
        vector_store._initialized_collections.add("test_collection")
        vector_store._vector_name = "test-vector"
        vector_store._embedding_model_name = "test-model"

        # Execute
        results = await vector_store.search("test_collection", "test query")

        # Assert
        assert len(results) == 1
        assert results[0]["content"] == ""
        assert results[0]["file_path"] == ""
