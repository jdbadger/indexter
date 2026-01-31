"""Tests for the Repo, CacheManager, and StoreManager classes."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient, models

from indexter.config import RepoSettings
from indexter.exceptions import RepoExistsError, RepoNotFoundError
from indexter.models import (
    DocumentMetadata,
    IndexResult,
    Node,
    NodeMetadata,
    RepoMetadata,
    SearchResult,
    SearchResults,
)
from indexter.repo import CacheManager, Repo, StoreManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo_dir(tmp_path):
    """Create a temporary git repo directory."""
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    return repo_dir


@pytest.fixture
def repo_settings(tmp_repo_dir):
    """Create RepoSettings pointing at the tmp repo."""
    return RepoSettings(path=tmp_repo_dir)


@pytest.fixture
def repo(repo_settings):
    """Create a Repo instance with tmp settings."""
    return Repo(settings=repo_settings)


@pytest.fixture
def mock_client():
    """Create a MagicMock QdrantClient."""
    client = MagicMock(spec=QdrantClient)
    client.collection_exists.return_value = False
    client.get_embedding_size.return_value = 384
    client.get_collections.return_value = MagicMock(collections=[])
    return client


@pytest.fixture
def sample_node():
    """Create a sample Node for testing."""
    return Node(
        id=uuid.uuid4(),
        content="def hello(): pass",
        metadata=NodeMetadata(
            repo="my_repo",
            repo_path="/tmp/my_repo",
            document_path="src/main.py",
            document_hash="abc123",
            language="python",
            node_type="function",
            node_name="hello",
            start_byte=0,
            end_byte=17,
            start_line=1,
            end_line=1,
        ),
    )


@pytest.fixture
def sample_doc_metadata():
    """Sample DocumentMetadata for testing."""
    return DocumentMetadata(
        repo="my_repo",
        repo_path="/tmp/my_repo",
        ext=".py",
        size_bytes=100,
        mtime=1700000000.0,
    )


# ---------------------------------------------------------------------------
# CacheManager Tests
# ---------------------------------------------------------------------------


class TestCacheManager:
    def test_get_returns_none_when_no_cache(self, repo):
        """get() returns None when cache key does not exist."""
        cache = CacheManager(repo)
        assert cache.get("nonexistent") is None

    def test_set_and_get_roundtrip(self, repo):
        """set() then get() returns the stored value."""
        cache = CacheManager(repo)
        cache.set("hashmap", '{"a": "1"}')
        result = cache.get("hashmap")
        assert result == '{"a": "1"}'

    def test_delete_existing_key(self, repo):
        """delete() returns True and removes the cached file."""
        cache = CacheManager(repo)
        cache.set("test", "data")
        assert cache.delete("test") is True
        assert cache.get("test") is None

    def test_delete_nonexistent_key(self, repo):
        """delete() returns False when key does not exist."""
        cache = CacheManager(repo)
        assert cache.delete("missing") is False

    def test_clear_removes_all_keys(self, repo):
        """clear() removes all cached data for the repo."""
        cache = CacheManager(repo)
        cache.set("key1", "val1")
        cache.set("key2", "val2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_clear_noop_when_empty(self, repo):
        """clear() does not raise when cache dir does not exist."""
        cache = CacheManager(repo)
        cache.clear()  # Should not raise

    def test_cache_dir_derives_from_settings(self, repo):
        """Cache dir includes the repo name."""
        cache = CacheManager(repo)
        assert repo.name in str(cache.cache_dir)


# ---------------------------------------------------------------------------
# StoreManager Tests
# ---------------------------------------------------------------------------


class TestStoreManagerProperties:
    @patch("indexter.repo.settings")
    def test_dense_model_name(self, mock_settings, repo):
        """dense_model_name returns the configured embedding model."""
        mock_settings.store.embedding_model = "test-model/dense"
        store = StoreManager(repo)
        assert store.dense_model_name == "test-model/dense"

    @patch("indexter.repo.settings")
    def test_sparse_model_name(self, mock_settings, repo):
        """sparse_model_name returns the configured sparse model."""
        mock_settings.store.sparse_embedding_model = "test-model/sparse"
        store = StoreManager(repo)
        assert store.sparse_model_name == "test-model/sparse"

    @patch("indexter.repo.settings")
    def test_dense_vector_name(self, mock_settings, repo):
        """dense_vector_name equals the embedding model name."""
        mock_settings.store.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        store = StoreManager(repo)
        assert store.dense_vector_name == "sentence-transformers/all-MiniLM-L6-v2"

    @patch("indexter.repo.settings")
    def test_sparse_vector_name(self, mock_settings, repo):
        """sparse_vector_name equals the sparse model name."""
        mock_settings.store.sparse_embedding_model = "Qdrant/bm25"
        store = StoreManager(repo)
        assert store.sparse_vector_name == "Qdrant/bm25"


class TestStoreManagerCreateCollection:
    def test_creates_collection_when_not_exists(self, repo, mock_client):
        """create_collection calls client.create_collection when collection missing."""
        mock_client.collection_exists.return_value = False
        store = StoreManager(repo)
        store.create_collection(mock_client)
        mock_client.create_collection.assert_called_once()
        kwargs = mock_client.create_collection.call_args
        assert kwargs.kwargs["collection_name"] == repo.collection_name

    def test_skips_creation_when_exists(self, repo, mock_client):
        """create_collection does nothing when collection already exists."""
        mock_client.collection_exists.return_value = True
        store = StoreManager(repo)
        store.create_collection(mock_client)
        mock_client.create_collection.assert_not_called()


class TestStoreManagerDeleteCollection:
    def test_deletes_collection(self, repo, mock_client):
        """delete_collection calls client.delete_collection."""
        store = StoreManager(repo)
        store.delete_collection(mock_client)
        mock_client.delete_collection.assert_called_once_with(
            collection_name=repo.collection_name,
        )


class TestStoreManagerEnsureCollection:
    def test_creates_when_missing(self, repo, mock_client):
        """ensure_collection creates collection if it doesn't exist and returns True."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        store = StoreManager(repo)
        result = store.ensure_collection(mock_client)
        assert result is True
        mock_client.create_collection.assert_called_once()

    def test_returns_false_when_exists(self, repo, mock_client):
        """ensure_collection returns False if collection already present."""
        existing = MagicMock()
        existing.name = repo.collection_name
        mock_client.get_collections.return_value = MagicMock(collections=[existing])
        store = StoreManager(repo)
        result = store.ensure_collection(mock_client)
        assert result is False


class TestStoreManagerUpsertNodes:
    def test_upserts_nodes_in_batches(self, repo, mock_client, sample_node):
        """upsert_nodes calls client.upsert for each batch."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        repo.settings.upsert_batch_size = 2
        nodes = [sample_node] * 5
        store = StoreManager(repo)
        count = store.upsert_nodes(mock_client, nodes, batch_size=2)
        assert count == 5
        # 3 batches: 2, 2, 1
        assert mock_client.upsert.call_count == 3

    def test_returns_zero_for_empty_list(self, repo, mock_client):
        """upsert_nodes returns 0 if nodes list is empty."""
        store = StoreManager(repo)
        count = store.upsert_nodes(mock_client, [])
        assert count == 0
        mock_client.upsert.assert_not_called()

    def test_uses_repo_batch_size_setting(self, repo, mock_client, sample_node):
        """upsert_nodes uses repo's upsert_batch_size when set."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        repo.settings.upsert_batch_size = 3
        nodes = [sample_node] * 6
        store = StoreManager(repo)
        store.upsert_nodes(mock_client, nodes, batch_size=100)
        # batch_size=3 → 2 batches of 3
        assert mock_client.upsert.call_count == 2


class TestStoreManagerDeleteNodes:
    def test_deletes_nodes_by_hash(self, repo, mock_client):
        """delete_nodes creates filter and calls client.delete."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.count.return_value = MagicMock(count=3)
        store = StoreManager(repo)
        deleted = store.delete_nodes(mock_client, ["hash1", "hash2"])
        assert deleted == 3
        mock_client.delete.assert_called_once()

    def test_returns_zero_for_empty_hashes(self, repo, mock_client):
        """delete_nodes returns 0 when hashes list is empty."""
        store = StoreManager(repo)
        deleted = store.delete_nodes(mock_client, [])
        assert deleted == 0
        mock_client.delete.assert_not_called()


class TestStoreManagerSearch:
    def test_returns_search_results(self, repo, mock_client):
        """search returns SearchResults with matched points."""
        # Collection exists
        existing = MagicMock()
        existing.name = repo.collection_name
        mock_client.get_collections.return_value = MagicMock(collections=[existing])

        mock_point = MagicMock()
        mock_point.payload = {"content": "def foo(): pass", "language": "python"}
        mock_point.score = 0.95
        mock_client.query_points.return_value = MagicMock(points=[mock_point])

        store = StoreManager(repo)
        results = store.search(mock_client, "find foo")
        assert isinstance(results, SearchResults)
        assert results.count == 1
        assert results.results[0].score == 0.95
        assert results.results[0].content == "def foo(): pass"

    def test_returns_empty_when_collection_just_created(self, repo, mock_client):
        """search returns empty results when collection was just created (empty)."""
        # ensure_collection returns True when it creates a new one
        mock_client.get_collections.return_value = MagicMock(collections=[])
        store = StoreManager(repo)
        results = store.search(mock_client, "anything")
        assert results.count == 0

    @pytest.mark.parametrize(
        "filter_kwargs,expected_key",
        [
            ({"language": "python"}, "language"),
            ({"node_type": "function"}, "node_type"),
            ({"node_name": "hello"}, "node_name"),
            ({"parent_scope": "MyClass"}, "parent_scope"),
            ({"document_path": "src/main.py"}, "document_path"),
        ],
    )
    def test_applies_filters(self, repo, mock_client, filter_kwargs, expected_key):
        """search passes filters through to query_points."""
        existing = MagicMock()
        existing.name = repo.collection_name
        mock_client.get_collections.return_value = MagicMock(collections=[existing])
        mock_client.query_points.return_value = MagicMock(points=[])

        store = StoreManager(repo)
        store.search(mock_client, "query", **filter_kwargs)
        mock_client.query_points.assert_called_once()
        _, kwargs = mock_client.query_points.call_args
        assert kwargs["query_filter"] is not None


class TestStoreManagerBuildFilterConditions:
    def test_empty_when_no_filters(self):
        """Returns empty list when all filter params are None."""
        conditions = StoreManager._build_filter_conditions()
        assert conditions == []

    def test_directory_prefix_filter(self):
        """Trailing / on document_path uses MatchText (prefix)."""
        conditions = StoreManager._build_filter_conditions(document_path="src/")
        assert len(conditions) == 1
        assert isinstance(conditions[0].match, models.MatchText)

    def test_exact_document_path_filter(self):
        """Exact document_path uses MatchValue."""
        conditions = StoreManager._build_filter_conditions(document_path="src/main.py")
        assert len(conditions) == 1
        assert isinstance(conditions[0].match, models.MatchValue)

    def test_has_documentation_true(self):
        """has_documentation=True filters for non-empty documentation."""
        conditions = StoreManager._build_filter_conditions(has_documentation=True)
        assert len(conditions) == 1

    def test_has_documentation_false(self):
        """has_documentation=False filters for empty documentation."""
        conditions = StoreManager._build_filter_conditions(has_documentation=False)
        assert len(conditions) == 1
        assert isinstance(conditions[0].match, models.MatchValue)

    def test_multiple_filters_combined(self):
        """Multiple filters produce multiple conditions."""
        conditions = StoreManager._build_filter_conditions(
            language="python",
            node_type="function",
            node_name="hello",
        )
        assert len(conditions) == 3


# ---------------------------------------------------------------------------
# Repo Tests
# ---------------------------------------------------------------------------


class TestRepoInit:
    def test_properties(self, repo, tmp_repo_dir):
        """Repo exposes name, path, collection_name from settings."""
        assert repo.name == tmp_repo_dir.name
        assert repo.path == str(tmp_repo_dir)
        assert repo.collection_name is not None

    def test_cache_property(self, repo):
        """cache property returns a CacheManager."""
        assert isinstance(repo.cache, CacheManager)

    def test_store_property(self, repo):
        """store property returns a StoreManager."""
        assert isinstance(repo.store, StoreManager)


class TestRepoInitMethod:
    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load", return_value=[])
    def test_init_new_repo(self, mock_load, mock_save, tmp_repo_dir):
        """init() creates and registers a new repository."""
        new_repo = Repo.init(tmp_repo_dir)
        assert new_repo.name == tmp_repo_dir.name
        mock_save.assert_called_once()

    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load")
    def test_init_returns_existing_for_same_path(self, mock_load, mock_save, tmp_repo_dir):
        """init() returns existing Repo if same path is already registered."""
        existing = RepoSettings(path=tmp_repo_dir)
        mock_load.return_value = [existing]
        result = Repo.init(tmp_repo_dir)
        assert result.name == tmp_repo_dir.name
        mock_save.assert_not_called()

    @patch("indexter.repo.RepoSettings.load")
    def test_init_raises_for_name_conflict(self, mock_load, tmp_path):
        """init() raises RepoExistsError for same name at different path."""
        repo_a = tmp_path / "my_repo"
        repo_a.mkdir()
        (repo_a / ".git").mkdir()

        repo_b = tmp_path / "other" / "my_repo"
        repo_b.mkdir(parents=True)
        (repo_b / ".git").mkdir()

        existing = RepoSettings(path=repo_a)
        mock_load.return_value = [existing]

        with pytest.raises(RepoExistsError, match="already exists"):
            Repo.init(repo_b)


class TestRepoGetOne:
    @patch("indexter.repo.RepoSettings.load")
    def test_get_one_found(self, mock_load, tmp_repo_dir):
        """get_one() returns the matching Repo."""
        s = RepoSettings(path=tmp_repo_dir)
        mock_load.return_value = [s]
        repo = Repo.get_one(tmp_repo_dir.name)
        assert repo.name == tmp_repo_dir.name

    @patch("indexter.repo.RepoSettings.load", return_value=[])
    def test_get_one_not_found(self, mock_load):
        """get_one() raises RepoNotFoundError."""
        with pytest.raises(RepoNotFoundError, match="not found"):
            Repo.get_one("nonexistent")


class TestRepoGetAll:
    @patch("indexter.repo.RepoSettings.load", return_value=[])
    def test_get_all_empty(self, mock_load):
        """get_all() returns empty list when no repos registered."""
        repos = Repo.get_all()
        assert repos == []

    @patch("indexter.repo.RepoSettings.load")
    def test_get_all_returns_repos(self, mock_load, tmp_repo_dir):
        """get_all() returns Repo instances for each registered repo."""
        s = RepoSettings(path=tmp_repo_dir)
        mock_load.return_value = [s]
        repos = Repo.get_all()
        assert len(repos) == 1
        assert repos[0].name == tmp_repo_dir.name


class TestRepoRemoveOne:
    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load")
    def test_remove_one_deletes_collection_and_config(self, mock_load, mock_save, tmp_repo_dir, mock_client):
        """remove_one() deletes collection, cache, and config entry."""
        s = RepoSettings(path=tmp_repo_dir)
        mock_load.return_value = [s]
        result = Repo.remove_one(tmp_repo_dir.name, mock_client)
        assert result is True
        mock_client.delete_collection.assert_called_once()
        mock_save.assert_called_once()

    @patch("indexter.repo.RepoSettings.load", return_value=[])
    def test_remove_one_raises_when_not_found(self, mock_load, mock_client):
        """remove_one() raises RepoNotFoundError for unknown repo."""
        with pytest.raises(RepoNotFoundError):
            Repo.remove_one("ghost", mock_client)


class TestRepoRemoveAll:
    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load", return_value=[])
    def test_remove_all_empty(self, mock_load, mock_save, mock_client):
        """remove_all() returns False when no repos registered."""
        result = Repo.remove_all(mock_client)
        assert result is False

    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load")
    def test_remove_all_deletes_all(self, mock_load, mock_save, tmp_repo_dir, mock_client):
        """remove_all() deletes all collections and returns True."""
        s = RepoSettings(path=tmp_repo_dir)
        mock_load.return_value = [s]
        result = Repo.remove_all(mock_client)
        assert result is True
        mock_client.delete_collection.assert_called_once()
        mock_save.assert_called_with([])


# ---------------------------------------------------------------------------
# Repo Staleness / Hashmap Tests
# ---------------------------------------------------------------------------


class TestRepoIsStale:
    def test_stale_when_no_cache(self, repo):
        """is_stale returns True when no cached hashmap exists."""
        assert repo.is_stale is True

    def test_not_stale_when_cache_matches(self, repo):
        """is_stale returns False when cached hashmap equals current."""
        hashmap = {"src/main.py": "hash1"}
        with (
            patch.object(Repo, "_get_cached_hashmap", return_value=hashmap),
            patch.object(Repo, "_get_hashmap", return_value=hashmap),
        ):
            assert repo.is_stale is False

    def test_stale_when_cache_differs(self, repo):
        """is_stale returns True when cached hashmap differs from current."""
        with (
            patch.object(Repo, "_get_cached_hashmap", return_value={"a": "1"}),
            patch.object(Repo, "_get_hashmap", return_value={"b": "2"}),
        ):
            assert repo.is_stale is True


class TestRepoHashmapHelpers:
    def test_get_hashes(self, repo):
        """_get_hashes extracts values from hashmap dict."""
        result = repo._get_hashes({"a": "h1", "b": "h2"})
        assert set(result) == {"h1", "h2"}

    def test_get_hashes_to_add(self, repo):
        """_get_hashes_to_add returns new hashes not in cached."""
        result = repo._get_hashes_to_add(["h1", "h2", "h3"], ["h1"])
        assert set(result) == {"h2", "h3"}

    def test_get_hashes_to_add_with_empty_cached(self, repo):
        """_get_hashes_to_add treats empty cached as all-new."""
        result = repo._get_hashes_to_add(["h1", "h2"], [])
        assert set(result) == {"h1", "h2"}

    def test_get_hashes_to_delete(self, repo):
        """_get_hashes_to_delete returns removed hashes missing from current."""
        result = repo._get_hashes_to_delete(["h1"], ["h1", "h2", "h3"])
        assert set(result) == {"h2", "h3"}

    def test_get_hashes_to_delete_with_empty_cached(self, repo):
        """_get_hashes_to_delete returns empty when no cached state."""
        result = repo._get_hashes_to_delete(["h1"], [])
        assert result == []

    def test_set_and_get_hashmap(self, repo):
        """_set_hashmap then _get_cached_hashmap roundtrips."""
        hm = {"src/main.py": "abc123"}
        repo._set_hashmap(hm)
        result = repo._get_cached_hashmap()
        assert result == hm

    def test_delete_hashmap(self, repo):
        """_delete_hashmap removes the hashmap from cache."""
        repo._set_hashmap({"a": "1"})
        assert repo._delete_hashmap() is True
        assert repo._get_cached_hashmap() == {}

    def test_delete_hashmap_nonexistent(self, repo):
        """_delete_hashmap returns False when no hashmap cached."""
        assert repo._delete_hashmap() is False


# ---------------------------------------------------------------------------
# Repo.index() Tests
# ---------------------------------------------------------------------------


class TestRepoIndex:
    @patch("indexter.repo.Walker")
    def test_index_no_changes(self, MockWalker, repo, mock_client):
        """index() returns early when current hashmap matches cached."""
        repo._set_hashmap({"src/main.py": "hash1"})
        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/main.py", "content", MagicMock()),
            ]
        )

        # Patch _get_hashmap to return same as cached
        with patch.object(Repo, "_get_hashmap", return_value={"src/main.py": "hash1"}):
            result = repo.index(mock_client)

        assert isinstance(result, IndexResult)
        assert result.nodes_added == 0
        assert result.nodes_deleted == 0

    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_index_full_rebuilds(self, MockWalker, MockParser, repo, mock_client, sample_doc_metadata):
        """index(full=True) deletes collection and re-indexes everything."""
        # No cached hashmap
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.count.return_value = MagicMock(count=0)

        node_meta = NodeMetadata(
            repo="my_repo",
            repo_path="/tmp/my_repo",
            document_path="src/main.py",
            document_hash="hash1",
            language="python",
            node_type="function",
            node_name="hello",
            start_byte=0,
            end_byte=17,
            start_line=1,
            end_line=1,
        )

        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/main.py", "def hello(): pass", sample_doc_metadata),
            ]
        )
        MockParser.return_value.parse.return_value = iter(
            [
                ("def hello(): pass", node_meta),
            ]
        )

        result = repo.index(mock_client, full=True)
        assert isinstance(result, IndexResult)
        # delete_collection should be called for full re-index
        mock_client.delete_collection.assert_called_once()

    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_index_incremental_adds_new_docs(self, MockWalker, MockParser, repo, mock_client, sample_doc_metadata):
        """index() incrementally adds new documents."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.count.return_value = MagicMock(count=0)

        node_meta = NodeMetadata(
            repo="my_repo",
            repo_path="/tmp/my_repo",
            document_path="src/new.py",
            document_hash="newhash",
            language="python",
            node_type="function",
            node_name="new_func",
            start_byte=0,
            end_byte=20,
            start_line=1,
            end_line=1,
        )

        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/new.py", "def new_func(): pass", sample_doc_metadata),
            ]
        )
        MockParser.return_value.parse.return_value = iter(
            [
                ("def new_func(): pass", node_meta),
            ]
        )

        with (
            patch.object(Repo, "_get_hashmap", return_value={"src/new.py": "newhash"}),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
        ):
            result = repo.index(mock_client)

        assert result.nodes_added > 0
        assert "src/new.py" in result.documents_indexed

    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_index_incremental_deletes_removed_docs(self, MockWalker, MockParser, repo, mock_client):
        """index() deletes nodes for documents no longer present."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.count.return_value = MagicMock(count=2)

        MockWalker.return_value.walk.return_value = iter([])

        with (
            patch.object(Repo, "_get_hashmap", return_value={}),
            patch.object(Repo, "_get_cached_hashmap", return_value={"src/old.py": "oldhash"}),
        ):
            result = repo.index(mock_client)

        assert result.nodes_deleted == 2
        assert "src/old.py" in result.documents_deleted

    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_index_handles_parse_errors(self, MockWalker, MockParser, repo, mock_client, sample_doc_metadata):
        """index() records errors for documents that fail to parse."""
        mock_client.get_collections.return_value = MagicMock(collections=[])

        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/bad.py", "invalid content", sample_doc_metadata),
            ]
        )
        MockParser.return_value.parse.side_effect = RuntimeError("Parse failed")

        with (
            patch.object(Repo, "_get_hashmap", return_value={"src/bad.py": "badhash"}),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
        ):
            result = repo.index(mock_client)

        assert len(result.errors) > 0
        assert "Parse failed" in result.errors[0]


# ---------------------------------------------------------------------------
# Repo.search() Tests
# ---------------------------------------------------------------------------


class TestRepoSearch:
    def test_search_delegates_to_store(self, repo, mock_client):
        """search() delegates to store.search and annotates results."""
        mock_results = SearchResults(
            results=[SearchResult(content="def foo(): pass", score=0.9, metadata={})],
            query="find foo",
            filters={},
        )
        with patch.object(StoreManager, "search", return_value=mock_results):
            results = repo.search(mock_client, "find foo")

        assert results.query == "find foo"
        assert results.repo == repo.name
        assert results.repo_path == repo.path
        assert results.count == 1

    def test_search_passes_all_filters(self, repo, mock_client):
        """search() passes all filter kwargs to store.search."""
        mock_results = SearchResults(results=[], query="q", filters={})
        with patch.object(StoreManager, "search", return_value=mock_results) as mock_search:
            repo.search(
                mock_client,
                "query",
                language="python",
                node_type="function",
                node_name="hello",
                document_path="src/main.py",
                parent_scope="MyClass",
                has_documentation=True,
                limit=5,
            )

        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["language"] == "python"
        assert call_kwargs.kwargs["node_type"] == "function"
        assert call_kwargs.kwargs["node_name"] == "hello"
        assert call_kwargs.kwargs["document_path"] == "src/main.py"
        assert call_kwargs.kwargs["parent_scope"] == "MyClass"
        assert call_kwargs.kwargs["has_documentation"] is True
        assert call_kwargs.kwargs["limit"] == 5

    def test_search_uses_default_limit(self, repo, mock_client):
        """search() uses repo's top_k when limit is not provided."""
        mock_results = SearchResults(results=[], query="q", filters={})
        with patch.object(StoreManager, "search", return_value=mock_results) as mock_search:
            repo.search(mock_client, "query")

        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["limit"] == repo.settings.top_k


# ---------------------------------------------------------------------------
# Repo.metadata Tests
# ---------------------------------------------------------------------------


class TestRepoMetadata:
    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_metadata_aggregates_docs(self, MockWalker, MockParser, repo, sample_doc_metadata):
        """metadata property aggregates document counts and languages."""
        node_meta = MagicMock()
        node_meta.node_type = "function"

        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/main.py", "content", sample_doc_metadata),
            ]
        )
        MockParser.return_value.parse.return_value = iter(
            [
                ("def foo(): pass", node_meta),
            ]
        )
        MockParser.return_value.language = "python"

        meta = repo.metadata
        assert isinstance(meta, RepoMetadata)
        assert meta.documents == 1
        assert meta.nodes == 1
        assert "src/main.py" in meta.document_paths

    @patch("indexter.repo.Walker")
    def test_metadata_empty_repo(self, MockWalker, repo):
        """metadata returns empty RepoMetadata for empty repo."""
        MockWalker.return_value.walk.return_value = iter([])
        meta = repo.metadata
        assert meta.documents == 0
        assert meta.nodes == 0

    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_metadata_respects_max_files(self, MockWalker, MockParser, repo, sample_doc_metadata):
        """metadata stops after max_files documents."""
        repo.settings.max_files = 1
        node_meta = MagicMock()
        node_meta.node_type = "function"

        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/a.py", "content", sample_doc_metadata),
                ("src/b.py", "content", sample_doc_metadata),
            ]
        )
        MockParser.return_value.parse.return_value = iter(
            [
                ("def foo(): pass", node_meta),
            ]
        )
        MockParser.return_value.language = "python"

        meta = repo.metadata
        assert meta.documents == 1

    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_metadata_skips_unparseable_docs(self, MockWalker, MockParser, repo, sample_doc_metadata):
        """metadata continues when a document fails to parse."""
        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/bad.py", "content", sample_doc_metadata),
            ]
        )
        MockParser.side_effect = RuntimeError("parse error")

        meta = repo.metadata
        assert meta.documents == 0


# ---------------------------------------------------------------------------
# Repo._get_hashmap Tests
# ---------------------------------------------------------------------------


class TestRepoGetHashmap:
    @patch("indexter.repo.Walker")
    def test_get_hashmap_respects_max_files(self, MockWalker, repo, sample_doc_metadata):
        """_get_hashmap stops walking after max_files."""
        repo.settings.max_files = 1
        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/a.py", "a", sample_doc_metadata),
                ("src/b.py", "b", sample_doc_metadata),
            ]
        )
        hashmap = repo._get_hashmap()
        assert len(hashmap) == 1

    @patch("indexter.repo.Walker")
    def test_get_hashmap_handles_doc_errors(self, MockWalker, repo):
        """_get_hashmap skips documents that raise during construction."""
        # Pass invalid metadata that will cause Document() to fail
        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/bad.py", "content", None),
            ]
        )
        hashmap = repo._get_hashmap()
        assert hashmap == {}


# ---------------------------------------------------------------------------
# Repo.remove_one / remove_all cache cleanup
# ---------------------------------------------------------------------------


class TestRepoRemoveCacheCleanup:
    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load")
    def test_remove_one_cleans_cache_dir(self, mock_load, mock_save, tmp_repo_dir, mock_client):
        """remove_one() deletes cache directory when it exists."""
        s = RepoSettings(path=tmp_repo_dir)
        mock_load.return_value = [s]

        # Pre-create some cache data
        repo = Repo(settings=s)
        repo.cache.set("hashmap", '{"a": "1"}')
        assert repo.cache.cache_dir.exists()

        Repo.remove_one(tmp_repo_dir.name, mock_client)
        assert not repo.cache.cache_dir.exists()

    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load")
    def test_remove_one_returns_false_when_already_gone(self, mock_load, mock_save, tmp_repo_dir, mock_client):
        """remove_one() returns False when repo already removed from config."""
        s = RepoSettings(path=tmp_repo_dir)
        # First call returns repo, second (after remove) returns empty
        mock_load.side_effect = [[s], []]
        result = Repo.remove_one(tmp_repo_dir.name, mock_client)
        # new_repo_settings == repo_settings (both empty after filtering)
        # This tests the `return False` branch
        assert result is False

    @patch("indexter.repo.RepoSettings.save")
    @patch("indexter.repo.RepoSettings.load")
    def test_remove_all_cleans_cache_dirs(self, mock_load, mock_save, tmp_repo_dir, mock_client):
        """remove_all() deletes cache directories."""
        s = RepoSettings(path=tmp_repo_dir)
        mock_load.return_value = [s]

        repo = Repo(settings=s)
        repo.cache.set("hashmap", '{"a": "1"}')

        Repo.remove_all(mock_client)
        assert not repo.cache.cache_dir.exists()


# ---------------------------------------------------------------------------
# Repo.index() mid-batch upsert
# ---------------------------------------------------------------------------


class TestRepoIndexBatchUpsert:
    @patch("indexter.repo.Parser")
    @patch("indexter.repo.Walker")
    def test_index_upserts_mid_batch(self, MockWalker, MockParser, repo, mock_client, sample_doc_metadata):
        """index() upserts mid-walk when node count exceeds batch size."""
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.count.return_value = MagicMock(count=0)
        repo.settings.upsert_batch_size = 1  # tiny batch to trigger mid-batch upsert

        node_meta = NodeMetadata(
            repo="my_repo",
            repo_path="/tmp/my_repo",
            document_path="src/a.py",
            document_hash="hash_a",
            language="python",
            node_type="function",
            node_name="a",
            start_byte=0,
            end_byte=10,
            start_line=1,
            end_line=1,
        )

        MockWalker.return_value.walk.return_value = iter(
            [
                ("src/a.py", "def a(): pass", sample_doc_metadata),
                ("src/b.py", "def b(): pass", sample_doc_metadata),
            ]
        )
        MockParser.return_value.parse.return_value = iter(
            [
                ("def a(): pass", node_meta),
            ]
        )

        with (
            patch.object(Repo, "_get_hashmap", return_value={"src/a.py": "ha", "src/b.py": "hb"}),
            patch.object(Repo, "_get_cached_hashmap", return_value={}),
        ):
            result = repo.index(mock_client)

        assert result.nodes_added >= 1
