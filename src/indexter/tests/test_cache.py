"""Tests for CacheManager."""

from unittest.mock import MagicMock, patch

import pytest

from indexter.cache import CacheManager


@pytest.fixture
def cache_manager(tmp_path):
    """Create a CacheManager with mocked settings and repo."""
    mock_repo = MagicMock()
    mock_repo.name = "test_repo"

    with patch("indexter.cache.settings") as mock_settings:
        mock_settings.cache_dir = tmp_path
        manager = CacheManager(mock_repo)
        yield manager


class TestCacheManager:
    """Test CacheManager class."""

    def test_should_initialize_with_repo(self, tmp_path):
        """Test CacheManager initializes with repo and sets cache_dir."""
        mock_repo = MagicMock()
        mock_repo.name = "test_repo"

        with patch("indexter.cache.settings") as mock_settings:
            mock_settings.cache_dir = tmp_path
            cache = CacheManager(mock_repo)

            assert cache.repo is mock_repo
            assert cache.cache_dir == tmp_path / "test_repo"

    def test_key_path_should_return_json_file_path(self, cache_manager):
        """Test _key_path returns correct path for a key with repo prefix."""
        result = cache_manager._key_path("test_key")

        assert result == cache_manager.cache_dir / "test_repo_test_key.json"

    def test_get_should_return_none_when_cache_miss(self, cache_manager):
        """Test get returns None when key doesn't exist."""
        result = cache_manager.get("nonexistent")

        assert result is None

    def test_get_should_return_data_when_cache_hit(self, cache_manager):
        """Test get returns cached data when key exists."""
        cache_manager.set("test_key", '{"data": "value"}')

        result = cache_manager.get("test_key")

        assert result == '{"data": "value"}'

    def test_set_should_create_cache_directory(self, cache_manager):
        """Test set creates cache directory if it doesn't exist."""
        assert not cache_manager.cache_dir.exists()

        cache_manager.set("test_key", "test_data")

        assert cache_manager.cache_dir.exists()
        assert cache_manager.cache_dir.is_dir()

    def test_set_should_write_data_to_file(self, cache_manager):
        """Test set writes data to the correct file."""
        cache_manager.set("test_key", '{"hello": "world"}')

        file_path = cache_manager._key_path("test_key")
        assert file_path.exists()
        assert file_path.read_text() == '{"hello": "world"}'

    def test_set_should_overwrite_existing_data(self, cache_manager):
        """Test set overwrites existing cached data."""
        cache_manager.set("test_key", "old_data")

        cache_manager.set("test_key", "new_data")

        assert cache_manager.get("test_key") == "new_data"

    def test_delete_should_return_true_when_key_exists(self, cache_manager):
        """Test delete returns True when key is deleted."""
        cache_manager.set("test_key", "data")

        result = cache_manager.delete("test_key")

        assert result is True
        assert cache_manager.get("test_key") is None

    def test_delete_should_return_false_when_key_not_exists(self, cache_manager):
        """Test delete returns False when key doesn't exist."""
        result = cache_manager.delete("nonexistent")

        assert result is False

    def test_clear_should_remove_all_cached_files(self, cache_manager):
        """Test clear removes all files in cache directory."""
        cache_manager.set("key1", "data1")
        cache_manager.set("key2", "data2")
        cache_manager.set("key3", "data3")

        cache_manager.clear()

        assert not cache_manager.cache_dir.exists()
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
        assert cache_manager.get("key3") is None

    def test_clear_should_handle_nonexistent_directory(self, cache_manager):
        """Test clear handles case where cache directory doesn't exist."""
        assert not cache_manager.cache_dir.exists()

        # Should not raise
        cache_manager.clear()

        assert not cache_manager.cache_dir.exists()

    def test_multiple_keys_should_be_independent(self, cache_manager):
        """Test different keys are stored independently."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") == "value2"

        cache_manager.delete("key1")

        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") == "value2"
