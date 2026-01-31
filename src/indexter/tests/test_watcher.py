"""Tests for the background file watcher."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import QdrantClient
from watchfiles import Change

from indexter.config.config import WatchSettings
from indexter.repo import Repo
from indexter.watcher import IndexterFilter, _group_changes_by_repo, watch_repos

# ---------------------------------------------------------------------------
# IndexterFilter
# ---------------------------------------------------------------------------


class TestIndexterFilter:
    """Test IndexterFilter construction from ignore patterns."""

    def test_should_add_directory_patterns_to_ignore_dirs(self):
        """Patterns ending in / become ignored directories."""
        filt = IndexterFilter(["__pycache__/", ".venv/", "node_modules/"])
        assert "__pycache__" in filt.ignore_dirs
        assert ".venv" in filt.ignore_dirs
        assert "node_modules" in filt.ignore_dirs

    def test_should_convert_glob_patterns_to_regex(self):
        """Glob patterns like *.pyc become regex entity patterns."""
        filt = IndexterFilter(["*.pyc", "*.log"])
        # fnmatch.translate produces a regex; check the tuple has entries
        assert len(filt.ignore_entity_patterns) >= 2
        # The pattern should match filenames
        import re

        combined = "|".join(filt.ignore_entity_patterns)
        assert re.search(combined, "module.pyc")
        assert re.search(combined, "app.log")

    def test_should_convert_bare_names_to_exact_match(self):
        """Bare names like .DS_Store become exact-match entity patterns."""
        filt = IndexterFilter([".DS_Store", "Thumbs.db"])
        import re

        combined = "|".join(filt.ignore_entity_patterns)
        assert re.search(combined, ".DS_Store")
        assert re.search(combined, "Thumbs.db")

    def test_should_preserve_default_filter_dirs(self):
        """DefaultFilter default dirs (.git, __pycache__) are preserved."""
        filt = IndexterFilter([])
        # DefaultFilter ignores .git by default
        assert ".git" in filt.ignore_dirs

    def test_should_merge_custom_with_defaults(self):
        """Custom dir patterns are merged with, not replacing, defaults."""
        filt = IndexterFilter(["custom_dir/"])
        assert "custom_dir" in filt.ignore_dirs
        assert ".git" in filt.ignore_dirs  # default still present

    def test_should_handle_empty_patterns(self):
        """Empty pattern list produces a valid filter."""
        filt = IndexterFilter([])
        assert callable(filt)

    def test_should_reject_ignored_directory(self, tmp_path):
        """Filter returns False for paths inside ignored directories."""
        filt = IndexterFilter(["build/"])
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        assert filt(Change.modified, str(build_dir)) is False

    def test_should_reject_ignored_file_pattern(self, tmp_path):
        """Filter returns False for files matching glob patterns."""
        filt = IndexterFilter(["*.pyc"])
        pyc_file = tmp_path / "module.pyc"
        pyc_file.touch()
        assert filt(Change.modified, str(pyc_file)) is False

    def test_should_accept_normal_file(self, tmp_path):
        """Filter returns True for files not matching any ignore pattern."""
        filt = IndexterFilter(["*.pyc", "build/"])
        py_file = tmp_path / "module.py"
        py_file.touch()
        assert filt(Change.modified, str(py_file)) is True


# ---------------------------------------------------------------------------
# _group_changes_by_repo
# ---------------------------------------------------------------------------


class TestGroupChangesByRepo:
    """Test change grouping by repository path."""

    def test_should_group_changes_to_correct_repo(self):
        """Changes are grouped by matching repo path prefix."""
        repo_a = MagicMock()
        repo_a.name = "repo_a"
        repo_a.path = "/home/user/repo_a"
        repo_b = MagicMock()
        repo_b.name = "repo_b"
        repo_b.path = "/home/user/repo_b"

        changes = {
            (Change.modified, "/home/user/repo_a/src/main.py"),
            (Change.added, "/home/user/repo_b/README.md"),
        }
        grouped = _group_changes_by_repo(changes, [repo_a, repo_b])

        assert "repo_a" in grouped
        assert "repo_b" in grouped
        assert len(grouped["repo_a"]) == 1
        assert len(grouped["repo_b"]) == 1

    def test_should_ignore_changes_outside_repos(self):
        """Changes not matching any repo path are dropped."""
        repo = MagicMock()
        repo.name = "my_repo"
        repo.path = "/home/user/my_repo"

        changes = {(Change.modified, "/other/path/file.py")}
        grouped = _group_changes_by_repo(changes, [repo])

        assert grouped == {}

    def test_should_handle_multiple_changes_in_same_repo(self):
        """Multiple changes in the same repo are grouped together."""
        repo = MagicMock()
        repo.name = "my_repo"
        repo.path = "/home/user/my_repo"

        changes = {
            (Change.modified, "/home/user/my_repo/a.py"),
            (Change.added, "/home/user/my_repo/b.py"),
            (Change.deleted, "/home/user/my_repo/c.py"),
        }
        grouped = _group_changes_by_repo(changes, [repo])

        assert len(grouped["my_repo"]) == 3

    def test_should_return_empty_dict_for_empty_changes(self):
        """Empty change set produces empty grouping."""
        repo = MagicMock()
        repo.name = "my_repo"
        repo.path = "/home/user/my_repo"
        assert _group_changes_by_repo(set(), [repo]) == {}


# ---------------------------------------------------------------------------
# watch_repos
# ---------------------------------------------------------------------------


class TestWatchRepos:
    """Test the main watch_repos coroutine."""

    @pytest.fixture
    def client(self):
        return MagicMock(spec=QdrantClient)

    @pytest.fixture
    def watch_settings(self):
        return WatchSettings(enabled=True, debounce_ms=100, poll_delay_ms=100)

    async def test_should_sleep_when_no_repos(self, client, watch_settings):
        """When no repos are registered, sleeps and re-checks."""
        stop = asyncio.Event()
        call_count = 0

        def fake_get_all():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                stop.set()
            return []

        with (
            patch.object(Repo, "get_all", side_effect=fake_get_all),
            patch("indexter.watcher._NO_REPOS_SLEEP", 0.05),
        ):
            await asyncio.wait_for(watch_repos(client, stop, watch_settings), timeout=5)

        assert call_count >= 2

    async def test_should_index_stale_repo_on_change(self, client, watch_settings):
        """Triggers incremental index for a stale repo when files change."""
        stop = asyncio.Event()
        repo = MagicMock()
        repo.name = "test_repo"
        repo.path = "/home/user/test_repo"
        repo.settings.ignore_patterns = [".git/", "*.pyc"]
        repo.is_stale = True

        changes = {(Change.modified, "/home/user/test_repo/main.py")}

        async def fake_awatch(*args, **kwargs):
            yield changes
            stop.set()

        with (
            patch.object(Repo, "get_all", return_value=[repo]),
            patch("indexter.watcher.awatch", side_effect=fake_awatch),
        ):
            await asyncio.wait_for(watch_repos(client, stop, watch_settings), timeout=5)

        repo.index.assert_called_once_with(client, full=False)

    async def test_should_skip_repo_not_stale(self, client, watch_settings):
        """Skips re-indexing when the repo is not stale."""
        stop = asyncio.Event()
        repo = MagicMock()
        repo.name = "test_repo"
        repo.path = "/home/user/test_repo"
        repo.settings.ignore_patterns = []
        repo.is_stale = False

        changes = {(Change.modified, "/home/user/test_repo/main.py")}

        async def fake_awatch(*args, **kwargs):
            yield changes
            stop.set()

        with (
            patch.object(Repo, "get_all", return_value=[repo]),
            patch("indexter.watcher.awatch", side_effect=fake_awatch),
        ):
            await asyncio.wait_for(watch_repos(client, stop, watch_settings), timeout=5)

        repo.index.assert_not_called()

    async def test_should_isolate_per_repo_errors(self, client, watch_settings):
        """An error indexing one repo does not block others."""
        stop = asyncio.Event()
        repo_a = MagicMock()
        repo_a.name = "repo_a"
        repo_a.path = "/home/user/repo_a"
        repo_a.settings.ignore_patterns = []
        repo_a.is_stale = True
        repo_a.index.side_effect = RuntimeError("boom")

        repo_b = MagicMock()
        repo_b.name = "repo_b"
        repo_b.path = "/home/user/repo_b"
        repo_b.settings.ignore_patterns = []
        repo_b.is_stale = True

        changes = {
            (Change.modified, "/home/user/repo_a/a.py"),
            (Change.modified, "/home/user/repo_b/b.py"),
        }

        async def fake_awatch(*args, **kwargs):
            yield changes
            stop.set()

        with (
            patch.object(Repo, "get_all", return_value=[repo_a, repo_b]),
            patch("indexter.watcher.awatch", side_effect=fake_awatch),
        ):
            await asyncio.wait_for(watch_repos(client, stop, watch_settings), timeout=5)

        # repo_a errored but repo_b should still be indexed
        repo_a.index.assert_called_once()
        repo_b.index.assert_called_once()

    async def test_should_stop_on_cancellation(self, client, watch_settings):
        """CancelledError exits the watcher cleanly."""
        stop = asyncio.Event()
        repo = MagicMock()
        repo.name = "test_repo"
        repo.path = "/home/user/test_repo"
        repo.settings.ignore_patterns = []

        async def fake_awatch(*args, **kwargs):
            raise asyncio.CancelledError
            yield  # make it an async generator  # pragma: no cover

        with (
            patch.object(Repo, "get_all", return_value=[repo]),
            patch("indexter.watcher.awatch", fake_awatch),
        ):
            # Should not raise
            await watch_repos(client, stop, watch_settings)

    async def test_should_recover_from_watcher_error(self, client, watch_settings):
        """General exceptions in awatch trigger a restart after sleep."""
        stop = asyncio.Event()
        repo = MagicMock()
        repo.name = "test_repo"
        repo.path = "/home/user/test_repo"
        repo.settings.ignore_patterns = []

        call_count = 0

        async def fake_awatch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("filesystem error")
            stop.set()
            # Need to yield at least once so awatch acts as async generator
            if False:
                yield  # pragma: no cover

        with (
            patch.object(Repo, "get_all", return_value=[repo]),
            patch("indexter.watcher.awatch", side_effect=fake_awatch),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            await asyncio.wait_for(watch_repos(client, stop, watch_settings), timeout=5)

        assert call_count >= 2

    async def test_should_handle_changes_outside_all_repos(self, client, watch_settings):
        """Changes that don't match any repo path are silently ignored."""
        stop = asyncio.Event()
        repo = MagicMock()
        repo.name = "my_repo"
        repo.path = "/home/user/my_repo"
        repo.settings.ignore_patterns = []

        changes = {(Change.modified, "/some/other/path/file.py")}

        async def fake_awatch(*args, **kwargs):
            yield changes
            stop.set()

        with (
            patch.object(Repo, "get_all", return_value=[repo]),
            patch("indexter.watcher.awatch", side_effect=fake_awatch),
        ):
            await asyncio.wait_for(watch_repos(client, stop, watch_settings), timeout=5)

        repo.index.assert_not_called()
