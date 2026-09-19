import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

import indexter.mcp.tools as tools
from indexter.config import Settings
from indexter.index.embed import FakeEmbedder, ModelAcquisitionError
from indexter.mcp.tests.conftest import index_repo, make_repo
from indexter.mcp.tools import RepositoryNotFound, ServerState, resolve_repository, run_neighbors, run_search, warm_up
from indexter.paths import canonical_repo_path
from indexter.paths import db_path as resolve_db_path


class TestResolveRepository:
    def test_working_directory_is_the_repository(self, indexed_repo):
        resolved = resolve_repository(None, default_repo=None, working_dir=indexed_repo)
        assert resolved == canonical_repo_path(indexed_repo)

    def test_subdirectory_resolves_upward(self, indexed_repo):
        resolved = resolve_repository(None, default_repo=None, working_dir=indexed_repo / "src" / "sub")
        assert resolved == canonical_repo_path(indexed_repo)

    def test_explicit_repo_wins(self, tmp_path, settings, embedder):
        a = index_repo(make_repo(tmp_path / "a"), settings, embedder)
        b = index_repo(make_repo(tmp_path / "b"), settings, embedder)

        resolved = resolve_repository(str(b), default_repo=a, working_dir=tmp_path)
        assert resolved == canonical_repo_path(b)

    def test_server_default_used(self, tmp_path, indexed_repo):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        resolved = resolve_repository(None, default_repo=indexed_repo, working_dir=elsewhere)
        assert resolved == canonical_repo_path(indexed_repo)

    def test_relative_repo_is_joined_to_working_directory(self, tmp_path, settings, embedder):
        index_repo(make_repo(tmp_path / "child"), settings, embedder)

        resolved = resolve_repository("child", default_repo=None, working_dir=tmp_path)
        assert resolved == canonical_repo_path(tmp_path / "child")

    def test_nothing_indexed_creates_no_database(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(RepositoryNotFound):
            resolve_repository(None, default_repo=None, working_dir=empty)

        assert not resolve_db_path(empty).exists()


class TestServerStateEmbedderCache:
    def test_one_embedder_per_configuration(self):
        created = []

        def factory(settings):
            created.append(settings)
            return object()

        state = ServerState(default_repo=None, working_dir=Path.cwd(), embedder_factory=factory)
        settings = Settings(embedding_dim=4)

        first = state.get_embedder(settings)
        second = state.get_embedder(settings)

        assert first is second
        assert len(created) == 1

    def test_different_model_gets_a_different_embedder(self):
        created = []

        def factory(settings):
            created.append(settings)
            return object()

        state = ServerState(default_repo=None, working_dir=Path.cwd(), embedder_factory=factory)
        a = Settings(embedding_dim=4, embedding_model="model-a")
        b = Settings(embedding_dim=4, embedding_model="model-b")

        first = state.get_embedder(a)
        second = state.get_embedder(b)

        assert first is not second
        assert len(created) == 2


class TestRunSearchAndRunNeighbors:
    def _state(self, indexed_repo, embedder):
        return ServerState(default_repo=None, working_dir=indexed_repo, embedder_factory=lambda settings: embedder)

    def test_run_search_returns_rendered_text(self, indexed_repo, embedder):
        state = self._state(indexed_repo, embedder)
        text = run_search(state, "helper")
        assert "helper" in text

    def test_run_neighbors_returns_rendered_text(self, indexed_repo, embedder):
        with closing(sqlite3.connect(str(resolve_db_path(indexed_repo)))) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id FROM nodes WHERE file_path = ? AND name = ?", ("src/walker.py", "helper")
            ).fetchone()

        state = self._state(indexed_repo, embedder)
        text = run_neighbors(state, row["id"], direction="in")
        assert "caller" in text

    def test_settings_loaded_on_every_call(self, indexed_repo, embedder, monkeypatch):
        calls = []
        original = tools.load_settings

        def spy(repo, **overrides):
            calls.append(repo)
            return original(repo, **overrides)

        monkeypatch.setattr(tools, "load_settings", spy)
        state = self._state(indexed_repo, embedder)

        run_search(state, "helper")
        run_search(state, "helper")

        assert len(calls) == 2

    def test_search_error_becomes_tool_error(self, indexed_repo, embedder):
        state = self._state(indexed_repo, embedder)
        with pytest.raises(ToolError):
            run_search(state, "   ")

    def test_neighbors_error_becomes_tool_error(self, indexed_repo, embedder):
        state = self._state(indexed_repo, embedder)
        with pytest.raises(ToolError):
            run_neighbors(state, "   ")

    def test_resolution_error_becomes_tool_error(self, tmp_path, embedder):
        state = ServerState(default_repo=None, working_dir=tmp_path / "empty", embedder_factory=lambda s: embedder)
        (tmp_path / "empty").mkdir()
        with pytest.raises(ToolError):
            run_search(state, "helper")

    def test_config_error_becomes_tool_error(self, indexed_repo, embedder):
        (indexed_repo / "indexter.toml").write_text("bogus_setting = 1\n")
        state = self._state(indexed_repo, embedder)
        with pytest.raises(ToolError):
            run_search(state, "helper")

    def test_database_error_becomes_tool_error(self, indexed_repo, embedder):
        with closing(sqlite3.connect(str(resolve_db_path(indexed_repo)))) as conn:
            conn.execute("UPDATE project_metadata SET value = ? WHERE key = ?", ("999", "schema_version"))
            conn.commit()

        state = self._state(indexed_repo, embedder)
        with pytest.raises(ToolError):
            run_search(state, "helper")

    def test_embedding_error_becomes_tool_error(self, indexed_repo):
        class FailingEmbedder:
            model_name = "failing"

            def tokenizer(self):
                raise AssertionError("not used")

            def embed(self, texts):
                raise ModelAcquisitionError("failing-model", "boom")

        state = ServerState(
            default_repo=None, working_dir=indexed_repo, embedder_factory=lambda settings: FailingEmbedder()
        )
        with pytest.raises(ToolError):
            run_search(state, "helper")

    def test_two_concurrent_calls_serialize(self, indexed_repo, embedder, monkeypatch):
        active = 0
        max_active = 0
        guard = threading.Lock()
        original_search = tools.search

        def tracking_search(*args, **kwargs):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.05)
                return original_search(*args, **kwargs)
            finally:
                with guard:
                    active -= 1

        monkeypatch.setattr(tools, "search", tracking_search)
        state = self._state(indexed_repo, embedder)

        threads = [threading.Thread(target=run_search, args=(state, "helper")) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert max_active == 1


class TestWarmUp:
    def test_success_loads_settings_and_embeds(self, indexed_repo):
        embedder = FakeEmbedder(dim=4)
        state = ServerState(
            default_repo=indexed_repo, working_dir=Path("/does-not-matter"), embedder_factory=lambda s: embedder
        )

        warm_up(state)

        assert embedder.model_loads == 1

    def test_failure_is_logged_and_does_not_raise(self, indexed_repo, capsys):
        class FailingEmbedder:
            model_name = "failing"

            def tokenizer(self):
                raise AssertionError("not used")

            def embed(self, texts):
                raise ModelAcquisitionError("failing-model", "boom")

        state = ServerState(
            default_repo=indexed_repo,
            working_dir=Path("/does-not-matter"),
            embedder_factory=lambda s: FailingEmbedder(),
        )

        warm_up(state)

        assert "warm-up failed" in capsys.readouterr().err

    def test_no_repository_is_a_noop(self, tmp_path):
        created = []
        empty = tmp_path / "empty"
        empty.mkdir()
        state = ServerState(
            default_repo=None,
            working_dir=empty,
            embedder_factory=lambda s: created.append(s),
        )

        warm_up(state)

        assert created == []
