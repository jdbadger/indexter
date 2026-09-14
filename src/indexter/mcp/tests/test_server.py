import asyncio
import functools
import sqlite3
import time

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import indexter.mcp.server as server_module
from indexter.mcp.server import build_server, run_server
from indexter.mcp.tools import ServerState
from indexter.parse.base import registered_languages
from indexter.paths import db_path as resolve_db_path
from indexter.search.hybrid import FILTERABLE_KINDS


def async_test(fn):
    """Run an async test body to completion with `asyncio.run`, so these
    tests need no async test-runner plugin."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        asyncio.run(fn(*args, **kwargs))

    return wrapper


def make_client(indexed_repo, embedder):
    state = ServerState(default_repo=None, working_dir=indexed_repo, embedder_factory=lambda settings: embedder)
    return Client(build_server(state))


class TestToolRegistration:
    @async_test
    async def test_exactly_two_tools(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            tools = await client.list_tools()
        assert {tool.name for tool in tools} == {"search", "neighbors"}

    @async_test
    async def test_both_tools_are_read_only_and_closed_world(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            tools = await client.list_tools()
        for tool in tools:
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.idempotent_hint is True
            assert tool.annotations.open_world_hint is False

    @async_test
    async def test_search_is_always_loaded(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            tools = await client.list_tools()
        search = next(tool for tool in tools if tool.name == "search")
        assert search.meta["anthropic/alwaysLoad"] is True

    @async_test
    async def test_neighbors_has_search_hint_and_no_always_load(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            tools = await client.list_tools()
        neighbors = next(tool for tool in tools if tool.name == "neighbors")
        assert isinstance(neighbors.meta["anthropic/searchHint"], str)
        assert "anthropic/alwaysLoad" not in neighbors.meta

    @async_test
    async def test_search_schema_lists_valid_kinds_and_languages(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            tools = await client.list_tools()
        search = next(tool for tool in tools if tool.name == "search")
        kind_description = search.input_schema["properties"]["kind"]["description"]
        language_description = search.input_schema["properties"]["language"]["description"]
        for kind in FILTERABLE_KINDS:
            assert kind in kind_description
        for language in registered_languages():
            assert language in language_description

    @async_test
    async def test_neighbors_schema_bounds_depth_and_limit(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            tools = await client.list_tools()
        neighbors = next(tool for tool in tools if tool.name == "neighbors")
        depth = neighbors.input_schema["properties"]["depth"]
        limit = neighbors.input_schema["properties"]["limit"]
        assert depth["minimum"] == 1
        assert depth["maximum"] == 3
        assert limit["minimum"] == 1
        assert limit["maximum"] == 100

    def test_instructions_name_both_tools(self, indexed_repo, embedder):
        state = ServerState(default_repo=None, working_dir=indexed_repo, embedder_factory=lambda settings: embedder)
        server = build_server(state)
        assert "search" in server.instructions
        assert "neighbors" in server.instructions
        assert len(server.instructions.split(". ")) <= 2


class TestToolCalls:
    @async_test
    async def test_search_returns_rendered_text(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            result = await client.call_tool("search", {"query": "helper"})
        assert "helper" in result.data

    @async_test
    async def test_neighbors_returns_rendered_text(self, indexed_repo, embedder):
        with sqlite3.connect(str(resolve_db_path(indexed_repo))) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id FROM nodes WHERE file_path = ? AND name = ?", ("src/walker.py", "helper")
            ).fetchone()

        async with make_client(indexed_repo, embedder) as client:
            result = await client.call_tool("neighbors", {"node_id": row["id"], "direction": "in"})
        assert "caller" in result.data

    @async_test
    async def test_invalid_input_is_a_tool_error(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            result = await client.call_tool("search", {"query": "helper", "kind": "func"}, raise_on_error=False)
        assert result.is_error

    @async_test
    async def test_invalid_input_raises_tool_error(self, indexed_repo, embedder):
        async with make_client(indexed_repo, embedder) as client:
            with pytest.raises(ToolError):
                await client.call_tool("search", {"query": "helper", "kind": "func"})


class TestRunServer:
    def test_builds_state_for_default_repo_and_runs_stdio(self, indexed_repo, monkeypatch):
        captured = {}

        class FakeServer:
            def run(self, transport, show_banner):
                captured["transport"] = transport
                captured["show_banner"] = show_banner

        def fake_build_server(state):
            captured["state"] = state
            return FakeServer()

        monkeypatch.setattr(server_module, "build_server", fake_build_server)

        run_server(indexed_repo)

        assert captured["state"].default_repo == indexed_repo
        assert captured["transport"] == "stdio"
        assert captured["show_banner"] is False


class TestLifespanWarmUp:
    @async_test
    async def test_warm_up_starts_in_the_background_on_connect(self, indexed_repo, embedder):
        state = ServerState(default_repo=indexed_repo, working_dir=indexed_repo, embedder_factory=lambda s: embedder)
        server = build_server(state)
        async with Client(server) as client:
            await client.list_tools()
            deadline = time.monotonic() + 2
            while embedder.model_loads < 1 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
        assert embedder.model_loads >= 1
