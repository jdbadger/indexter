import struct
import time

import pytest

from indexter.config import Settings


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "sample-repo"
    d.mkdir()
    return d


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "data" / "sample.db"


@pytest.fixture
def settings():
    return Settings(embedding_dim=4)


def f32(values: list[float]) -> bytes:
    """Pack floats as the raw vector bytes sqlite-vec expects."""
    return struct.pack(f"{len(values)}f", *values)


def insert_file(conn, path="a.py", **overrides):
    defaults = dict(
        path=path,
        content_hash="hash1",
        language="python",
        size=100,
        mtime=time.time(),
        indexed_at=time.time(),
        node_count=1,
        errors=None,
    )
    defaults.update(overrides)
    cols = ", ".join(defaults)
    placeholders = ", ".join("?" for _ in defaults)
    conn.execute(f"INSERT INTO files ({cols}) VALUES ({placeholders})", tuple(defaults.values()))


def insert_node(conn, id, **overrides):  # noqa: A002 - matches the column name
    defaults = dict(
        id=id,
        kind="function",
        name="foo",
        name_words="foo",
        qualified_name="module.foo",
        file_path="a.py",
        language="python",
        start_line=1,
        end_line=5,
        start_byte=0,
        end_byte=50,
        signature="def foo():",
        docstring=None,
        parent_id=None,
        embed_text="foo function",
        embed_hash="hash2",
        degree=0,
        updated_at=time.time(),
    )
    defaults.update(overrides)
    cols = ", ".join(defaults)
    placeholders = ", ".join("?" for _ in defaults)
    cur = conn.execute(f"INSERT INTO nodes ({cols}) VALUES ({placeholders})", tuple(defaults.values()))
    return cur.lastrowid


def insert_ref(conn, from_node_id, **overrides):
    defaults = dict(
        from_node_id=from_node_id,
        raw_name="bar",
        head="bar",
        ref_kind="call",
        line=3,
        col=4,
        status="unresolved",
        resolved_target_id=None,
        confidence=None,
        candidates=None,
    )
    defaults.update(overrides)
    cols = ", ".join(defaults)
    placeholders = ", ".join("?" for _ in defaults)
    cur = conn.execute(f"INSERT INTO refs ({cols}) VALUES ({placeholders})", tuple(defaults.values()))
    return cur.lastrowid


def insert_edge(conn, source, target, **overrides):
    defaults = dict(source=source, target=target, kind="calls", line=3, confidence="exact")
    defaults.update(overrides)
    cols = ", ".join(defaults)
    placeholders = ", ".join("?" for _ in defaults)
    cur = conn.execute(f"INSERT INTO edges ({cols}) VALUES ({placeholders})", tuple(defaults.values()))
    return cur.lastrowid
