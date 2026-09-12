import struct
import time
from pathlib import Path

import pytest

from indexter.config import Settings
from indexter.index.compose import compose_file
from indexter.index.embed import FakeEmbedder
from indexter.parse.base import parse_file
from indexter.walk import WalkedFile, compute_hash


@pytest.fixture
def settings():
    return Settings(embedding_dim=4, embed_max_tokens=256)


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "sample-repo"
    d.mkdir()
    return d


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "data" / "sample.db"


@pytest.fixture
def tokenizer():
    return FakeEmbedder().tokenizer()


def make_walked(path: str, content: str, mtime: float | None = None) -> WalkedFile:
    data = content.encode("utf-8")
    return WalkedFile(
        path=path,
        size=len(data),
        mtime=mtime if mtime is not None else time.time(),
        extension=Path(path).suffix.lower(),
    )


def sync_source(conn, path: str, content: str, settings, tokenizer):
    """Parse, compose, and write one file's content, as `sync_repo` will do
    in group 6 -- used here to exercise `write_file` against realistic
    parser/composer output instead of hand-built fixtures.
    """
    from indexter.index.sync import write_file

    parse_result = parse_file(path, content, settings=settings)
    composed = compose_file(path, content, parse_result, tokenizer, settings.embed_max_tokens)
    walked = make_walked(path, content)
    content_hash = compute_hash(path, content)
    write_file(conn, walked, content_hash, parse_result, composed)
    return parse_result


def insert_vector(conn, rowid: int, dim: int = 4) -> None:
    vec = struct.pack(f"{dim}f", *([0.1] * dim))
    conn.execute(
        "INSERT INTO vectors (node_rowid, kind, language, emb) VALUES (?, ?, ?, ?)",
        (rowid, "function", "python", vec),
    )


def node_row(conn, node_id: str):
    return conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
