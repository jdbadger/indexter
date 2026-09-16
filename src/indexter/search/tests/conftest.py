import time

import pytest

from indexter.config import Settings
from indexter.db.connection import open_db
from indexter.index.embed import FakeEmbedder
from indexter.index.sync import sync_repo


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path):
    """The `search`/`neighbors` entry points resolve their own database path
    from the XDG data directory, so without this they write into the user's
    real one."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))


@pytest.fixture
def settings():
    return Settings(embedding_dim=4)


@pytest.fixture
def embedder(settings):
    return FakeEmbedder(dim=settings.embedding_dim)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def repo(tmp_path):
    """A small, realistic repo: a module with a function and a class, its
    test, two sibling directories for path-filter boundary checks, a
    markdown file for language filtering, and an import that resolves to an
    external module (never a candidate)."""
    repo_dir = tmp_path / "repo"

    write(
        repo_dir / "src" / "walker.py",
        "import os\n\n\ndef helper():\n    return 1\n\n\nclass Store:\n    def add(self, item):\n        return item\n",
    )
    write(
        repo_dir / "tests" / "test_walker.py",
        "def test_helper():\n    return 1\n",
    )
    write(repo_dir / "src" / "auth" / "login.py", "def login():\n    return True\n")
    write(repo_dir / "src" / "authz.py", "def authorize():\n    return True\n")
    write(repo_dir / "README.md", "# Walker\n\nWalks stuff.\n")

    return repo_dir


@pytest.fixture
def conn(repo, tmp_path, settings, embedder):
    db_path = tmp_path / "data" / "sample.db"
    with open_db(db_path, repo=repo, settings=settings) as connection:
        sync_repo(connection, repo, settings, embedder)
        yield connection


@pytest.fixture
def graph_conn(tmp_path, settings):
    """An empty, schema-only database for hand-built node/edge graph tests
    (design.md decision 10: `hit_context`/`expand` are pure SQL reads, tested
    over hand-built sets rather than a synced repo)."""
    repo_dir = tmp_path / "graph_repo"
    repo_dir.mkdir()
    db_path = tmp_path / "data" / "graph.db"
    with open_db(db_path, repo=repo_dir, settings=settings) as connection:
        yield connection


def insert_node(
    conn,
    *,
    id,
    kind="function",
    name=None,
    qualified_name=None,
    file_path="src/a.py",
    start_line=None,
    end_line=None,
    parent_id=None,
    degree=0,
):
    conn.execute(
        "INSERT INTO nodes (id, kind, name, qualified_name, file_path, start_line, end_line, parent_id, degree, "
        "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (id, kind, name or id, qualified_name or id, file_path, start_line, end_line, parent_id, degree, time.time()),
    )


def insert_edge(conn, *, source, target, kind="calls", confidence="exact", line=None):
    conn.execute(
        "INSERT INTO edges (source, target, kind, line, confidence) VALUES (?, ?, ?, ?, ?)",
        (source, target, kind, line, confidence),
    )
