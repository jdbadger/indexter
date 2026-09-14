import pytest

from indexter.config import Settings
from indexter.db.connection import open_db
from indexter.index.embed import FakeEmbedder
from indexter.index.sync import sync_repo
from indexter.paths import data_dir
from indexter.paths import db_path as resolve_db_path


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path):
    """Every mcp test writes databases under an isolated XDG data directory,
    never the user's real one."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    return data_dir()


@pytest.fixture
def settings():
    """Default settings -- `run_search`/`run_neighbors` re-derive settings
    from the repository via `load_settings`, which returns these same
    defaults absent an `indexter.toml`, so the embedder built for indexing
    must match its dimension (384)."""
    return Settings()


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=Settings().embedding_dim)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def make_repo(root):
    """A small repo: a function, a caller of it, and a subdirectory -- a
    `search` hit and a `neighbors` caller edge, and a place to resolve
    upward from.
    """
    write(
        root / "src" / "walker.py",
        "def helper():\n    return 1\n\n\ndef caller():\n    return helper()\n",
    )
    write(root / "src" / "sub" / "extra.py", "def extra():\n    return 2\n")
    return root


def index_repo(root, settings, embedder):
    """Sync `root` and close the connection, leaving a database on disk for
    `run_search`/`run_neighbors` to open themselves."""
    with open_db(resolve_db_path(root), repo=root, settings=settings) as conn:
        sync_repo(conn, root, settings, embedder)
    return root


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


@pytest.fixture
def indexed_repo(repo, settings, embedder):
    return index_repo(repo, settings, embedder)
