import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indexter.cli import _format_size, app
from indexter.config import Settings
from indexter.db.connection import IndexterDBError, open_db
from indexter.paths import data_dir, db_path

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-home"))
    return data_dir()


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "some-repo"
    d.mkdir()
    return d


def _index(repo, settings=None):
    settings = settings or Settings(embedding_dim=4)
    path = db_path(repo)
    with open_db(path, repo=repo, settings=settings):
        pass
    return path


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0KB"

    def test_gigabytes(self):
        assert _format_size(3 * 1024**3) == "3.0GB"


class TestHelp:
    def test_help_lists_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "remove" in result.output

    def test_no_args_shows_help(self):
        # Click's no_args_is_help prints usage and exits 2 (a usage error) --
        # only `--help` itself is specified to exit zero.
        result = runner.invoke(app, [])
        assert "list" in result.output
        assert result.exit_code != 0

    def test_unknown_command_fails(self):
        result = runner.invoke(app, ["frobnicate"])
        assert result.exit_code != 0


class TestList:
    def test_no_repositories_indexed(self):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No repositories indexed" in result.output

    def test_absent_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "does-not-exist"))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No repositories indexed" in result.output

    def test_lists_indexed_repository(self, repo):
        _index(repo)
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert str(repo.resolve()) in result.output
        assert "nodes=0" in result.output
        assert "indexed_at=never" in result.output

    def test_lists_last_indexed_time(self, repo):
        from datetime import datetime

        path = _index(repo)
        timestamp = 1700000000
        raw = sqlite3.connect(str(path))
        raw.execute(
            "INSERT INTO files (path, content_hash, language, size, mtime, indexed_at, node_count, errors) "
            "VALUES ('a.py', 'h', 'python', 10, 0, ?, 0, NULL)",
            (timestamp,),
        )
        raw.commit()
        raw.close()

        expected = datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")  # noqa: DTZ006
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert f"indexed_at={expected}" in result.output

    def test_missing_repository_marked(self, repo):
        _index(repo)
        repo.rmdir()
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "[missing]" in result.output

    def test_schema_version_mismatch_still_listed(self, repo):
        path = _index(repo)
        raw = sqlite3.connect(str(path))
        raw.execute("UPDATE project_metadata SET value = '999' WHERE key = 'schema_version'")
        raw.commit()
        raw.close()

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "schema_version=999" in result.output

    def test_corrupt_database_reported(self, tmp_path):
        bad = data_dir()
        bad.mkdir(parents=True)
        (bad / "broken.db").write_bytes(b"not a real database")

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "corrupt" in result.output.lower()


class TestRemove:
    def test_remove_by_repo_path_with_confirmation(self, repo):
        path = _index(repo)
        result = runner.invoke(app, ["remove", str(repo), "--yes"])
        assert result.exit_code == 0
        assert not path.exists()
        assert repo.is_dir()  # the repo itself is untouched

    def test_remove_by_database_filename(self, repo):
        path = _index(repo)
        result = runner.invoke(app, ["remove", path.name, "--yes"])
        assert result.exit_code == 0
        assert not path.exists()

    def test_declined_confirmation_deletes_nothing(self, repo):
        path = _index(repo)
        result = runner.invoke(app, ["remove", str(repo)], input="n\n")
        assert result.exit_code == 0
        assert path.exists()

    def test_confirmed_prompt_deletes(self, repo):
        path = _index(repo)
        result = runner.invoke(app, ["remove", str(repo)], input="y\n")
        assert result.exit_code == 0
        assert not path.exists()

    def test_no_matching_database_fails(self, repo):
        result = runner.invoke(app, ["remove", str(repo), "--yes"])
        assert result.exit_code != 0
        assert "No indexed database found" in result.output

    def test_removes_sidecars(self, repo):
        path = _index(repo)
        wal = Path(f"{path}-wal")
        shm = Path(f"{path}-shm")
        wal.write_bytes(b"wal")
        shm.write_bytes(b"shm")

        result = runner.invoke(app, ["remove", str(repo), "--yes"])
        assert result.exit_code == 0
        assert not path.exists()
        assert not wal.exists()
        assert not shm.exists()

    def test_deleted_repository_still_removable(self, repo):
        path = _index(repo)
        repo.rmdir()
        result = runner.invoke(app, ["remove", str(repo), "--yes"])
        assert result.exit_code == 0
        assert not path.exists()

    def test_db_error_rendered_as_one_line_and_exits_nonzero(self, repo, monkeypatch):
        import indexter.cli as cli_mod

        def boom(target):
            raise IndexterDBError("simulated database error")

        monkeypatch.setattr(cli_mod, "_resolve_target", boom)
        result = runner.invoke(app, ["remove", str(repo), "--yes"])
        assert result.exit_code == 1
        assert "simulated database error" in result.output
        assert "Traceback" not in result.output
