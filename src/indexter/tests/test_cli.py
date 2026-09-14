import asyncio
import importlib.metadata
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from typer.testing import CliRunner

from indexter.cli import _format_size, _skill_content, app
from indexter.config import Settings
from indexter.db.connection import IndexterDBError, open_db
from indexter.index.embed import FakeEmbedder
from indexter.paths import data_dir, db_path

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    return data_dir()


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "some-repo"
    d.mkdir()
    return d


@pytest.fixture
def fake_embedder(monkeypatch):
    """`init`/`reindex` tests inject a `FakeEmbedder` so no real model or
    tokenizer is ever loaded.
    """
    monkeypatch.setattr("indexter.cli.make_embedder", lambda settings: FakeEmbedder(dim=settings.embedding_dim))


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


def write(repo: Path, relpath: str, content: str) -> None:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


SRC_A = "def helper():\n    return 1\n"
SRC_A_EDITED = "def helper():\n    return 2\n"


class TestHelp:
    def test_help_lists_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "remove" in result.output

    def test_help_lists_init_and_reindex(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "reindex" in result.output

    def test_no_args_shows_help(self):
        # Click's no_args_is_help prints usage and exits 2 (a usage error) --
        # only `--help` itself is specified to exit zero.
        result = runner.invoke(app, [])
        assert "list" in result.output
        assert result.exit_code != 0

    def test_unknown_command_fails(self):
        result = runner.invoke(app, ["frobnicate"])
        assert result.exit_code != 0


class TestVersion:
    def test_version_matches_installed_package(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.output.strip() == f"indexter {importlib.metadata.version('indexter')}"

    def test_version_takes_precedence_over_a_command(self):
        result = runner.invoke(app, ["--version", "list"])
        assert result.exit_code == 0
        assert result.output.strip() == f"indexter {importlib.metadata.version('indexter')}"


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


@pytest.mark.usefixtures("fake_embedder")
class TestInit:
    def test_creates_and_summarizes(self, repo):
        write(repo, "a.py", SRC_A)
        result = runner.invoke(app, ["init", str(repo)])
        assert result.exit_code == 0
        assert "Initialized" in result.output
        assert "added=1" in result.output
        assert "texts_embedded=" in result.output
        assert db_path(repo).is_file()

    def test_on_existing_syncs_and_says_so(self, repo):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])
        result = runner.invoke(app, ["init", str(repo)])
        assert result.exit_code == 0
        assert "already initialized" in result.output
        assert "unchanged=1" in result.output

    def test_first_init_prints_resolution_line(self, repo):
        write(
            repo,
            "a.py",
            "import os\n\n\ndef helper():\n    return 1\n\n\ndef main():\n    return helper()\n",
        )
        result = runner.invoke(app, ["init", str(repo)])
        assert result.exit_code == 0
        assert "resolution: edges[" in result.output
        assert "calls[" in result.output
        assert "external_modules=" in result.output

    def test_noop_init_omits_resolution_line(self, repo):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])
        result = runner.invoke(app, ["init", str(repo)])
        assert result.exit_code == 0
        assert "resolution:" not in result.output

    def test_on_non_directory_exits_nonzero_and_creates_nothing(self, tmp_path):
        not_a_dir = tmp_path / "not-a-dir.txt"
        not_a_dir.write_text("hello")
        result = runner.invoke(app, ["init", str(not_a_dir)])
        assert result.exit_code != 0
        assert not db_path(not_a_dir).exists()

    def test_parse_errors_listed_with_exit_zero(self, repo, monkeypatch):
        import indexter.index.sync as sync_mod

        write(repo, "a.py", SRC_A)

        original_parse_file = sync_mod.parse_file

        def failing_parse_file(relpath, content, *, settings=None):
            result = original_parse_file(relpath, content, settings=settings)
            if relpath == "a.py":
                result.errors.append("simulated parse error")
            return result

        monkeypatch.setattr(sync_mod, "parse_file", failing_parse_file)

        result = runner.invoke(app, ["init", str(repo)])
        assert result.exit_code == 0
        assert "error: a.py: simulated parse error" in result.output

    def test_config_error_rendered_as_one_line_and_exits_nonzero(self, repo):
        write(repo, "indexter.toml", "not_a_real_setting = 1\n")
        result = runner.invoke(app, ["init", str(repo)])
        assert result.exit_code == 1
        assert "not_a_real_setting" in result.output
        assert "Traceback" not in result.output

    def test_repo_path_mismatch_exits_nonzero(self, repo, tmp_path, monkeypatch):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])

        other = tmp_path / "other-repo"
        other.mkdir()
        write(other, "a.py", SRC_A)
        monkeypatch.setattr("indexter.index.sync.resolve_db_path", lambda _repo: db_path(repo))

        result = runner.invoke(app, ["init", str(other)])
        assert result.exit_code == 1
        assert "does not match" in result.output


@pytest.mark.usefixtures("fake_embedder")
class TestReindex:
    def test_on_non_directory_exits_nonzero(self, tmp_path):
        not_a_dir = tmp_path / "not-a-dir.txt"
        not_a_dir.write_text("hello")
        result = runner.invoke(app, ["reindex", str(not_a_dir)])
        assert result.exit_code != 0
        assert "is not a directory" in result.output

    def test_without_database_suggests_init(self, repo):
        result = runner.invoke(app, ["reindex", str(repo)])
        assert result.exit_code != 0
        assert "indexter init" in result.output

    def test_noop_summary(self, repo):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])
        result = runner.invoke(app, ["reindex", str(repo)])
        assert result.exit_code == 0
        assert "unchanged=1" in result.output
        assert "texts_embedded=0" in result.output

    def test_edit_reparses_and_reembeds_only_changed_file(self, repo):
        write(repo, "a.py", SRC_A)
        write(repo, "b.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])

        write(repo, "a.py", SRC_A_EDITED)
        result = runner.invoke(app, ["reindex", str(repo)])
        assert result.exit_code == 0
        assert "changed=1" in result.output
        assert "unchanged=1" in result.output

    def test_full_rebuilds(self, repo):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])

        result = runner.invoke(app, ["reindex", str(repo), "--full"])
        assert result.exit_code == 0
        assert "Rebuilt" in result.output
        assert "added=1" in result.output

    def test_schema_mismatch_rebuild_message(self, repo):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])

        path = db_path(repo)
        raw = sqlite3.connect(str(path))
        raw.execute("UPDATE project_metadata SET value = '999' WHERE key = 'schema_version'")
        raw.commit()
        raw.close()

        result = runner.invoke(app, ["reindex", str(repo)])
        assert result.exit_code == 0
        assert "Rebuilt" in result.output

    def test_repo_path_mismatch_exits_nonzero(self, repo):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])

        path = db_path(repo)
        raw = sqlite3.connect(str(path))
        raw.execute("UPDATE project_metadata SET value = '/somewhere/else' WHERE key = 'repo_path'")
        raw.commit()
        raw.close()

        result = runner.invoke(app, ["reindex", str(repo)])
        assert result.exit_code == 1
        assert "does not match" in result.output

    def test_embedder_error_rendered_as_one_line_and_exits_nonzero(self, repo, monkeypatch):
        write(repo, "a.py", SRC_A)
        runner.invoke(app, ["init", str(repo)])
        write(repo, "b.py", SRC_A)  # something new to embed on the next sync

        from indexter.index.embed import EmbeddingError

        def boom(settings):
            raise EmbeddingError("simulated embedder error")

        monkeypatch.setattr("indexter.cli.make_embedder", boom)
        result = runner.invoke(app, ["reindex", str(repo)])
        assert result.exit_code == 1
        assert "simulated embedder error" in result.output
        assert "Traceback" not in result.output


class TestMcpCommand:
    def test_missing_repo_exits_nonzero_and_never_starts_server(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr("indexter.cli.run_server", lambda repo: calls.append(repo))
        missing = tmp_path / "does-not-exist"

        result = runner.invoke(app, ["mcp", "--repo", str(missing)])

        assert result.exit_code != 0
        assert "is not a directory" in result.output
        assert calls == []

    def test_valid_repo_starts_the_server(self, repo, monkeypatch):
        calls = []
        monkeypatch.setattr("indexter.cli.run_server", lambda r: calls.append(r))

        result = runner.invoke(app, ["mcp", "--repo", str(repo)])

        assert result.exit_code == 0
        assert calls == [repo]

    def test_no_repo_option_passes_none(self, monkeypatch):
        calls = []
        monkeypatch.setattr("indexter.cli.run_server", lambda r: calls.append(r))

        result = runner.invoke(app, ["mcp"])

        assert result.exit_code == 0
        assert calls == [None]


class TestSkillCommand:
    def test_prints_the_packaged_skill_byte_for_byte(self):
        result = runner.invoke(app, ["skill"])
        assert result.exit_code == 0
        assert result.output == _skill_content()

    def test_dir_without_install_is_rejected(self, tmp_path):
        result = runner.invoke(app, ["skill", "--dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_force_without_install_is_rejected(self):
        result = runner.invoke(app, ["skill", "--force"])
        assert result.exit_code != 0

    def test_install_writes_under_claude_config_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))

        result = runner.invoke(app, ["skill", "--install"])

        assert result.exit_code == 0
        target = tmp_path / "claude-config" / "skills" / "indexter" / "SKILL.md"
        assert target.read_text() == _skill_content()

    def test_install_falls_back_to_home_claude_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        result = runner.invoke(app, ["skill", "--install"])

        assert result.exit_code == 0
        assert (tmp_path / ".claude" / "skills" / "indexter" / "SKILL.md").is_file()

    def test_install_to_explicit_dir(self, tmp_path):
        target_dir = tmp_path / "custom-skills"

        result = runner.invoke(app, ["skill", "--install", "--dir", str(target_dir)])

        assert result.exit_code == 0
        assert (target_dir / "SKILL.md").read_text() == _skill_content()

    def test_reinstalling_identical_content_is_up_to_date(self, tmp_path):
        target_dir = tmp_path / "skills"
        runner.invoke(app, ["skill", "--install", "--dir", str(target_dir)])

        result = runner.invoke(app, ["skill", "--install", "--dir", str(target_dir)])

        assert result.exit_code == 0
        assert "up to date" in result.output

    def test_edited_file_is_protected_without_force(self, tmp_path):
        target_dir = tmp_path / "skills"
        runner.invoke(app, ["skill", "--install", "--dir", str(target_dir)])
        target = target_dir / "SKILL.md"
        target.write_text("edited by the user\n")

        result = runner.invoke(app, ["skill", "--install", "--dir", str(target_dir)])

        assert result.exit_code != 0
        assert target.read_text() == "edited by the user\n"

    def test_force_overwrites_an_edited_file(self, tmp_path):
        target_dir = tmp_path / "skills"
        runner.invoke(app, ["skill", "--install", "--dir", str(target_dir)])
        target = target_dir / "SKILL.md"
        target.write_text("edited by the user\n")

        result = runner.invoke(app, ["skill", "--install", "--dir", str(target_dir), "--force"])

        assert result.exit_code == 0
        assert target.read_text() == _skill_content()


class TestSkillContent:
    def test_frontmatter_names_and_describes_the_skill(self):
        content = _skill_content()
        lines = content.splitlines()
        assert lines[0] == "---"
        assert lines[1] == "name: indexter"
        assert lines[2].startswith("description:")

    def test_names_both_tools_and_the_claude_code_form(self):
        content = _skill_content()
        assert "`search`" in content
        assert "`neighbors`" in content
        assert "mcp__indexter__search" in content
        assert "mcp__indexter__neighbors" in content

    def test_documents_every_search_parameter(self):
        content = _skill_content()
        for param in ("query", "repo", "kind", "language", "path", "limit"):
            assert f"`{param}`" in content

    def test_documents_every_neighbors_parameter(self):
        content = _skill_content()
        for param in ("node_id", "direction", "edges", "depth", "limit"):
            assert f"`{param}`" in content

    def test_mentions_asking_the_user_before_init(self):
        content = " ".join(_skill_content().lower().split())
        assert "indexter init" in content
        assert "tell the user" in content


class TestMcpSubprocess:
    def test_stdio_transport_serves_exactly_two_tools_with_clean_stdout(self, tmp_path):
        indexter_bin = shutil.which("indexter") or str(Path(sys.executable).parent / "indexter")
        empty = tmp_path / "empty"
        empty.mkdir()
        env = {
            **os.environ,
            "XDG_DATA_HOME": str(tmp_path / "data-home"),
            "XDG_CONFIG_HOME": str(tmp_path / "config-home"),
        }
        transport = StdioTransport(indexter_bin, ["mcp", "--repo", str(empty)], env=env)

        async def list_tools():
            # MCP's stdio protocol is newline-delimited JSON-RPC: any stray
            # text on stdout (a stray print, a warning) breaks framing and
            # this raises instead of returning a tool list.
            async with Client(transport) as client:
                return await client.list_tools()

        tools = asyncio.run(list_tools())
        assert {tool.name for tool in tools} == {"search", "neighbors"}
