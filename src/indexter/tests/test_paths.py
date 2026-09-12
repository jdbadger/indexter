import os
from pathlib import Path

import pytest

from indexter.paths import canonical_repo_path, config_dir, data_dir, db_path, slugify


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "my-repo"
    d.mkdir()
    return d


class TestDataDir:
    def test_xdg_env_var_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert data_dir() == tmp_path / "indexter"

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        assert data_dir() == Path.home().joinpath(".local", "share", "indexter")

    def test_default_when_empty(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "")
        assert data_dir() == Path.home().joinpath(".local", "share", "indexter")

    def test_lookup_has_no_side_effects(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        path = data_dir()
        assert not path.exists()


class TestConfigDir:
    def test_xdg_env_var_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert config_dir() == tmp_path / "indexter"

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert config_dir() == Path.home().joinpath(".config", "indexter")


class TestCanonicalRepoPath:
    def test_relative_path(self, repo, monkeypatch):
        monkeypatch.chdir(repo.parent)
        assert canonical_repo_path(repo.name) == repo.resolve()

    def test_dot_segments(self, repo):
        noisy = repo / ".." / repo.name / "." / ".."
        assert canonical_repo_path(noisy) == repo.resolve().parent

    def test_trailing_slash(self, repo):
        assert canonical_repo_path(str(repo) + os.sep) == repo.resolve()

    def test_symlink(self, repo, tmp_path):
        link = tmp_path / "link-to-repo"
        link.symlink_to(repo)
        assert canonical_repo_path(link) == repo.resolve()


class TestSlugify:
    def test_only_lowercase_alnum_and_hyphens(self):
        assert slugify("My Repo.v2!!") == "my-repo-v2"

    def test_already_clean(self):
        assert slugify("my-repo") == "my-repo"

    def test_collapses_runs_and_trims_edges(self):
        assert slugify("--Foo   Bar--") == "foo-bar"

    def test_empty_after_slug_falls_back(self):
        assert slugify("!!!") == "repo"


class TestDbPath:
    def test_deterministic_across_calls(self, repo):
        assert db_path(repo) == db_path(repo)

    def test_distinct_for_same_name_different_parent(self, tmp_path):
        a = tmp_path / "a" / "repo"
        b = tmp_path / "b" / "repo"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        assert db_path(a) != db_path(b)
        # same slug prefix, different hash suffix
        assert db_path(a).name.rsplit("-", 1)[0] == db_path(b).name.rsplit("-", 1)[0]

    def test_canonicalization_equivalence(self, repo, monkeypatch):
        monkeypatch.chdir(repo.parent)
        relative = repo.name
        dotted = repo / ".." / repo.name
        trailing = str(repo) + os.sep
        expected = db_path(repo.resolve())
        assert db_path(relative) == expected
        assert db_path(dotted) == expected
        assert db_path(trailing) == expected

    def test_filename_format(self, repo, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        path = db_path(repo)
        assert path.parent == tmp_path / "indexter"
        name = path.stem
        slug, digest = name.rsplit("-", 1)
        assert slug == "my-repo"
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)
