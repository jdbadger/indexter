import os

import pytest

from indexter.config import Settings
from indexter.walk import IgnorePatternMatcher, Walker, compute_hash, read_file


@pytest.fixture
def repo(tmp_path):
    return tmp_path


@pytest.fixture
def settings():
    return Settings()


def write(path, content="", binary=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary is not None:
        path.write_bytes(binary)
    else:
        path.write_text(content)


class TestTraversal:
    def test_relative_forward_slash_paths(self, repo, settings):
        write(repo / "src" / "sub" / "a.py", "x = 1\n")
        [result] = list(Walker(repo, settings).walk())
        assert result.path == "src/sub/a.py"
        assert "\\" not in result.path

    def test_lazy_generator_no_event_loop(self, repo, settings):
        write(repo / "a.py", "x = 1\n")
        gen = Walker(repo, settings).walk()
        # No await, no asyncio.run -- plain next() works.
        first = next(gen)
        assert first.path == "a.py"

    def test_stat_fields_populated(self, repo, settings):
        write(repo / "a.py", "x = 1\n")
        [result] = list(Walker(repo, settings).walk())
        assert result.size == (repo / "a.py").stat().st_size
        assert result.mtime > 0
        assert result.extension == ".py"

    def test_empty_repository(self, repo, settings):
        assert list(Walker(repo, settings).walk()) == []

    def test_walking_does_not_read_contents(self, repo, settings, monkeypatch):
        write(repo / "a.py", "x = 1\n")

        def boom(*args, **kwargs):
            raise AssertionError("walk() must not read file contents")

        import pathlib

        monkeypatch.setattr(pathlib.Path, "read_text", boom)
        monkeypatch.setattr(pathlib.Path, "read_bytes", boom)
        results = list(Walker(repo, settings).walk())
        assert len(results) == 1


class TestFiltering:
    def test_configured_ignore_patterns(self, repo, settings):
        write(repo / "keep.py", "x = 1\n")
        write(repo / "skip.py", "x = 1\n")
        settings = settings.model_copy(update={"ignore_patterns": ("skip.py",)})
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_gitignore_is_honoured(self, repo, settings):
        write(repo / ".gitignore", "ignored.py\n")
        write(repo / "keep.py", "x = 1\n")
        write(repo / "ignored.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py", ".gitignore"}
        assert "ignored.py" not in paths

    def test_ignored_directory_is_pruned_not_descended(self, repo, settings):
        write(repo / ".gitignore", "build/\n")
        write(repo / "build" / "generated.py", "x = 1\n")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py", ".gitignore"}
        assert "build/generated.py" not in paths

    def test_missing_gitignore_is_fine(self, repo, settings):
        write(repo / "a.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"a.py"}

    def test_git_directory_is_always_ignored(self, repo, settings):
        # A repo's own .gitignore has no reason to list ".git/" -- git
        # itself never walks into it -- so this exclusion can't come from
        # user-configured patterns or the .gitignore file; it must be built
        # in, the same as the old repo's hardcoded VERSION_CONTROL list.
        write(repo / ".git" / "config", "[core]\n")
        write(repo / ".git" / "hooks" / "pre-commit.sample", "#!/bin/sh\n")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_binary_extension_skipped(self, repo, settings):
        write(repo / "image.png", binary=b"\x89PNG\r\n")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_minified_skipped(self, repo, settings):
        write(repo / "app.min.js", "x=1;")
        write(repo / "keep.js", "x = 1;")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.js"}

    def test_oversized_skipped(self, repo, settings):
        settings = settings.model_copy(update={"max_file_size_bytes": 10})
        write(repo / "big.py", "x" * 100)
        write(repo / "small.py", "x")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"small.py"}

    def test_empty_file_skipped(self, repo, settings):
        write(repo / "empty.py", "")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}


class TestSafety:
    def test_symlink_escaping_repo_not_followed(self, repo, settings, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        write(outside / "secret.py", "x = 1\n")
        (repo / "escape").symlink_to(outside)
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_symlink_within_repo_is_fine(self, repo, settings):
        write(repo / "real" / "a.py", "x = 1\n")
        (repo / "link").symlink_to(repo / "real")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"real/a.py", "link/a.py"}

    def test_symlink_relative_escape_not_followed(self, repo, settings, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        write(outside / "secret.py", "x = 1\n")
        write(repo / "keep.py", "x = 1\n")
        (repo / "sub").mkdir()
        target = os.path.relpath(outside / "secret.py", start=repo / "sub")
        (repo / "sub" / "escape.py").symlink_to(target)
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_symlink_absolute_escape_not_followed(self, repo, settings, tmp_path_factory, monkeypatch):
        outside = tmp_path_factory.mktemp("outside")
        write(outside / "secret.py", "x = 1\n")
        (repo / "escape.py").symlink_to(outside / "secret.py")
        write(repo / "keep.py", "x = 1\n")

        original_consider = Walker._consider_file

        def guarded(self, entry, relative_str):
            assert relative_str != "escape.py", "escaping symlink target must never be stat'd"
            return original_consider(self, entry, relative_str)

        monkeypatch.setattr(Walker, "_consider_file", guarded)
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_symlink_chain_escaping_not_followed(self, repo, settings, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        write(outside / "secret.py", "x = 1\n")
        (repo / "inner").symlink_to(outside / "secret.py")
        (repo / "outer").symlink_to(repo / "inner")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_symlink_to_gitignored_file_not_followed(self, repo, settings):
        write(repo / ".gitignore", ".env\n")
        write(repo / ".env", "SECRET=1\n")
        (repo / "notes.md").symlink_to(repo / ".env")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py", ".gitignore"}

    def test_symlink_to_file_under_gitignored_directory_not_followed(self, repo, settings):
        write(repo / ".gitignore", "secrets/\n")
        write(repo / "secrets" / "key.txt", "shh\n")
        (repo / "link.txt").symlink_to(repo / "secrets" / "key.txt")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py", ".gitignore"}

    def test_symlink_to_configured_ignore_pattern_not_followed(self, repo, settings):
        write(repo / "skip.py", "x = 1\n")
        (repo / "link.py").symlink_to(repo / "skip.py")
        write(repo / "keep.py", "x = 1\n")
        settings = settings.model_copy(update={"ignore_patterns": ("skip.py",)})
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_symlinked_git_directory_not_descended(self, repo, settings):
        write(repo / ".git" / "config", "[core]\n")
        (repo / "link").symlink_to(repo / ".git")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_symlinked_git_config_file_not_followed(self, repo, settings):
        write(repo / ".git" / "config", "[core]\n")
        (repo / "cfg").symlink_to(repo / ".git" / "config")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_symlink_to_file_within_repo_is_fine(self, repo, settings):
        write(repo / "real.py", "x = 1\n")
        (repo / "link.py").symlink_to(repo / "real.py")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"real.py", "link.py"}

    def test_symlinked_repo_root_still_admits_in_repo_links(self, repo, settings, tmp_path_factory):
        actual = tmp_path_factory.mktemp("actual")
        write(actual / "real.py", "x = 1\n")
        (actual / "link.py").symlink_to(actual / "real.py")
        root = tmp_path_factory.mktemp("via_root_parent") / "root"
        root.symlink_to(actual)
        paths = {r.path for r in Walker(root, settings).walk()}
        assert paths == {"real.py", "link.py"}

    def test_broken_symlink_skipped(self, repo, settings):
        (repo / "broken").symlink_to(repo / "does-not-exist")
        write(repo / "keep.py", "x = 1\n")
        paths = {r.path for r in Walker(repo, settings).walk()}
        assert paths == {"keep.py"}

    def test_unreadable_directory_skipped(self, repo, settings):
        blocked = repo / "blocked"
        blocked.mkdir()
        write(blocked / "a.py", "x = 1\n")
        write(repo / "keep.py", "x = 1\n")
        blocked.chmod(0o000)
        try:
            paths = {r.path for r in Walker(repo, settings).walk()}
        finally:
            blocked.chmod(0o755)
        assert paths == {"keep.py"}

    def test_file_vanishing_mid_walk(self, repo, settings, monkeypatch):
        write(repo / "a.py", "x = 1\n")
        import pathlib

        original_stat = pathlib.Path.stat

        def flaky_stat(self, *args, **kwargs):
            if self.name == "a.py":
                raise OSError("vanished")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "stat", flaky_stat)
        assert list(Walker(repo, settings).walk()) == []


class TestReading:
    def test_content_and_hash_returned(self, repo):
        write(repo / "a.py", "x = 1\n")
        result = read_file(repo, "a.py")
        assert result is not None
        content, digest = result
        assert content == "x = 1\n"
        assert digest == compute_hash("a.py", "x = 1\n")

    def test_hash_changes_with_content(self, repo):
        write(repo / "a.py", "x = 1\n")
        _, digest1 = read_file(repo, "a.py")
        write(repo / "a.py", "x = 2\n")
        _, digest2 = read_file(repo, "a.py")
        assert digest1 != digest2

    def test_hash_changes_with_path(self, repo):
        write(repo / "a.py", "same content")
        write(repo / "b.py", "same content")
        _, digest_a = read_file(repo, "a.py")
        _, digest_b = read_file(repo, "b.py")
        assert digest_a != digest_b

    def test_latin1_fallback(self, repo):
        # 0xFF is invalid as a UTF-8 continuation/lead byte but valid Latin-1 (ÿ).
        write(repo / "legacy.py", binary=b"# caf\xe9\n")
        result = read_file(repo, "legacy.py")
        assert result is not None
        content, _ = result
        assert "caf" in content

    def test_missing_file_returns_none(self, repo):
        assert read_file(repo, "does-not-exist.py") is None

    def test_escaping_symlink_returns_none(self, repo, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        write(outside / "secret.py", "x = 1\n")
        (repo / "escape.py").symlink_to(outside / "secret.py")
        assert read_file(repo, "escape.py") is None

    def test_relpath_escape_returns_none(self, repo, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        write(outside / "secret.py", "x = 1\n")
        relpath = os.path.relpath(outside / "secret.py", start=repo)
        assert read_file(repo, relpath) is None

    def test_broken_symlink_returns_none(self, repo):
        (repo / "broken.py").symlink_to(repo / "does-not-exist.py")
        assert read_file(repo, "broken.py") is None

    def test_contained_dotdot_path_still_reads(self, repo):
        write(repo / "sub" / "a.py", "x = 1\n")
        write(repo / "b.py", "y = 2\n")
        result = read_file(repo, "sub/../b.py")
        assert result is not None
        content, _ = result
        assert content == "y = 2\n"

    def test_undecodable_returns_none(self, repo, monkeypatch):
        write(repo / "a.py", "x = 1\n")
        import pathlib

        calls = {"n": 0}

        def flaky_read_text(self, encoding=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            raise UnicodeDecodeError("latin-1", b"", 0, 1, "bad")

        monkeypatch.setattr(pathlib.Path, "read_text", flaky_read_text)
        assert read_file(repo, "a.py") is None


class TestIgnorePatternMatcher:
    def test_add_patterns_from_missing_file_is_a_noop(self, tmp_path):
        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(tmp_path / "does-not-exist")
        assert matcher.should_ignore("anything.py") is False

    def test_add_patterns_from_file(self, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\n")
        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(gitignore)
        assert matcher.should_ignore("debug.log") is True
        assert matcher.should_ignore("keep.py") is False

    def test_add_patterns_from_unreadable_file_is_a_noop(self, tmp_path, monkeypatch):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\n")
        import pathlib

        def boom(self, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(pathlib.Path, "read_text", boom)
        matcher = IgnorePatternMatcher()
        matcher.add_patterns_from_file(gitignore)
        assert matcher.should_ignore("debug.log") is False


class TestDefensiveBranches:
    """Exercises the walker's belt-and-suspenders error handling that isn't
    reachable through ordinary filesystem setups.
    """

    def test_repo_path_cannot_be_resolved(self, repo, settings, monkeypatch):
        import pathlib

        def boom(self, *args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(pathlib.Path, "resolve", boom)
        assert list(Walker(repo, settings).walk()) == []

    def test_generic_os_error_reading_directory(self, repo, settings, monkeypatch):
        write(repo / "a.py", "x = 1\n")
        import pathlib

        def boom(self, *args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(pathlib.Path, "iterdir", boom)
        assert list(Walker(repo, settings).walk()) == []

    def test_relative_to_value_error_is_skipped(self, repo, settings, monkeypatch):
        write(repo / "a.py", "x = 1\n")
        import pathlib

        def boom(self, *args, **kwargs):
            raise ValueError("not relative")

        monkeypatch.setattr(pathlib.Path, "relative_to", boom)
        assert list(Walker(repo, settings).walk()) == []

    def test_is_file_os_error_is_skipped(self, repo, settings, monkeypatch):
        write(repo / "a.py", "x = 1\n")
        import pathlib

        original_is_file = pathlib.Path.is_file

        def flaky_is_file(self, *args, **kwargs):
            if self.name == "a.py":
                raise OSError("boom")
            return original_is_file(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "is_file", flaky_is_file)
        assert list(Walker(repo, settings).walk()) == []

    def test_consider_file_stat_os_error_is_skipped(self, repo, settings):
        # pathlib's is_dir()/is_file() swallow OSError from their own internal
        # stat() calls, so a global Path.stat patch never reaches
        # _consider_file's own stat() call during a real walk. Exercise it
        # directly with a duck-typed entry instead.
        class FakeEntry:
            name = "a.py"
            suffix = ".py"

            def stat(self):
                raise OSError("boom")

        walker = Walker(repo, settings)
        assert walker._consider_file(FakeEntry(), "a.py") is None
