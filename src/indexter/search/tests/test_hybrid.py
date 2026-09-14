import struct

import pytest

import indexter.search.hybrid as hybrid_module
from indexter.db.connection import open_db
from indexter.index.embed import FakeEmbedder
from indexter.paths import db_path
from indexter.search.hybrid import (
    EmptyQuery,
    IndexNotFound,
    InvalidFilter,
    InvalidLimit,
    build_keyword_expression,
    fuse,
    is_test_file,
    keyword_candidates,
    normalize_kind,
    normalize_language,
    normalize_path,
    search,
    search_repo,
    validate_limit,
    validate_query,
    vector_candidates,
)
from indexter.search.results import render, render_entry_chunk


class TestValidateQuery:
    def test_passes_through_a_real_query(self):
        assert validate_query("find the walker") == "find the walker"

    def test_strips_outer_whitespace(self):
        assert validate_query("  find it  ") == "find it"

    def test_blank_query_rejected(self):
        with pytest.raises(EmptyQuery):
            validate_query("   ")

    def test_empty_string_rejected(self):
        with pytest.raises(EmptyQuery):
            validate_query("")


class TestValidateLimit:
    def test_default_used_when_none(self):
        assert validate_limit(None, default=10) == 10

    def test_value_in_range_kept(self):
        assert validate_limit(25, default=10) == 25

    def test_minimum_allowed(self):
        assert validate_limit(1, default=10) == 1

    def test_maximum_allowed(self):
        assert validate_limit(50, default=10) == 50

    def test_zero_rejected(self):
        with pytest.raises(InvalidLimit):
            validate_limit(0, default=10)

    def test_above_max_rejected(self):
        with pytest.raises(InvalidLimit):
            validate_limit(51, default=10)


class TestNormalizeKind:
    def test_none_means_no_filter(self):
        assert normalize_kind(None) is None

    def test_single_value(self):
        assert normalize_kind("class") == ("class",)

    def test_list_of_values(self):
        assert normalize_kind(["function", "method"]) == ("function", "method")

    def test_external_module_rejected(self):
        with pytest.raises(InvalidFilter) as exc_info:
            normalize_kind("external_module")
        assert exc_info.value.filter_name == "kind"

    def test_unknown_kind_rejected_and_lists_valid_values(self):
        with pytest.raises(InvalidFilter) as exc_info:
            normalize_kind("func")
        assert exc_info.value.value == "func"
        assert "class" in exc_info.value.valid
        assert "external_module" not in exc_info.value.valid


class TestNormalizeLanguage:
    def test_none_means_no_filter(self):
        assert normalize_language(None) is None

    def test_known_language(self):
        assert normalize_language("rust") == ("rust",)

    def test_list_of_languages(self):
        assert normalize_language(["python", "rust"]) == ("python", "rust")

    def test_unknown_language_rejected(self):
        with pytest.raises(InvalidFilter) as exc_info:
            normalize_language("cobol")
        assert exc_info.value.filter_name == "language"
        assert "python" in exc_info.value.valid


class TestNormalizePath:
    def test_none_means_no_filter(self, tmp_path):
        assert normalize_path(None, tmp_path) is None

    def test_empty_string_means_no_filter(self, tmp_path):
        assert normalize_path("", tmp_path) is None

    def test_dot_means_no_filter(self, tmp_path):
        assert normalize_path(".", tmp_path) is None

    def test_leading_dot_slash_stripped(self, tmp_path):
        assert normalize_path("./src/auth", tmp_path) == "src/auth"

    def test_trailing_slash_stripped(self, tmp_path):
        assert normalize_path("src/auth/", tmp_path) == "src/auth"

    def test_plain_relative_path(self, tmp_path):
        assert normalize_path("src/auth", tmp_path) == "src/auth"

    def test_absolute_path_inside_repo_made_relative(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "src" / "auth").mkdir(parents=True)
        assert normalize_path(str(repo / "src" / "auth"), repo) == "src/auth"

    def test_absolute_path_equal_to_repo_root_means_no_filter(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        assert normalize_path(str(repo), repo) is None

    def test_absolute_path_outside_repo_rejected(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        other = tmp_path / "other"
        with pytest.raises(InvalidFilter):
            normalize_path(str(other), repo)

    def test_dotdot_escape_rejected(self, tmp_path):
        with pytest.raises(InvalidFilter):
            normalize_path("../other", tmp_path)


class TestBuildKeywordExpression:
    def test_plain_words_quoted_and_ored(self):
        expr = build_keyword_expression("find walker")
        assert expr == '"find" OR "walker"'

    def test_stopwords_dropped(self):
        expr = build_keyword_expression("where is the walker")
        assert expr == '"walker"'

    def test_stopword_only_query_keeps_all_words(self):
        expr = build_keyword_expression("how is it")
        assert expr is not None
        assert '"how"' in expr
        assert '"is"' in expr
        assert '"it"' in expr

    def test_no_word_characters_yields_no_expression(self):
        assert build_keyword_expression("?!") is None

    def test_split_identifier_words_included(self):
        expr = build_keyword_expression("getUserByEmail")
        assert '"getuserbyemail"' in expr
        assert '"get"' in expr
        assert '"user"' in expr
        assert '"email"' in expr

    def test_underscored_identifier_split(self):
        expr = build_keyword_expression("get_user_by_email")
        assert '"user"' in expr
        assert '"email"' in expr

    def test_syntax_looking_query_treated_as_text(self):
        expr = build_keyword_expression('parse AND NOT "config*')
        assert expr is not None
        assert "config" in expr
        for term in expr.split(" OR "):
            assert term.startswith('"')
            assert term.endswith('"')

    def test_duplicate_terms_deduplicated(self):
        expr = build_keyword_expression("walker walker")
        assert expr == '"walker"'

    def test_term_cap(self):
        query = " ".join(f"word{i}" for i in range(40))
        expr = build_keyword_expression(query)
        assert expr.count(" OR ") == 31  # 32 terms joined by 31 " OR "s


class TestIsTestFile:
    def test_tests_directory_component(self):
        assert is_test_file("tests/test_walker.py")

    def test_test_directory_component(self):
        assert is_test_file("test/walker.py")

    def test_dunder_tests_directory_component(self):
        assert is_test_file("__tests__/walker.js")

    def test_test_prefix_filename(self):
        assert is_test_file("src/test_walker.py")

    def test_test_suffix_filename(self):
        assert is_test_file("src/walker_test.py")

    def test_dot_test_dot_filename(self):
        assert is_test_file("src/walker.test.js")

    def test_dot_spec_dot_filename(self):
        assert is_test_file("src/walker.spec.ts")

    def test_conftest(self):
        assert is_test_file("src/conftest.py")

    def test_ordinary_file_is_not_a_test(self):
        assert not is_test_file("src/walker.py")

    def test_directory_named_like_test_word_but_not_a_component(self):
        assert not is_test_file("src/attestation.py")


class TestFuse:
    def test_both_signals_beat_one(self):
        # A: vector rank 3, keyword rank 3. B: vector rank 1 only.
        vector_ids = ["b", "x", "a"]
        keyword_ids = ["y", "z", "a"]
        ranked = fuse(vector_ids, keyword_ids, is_test=lambda _: False)
        ids = [n.node_id for n in ranked]
        assert ids.index("a") < ids.index("b")

    def test_match_reasons_recorded(self):
        ranked = fuse([], ["x", "a"], is_test=lambda _: False)
        node = next(n for n in ranked if n.node_id == "a")
        assert node.reasons.vector_rank is None
        assert node.reasons.keyword_rank == 2

    def test_test_files_demoted_below_equal_score(self):
        ranked = fuse(["test_node"], ["src_node"], is_test=lambda node_id: node_id == "test_node")
        ids = [n.node_id for n in ranked]
        assert ids.index("src_node") < ids.index("test_node")

    def test_strong_test_match_still_surfaces(self):
        vector_ids = ["test_node"] + [f"other{i}" for i in range(39)]
        keyword_ids = ["test_node"]
        ranked = fuse(vector_ids, keyword_ids, is_test=lambda node_id: node_id == "test_node")
        ids = [n.node_id for n in ranked]
        assert ids.index("test_node") < ids.index("other38")

    def test_deterministic_ties_broken_by_node_id(self):
        ranked = fuse(["b", "a"], [], is_test=lambda _: False)
        # equal vector-only presence would tie only if ranks equal; use two
        # separate lists giving equal scores instead
        ranked_equal = fuse(["a"], ["b"], is_test=lambda _: False)
        ids = [n.node_id for n in ranked_equal]
        assert ids == ["a", "b"]
        assert [n.node_id for n in ranked] == ["b", "a"]  # not a tie: different ranks

    def test_repeated_fuse_is_stable(self):
        first = fuse(["a", "b"], ["b", "a"], is_test=lambda _: False)
        second = fuse(["a", "b"], ["b", "a"], is_test=lambda _: False)
        assert [n.node_id for n in first] == [n.node_id for n in second]


def _node_row(conn, node_id):
    return conn.execute("SELECT kind, language, file_path FROM nodes WHERE id = ?", (node_id,)).fetchone()


class TestVectorCandidates:
    def test_returns_some_candidates(self, conn, embedder):
        ids = vector_candidates(conn, embedder, "helper")
        assert ids
        assert len(ids) <= 50

    def test_external_module_never_a_candidate(self, conn, embedder):
        ids = vector_candidates(conn, embedder, "os")
        assert "external::os" not in ids

    def test_kind_filter_matching_nothing_yields_no_candidates(self, conn, embedder):
        assert vector_candidates(conn, embedder, "helper", kind=("interface",)) == []

    def test_kind_filter(self, conn, embedder):
        ids = vector_candidates(conn, embedder, "helper", kind=("class",))
        assert ids
        for node_id in ids:
            assert _node_row(conn, node_id)["kind"] == "class"

    def test_language_filter(self, conn, embedder):
        ids = vector_candidates(conn, embedder, "helper", language=("markdown",))
        assert ids
        for node_id in ids:
            assert _node_row(conn, node_id)["language"] == "markdown"

    def test_path_filter_respects_component_boundary(self, conn, embedder):
        ids = vector_candidates(conn, embedder, "helper", path="src/auth")
        assert ids
        for node_id in ids:
            file_path = _node_row(conn, node_id)["file_path"]
            assert file_path == "src/auth" or file_path.startswith("src/auth/")


class TestKeywordCandidates:
    def test_matches_function_by_name(self, conn):
        ids = keyword_candidates(conn, "helper")
        assert "src/walker.py::helper#function" in ids

    def test_external_module_excluded(self, conn):
        ids = keyword_candidates(conn, "os")
        assert "external::os" not in ids
        assert any(_node_row(conn, i)["file_path"] == "src/walker.py" for i in ids)

    def test_no_terms_yields_no_candidates(self, conn):
        assert keyword_candidates(conn, "?!") == []

    def test_syntax_looking_query_runs_without_error(self, conn):
        ids = keyword_candidates(conn, 'parse AND NOT "config*')
        assert isinstance(ids, list)

    def test_kind_filter(self, conn):
        ids = keyword_candidates(conn, "return", kind=("function",))
        for node_id in ids:
            assert _node_row(conn, node_id)["kind"] == "function"

    def test_language_filter_includes_markdown_only_term(self, conn):
        ids = keyword_candidates(conn, "walks")
        assert any(_node_row(conn, i)["file_path"] == "README.md" for i in ids)

        ids_python_only = keyword_candidates(conn, "walks", language=("python",))
        assert ids_python_only == []

    def test_path_filter_finds_matches_outside_top_pool(self, conn):
        ids = keyword_candidates(conn, "true", path="src/auth")
        assert ids
        for node_id in ids:
            file_path = _node_row(conn, node_id)["file_path"]
            assert file_path == "src/auth" or file_path.startswith("src/auth/")
        assert not any(_node_row(conn, i)["file_path"] == "src/authz.py" for i in ids)


def _entries_text(entries):
    """Rebuild the joined entry text `search_repo` measures related's
    remaining budget against, from a response's own admitted entries."""
    seen_files = set()
    chunks = []
    for entry in entries:
        heading = f"## {entry.file_path}" if entry.file_path not in seen_files else None
        seen_files.add(entry.file_path)
        chunks.append(render_entry_chunk(entry, heading=heading))
    return "\n\n".join(chunks)


class TestSearchRepoBudget:
    def test_tiny_budget_returns_fewer_entries_as_a_prefix(self, conn, repo, settings, embedder):
        full = search_repo(conn, repo, "return", settings, embedder, limit=10)
        assert len(full.entries) >= 2

        tiny_settings = settings.model_copy(update={"search_max_chars": 200})
        tiny = search_repo(conn, repo, "return", tiny_settings, embedder, limit=10)

        assert 1 <= len(tiny.entries) < len(full.entries)
        assert [e.node_id for e in tiny.entries] == [e.node_id for e in full.entries[: len(tiny.entries)]]
        assert tiny.entries_omitted == len(full.entries) - len(tiny.entries)

    def test_oversized_first_entry_is_included_alone(self, conn, repo, settings, embedder):
        tiny_settings = settings.model_copy(update={"search_max_chars": 1})
        result = search_repo(conn, repo, "helper", tiny_settings, embedder, limit=10)
        assert len(result.entries) == 1

    def test_related_yields_to_hits_when_budget_is_tight(self, conn, repo, settings, embedder):
        # `kind="class"` narrows candidates to the fixture's one class, `Store`,
        # so `Store.add` never independently ranks and is only ever reached
        # as `related` -- a deterministic setup regardless of embedder.
        generous = search_repo(conn, repo, "Store", settings, embedder, kind="class", limit=10)
        assert generous.related  # Store.add, reached as Store's member

        exact_budget = len(_entries_text(generous.entries))
        tight_settings = settings.model_copy(update={"search_max_chars": exact_budget})
        result = search_repo(conn, repo, "Store", tight_settings, embedder, kind="class", limit=10)

        assert [e.node_id for e in result.entries] == [e.node_id for e in generous.entries]
        assert result.related == ()
        assert result.related_omitted == len(generous.related)

    def test_seeds_come_only_from_admitted_entries(self, conn, repo, settings, embedder, monkeypatch):
        captured = {}
        original_expand = hybrid_module.expand

        def capturing_expand(conn_, seeds, excluded):
            captured["seeds"] = seeds
            return original_expand(conn_, seeds, excluded)

        monkeypatch.setattr(hybrid_module, "expand", capturing_expand)

        full = search_repo(conn, repo, "return", settings, embedder, limit=10)
        assert len(full.entries) >= 2

        tiny_settings = settings.model_copy(update={"search_max_chars": 1})
        result = search_repo(conn, repo, "return", tiny_settings, embedder, limit=10)

        assert len(result.entries) == 1
        assert len(captured["seeds"]) == 1
        assert captured["seeds"][0].node_id == result.entries[0].node_id
        assert captured["seeds"][0].weight == pytest.approx(1.0 / 61)


class TestSearchRepoRendering:
    def test_entries_from_different_files_get_separate_headings(self, conn, repo, settings, embedder):
        result = search_repo(conn, repo, "true", settings, embedder, kind="function", limit=10)
        files = {e.file_path for e in result.entries}
        assert len(files) >= 2

        text = render(result)
        for file_path in files:
            assert f"## {file_path}" in text

    def test_repeated_search_renders_identically(self, conn, repo, settings, embedder):
        first = search_repo(conn, repo, "helper", settings, embedder, limit=10)
        second = search_repo(conn, repo, "helper", settings, embedder, limit=10)
        assert render(first) == render(second)


class TestSearchRepoSync:
    def test_added_function_is_searchable_immediately(self, conn, repo, settings, embedder):
        (repo / "src" / "new_module.py").write_text("def brand_new_symbol():\n    return 42\n")

        result = search_repo(conn, repo, "brand_new_symbol", settings, embedder)

        assert any(e.qualified_name == "brand_new_symbol" for e in result.entries)

    def test_deleted_file_is_not_returned(self, conn, repo, settings, embedder):
        (repo / "src" / "authz.py").unlink()

        result = search_repo(conn, repo, "authorize", settings, embedder)

        assert all(e.file_path != "src/authz.py" for e in result.entries)

    def test_unchanged_repo_sync_report_shows_no_work(self, conn, repo, settings, embedder):
        result = search_repo(conn, repo, "helper", settings, embedder)

        assert result.sync_report.added == ()
        assert result.sync_report.changed == ()
        assert result.sync_report.removed == ()
        assert result.sync_report.texts_embedded == 0

    def test_invalid_filter_costs_no_sync(self, conn, repo, settings, embedder, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("sync_repo should not be called for an invalid filter")

        monkeypatch.setattr(hybrid_module, "sync_repo", boom)

        with pytest.raises(InvalidFilter):
            search_repo(conn, repo, "helper", settings, embedder, kind="not-a-kind")


class TestSearchRepoErrors:
    def test_empty_query_raises(self, conn, repo, settings, embedder):
        with pytest.raises(EmptyQuery):
            search_repo(conn, repo, "   ", settings, embedder)


class TestSearch:
    def test_unindexed_repo_raises_and_creates_no_database(self, tmp_path, settings, embedder):
        repo_dir = tmp_path / "fresh_repo"
        repo_dir.mkdir()
        (repo_dir / "a.py").write_text("def f():\n    pass\n")

        with pytest.raises(IndexNotFound):
            search(repo_dir, "f", settings, embedder)

        assert not db_path(repo_dir).exists()

    def test_opens_existing_database_and_delegates(self, repo, settings, embedder, monkeypatch, tmp_path):
        target_db = tmp_path / "data" / "search.db"
        monkeypatch.setattr(hybrid_module, "resolve_db_path", lambda _repo: target_db)
        with open_db(target_db, repo=repo, settings=settings):
            pass  # create the database ahead of the search

        result = search(repo, "helper", settings, embedder)

        assert any(e.qualified_name == "helper" for e in result.entries)


class _ConstantEmbedder:
    """A stub embedder mapping every text to the same fixed vector (design.md
    decision 10): with vector distances all tied, order is fully controlled
    by the deterministic node-ID tie-break, making the snapshot below exact
    and reproducible without a real model.
    """

    model_name = "constant-stub"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._tokenizer_source = FakeEmbedder(dim=dim)

    def tokenizer(self):
        return self._tokenizer_source.tokenizer()

    def embed(self, texts):
        vector = struct.pack(f"{self.dim}f", *([1.0] * self.dim))
        return [vector for _ in texts]


class TestSearchRepoSnapshot:
    def test_full_rendered_response_over_the_fixture_repo(self, repo, settings, tmp_path):
        embedder = _ConstantEmbedder(dim=settings.embedding_dim)
        database = tmp_path / "data" / "snapshot.db"

        with open_db(database, repo=repo, settings=settings) as conn:
            result = search_repo(conn, repo, "helper", settings, embedder, kind="function", limit=5)
            text = render(result)

        assert text == (
            '4 results for "helper"\n'
            "\n"
            "## src/walker.py\n"
            "\n"
            "### helper — function — src/walker.py:4-5 — vector, keyword\n"
            "id: src/walker.py::helper#function\n"
            "def helper()\n"
            "def helper():\n"
            "    return 1\n"
            "\n"
            "## src/auth/login.py\n"
            "\n"
            "### login — function — src/auth/login.py:1-2 — vector\n"
            "id: src/auth/login.py::login#function\n"
            "def login()\n"
            "def login():\n"
            "    return True\n"
            "\n"
            "## src/authz.py\n"
            "\n"
            "### authorize — function — src/authz.py:1-2 — vector\n"
            "id: src/authz.py::authorize#function\n"
            "def authorize()\n"
            "def authorize():\n"
            "    return True\n"
            "\n"
            "## tests/test_walker.py\n"
            "\n"
            "### test_helper — function — tests/test_walker.py:1-2 — vector, keyword\n"
            "id: tests/test_walker.py::test_helper#function\n"
            "def test_helper()\n"
            "def test_helper():\n"
            "    return 1"
        )
