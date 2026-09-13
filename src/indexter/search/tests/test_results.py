from indexter.index.sync import SyncReport
from indexter.search.results import (
    admit_entries,
    admit_related,
    read_snippets,
    render,
    render_entry_chunk,
    render_related_item,
    select_entries,
)
from indexter.search.types import (
    ContextRef,
    Entry,
    GraphContext,
    MatchReasons,
    Member,
    NodeRow,
    RankedNode,
    Related,
    SearchResponse,
    Selection,
    Timings,
)


def _row(node_id, *, kind="function", qualified_name=None, file_path="src/a.py",
         start_line=1, end_line=1, start_byte=0, end_byte=10, signature=None,
         docstring=None, parent_id=None):
    return NodeRow(
        id=node_id,
        kind=kind,
        qualified_name=qualified_name or node_id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        start_byte=start_byte,
        end_byte=end_byte,
        signature=signature,
        docstring=docstring,
        parent_id=parent_id,
    )


def _ranked(node_id, score, *, vector_rank=None, keyword_rank=None):
    reasons = MatchReasons(vector_rank=vector_rank, keyword_rank=keyword_rank)
    return RankedNode(node_id=node_id, score=score, reasons=reasons)


class TestSelectEntries:
    def test_empty_ranking_yields_no_entries(self):
        assert select_entries([], {}.__getitem__, limit=10) == ()

    def test_single_node_with_no_container_is_a_plain_hit(self):
        rows = {"a": _row("a")}
        ranked = [_ranked("a", 1.0, vector_rank=1)]

        entries = select_entries(ranked, rows.__getitem__, limit=10)

        assert len(entries) == 1
        assert entries[0].node_id == "a"
        assert entries[0].members == ()
        assert entries[0].reasons == MatchReasons(vector_rank=1)

    def test_two_methods_collapse_and_fill_the_freed_slot(self):
        rows = {
            "Store": _row("Store", kind="class", start_byte=0, end_byte=100),
            "Store.add": _row("Store.add", kind="method", parent_id="Store", start_byte=10, end_byte=20),
            "Store.remove": _row("Store.remove", kind="method", parent_id="Store", start_byte=30, end_byte=40),
            "x": _row("x"),
            "y": _row("y"),
        }
        ranked = [
            _ranked("Store.add", 0.99),
            _ranked("Store.remove", 0.98),
            _ranked("x", 0.97),
            _ranked("y", 0.96),
        ]

        entries = select_entries(ranked, rows.__getitem__, limit=2)

        assert [e.node_id for e in entries] == ["Store", "x"]
        assert [m.node_id for m in entries[0].members] == ["Store.add", "Store.remove"]

    def test_class_absorbs_its_own_single_method_hit(self):
        rows = {
            "Store": _row("Store", kind="class"),
            "Store.add": _row("Store.add", kind="method", parent_id="Store"),
        }
        ranked = [_ranked("Store", 1.0), _ranked("Store.add", 0.9)]

        entries = select_entries(ranked, rows.__getitem__, limit=10)

        assert len(entries) == 1
        assert entries[0].node_id == "Store"
        assert [m.node_id for m in entries[0].members] == ["Store.add"]

    def test_single_method_without_a_class_hit_stays_a_hit(self):
        rows = {
            "Store": _row("Store", kind="class"),
            "Store.add": _row("Store.add", kind="method", parent_id="Store", qualified_name="Store.add"),
            "other": _row("other"),
        }
        ranked = [_ranked("Store.add", 1.0), _ranked("other", 0.9)]

        entries = select_entries(ranked, rows.__getitem__, limit=10)

        assert [e.node_id for e in entries] == ["Store.add", "other"]
        assert entries[0].members == ()
        assert entries[0].kind == "method"

    def test_limit_counts_entries_not_raw_ranked_nodes(self):
        rows = {"a": _row("a"), "b": _row("b"), "c": _row("c")}
        ranked = [_ranked("a", 1.0), _ranked("b", 0.9), _ranked("c", 0.8)]

        entries = select_entries(ranked, rows.__getitem__, limit=2)

        assert [e.node_id for e in entries] == ["a", "b"]

    def test_reasons_are_unioned_across_the_group(self):
        rows = {
            "Store": _row("Store", kind="class"),
            "Store.add": _row("Store.add", kind="method", parent_id="Store"),
        }
        ranked = [_ranked("Store", 1.0, vector_rank=5), _ranked("Store.add", 0.9, keyword_rank=2)]

        entries = select_entries(ranked, rows.__getitem__, limit=10)

        assert entries[0].reasons == MatchReasons(vector_rank=5, keyword_rank=2)

    def test_reasons_union_keeps_the_better_rank_when_both_matched_the_same_way(self):
        rows = {
            "Store": _row("Store", kind="class"),
            "Store.add": _row("Store.add", kind="method", parent_id="Store"),
        }
        ranked = [_ranked("Store", 1.0, vector_rank=8), _ranked("Store.add", 0.9, vector_rank=3)]

        entries = select_entries(ranked, rows.__getitem__, limit=10)

        assert entries[0].reasons == MatchReasons(vector_rank=3)

    def test_snippet_follows_the_best_ranked_node_in_the_group(self):
        rows = {
            "Store": _row("Store", kind="class", start_byte=0, end_byte=100),
            "Store.add": _row("Store.add", kind="method", parent_id="Store", start_byte=10, end_byte=20),
        }
        ranked = [_ranked("Store.add", 1.0), _ranked("Store", 0.9)]

        entries = select_entries(ranked, rows.__getitem__, limit=10)

        assert entries[0].node_id == "Store"
        assert entries[0].snippet_node_id == "Store.add"
        assert (entries[0].snippet_start_byte, entries[0].snippet_end_byte) == (10, 20)

    def test_snippet_is_the_class_when_the_class_itself_ranked_best(self):
        rows = {
            "Store": _row("Store", kind="class", start_byte=0, end_byte=100),
            "Store.add": _row("Store.add", kind="method", parent_id="Store", start_byte=10, end_byte=20),
        }
        ranked = [_ranked("Store", 1.0), _ranked("Store.add", 0.9)]

        entries = select_entries(ranked, rows.__getitem__, limit=10)

        assert entries[0].snippet_node_id == "Store"
        assert (entries[0].snippet_start_byte, entries[0].snippet_end_byte) == (0, 100)


def _selection(node_id, *, file_path, start_byte, end_byte):
    return Selection(
        node_id=node_id,
        qualified_name=node_id,
        kind="function",
        file_path=file_path,
        start_line=1,
        end_line=1,
        signature=None,
        docstring=None,
        reasons=MatchReasons(),
        members=(),
        snippet_node_id=node_id,
        snippet_start_byte=start_byte,
        snippet_end_byte=end_byte,
    )


class TestReadSnippets:
    def test_source_under_max_lines_is_returned_exactly(self, tmp_path):
        content = "def f():\n    return 1"
        (tmp_path / "a.py").write_text(content)
        end = len(content.encode("utf-8"))
        selection = _selection("a", file_path="a.py", start_byte=0, end_byte=end)

        snippets = read_snippets(tmp_path, [selection], max_lines=40)

        assert snippets["a"] == content

    def test_100_lines_elided_to_exactly_40(self, tmp_path):
        content = "\n".join(f"line{i}" for i in range(100))
        (tmp_path / "a.py").write_text(content)
        end = len(content.encode("utf-8"))
        selection = _selection("a", file_path="a.py", start_byte=0, end_byte=end)

        snippets = read_snippets(tmp_path, [selection], max_lines=40)

        lines = snippets["a"].split("\n")
        assert len(lines) == 40
        assert lines[0] == "line0"
        assert lines[-1] == "line99"
        assert "61 lines elided" in lines[20]

    def test_long_line_is_cut_at_240_characters(self, tmp_path):
        content = "x" * 500
        (tmp_path / "a.py").write_text(content)
        selection = _selection("a", file_path="a.py", start_byte=0, end_byte=len(content))

        snippets = read_snippets(tmp_path, [selection], max_lines=40)

        assert snippets["a"] == "x" * 240 + "…"

    def test_multi_byte_content_slices_on_utf8_boundaries(self, tmp_path):
        content = "def greet():\n    return '日本語'\n"
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        encoded = content.encode("utf-8")
        selection = _selection("a", file_path="a.py", start_byte=0, end_byte=len(encoded))

        snippets = read_snippets(tmp_path, [selection], max_lines=40)

        assert "日本語" in snippets["a"]

    def test_byte_range_splitting_a_multi_byte_character_yields_no_snippet(self, tmp_path):
        content = "café"
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        # "é" is 2 UTF-8 bytes; ending one byte early splits it.
        selection = _selection("a", file_path="a.py", start_byte=0, end_byte=len(content.encode("utf-8")) - 1)

        snippets = read_snippets(tmp_path, [selection], max_lines=40)

        assert snippets["a"] is None

    def test_vanished_file_yields_no_snippet(self, tmp_path):
        selection = _selection("a", file_path="missing.py", start_byte=0, end_byte=10)

        snippets = read_snippets(tmp_path, [selection], max_lines=40)

        assert snippets["a"] is None

    def test_each_file_is_read_once(self, tmp_path, monkeypatch):
        content = "def f():\n    return 1\ndef g():\n    return 2\n"
        (tmp_path / "a.py").write_text(content)
        encoded = content.encode("utf-8")

        import indexter.search.results as results_module

        calls = []
        original = results_module.read_file

        def counting_read_file(repo, relpath):
            calls.append(relpath)
            return original(repo, relpath)

        monkeypatch.setattr(results_module, "read_file", counting_read_file)

        second = content.index("def g")
        selections = [
            _selection("f", file_path="a.py", start_byte=0, end_byte=len(content[:second].encode("utf-8"))),
            _selection("g", file_path="a.py", start_byte=len(content[:second].encode("utf-8")), end_byte=len(encoded)),
        ]

        results_module.read_snippets(tmp_path, selections, max_lines=40)

        assert calls == ["a.py"]


def _entry(
    node_id,
    *,
    qualified_name=None,
    kind="function",
    file_path="src/a.py",
    start_line=1,
    end_line=1,
    reasons=None,
    signature=None,
    docstring=None,
    snippet="return 1",
    snippet_unavailable=False,
    members=(),
    context=None,
):
    return Entry(
        node_id=node_id,
        qualified_name=qualified_name or node_id,
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        reasons=reasons or MatchReasons(vector_rank=1),
        signature=signature,
        docstring=docstring,
        snippet=snippet,
        snippet_unavailable=snippet_unavailable,
        members=members,
        context=context or GraphContext(),
    )


def _related(
    node_id,
    *,
    qualified_name=None,
    kind="function",
    file_path="src/a.py",
    start_line=1,
    end_line=1,
    reason="calls x, which matched",
):
    return Related(
        node_id=node_id,
        qualified_name=qualified_name or node_id,
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        confidence="exact",
        reason=reason,
    )


def _sync_report():
    return SyncReport(
        added=(),
        changed=(),
        removed=(),
        unchanged=(),
        nodes_written=0,
        nodes_deleted=0,
        refs_written=0,
        texts_embedded=0,
        errors={},
        elapsed_seconds=0.0,
    )


def _timings():
    return Timings(
        sync_seconds=0.0,
        embed_seconds=0.0,
        candidates_seconds=0.0,
        fusion_seconds=0.0,
        expansion_seconds=0.0,
        render_seconds=0.0,
    )


def _response(*, query="q", entries=(), related=(), entries_omitted=0, related_omitted=0):
    return SearchResponse(
        query=query,
        kind=None,
        language=None,
        path=None,
        limit=10,
        entries=entries,
        related=related,
        entries_omitted=entries_omitted,
        related_omitted=related_omitted,
        sync_report=_sync_report(),
        timings=_timings(),
    )


class TestRenderEntryChunk:
    def test_header_names_qualified_name_kind_and_location(self):
        entry = _entry("a", qualified_name="Store.add", kind="method", file_path="src/a.py", start_line=10, end_line=12)

        chunk = render_entry_chunk(entry, heading=None)

        assert "### Store.add — method — src/a.py:10-12" in chunk

    def test_reasons_list_vector_then_keyword(self):
        entry = _entry("a", reasons=MatchReasons(vector_rank=1, keyword_rank=2))
        assert "vector, keyword" in render_entry_chunk(entry, heading=None)

    def test_keyword_only_reason(self):
        entry = _entry("a", reasons=MatchReasons(keyword_rank=1))
        chunk = render_entry_chunk(entry, heading=None)
        assert "keyword" in chunk
        assert "vector" not in chunk

    def test_id_line_present(self):
        entry = _entry("src/a.py::foo#function")
        assert "id: src/a.py::foo#function" in render_entry_chunk(entry, heading=None)

    def test_container_shown_in_context_line(self):
        context = GraphContext(container=ContextRef(node_id="c", qualified_name="Store"))
        entry = _entry("a", context=context)
        assert "in: Store" in render_entry_chunk(entry, heading=None)

    def test_callers_fully_shown_have_no_of_suffix(self):
        context = GraphContext(callers=(ContextRef(node_id="x", qualified_name="caller_one"),), caller_total=1)
        entry = _entry("a", context=context)
        assert "callers (1): caller_one" in render_entry_chunk(entry, heading=None)

    def test_callees_partially_shown_note_the_total(self):
        callees = tuple(ContextRef(node_id=f"c{i}", qualified_name=f"callee{i}") for i in range(3))
        context = GraphContext(callees=callees, callee_total=4)
        entry = _entry("a", context=context)
        assert "callees (3 of 4): callee0, callee1, callee2" in render_entry_chunk(entry, heading=None)

    def test_ambiguous_caller_is_marked(self):
        context = GraphContext(
            callers=(ContextRef(node_id="x", qualified_name="maybe_caller", confidence="ambiguous"),), caller_total=1
        )
        entry = _entry("a", context=context)
        assert "maybe_caller?" in render_entry_chunk(entry, heading=None)

    def test_no_context_line_when_context_is_empty(self):
        entry = _entry("a", context=GraphContext())
        chunk = render_entry_chunk(entry, heading=None)
        assert "in:" not in chunk

    def test_signature_and_docstring_included(self):
        entry = _entry("a", signature="def foo() -> int", docstring="Does a thing.")
        chunk = render_entry_chunk(entry, heading=None)
        assert "def foo() -> int" in chunk
        assert "Does a thing." in chunk

    def test_matched_members_block(self):
        members = (
            Member(node_id="a.add", qualified_name="Store.add", start_line=2, end_line=3, signature="def add(self)"),
        )
        entry = _entry("a", file_path="src/a.py", members=members)
        chunk = render_entry_chunk(entry, heading=None)
        assert "matched members:" in chunk
        assert "- Store.add — src/a.py:2-3 — def add(self)" in chunk

    def test_snippet_included(self):
        entry = _entry("a", snippet="def foo():\n    return 1")
        assert "def foo():\n    return 1" in render_entry_chunk(entry, heading=None)

    def test_unavailable_snippet_note(self):
        entry = _entry("a", snippet=None, snippet_unavailable=True)
        assert "snippet unavailable" in render_entry_chunk(entry, heading=None)

    def test_heading_prefixes_the_block(self):
        entry = _entry("a", file_path="src/a.py")
        chunk = render_entry_chunk(entry, heading="## src/a.py")
        assert chunk.startswith("## src/a.py\n\n### ")

    def test_no_heading_when_none(self):
        entry = _entry("a")
        chunk = render_entry_chunk(entry, heading=None)
        assert not chunk.startswith("## ")


class TestRenderRelatedItem:
    def test_format(self):
        item = _related(
            "b",
            qualified_name="helper",
            kind="function",
            file_path="src/b.py",
            start_line=5,
            end_line=8,
            reason="calls foo, which matched",
        )
        assert render_related_item(item) == "- helper — function — src/b.py:5-8 — calls foo, which matched"


class TestAdmitEntries:
    def test_first_chunk_always_admitted_even_if_oversized(self):
        assert admit_entries(["x" * 1000], budget=10) == 1

    def test_all_fit(self):
        assert admit_entries(["a", "b", "c"], budget=1000) == 3

    def test_stops_at_first_misfit(self):
        chunks = ["a" * 5, "b" * 100, "c" * 1]
        assert admit_entries(chunks, budget=10) == 1

    def test_no_pull_forward_of_a_smaller_later_chunk(self):
        chunks = ["a" * 5, "b" * 100, "c" * 1]
        assert admit_entries(chunks, budget=10) == 1

    def test_empty_input(self):
        assert admit_entries([], budget=100) == 0


class TestAdmitRelated:
    def test_no_related_chunks_yields_zero(self):
        assert admit_related("entries", [], budget=1000) == 0

    def test_all_fit_after_entries(self):
        assert admit_related("entries text", ["- a", "- b"], budget=1000) == 2

    def test_stops_at_first_misfit(self):
        assert admit_related("x" * 50, ["- a", "- " + "b" * 100, "- c"], budget=70) == 1

    def test_related_omitted_when_none_fit(self):
        assert admit_related("x" * 990, ["- " + "a" * 100], budget=1000) == 0

    def test_empty_entries_text(self):
        assert admit_related("", ["- a"], budget=1000) == 1


class TestRender:
    def test_no_results(self):
        response = _response(query="nothing matches this")
        assert render(response) == 'no results for "nothing matches this"'

    def test_no_results_has_no_related_section(self):
        response = _response(query="q")
        assert "related" not in render(response)

    def test_header_line_names_count_and_query(self):
        response = _response(query="find it", entries=(_entry("a"),))
        assert render(response).startswith('1 results for "find it"')

    def test_header_line_reports_omitted_count(self):
        response = _response(entries=(_entry("a"),), entries_omitted=4)
        assert "(4 omitted for budget)" in render(response).splitlines()[0]

    def test_grouped_by_file_out_of_rank_order(self):
        entries = (
            _entry("a1", file_path="a.py"),
            _entry("b1", file_path="b.py"),
            _entry("a2", file_path="a.py"),
        )
        response = _response(entries=entries)

        text = render(response)

        a_heading = text.index("## a.py")
        b_heading = text.index("## b.py")
        a1_pos = text.index("### a1")
        a2_pos = text.index("### a2")
        b1_pos = text.index("### b1")
        assert a_heading < a1_pos < a2_pos
        assert b_heading < b1_pos
        # a.py's two entries are grouped together, before b.py's heading.
        assert a2_pos < b_heading

    def test_only_one_heading_per_file(self):
        entries = (_entry("a1", file_path="a.py"), _entry("a2", file_path="a.py"))
        response = _response(entries=entries)
        assert render(response).count("## a.py") == 1

    def test_related_section_lists_each_item(self):
        response = _response(entries=(_entry("a"),), related=(_related("b", qualified_name="helper"),))
        text = render(response)
        assert "## related" in text
        assert "- helper" in text

    def test_repeated_render_is_stable(self):
        response = _response(entries=(_entry("a"), _entry("b", file_path="b.py")), related=(_related("c"),))
        assert render(response) == render(response)
