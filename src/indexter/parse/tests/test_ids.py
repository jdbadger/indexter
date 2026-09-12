from indexter.parse.ids import assign_ids, build_id, link_parents, link_refs
from indexter.parse.models import Kind, ParsedNode, RawRef, RefKind


def node(*, kind=Kind.FUNCTION, name="foo", scope_path=(), start_line=1, end_line=2, start_byte=0, end_byte=10, **kw):
    return ParsedNode(
        kind=kind,
        name=name,
        scope_path=scope_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        start_byte=start_byte,
        end_byte=end_byte,
        **kw,
    )


class TestBuildId:
    def test_method_inside_class(self):
        assert build_id("src/auth/handlers.py", ("AuthHandler",), "login", Kind.METHOD) == (
            "src/auth/handlers.py::AuthHandler.login#method"
        )

    def test_function_at_file_scope(self):
        assert build_id("src/auth/handlers.py", (), "login", Kind.FUNCTION) == "src/auth/handlers.py::login#function"

    def test_file_node(self):
        assert build_id("src/auth/handlers.py", (), "", Kind.FILE) == "src/auth/handlers.py::#file"

    def test_reproducible(self):
        a = build_id("a.py", ("Foo",), "bar", Kind.METHOD)
        b = build_id("a.py", ("Foo",), "bar", Kind.METHOD)
        assert a == b

    def test_nested_scope_joined_with_dots(self):
        assert build_id("a.py", ("Outer", "Inner"), "m", Kind.METHOD) == "a.py::Outer.Inner.m#method"


class TestStability:
    def test_start_line_does_not_affect_id(self):
        n1 = node(start_line=5)
        n2 = node(start_line=50)
        [a] = assign_ids("a.py", [n1])
        [b] = assign_ids("a.py", [n2])
        assert a.id == b.id

    def test_docstring_and_signature_do_not_affect_id(self):
        n1 = node(docstring="one", signature="def foo():")
        n2 = node(docstring="two", signature="def foo(x):")
        [a] = assign_ids("a.py", [n1])
        [b] = assign_ids("a.py", [n2])
        assert a.id == b.id

    def test_renaming_changes_id(self):
        [a] = assign_ids("a.py", [node(name="login")])
        [b] = assign_ids("a.py", [node(name="signin")])
        assert a.id != b.id

    def test_moving_into_a_class_changes_id(self):
        [a] = assign_ids("a.py", [node(name="login", scope_path=(), kind=Kind.FUNCTION)])
        [b] = assign_ids("a.py", [node(name="login", scope_path=("Handler",), kind=Kind.METHOD)])
        assert a.id != b.id


class TestDuplicates:
    def test_second_occurrence_gets_suffix(self):
        first = node(name="login", start_line=5)
        second = node(name="login", start_line=20)
        result = assign_ids("a.py", [first, second])
        ids = {n.start_line: n.id for n in result}
        assert ids[5] == "a.py::login#function"
        assert ids[20] == "a.py::login#function~2"

    def test_numbering_follows_line_order_not_emission_order(self):
        later = node(name="login", start_line=20)
        earlier = node(name="login", start_line=5)
        # emitted later-first, earlier-second
        result = assign_ids("a.py", [later, earlier])
        by_line = {n.start_line: n.id for n in result}
        assert by_line[5] == "a.py::login#function"
        assert by_line[20] == "a.py::login#function~2"

    def test_three_duplicates(self):
        nodes = [node(name="login", start_line=line) for line in (30, 10, 20)]
        result = assign_ids("a.py", nodes)
        by_line = {n.start_line: n.id for n in result}
        assert by_line[10] == "a.py::login#function"
        assert by_line[20] == "a.py::login#function~2"
        assert by_line[30] == "a.py::login#function~3"

    def test_distinct_kinds_are_not_duplicates(self):
        cls = node(name="Login", kind=Kind.CLASS, start_line=1)
        fn = node(name="Login", kind=Kind.FUNCTION, start_line=10)
        result = assign_ids("a.py", [cls, fn])
        assert all("~" not in n.id for n in result)

    def test_unique_names_never_suffixed(self):
        a = node(name="foo", start_line=1)
        b = node(name="bar", start_line=2)
        result = assign_ids("a.py", [a, b])
        assert all("~" not in n.id for n in result)


class TestLinkParents:
    def test_method_links_to_class(self):
        cls = node(name="Handler", kind=Kind.CLASS, scope_path=(), start_byte=0, end_byte=100, start_line=1)
        method = node(
            name="login", kind=Kind.METHOD, scope_path=("Handler",), start_byte=10, end_byte=50, start_line=2
        )
        with_ids = assign_ids("a.py", [cls, method])
        linked = link_parents(with_ids)
        by_name = {n.name: n for n in linked}
        assert by_name["login"].parent_id == by_name["Handler"].id

    def test_file_scope_symbol_links_to_file_node(self):
        file_node = node(name="", kind=Kind.FILE, scope_path=(), start_byte=0, end_byte=100, start_line=1)
        fn = node(name="foo", kind=Kind.FUNCTION, scope_path=(), start_byte=10, end_byte=50, start_line=2)
        with_ids = assign_ids("a.py", [file_node, fn])
        linked = link_parents(with_ids)
        by_name = {n.name: n for n in linked}
        assert by_name["foo"].parent_id == by_name[""].id

    def test_file_node_has_no_parent(self):
        file_node = node(name="", kind=Kind.FILE, scope_path=(), start_byte=0, end_byte=100, start_line=1)
        with_ids = assign_ids("a.py", [file_node])
        linked = link_parents(with_ids)
        assert linked[0].parent_id is None

    def test_deeply_nested_links_to_immediate_parent(self):
        file_node = node(name="", kind=Kind.FILE, scope_path=(), start_byte=0, end_byte=100, start_line=1)
        outer = node(name="outer", kind=Kind.FUNCTION, scope_path=(), start_byte=0, end_byte=90, start_line=1)
        inner = node(name="inner", kind=Kind.FUNCTION, scope_path=("outer",), start_byte=10, end_byte=80, start_line=2)
        innermost = node(
            name="innermost",
            kind=Kind.FUNCTION,
            scope_path=("outer", "inner"),
            start_byte=20,
            end_byte=70,
            start_line=3,
        )
        with_ids = assign_ids("a.py", [file_node, outer, inner, innermost])
        linked = link_parents(with_ids)
        by_name = {n.name: n for n in linked}
        assert by_name["innermost"].parent_id == by_name["inner"].id
        assert by_name["inner"].parent_id == by_name["outer"].id
        assert by_name["outer"].parent_id == by_name[""].id


class TestLinkRefs:
    def test_ref_resolves_to_innermost_containing_node(self):
        file_node = node(name="", kind=Kind.FILE, scope_path=(), start_byte=0, end_byte=100, start_line=1)
        method = node(
            name="login", kind=Kind.METHOD, scope_path=("Handler",), start_byte=10, end_byte=50, start_line=2
        )
        with_ids = assign_ids("a.py", [file_node, method])
        raw = RawRef(origin_byte=20, raw_name="self.validate", head="self", ref_kind=RefKind.CALLS, line=3, col=8)
        [ref] = link_refs(with_ids, [raw])
        by_name = {n.name: n for n in with_ids}
        assert ref.from_node_id == by_name["login"].id
        assert ref.raw_name == "self.validate"
        assert ref.head == "self"
        assert ref.ref_kind == RefKind.CALLS
        assert ref.line == 3
        assert ref.col == 8

    def test_ref_at_file_scope_resolves_to_file_node(self):
        file_node = node(name="", kind=Kind.FILE, scope_path=(), start_byte=0, end_byte=100, start_line=1)
        method = node(
            name="login", kind=Kind.METHOD, scope_path=("Handler",), start_byte=10, end_byte=50, start_line=2
        )
        with_ids = assign_ids("a.py", [file_node, method])
        raw = RawRef(origin_byte=60, raw_name="setup", head="setup", ref_kind=RefKind.CALLS, line=8, col=1)
        [ref] = link_refs(with_ids, [raw])
        by_name = {n.name: n for n in with_ids}
        assert ref.from_node_id == by_name[""].id

    def test_ref_matches_a_zero_width_node_at_the_exact_point(self):
        # An empty file's file node has start_byte == end_byte == 0.
        empty_file_node = node(name="", kind=Kind.FILE, scope_path=(), start_byte=0, end_byte=0, start_line=1)
        with_ids = assign_ids("a.py", [empty_file_node])
        raw = RawRef(origin_byte=0, raw_name="x", head="x", ref_kind=RefKind.CALLS, line=1, col=1)
        [ref] = link_refs(with_ids, [raw])
        assert ref.from_node_id == with_ids[0].id

    def test_ref_with_no_containing_node_falls_back_to_empty_origin(self):
        # Defensive case: shouldn't happen in practice since the file node
        # always spans the whole file, but a ref outside every node's range
        # must not raise.
        narrow = node(name="foo", start_byte=10, end_byte=20, start_line=1)
        with_ids = assign_ids("a.py", [narrow])
        raw = RawRef(origin_byte=999, raw_name="x", head="x", ref_kind=RefKind.CALLS, line=1, col=1)
        [ref] = link_refs(with_ids, [raw])
        assert ref.from_node_id == ""

    def test_every_ref_names_a_node_from_the_same_parse(self):
        file_node = node(name="", kind=Kind.FILE, scope_path=(), start_byte=0, end_byte=100, start_line=1)
        with_ids = assign_ids("a.py", [file_node])
        raws = [
            RawRef(origin_byte=5, raw_name="a", head="a", ref_kind=RefKind.CALLS, line=1, col=1),
            RawRef(origin_byte=50, raw_name="b", head="b", ref_kind=RefKind.IMPORTS, line=2, col=1),
        ]
        refs = link_refs(with_ids, raws)
        node_ids = {n.id for n in with_ids}
        assert all(r.from_node_id in node_ids for r in refs)
