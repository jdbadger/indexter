import tree_sitter_language_pack as tlp
from tree_sitter import Node

from indexter.parse import base as base_module
from indexter.parse.base import BaseLanguageParser, head_identifier, parse_file, register_parser
from indexter.parse.models import Kind, ParsedNode, ParsedRef, ParseResult, RawRef, RefKind


def _first_call_function(lang: str, src: bytes) -> Node:
    """Find the first node exposing a `function` field, in document order --
    for a chained call this is the outermost call.
    """
    parser = tlp.get_parser(lang)
    tree = parser.parse(src)
    found: list[Node] = []

    def walk(n: Node) -> None:
        fn = n.child_by_field_name("function")
        if fn is not None:
            found.append(fn)
        for child in n.children:
            walk(child)

    walk(tree.root_node)
    return found[0]


class TestHeadIdentifier:
    def test_bare_call(self):
        node = _first_call_function("python", b"foo(x)\n")
        assert head_identifier(node) == "foo"

    def test_python_attribute_chain(self):
        node = _first_call_function("python", b"os.path.join(a, b)\n")
        assert head_identifier(node) == "os"

    def test_python_self(self):
        node = _first_call_function("python", b"self.validate(user)\n")
        assert head_identifier(node) == "self"

    def test_js_member_chain(self):
        node = _first_call_function("javascript", b"os.path.join(a, b);\n")
        assert head_identifier(node) == "os"

    def test_js_this(self):
        node = _first_call_function("javascript", b"this.x.y();\n")
        assert head_identifier(node) == "this"

    def test_js_chained_call_has_no_head(self):
        node = _first_call_function("javascript", b"build().run();\n")
        assert head_identifier(node) is None

    def test_rust_field_chain_self(self):
        src = b"fn f() { self.x.y(); }"
        node = _first_call_function("rust", src)
        assert head_identifier(node) == "self"

    def test_rust_scoped_identifier(self):
        src = b"fn f() { Self::new(); }"
        node = _first_call_function("rust", src)
        assert head_identifier(node) == "Self"

    def test_literal_argument_has_no_head(self):
        # The argument itself (not the call) -- a plain literal is not a chain at all.
        parser = tlp.get_parser("python")
        tree = parser.parse(b"foo(5)\n")
        found = []

        def walk(n):
            if n.type == "integer":
                found.append(n)
            for c in n.children:
                walk(c)

        walk(tree.root_node)
        assert head_identifier(found[0]) is None

    def test_subscript_base_has_no_head(self):
        node = _first_call_function("python", b"arr[0].method()\n")
        assert head_identifier(node) is None

    def test_descend_field_missing_returns_none(self):
        # An `attribute` node whose `object` field is somehow absent -- the
        # inner defensive branch, not reachable through normal source text.
        class _NoObjectField:
            type = "attribute"

            def child_by_field_name(self, name):
                return None

        assert head_identifier(_NoObjectField()) is None


class _StubParser(BaseLanguageParser):
    """Minimal concrete parser used to exercise the framework mechanics
    without depending on the real (not-yet-lifted) language parsers.
    """

    language = "python"
    compile_count: int = 0

    def __init__(self) -> None:
        _StubParser.compile_count += 1
        super().__init__()

    @property
    def definitions_query_str(self) -> str:
        return "(function_definition name: (identifier) @name) @def"

    @property
    def references_query_str(self) -> str | None:
        return "(call function: (identifier) @callee) @call"

    def process_definition_match(self, match, source_bytes):
        def_node = match["def"][0]
        name = match["name"][0].text.decode()
        return ParsedNode(
            kind=Kind.FUNCTION,
            name=name,
            scope_path=self.build_scope_path(def_node),
            language=self.language,
            start_line=def_node.start_point[0] + 1,
            end_line=def_node.end_point[0] + 1,
            start_byte=def_node.start_byte,
            end_byte=def_node.end_byte,
        )

    def process_reference_match(self, match, source_bytes):
        call_node = match["call"][0]
        callee = match["callee"][0]
        return RawRef(
            origin_byte=call_node.start_byte,
            raw_name=callee.text.decode(),
            head=head_identifier(callee),
            ref_kind=RefKind.CALLS,
            line=call_node.start_point[0] + 1,
            col=call_node.start_point[1] + 1,
        )


class _RaisingDefinitionParser(BaseLanguageParser):
    language = "python"

    @property
    def definitions_query_str(self) -> str:
        return "(function_definition name: (identifier) @name) @def"

    def process_definition_match(self, match, source_bytes):
        raise RuntimeError("boom")


class _ScopedStubParser(_StubParser):
    """Overrides scope_segment to actually contribute a segment, for testing
    build_scope_path's append branch.
    """

    def scope_segment(self, ancestor):
        if ancestor.type == "block":
            return "inside-a-block"
        return None


class _QuietReferenceParser(BaseLanguageParser):
    """references_query_str is set but process_reference_match is left at
    its default (returns None) -- every match is dropped.
    """

    language = "python"

    @property
    def definitions_query_str(self) -> str:
        return "(function_definition name: (identifier) @name) @def"

    @property
    def references_query_str(self) -> str | None:
        return "(call function: (identifier) @callee) @call"

    def process_definition_match(self, match, source_bytes):
        return None


def _reset_stub_compile_count():
    _StubParser.compile_count = 0


class TestBaseLanguageParser:
    def test_query_compiled_once_across_several_parses(self):
        _reset_stub_compile_count()
        parser = _StubParser()
        parser.parse("a.py", "def a(): pass\n")
        parser.parse("a.py", "def b(): pass\n")
        parser.parse("a.py", "def c(): pass\n")
        assert _StubParser.compile_count == 1

    def test_repeated_parses_are_independent(self):
        parser = _StubParser()
        result1 = parser.parse("a.py", "def one(): pass\n")
        result2 = parser.parse("a.py", "def two(): pass\n")
        names1 = {n.name for n in result1.nodes}
        names2 = {n.name for n in result2.nodes}
        assert "one" in names1
        assert "one" not in names2
        assert "two" in names2
        assert "two" not in names1

    def test_file_node_is_always_present(self):
        parser = _StubParser()
        result = parser.parse("a.py", "x = 1\n")
        file_nodes = [n for n in result.nodes if n.kind == Kind.FILE]
        assert len(file_nodes) == 1
        assert file_nodes[0].start_byte == 0
        assert file_nodes[0].end_byte == len(b"x = 1\n")

    def test_empty_file_yields_cleanly(self):
        parser = _StubParser()
        result = parser.parse("a.py", "")
        assert result.errors == []
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_syntax_error_file_still_yields_parsed_constructs(self):
        parser = _StubParser()
        # tree-sitter is error-tolerant: the well-formed function still parses
        # even though the file overall has a syntax error.
        result = parser.parse("a.py", "def good(): pass\ndef bad(:::\n")
        names = {n.name for n in result.nodes}
        assert "good" in names

    def test_raising_definition_handler_is_contained(self):
        parser = _RaisingDefinitionParser()
        result = parser.parse("a.py", "def one(): pass\ndef two(): pass\n")
        assert result.errors  # the failure was recorded
        assert any("boom" in e for e in result.errors)
        # File node still comes through even though every definition match raised.
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_scope_segment_default_is_no_scoping(self):
        parser = _StubParser()
        result = parser.parse("a.py", "class C:\n    def m(self): pass\n")
        # _StubParser doesn't override scope_segment, so even a method inside
        # a class (which the stub's query doesn't even capture as a class)
        # gets an empty scope path from build_scope_path by default.
        method = next(n for n in result.nodes if n.name == "m")
        assert method.scope_path == ()

    def test_references_are_linked_to_their_origin(self):
        parser = _StubParser()
        result = parser.parse("a.py", "def outer():\n    helper()\n")
        outer = next(n for n in result.nodes if n.name == "outer")
        [ref] = result.refs
        assert ref.from_node_id == outer.id
        assert ref.raw_name == "helper"
        assert ref.head == "helper"

    def test_missing_language_raises(self):
        class _NoLanguage(BaseLanguageParser):
            @property
            def definitions_query_str(self):
                return ""

            def process_definition_match(self, match, source_bytes):
                return None

        try:
            _NoLanguage()
        except ValueError as e:
            assert "language" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_scope_segment_override_contributes_a_segment(self):
        parser = _ScopedStubParser()
        result = parser.parse("a.py", "def outer():\n    def inner(): pass\n")
        inner = next(n for n in result.nodes if n.name == "inner")
        assert inner.scope_path == ("inside-a-block",)

    def test_default_process_reference_match_drops_every_match(self):
        parser = _QuietReferenceParser()
        result = parser.parse("a.py", "def outer():\n    helper()\n")
        assert result.refs == []

    def test_tree_sitter_crash_is_recorded_as_an_error(self, monkeypatch):
        parser = _StubParser()

        class _CrashingTsParser:
            def parse(self, *args, **kwargs):
                raise RuntimeError("simulated tree-sitter crash")

        monkeypatch.setattr(parser, "_ts_parser", _CrashingTsParser())
        result = parser.parse("a.py", "def foo(): pass\n")
        assert result.errors
        assert any("simulated tree-sitter crash" in e for e in result.errors)
        assert [n.kind for n in result.nodes] == [Kind.FILE]

    def test_query_cursor_crash_is_recorded_as_an_error(self, monkeypatch):
        parser = _StubParser()

        def boom_cursor(query):
            raise RuntimeError("cursor boom")

        monkeypatch.setattr(base_module, "QueryCursor", boom_cursor)
        result = parser.parse("a.py", "def foo(): pass\n")
        assert any("cursor boom" in e for e in result.errors)


class TestRegistryAndDispatch:
    def test_case_insensitive_extension_match(self, monkeypatch):
        monkeypatch.setitem(base_module._EXTENSION_REGISTRY, ".stub", _StubParser)
        result = parse_file("A.STUB", "def foo(): pass\n")
        assert any(n.name == "foo" for n in result.nodes)

    def test_parser_instances_are_reused(self, monkeypatch):
        monkeypatch.setitem(base_module._EXTENSION_REGISTRY, ".stub", _StubParser)
        monkeypatch.setattr(base_module, "_instances", {})
        parse_file("a.stub", "def a(): pass\n")
        parse_file("b.stub", "def b(): pass\n")
        assert len(base_module._instances) == 1

    def test_unregistered_extension_falls_back_to_chunking(self):
        result = parse_file("a.zzz-not-registered", "some content that is not code\n")
        assert any(n.kind == Kind.CHUNK for n in result.nodes)

    def test_register_parser_helper(self):
        register_parser([".teststub"], _StubParser)
        try:
            assert base_module._EXTENSION_REGISTRY[".teststub"] is _StubParser
        finally:
            del base_module._EXTENSION_REGISTRY[".teststub"]


class TestParseResultType:
    def test_parse_file_returns_parse_result(self):
        result = parse_file("a.zzz", "text\n")
        assert isinstance(result, ParseResult)


class TestDropUnshadowedBuiltins:
    def _ref(self, head, ref_kind=RefKind.CALLS, raw_name="x", imported_name=None):
        return ParsedRef(
            from_node_id="a.py::f#function",
            raw_name=raw_name,
            head=head,
            ref_kind=ref_kind,
            line=1,
            col=1,
            imported_name=imported_name,
        )

    def test_unknown_language_is_a_no_op(self):
        refs = [self._ref("len")]
        assert base_module._drop_unshadowed_builtins("cobol", [], refs) == refs

    def test_builtin_call_dropped(self):
        refs = [self._ref("len")]
        assert base_module._drop_unshadowed_builtins("python", [], refs) == []

    def test_non_builtin_call_kept(self):
        refs = [self._ref("helper")]
        assert base_module._drop_unshadowed_builtins("python", [], refs) == refs

    def test_ref_with_no_head_kept(self):
        refs = [self._ref(None)]
        assert base_module._drop_unshadowed_builtins("python", [], refs) == refs

    def test_import_never_dropped(self):
        refs = [self._ref("len", ref_kind=RefKind.IMPORTS)]
        assert base_module._drop_unshadowed_builtins("python", [], refs) == refs

    def test_builtin_kept_when_file_defines_it(self):
        node = ParsedNode(
            kind=Kind.FUNCTION,
            name="open",
            scope_path=(),
            language="python",
            start_line=1,
            end_line=1,
            start_byte=0,
            end_byte=1,
        )
        refs = [self._ref("open")]
        assert base_module._drop_unshadowed_builtins("python", [node], refs) == refs

    def test_builtin_kept_when_imported(self):
        refs = [self._ref("filter", ref_kind=RefKind.IMPORTS, imported_name="filter"), self._ref("filter")]
        result = base_module._drop_unshadowed_builtins("python", [], refs)
        assert result == refs
