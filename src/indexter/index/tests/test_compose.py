"""Tests for `index/compose.py`.

Most tests parse real (tiny, inline) source snippets rather than hand-built
`ParsedNode`s, since the byte-range slicing compose.py relies on only means
something when it lines up with what a real parser produced.

See `test_compose_snapshots.py` for whole-output snapshots over every M2 fixture file.
"""

from __future__ import annotations

from pathlib import Path

from indexter.config import Settings
from indexter.index.compose import (
    INDEX_FORMAT_VERSION,
    _truncate,
    compose_file,
    split_docstring,
    split_identifier,
    split_path_words,
)
from indexter.index.embed import FakeEmbedder
from indexter.parse.base import parse_file
from indexter.parse.chunk import ChunkParser

FIXTURES = Path(__file__).parent.parent.parent / "parse" / "tests" / "fixtures"
TOKENIZER = FakeEmbedder().tokenizer()


def _compose(relpath: str, content: str, budget: int = 256):
    result = parse_file(relpath, content)
    return result, compose_file(relpath, content, result, TOKENIZER, budget)


def _find(result, name: str, kind=None):
    for node in result.nodes:
        if node.name == name and (kind is None or node.kind == kind):
            return node
    raise AssertionError(f"no node named {name!r} (kind={kind})")


class TestFormatVersion:
    def test_pinned(self):
        # Bump this alongside a real composer format change -- and
        # regenerate test_compose_snapshots.py's values.
        assert INDEX_FORMAT_VERSION == 2


class TestIdentifierSplitting:
    def test_snake_case(self):
        assert split_identifier("get_user_by_email") == ["get", "user", "by", "email"]

    def test_kebab_case(self):
        assert split_identifier("my-component-name") == ["my", "component", "name"]

    def test_camel_case(self):
        assert split_identifier("getUserByEmail") == ["get", "user", "by", "email"]

    def test_pascal_case(self):
        assert split_identifier("UserAccount") == ["user", "account"]

    def test_acronym_run_then_capitalized_word(self):
        assert split_identifier("XMLParser") == ["xml", "parser"]

    def test_acronym_run_and_digit(self):
        assert split_identifier("HTTPServer2") == ["http", "server", "2"]

    def test_single_lowercase_word(self):
        assert split_identifier("login") == ["login"]

    def test_pure_acronym(self):
        assert split_identifier("XML") == ["xml"]

    def test_leading_digits(self):
        assert split_identifier("2fast") == ["2", "fast"]

    def test_multi_digit_run(self):
        assert split_identifier("item42") == ["item", "42"]

    def test_empty_string(self):
        assert split_identifier("") == []

    def test_path_words_drops_extension(self):
        assert split_path_words("src/auth/handlers.py") == ["src", "auth", "handlers"]

    def test_path_words_single_segment(self):
        assert split_path_words("README.md") == ["readme"]

    def test_path_words_no_extension(self):
        assert split_path_words("Makefile") == ["makefile"]

    def test_path_words_split_hyphenated_filename(self):
        assert split_path_words("src/my-component.tsx") == ["src", "my", "component"]


class TestDocstringSplitting:
    def test_google_style_args_returns(self):
        doc = "Authenticate a user.\n\nArgs:\n    user: the user.\n\nReturns:\n    bool: whether it worked."
        prose, structured = split_docstring(doc)
        assert prose == "Authenticate a user."
        assert structured.startswith("Args:")
        assert "Returns:" in structured

    def test_numpy_style_parameters(self):
        doc = "Do a thing.\n\nParameters\n----------\nx : int\n    a value."
        prose, structured = split_docstring(doc)
        assert prose == "Do a thing."
        assert structured.startswith("Parameters")

    def test_jsdoc_tags(self):
        doc = "Handles the event.\n@param {Event} event - the event.\n@returns {void}"
        prose, structured = split_docstring(doc)
        assert prose == "Handles the event."
        assert structured.startswith("@param")

    def test_rustdoc_sections(self):
        doc = "Compute a sum.\n\n# Arguments\n\n* `a` - first.\n\n# Errors\n\nNever."
        prose, structured = split_docstring(doc)
        assert prose == "Compute a sum."
        assert "# Arguments" in structured
        assert "# Errors" in structured

    def test_no_structured_block_is_all_prose(self):
        doc = "Just a plain docstring with no sections at all."
        prose, structured = split_docstring(doc)
        assert prose == doc
        assert structured == ""

    def test_empty_docstring(self):
        assert split_docstring("") == ("", "")

    def test_purely_structured_docstring(self):
        doc = "Args:\n    x: a value."
        prose, structured = split_docstring(doc)
        assert prose == ""
        assert structured.startswith("Args:")

    def test_unrecognized_markdown_header_is_not_structured(self):
        doc = "Overview.\n\n# Overview\n\nMore prose."
        prose, structured = split_docstring(doc)
        assert structured == ""
        assert prose == doc


class TestSectionOrder:
    def test_documented_method_section_order(self):
        content = (
            "class Handler:\n"
            "    def login(self, user):\n"
            '        """Authenticate a user.\n\n'
            "        Args:\n"
            "            user: the user to authenticate.\n"
            '        """\n'
            "        return True\n"
        )
        result, composed = _compose("h.py", content)
        node = _find(result, "login")
        text = composed[node.id].embed_text
        lines = text.split("\n")
        assert lines[0].startswith("method Handler.login |")
        assert "def login(self, user)" in lines[1]
        assert text.index("Authenticate a user.") < text.index("Args:")
        assert text.index("return True") < text.index("Args:")

    def test_body_excludes_docstring(self):
        content = 'def standalone():\n    """A one-line docstring."""\n    return 42\n'
        result, composed = _compose("m.py", content)
        node = _find(result, "standalone")
        text = composed[node.id].embed_text
        assert text.count("A one-line docstring.") == 1
        assert "return 42" in text

    def test_string_literals_kept_in_body(self):
        content = 'def greet():\n    return "hello there"\n'
        result, composed = _compose("m.py", content)
        node = _find(result, "greet")
        assert '"hello there"' in composed[node.id].embed_text

    def test_blank_line_runs_collapsed(self):
        content = "def spaced():\n    a = 1\n\n\n\n    b = 2\n"
        result, composed = _compose("m.py", content)
        node = _find(result, "spaced")
        text = composed[node.id].embed_text
        assert "\n\n\n" not in text

    def test_jsdoc_structured_block_is_last_section(self):
        content = (
            "/**\n"
            " * Adds two numbers.\n"
            " * @param {number} a - the first number.\n"
            " * @returns {number} the sum.\n"
            " */\n"
            "function add(a, b) {\n"
            "  return a + b;\n"
            "}\n"
        )
        result, composed = _compose("m.js", content)
        node = _find(result, "add")
        text = composed[node.id].embed_text
        sections = text.split("\n")
        assert sections[-1].startswith("@param") or "@param" in text
        assert text.index("Adds two numbers.") < text.index("@param")
        assert text.rstrip().endswith("the sum.")

    def test_rustdoc_structured_block_is_last_section(self):
        content = (
            "/// Computes a sum.\n"
            "///\n"
            "/// # Errors\n"
            "///\n"
            "/// Never fails.\n"
            "pub fn add(a: i32, b: i32) -> i32 {\n"
            "    a + b\n"
            "}\n"
        )
        result, composed = _compose("m.rs", content)
        node = _find(result, "add")
        text = composed[node.id].embed_text
        assert text.index("Computes a sum.") < text.index("# Errors")
        assert text.rstrip().endswith("Never fails.")


class TestPerKindVariants:
    def test_container_lists_members_by_kind(self):
        content = "class Handler:\n    def login(self):\n        pass\n\n    def logout(self):\n        pass\n"
        result, composed = _compose("m.py", content)
        node = _find(result, "Handler")
        text = composed[node.id].embed_text
        assert "methods: login, logout" in text
        assert "pass" not in text

    def test_file_lists_top_level_symbols_then_residue(self):
        content = "import os\n\ndef standalone():\n    return os.getcwd()\n"
        result, composed = _compose("m.py", content)
        file_node = next(n for n in result.nodes if n.kind.value == "file")
        text = composed[file_node.id].embed_text
        assert "functions: standalone" in text
        assert "import os" in text
        assert "return os.getcwd()" not in text  # inside the function, not file residue

    def test_section_uses_heading_and_prose(self):
        content = "# Title\n\nSome introductory prose.\n\n## Sub\n\nMore text.\n"
        result, composed = _compose("m.md", content)
        node = _find(result, "Title")
        text = composed[node.id].embed_text
        assert text.startswith("section Title |")
        assert "Some introductory prose." in text

    def test_data_uses_key_path_and_slice(self):
        content = '{"settings": {"enabled": true}}'
        result, composed = _compose("m.json", content)
        node = _find(result, "settings")
        composed_node = composed[node.id]
        assert composed_node.qualified_name == "settings"
        assert "enabled" in composed_node.embed_text

    def test_chunk_uses_path_line_range_and_raw_text(self):
        settings = Settings(chunk_size=40, chunk_overlap=5)
        content = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        relpath = "notes.txt"
        result = ChunkParser(settings).parse(relpath, content)
        composed = compose_file(relpath, content, result, TOKENIZER, 256)
        chunk_node = next(n for n in result.nodes if n.kind.value == "chunk")
        c = composed[chunk_node.id]
        assert c.qualified_name == f"{relpath}:{chunk_node.start_line}-{chunk_node.end_line}"
        assert c.embed_text.startswith(c.qualified_name + "\n")
        assert "alpha" in c.embed_text


class TestFtsResidue:
    def test_method_residue_excludes_nested_function(self):
        content = Path(FIXTURES / "python" / "scopes.py").read_text()
        result = parse_file("python/scopes.py", content)
        composed = compose_file("python/scopes.py", content, result, TOKENIZER, 256)
        outer = _find(result, "outer_a")
        body = composed[outer.id].body
        assert 'return "a"' not in body
        assert "return inner" in body

    def test_class_residue_excludes_method_bodies(self):
        content = Path(FIXTURES / "python" / "sample.py").read_text()
        result = parse_file("python/sample.py", content)
        composed = compose_file("python/sample.py", content, result, TOKENIZER, 256)
        handler = _find(result, "Handler")
        body = composed[handler.id].body
        assert "self.validate(user)" not in body
        assert "Handles requests" in body

    def test_file_residue_excludes_top_level_symbols(self):
        content = Path(FIXTURES / "python" / "sample.py").read_text()
        result = parse_file("python/sample.py", content)
        composed = compose_file("python/sample.py", content, result, TOKENIZER, 256)
        file_node = next(n for n in result.nodes if n.kind.value == "file")
        body = composed[file_node.id].body
        assert "import os" in body
        assert "class Handler" not in body

    def test_script_style_file_indexes_whole_content_on_file_node(self):
        content = "0123456789" * 4  # 40 bytes: two contiguous, non-overlapping chunks
        relpath = "notes.txt"
        result = ChunkParser(Settings(chunk_size=20, chunk_overlap=0)).parse(relpath, content)
        composed = compose_file(relpath, content, result, TOKENIZER, 256)
        file_node = next(n for n in result.nodes if n.kind.value == "file")
        chunk_nodes = [n for n in result.nodes if n.kind.value == "chunk"]
        assert len(chunk_nodes) == 2
        # The chunks fully cover the file with no gap, so its own residue is empty.
        assert composed[file_node.id].body == ""
        assert content in composed[chunk_nodes[0].id].body + composed[chunk_nodes[1].id].body


class TestReproducibility:
    def test_same_input_composes_identically(self):
        content = Path(FIXTURES / "python" / "sample.py").read_text()
        result = parse_file("python/sample.py", content)
        first = compose_file("python/sample.py", content, result, TOKENIZER, 256)
        second = compose_file("python/sample.py", content, result, TOKENIZER, 256)
        assert {k: v.embed_text for k, v in first.items()} == {k: v.embed_text for k, v in second.items()}
        assert {k: v.embed_hash for k, v in first.items()} == {k: v.embed_hash for k, v in second.items()}


class TestLineShiftInvariance:
    def test_unaffected_node_text_survives_a_line_shift(self):
        before = "def first():\n    return 1\n\n\ndef second():\n    return 2\n"
        after = "def first():\n    return 1\n\n\n\n\ndef second():\n    return 2\n"
        result_before, composed_before = _compose("m.py", before)
        result_after, composed_after = _compose("m.py", after)
        second_before = _find(result_before, "second")
        second_after = _find(result_after, "second")
        assert second_before.id == second_after.id
        assert second_before.start_line != second_after.start_line
        assert composed_before[second_before.id].embed_text == composed_after[second_after.id].embed_text
        assert composed_before[second_before.id].embed_hash == composed_after[second_after.id].embed_hash


_ARGS_DOC = (
    "def calc(x):\n"
    '    """Compute something.\n\n'
    "    Args:\n"
    "        x: a number one two three four five six seven eight nine ten.\n"
    '    """\n'
    "    step_one_alpha\n"
    "    step_two_beta\n"
    "    step_three_gamma\n"
    "    step_four_delta\n"
    "    step_five_epsilon\n"
)
_NO_ARGS_DOC = (
    "def calc(x):\n"
    '    """Compute something."""\n'
    "    step_one_alpha\n"
    "    step_two_beta\n"
    "    step_three_gamma\n"
    "    step_four_delta\n"
    "    step_five_epsilon\n"
)


class TestTruncation:
    def test_within_budget_is_untouched(self):
        result, composed = _compose("m.py", _ARGS_DOC, budget=256)
        node = _find(result, "calc")
        text = composed[node.id].embed_text
        assert "Args:" in text
        assert "step_five_epsilon" in text
        ids = TOKENIZER.encode(text, add_special_tokens=False).ids
        assert len(ids) <= 254

    def test_over_budget_drops_structured_block_first(self):
        result_full, composed_full = _compose("m.py", _ARGS_DOC, budget=256)
        node_full = _find(result_full, "calc")
        full_text = composed_full[node_full.id].embed_text

        result_prefix, composed_prefix = _compose("m.py", _NO_ARGS_DOC, budget=256)
        node_prefix = _find(result_prefix, "calc")
        prefix_text = composed_prefix[node_prefix.id].embed_text
        prefix_ids = TOKENIZER.encode(prefix_text, add_special_tokens=False).ids

        budget = len(prefix_ids) + 2
        assert budget < len(TOKENIZER.encode(full_text, add_special_tokens=False).ids) + 2

        _, composed_at_budget = _compose("m.py", _ARGS_DOC, budget=budget)
        truncated = composed_at_budget[node_full.id].embed_text
        assert "Args:" not in truncated
        assert "step_five_epsilon" in truncated
        assert len(TOKENIZER.encode(truncated, add_special_tokens=False).ids) <= budget - 2

    def test_over_budget_also_cuts_the_body(self):
        result, composed_full = _compose("m.py", _ARGS_DOC, budget=256)
        node = _find(result, "calc")

        _, composed_small = _compose("m.py", _ARGS_DOC, budget=10)
        truncated = composed_small[node.id].embed_text
        full_text = composed_full[node.id].embed_text

        assert truncated != full_text
        assert "step_five_epsilon" not in truncated
        ids = TOKENIZER.encode(truncated, add_special_tokens=False).ids
        assert len(ids) <= 8
        # Re-tokenizing the result must land back within budget.
        assert full_text.startswith(truncated)

    def test_truncate_empty_text_is_a_no_op(self):
        assert _truncate("", TOKENIZER, 256) == ""

    def test_truncate_zero_budget_yields_empty_text(self):
        assert _truncate("one two three", TOKENIZER, 2) == ""


