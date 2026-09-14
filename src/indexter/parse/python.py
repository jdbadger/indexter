"""Python parser: functions, methods, classes, module constants, and
decorated definitions as nodes; calls, imports, and base classes as
references.

Lifted from `~/dev/indexter`'s Python parser, with the scope walk fixed to
return the full ancestor path (nested functions no longer collide -- see
design.md) and a references query added.
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, head_identifier, node_text
from indexter.parse.models import Kind, ParsedNode, RawRef, RefKind

_DEFINITIONS_QUERY = """
(function_definition
    name: (identifier) @name
) @def

(class_definition
    name: (identifier) @name
) @def

(decorated_definition
    (decorator)+
    definition: [
        (function_definition name: (identifier) @name) @inner
        (class_definition name: (identifier) @name) @inner
    ]
) @def

(module
    (assignment
        left: (identifier) @name
        right: (_)
    ) @def
)
"""

_REFERENCES_QUERY = """
(import_statement name: (dotted_name) @plain_import)
(import_statement name: (aliased_import name: (dotted_name) @aliased_module alias: (identifier) @aliased_alias))

(import_from_statement
    module_name: (_) @from_module
    name: (dotted_name) @from_name)
(import_from_statement
    module_name: (_) @from_module
    name: (aliased_import name: (dotted_name) @from_aliased_name alias: (identifier) @from_alias))
(import_from_statement
    module_name: (_) @from_module_wc
    (wildcard_import) @from_wildcard)

(call function: (identifier) @callee) @call
(call function: (attribute) @callee) @call

(class_definition
    superclasses: (argument_list (identifier) @base)
)
"""

# Ancestor node types that end a class body / function body -- the scope
# walk skips through these transparently, since they contribute no name of
# their own.
_SCOPE_FORMING = frozenset({"function_definition", "class_definition"})


def _is_constant_name(name: str) -> bool:
    """UPPER_CASE or UPPER_SNAKE_CASE, matching the existing convention."""
    return name.isupper() or ("_" in name and name.replace("_", "").isupper())


class PythonParser(BaseLanguageParser):
    language = "python"

    @property
    def definitions_query_str(self) -> str:
        return _DEFINITIONS_QUERY

    @property
    def references_query_str(self) -> str | None:
        return _REFERENCES_QUERY

    def scope_segment(self, ancestor: Node) -> str | None:
        if ancestor.type in _SCOPE_FORMING:
            name_node = ancestor.child_by_field_name("name")
            return node_text(name_node) if name_node else None
        return None

    def process_definition_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> ParsedNode | None:
        def_nodes = match.get("def")
        if not def_nodes:
            return None
        node = def_nodes[0]

        # function_definition/class_definition nested directly under a
        # decorated_definition are handled by the decorated pattern instead.
        if (
            node.type in ("function_definition", "class_definition")
            and node.parent
            and node.parent.type == "decorated_definition"
        ):
            return None

        inner_nodes = match.get("inner")
        actual_def = inner_nodes[0] if inner_nodes else node

        name_nodes = match.get("name")
        if not name_nodes:
            return None
        name = node_text(name_nodes[0])

        if actual_def.type == "assignment" and not _is_constant_name(name):
            return None

        kind = self._kind_for(actual_def)
        return ParsedNode(
            kind=kind,
            name=name,
            scope_path=self.build_scope_path(actual_def),
            language=self.language,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            signature=self._signature(actual_def, source_bytes),
            docstring=self._docstring(actual_def, source_bytes),
        )

    def process_reference_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> RawRef | None:
        if "call" in match:
            call_node = match["call"][0]
            callee = match["callee"][0]
            return RawRef(
                origin_byte=call_node.start_byte,
                raw_name=node_text(callee),
                head=head_identifier(callee),
                ref_kind=RefKind.CALLS,
                line=call_node.start_point[0] + 1,
                col=call_node.start_point[1] + 1,
            )
        if "base" in match:
            base = match["base"][0]
            return RawRef(
                origin_byte=base.start_byte,
                raw_name=node_text(base),
                head=head_identifier(base),
                ref_kind=RefKind.INHERITS,
                line=base.start_point[0] + 1,
                col=base.start_point[1] + 1,
            )
        if "plain_import" in match:
            node = match["plain_import"][0]
            return self._import_ref(node, node_text(node), head=_first_segment(node))
        if "aliased_module" in match:
            module = match["aliased_module"][0]
            alias = match["aliased_alias"][0]
            return self._import_ref(module, node_text(module), head=node_text(alias))
        if "from_name" in match:
            module = match["from_module"][0]
            name = match["from_name"][0]
            imported_name = node_text(name)
            return self._import_ref(name, node_text(module), head=imported_name, imported_name=imported_name)
        if "from_aliased_name" in match:
            module = match["from_module"][0]
            name = match["from_aliased_name"][0]
            alias = match["from_alias"][0]
            return self._import_ref(alias, node_text(module), head=node_text(alias), imported_name=node_text(name))
        if "from_wildcard" in match:
            module = match["from_module_wc"][0]
            wildcard = match["from_wildcard"][0]
            return self._import_ref(wildcard, node_text(module), head=None, imported_name="*")
        return None

    def _import_ref(self, node: Node, raw_name: str, *, head: str | None, imported_name: str | None = None) -> RawRef:
        return RawRef(
            origin_byte=node.start_byte,
            raw_name=raw_name,
            head=head,
            ref_kind=RefKind.IMPORTS,
            line=node.start_point[0] + 1,
            col=node.start_point[1] + 1,
            imported_name=imported_name,
        )

    def _kind_for(self, def_node: Node) -> Kind:
        if def_node.type == "class_definition":
            return Kind.CLASS
        if def_node.type == "assignment":
            return Kind.CONSTANT
        # function_definition: a method if its nearest enclosing definition
        # (skipping transparent wrappers) is a class, else a function --
        # including when nested inside another function.
        return Kind.METHOD if self._nearest_enclosing_type(def_node) == "class_definition" else Kind.FUNCTION

    @staticmethod
    def _nearest_enclosing_type(node: Node) -> str | None:
        current = node.parent
        while current is not None:
            if current.type in _SCOPE_FORMING:
                return current.type
            current = current.parent
        return None

    @staticmethod
    def _docstring(node: Node, source: bytes) -> str | None:
        if node.type not in ("function_definition", "class_definition"):
            return None
        body = node.child_by_field_name("body")
        if not body or not body.children:
            return None
        first = body.children[0]
        if first.type == "string" and first.text:
            return _strip_docstring(node_text(first))
        if first.type == "expression_statement" and first.children:
            expr = first.children[0]
            if expr.type == "string" and expr.text:
                return _strip_docstring(node_text(expr))
        return None

    @staticmethod
    def _signature(node: Node, source: bytes) -> str | None:
        if node.type != "function_definition":
            return None
        body = node.child_by_field_name("body")
        end = body.start_byte if body else node.end_byte
        return source[node.start_byte : end].decode().rstrip().rstrip(":")


def _first_segment(dotted_name: Node) -> str | None:
    """The leftmost identifier of a `dotted_name` node (`os.path` -> `os`)."""
    first = dotted_name.children[0] if dotted_name.child_count else None
    return node_text(first) if first is not None and first.type == "identifier" else None


def _strip_docstring(text: str) -> str:
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote) and text.endswith(quote) and len(text) >= 2 * len(quote):
            text = text[len(quote) : -len(quote)]
            break
    return text.strip()
