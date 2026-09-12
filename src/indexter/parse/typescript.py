"""TypeScript parser: functions, methods, classes, interfaces, type aliases,
enums, and constants as nodes; calls, imports, `extends`, and `implements`
as references.

Lifted from `~/dev/indexter`'s TypeScript parser. Export-wrapped
declarations need no special duplicate suppression here -- unlike the old
code, there is no competing `(export_statement) @def` pattern (exports
aren't nodes), and `export class Foo {}` matches `class_declaration`
directly regardless of the `export`/`export default` wrapper, one node
either way.
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, head_identifier, node_text
from indexter.parse.models import Kind, ParsedNode, RawRef, RefKind

_DEFINITIONS_QUERY = """
(function_declaration name: (identifier) @name) @def
(generator_function_declaration name: (identifier) @name) @def
(function_expression name: (identifier) @name) @def

(lexical_declaration
    (variable_declarator
        name: (identifier) @name
        value: (arrow_function) @func_value
    )
) @def

(class_declaration name: (type_identifier) @name) @def
(abstract_class_declaration name: (type_identifier) @name) @def

(interface_declaration name: (type_identifier) @name) @def
(type_alias_declaration name: (type_identifier) @name) @def
(enum_declaration name: (identifier) @name) @def

(method_definition name: (property_identifier) @name) @def
(method_signature name: (property_identifier) @name) @def

(lexical_declaration
    (variable_declarator
        name: (identifier) @name
        value: (_) @value
    )
) @def
"""

_REFERENCES_QUERY = """
(import_statement source: (string (string_fragment) @specifier)) @import_stmt

(call_expression
    function: (identifier) @fn
    arguments: (arguments (string (string_fragment) @specifier))
    (#eq? @fn "require")) @require_call

(call_expression function: (identifier) @callee) @call
(call_expression function: (member_expression) @callee) @call

(extends_clause value: (identifier) @base)
(implements_clause (type_identifier) @base)
"""

_CLASS_TYPES = frozenset({"class_declaration", "abstract_class_declaration"})
_NAMED_CONSTRUCTS = frozenset(
    _CLASS_TYPES
    | {
        "interface_declaration",
        "method_definition",
        "method_signature",
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
    }
)
_FUNCTION_LIKE = frozenset({"arrow_function", "function", "function_expression"})


def _is_constant_name(name: str) -> bool:
    return name.isupper() or ("_" in name and name.replace("_", "").isupper())


class TypeScriptParser(BaseLanguageParser):
    language = "typescript"

    @property
    def definitions_query_str(self) -> str:
        return _DEFINITIONS_QUERY

    @property
    def references_query_str(self) -> str | None:
        return _REFERENCES_QUERY

    def scope_segment(self, ancestor: Node) -> str | None:
        if ancestor.type in _NAMED_CONSTRUCTS:
            name_node = ancestor.child_by_field_name("name")
            return node_text(name_node) if name_node else None
        if ancestor.type == "variable_declarator":
            value = ancestor.child_by_field_name("value")
            if value is not None and value.type == "object":
                name_node = ancestor.child_by_field_name("name")
                return node_text(name_node) if name_node else None
        return None

    def process_definition_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> ParsedNode | None:
        def_nodes = match.get("def")
        if not def_nodes:
            return None
        node = def_nodes[0]
        name_nodes = match.get("name")
        if not name_nodes:
            return None
        name = node_text(name_nodes[0])

        func_value_nodes = match.get("func_value")
        if func_value_nodes:
            actual_def = func_value_nodes[0]
            kind = self._kind_for(actual_def)
        elif node.type == "lexical_declaration":
            value_nodes = match.get("value")
            value_node = value_nodes[0] if value_nodes else None
            if value_node is not None and value_node.type in _FUNCTION_LIKE:
                return None
            if not _is_constant_name(name):
                return None
            actual_def = node
            kind = Kind.CONSTANT
        else:
            actual_def = node
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
        if "import_stmt" in match:
            node = match["import_stmt"][0]
            return self._import_ref(node, node_text(match["specifier"][0]))
        if "require_call" in match:
            node = match["require_call"][0]
            return self._import_ref(node, node_text(match["specifier"][0]))
        return None

    @staticmethod
    def _import_ref(node: Node, specifier: str) -> RawRef:
        return RawRef(
            origin_byte=node.start_byte,
            raw_name=specifier,
            head=None,
            ref_kind=RefKind.IMPORTS,
            line=node.start_point[0] + 1,
            col=node.start_point[1] + 1,
        )

    def _kind_for(self, def_node: Node) -> Kind:
        if def_node.type in _CLASS_TYPES:
            return Kind.CLASS
        if def_node.type == "interface_declaration":
            return Kind.INTERFACE
        if def_node.type == "type_alias_declaration":
            return Kind.TYPE_ALIAS
        if def_node.type == "enum_declaration":
            return Kind.ENUM
        if def_node.type in ("method_definition", "method_signature"):
            return Kind.METHOD
        return Kind.METHOD if self._nearest_enclosing_type(def_node) in _CLASS_TYPES else Kind.FUNCTION

    @staticmethod
    def _nearest_enclosing_type(node: Node) -> str | None:
        current = node.parent
        while current is not None:
            if current.type in _NAMED_CONSTRUCTS:
                return current.type
            current = current.parent
        return None

    @staticmethod
    def _signature(node: Node, source: bytes) -> str | None:
        if node.type == "arrow_function":
            params = node.child_by_field_name("parameters") or node.child_by_field_name("parameter")
            if params is not None:
                for child in node.children:
                    if child.type == "=>":
                        return source[params.start_byte : child.end_byte].decode().strip()
            body = node.child_by_field_name("body")
            if body is not None:
                return source[node.start_byte : body.start_byte].decode().strip()
            return None
        if node.type not in (
            "function_declaration",
            "generator_function_declaration",
            "function_expression",
            "method_definition",
            "method_signature",
        ):
            return None
        body = node.child_by_field_name("body")
        end = body.start_byte if body else node.end_byte
        return source[node.start_byte : end].decode().strip()
