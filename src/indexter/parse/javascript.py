"""JavaScript parser: functions, methods, classes, and (non-uppercase-only)
constants as nodes; calls, ES imports, `require()`, and `extends` as
references.

Lifted from `~/dev/indexter`'s JavaScript parser, with the scope walk fixed
to return the full ancestor path -- a named callback inside a method now
scopes to `Class.method`, and an object-literal method scopes to its
variable's name -- both previously landed at file scope. Import/export
statements no longer produce nodes (imports/exports are edges, not nodes).
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

(class_declaration name: (identifier) @name) @def

(method_definition name: (property_identifier) @name) @def

(lexical_declaration
    (variable_declarator
        name: (identifier) @name
        value: (_) @value
    )
) @def
"""

_REFERENCES_QUERY = """
(import_clause (identifier) @default_name)
(import_clause (namespace_import (identifier) @ns_name))
(import_specifier name: (identifier) @named_name alias: (identifier)? @named_alias)
(import_statement source: (string (string_fragment) @side_effect_specifier)) @side_effect_import

(export_specifier name: (identifier) @export_name alias: (identifier)? @export_alias)
(export_statement "*" source: (string (string_fragment) @export_star_specifier))
(export_statement
    (namespace_export (identifier) @export_ns_name)
    source: (string (string_fragment) @export_ns_specifier))

(call_expression
    function: (identifier) @fn
    arguments: (arguments (string (string_fragment) @specifier))
    (#eq? @fn "require")) @require_call

(call_expression function: (identifier) @callee) @call
(call_expression function: (member_expression) @callee) @call

(class_heritage (identifier) @base)
"""

# Node types recognized when walking ancestors for both scope-path segments
# and method-vs-function kind determination.
_NAMED_CONSTRUCTS = frozenset(
    {
        "class_declaration",
        "method_definition",
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
    }
)
_FUNCTION_LIKE = frozenset({"arrow_function", "function", "function_expression"})


def _is_constant_name(name: str) -> bool:
    return name.isupper() or ("_" in name and name.replace("_", "").isupper())


def _enclosing(node: Node, type_name: str) -> Node | None:
    current = node.parent
    while current is not None and current.type != type_name:
        current = current.parent
    return current


def _import_source(stmt: Node | None) -> str | None:
    """The module specifier text an `import_statement`/`export_statement`'s
    `source` field carries, or `None` when there is no such field (a local
    `export { x };` with no module).
    """
    if stmt is None:
        return None
    source = stmt.child_by_field_name("source")
    if source is None:
        return None
    for child in source.children:
        if child.type == "string_fragment":
            return node_text(child)
    return ""


def _has_import_clause(stmt: Node) -> bool:
    return any(child.type == "import_clause" for child in stmt.children)


def _assigned_variable(call_node: Node) -> str | None:
    """The variable name a `require(...)` call is assigned to, if any."""
    parent = call_node.parent
    if parent is None or parent.type != "variable_declarator":
        return None
    if parent.child_by_field_name("value") != call_node:
        return None
    name_node = parent.child_by_field_name("name")
    return node_text(name_node) if name_node is not None else None


class JavaScriptParser(BaseLanguageParser):
    language = "javascript"

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
            # Only a real scope boundary when the declarator wraps a
            # container (an object literal) that something else is nested
            # inside -- not when the declarator's value IS the definition
            # itself (e.g. `const double = () => ...`), which would
            # otherwise put a function's own name in its own scope path.
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
            # The generic "any declarator value" pattern -- skip anything
            # already handled by the func_value pattern above, so a
            # `const fn = () => {}` isn't double-emitted.
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
            docstring=self._jsdoc(node, source_bytes),
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
        if "default_name" in match:
            name = match["default_name"][0]
            source = _import_source(_enclosing(name, "import_statement"))
            return self._import_ref(name, source, head=node_text(name), imported_name="default")
        if "ns_name" in match:
            name = match["ns_name"][0]
            source = _import_source(_enclosing(name, "import_statement"))
            return self._import_ref(name, source, head=node_text(name))
        if "named_name" in match:
            name = match["named_name"][0]
            alias = match.get("named_alias")
            source = _import_source(_enclosing(name, "import_statement"))
            imported = node_text(name)
            head = node_text(alias[0]) if alias else imported
            return self._import_ref(name, source, head=head, imported_name=imported)
        if "side_effect_import" in match:
            stmt = match["side_effect_import"][0]
            if _has_import_clause(stmt):
                return None  # handled per-binding by the patterns above
            specifier = match["side_effect_specifier"][0]
            return self._import_ref(stmt, node_text(specifier), head=None)
        if "export_name" in match:
            name = match["export_name"][0]
            source = _import_source(_enclosing(name, "export_statement"))
            if source is None:
                return None  # a local re-export, not a module reference
            alias = match.get("export_alias")
            imported = node_text(name)
            head = node_text(alias[0]) if alias else imported
            return self._import_ref(name, source, head=head, imported_name=imported)
        if "export_star_specifier" in match:
            specifier = match["export_star_specifier"][0]
            stmt = _enclosing(specifier, "export_statement")
            return self._import_ref(stmt or specifier, node_text(specifier), head=None, imported_name="*")
        if "export_ns_name" in match:
            name = match["export_ns_name"][0]
            specifier = match["export_ns_specifier"][0]
            return self._import_ref(name, node_text(specifier), head=node_text(name))
        if "require_call" in match:
            call_node = match["require_call"][0]
            specifier = match["specifier"][0]
            head = _assigned_variable(call_node)
            return self._import_ref(call_node, node_text(specifier), head=head)
        return None

    @staticmethod
    def _import_ref(
        node: Node, raw_name: str | None, *, head: str | None, imported_name: str | None = None
    ) -> RawRef:
        return RawRef(
            origin_byte=node.start_byte,
            raw_name=raw_name or "",
            head=head,
            ref_kind=RefKind.IMPORTS,
            line=node.start_point[0] + 1,
            col=node.start_point[1] + 1,
            imported_name=imported_name,
        )

    def _kind_for(self, def_node: Node) -> Kind:
        if def_node.type == "class_declaration":
            return Kind.CLASS
        if def_node.type == "method_definition":
            return Kind.METHOD
        return Kind.METHOD if self._nearest_enclosing_type(def_node) == "class_declaration" else Kind.FUNCTION

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
        ):
            return None
        body = node.child_by_field_name("body")
        end = body.start_byte if body else node.end_byte
        return source[node.start_byte : end].decode().strip()

    @staticmethod
    def _jsdoc(node: Node, source_bytes: bytes) -> str | None:
        parent = node.parent
        if parent is None:
            return None
        for i, child in enumerate(parent.children):
            if child == node:
                if i > 0:
                    prev = parent.children[i - 1]
                    if prev.type == "comment" and prev.text:
                        return _parse_jsdoc(node_text(prev))
                break
        return None


def _parse_jsdoc(comment: str) -> str | None:
    if not (comment.startswith("/**") and comment.endswith("*/")):
        return None
    content = comment[3:-2]
    lines = []
    for line in content.split("\n"):
        cleaned = line.strip()
        if cleaned.startswith("*"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines) if lines else None
