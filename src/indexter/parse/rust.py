"""Rust parser: functions, methods, structs, enums, traits, and constants as
nodes; calls, `use` declarations, and `impl ... for ...` as references.

Lifted from `~/dev/indexter`'s Rust parser. `impl` blocks and modules are
not nodes themselves (no `impl`/`module` kind exists in the closed
vocabulary) -- an `impl` block only contributes a scope segment to the
methods defined inside it, and that segment includes the implemented
trait's last path segment: `impl Display for Foo` and `impl Debug for Foo`
both defining `fmt` previously collided into one `Foo.fmt` identity; they
now land at `Foo<Display>.fmt` and `Foo<Debug>.fmt`.
"""

from __future__ import annotations

from tree_sitter import Node

from indexter.parse.base import BaseLanguageParser, head_identifier, node_text
from indexter.parse.models import Kind, ParsedNode, RawRef, RefKind

_DEFINITIONS_QUERY = """
(function_item name: (identifier) @name) @def
(struct_item name: (type_identifier) @name) @def
(enum_item name: (type_identifier) @name) @def
(trait_item name: (type_identifier) @name) @def
(const_item name: (identifier) @name) @def
(static_item name: (identifier) @name) @def
(type_item name: (type_identifier) @name) @def
"""

_REFERENCES_QUERY = """
(call_expression function: (identifier) @callee) @call
(call_expression function: (field_expression) @callee) @call
(call_expression function: (scoped_identifier) @callee) @call

(use_declaration argument: (scoped_identifier) @use_simple)
(use_declaration argument: (identifier) @use_simple)
(use_declaration argument: (use_as_clause) @use_aliased)
(use_declaration argument: (use_wildcard) @use_wildcard)
(use_declaration
    argument: (scoped_use_list
        path: (_) @use_group_prefix
        list: (use_list [(identifier) (scoped_identifier) (use_as_clause)] @use_group_item)))

(impl_item) @impl_block
"""

# Containers whose methods count as `method` rather than `function`.
_METHOD_CONTAINERS = frozenset({"impl_item", "trait_item"})


def _split_path(node: Node) -> tuple[str | None, str]:
    """Split a `use` target's `identifier`/`scoped_identifier` path into
    `(prefix, last_segment)`. `prefix` is `None` for a single-segment path.
    """
    if node.type == "scoped_identifier":
        path_field = node.child_by_field_name("path")
        name_field = node.child_by_field_name("name")
        prefix = node_text(path_field) if path_field is not None else None
        last = node_text(name_field) if name_field is not None else ""
        return prefix, last
    return None, node_text(node)


def _group_item_binding(item_node: Node, group_prefix: str) -> tuple[str, str | None, str] | None:
    """`(raw_name, imported_name, head)` for one member of a `use a::{...}`
    group, given the group's shared prefix text. `None` for a shape this
    parser doesn't recognize (e.g. a nested group).
    """
    alias_text: str | None = None
    inner = item_node
    if item_node.type == "use_as_clause":
        path_field = item_node.child_by_field_name("path")
        alias_field = item_node.child_by_field_name("alias")
        if path_field is None or alias_field is None:
            return None
        inner = path_field
        alias_text = node_text(alias_field)
    if inner.type not in ("identifier", "scoped_identifier"):
        return None
    prefix, last = _split_path(inner)
    raw_name = f"{group_prefix}::{prefix}" if prefix is not None else group_prefix
    return raw_name, last, alias_text if alias_text is not None else last


class RustParser(BaseLanguageParser):
    language = "rust"

    @property
    def definitions_query_str(self) -> str:
        return _DEFINITIONS_QUERY

    @property
    def references_query_str(self) -> str | None:
        return _REFERENCES_QUERY

    def scope_segment(self, ancestor: Node) -> str | None:
        if ancestor.type == "impl_item":
            type_node = ancestor.child_by_field_name("type")
            if type_node is None or not type_node.text:
                return None
            type_name = node_text(type_node)
            trait_node = ancestor.child_by_field_name("trait")
            if trait_node is not None and trait_node.text:
                trait_last = node_text(trait_node).rsplit("::", 1)[-1]
                return f"{type_name}<{trait_last}>"
            return type_name
        if ancestor.type == "trait_item":
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

        return ParsedNode(
            kind=self._kind_for(node),
            name=name,
            scope_path=self.build_scope_path(node),
            language=self.language,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            signature=self._signature(node, source_bytes),
            docstring=self._doc_comment(node),
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
        if "impl_block" in match:
            node = match["impl_block"][0]
            trait_node = node.child_by_field_name("trait")
            if trait_node is None or not trait_node.text:
                return None  # inherent impl -- nothing to record
            type_node = node.child_by_field_name("type")
            return RawRef(
                origin_byte=node.start_byte,
                raw_name=node_text(trait_node),
                head=head_identifier(trait_node),
                ref_kind=RefKind.INHERITS,
                line=node.start_point[0] + 1,
                col=node.start_point[1] + 1,
                for_type=node_text(type_node) if type_node is not None else None,
            )
        if "use_simple" in match:
            node = match["use_simple"][0]
            prefix, last = _split_path(node)
            raw_name = prefix if prefix is not None else last
            imported_name = last if prefix is not None else None
            return self._import_ref(node, raw_name, head=last, imported_name=imported_name)
        if "use_aliased" in match:
            node = match["use_aliased"][0]
            path_node = node.child_by_field_name("path")
            alias_node = node.child_by_field_name("alias")
            if path_node is None or alias_node is None:
                return None
            prefix, last = _split_path(path_node)
            raw_name = prefix if prefix is not None else last
            imported_name = last if prefix is not None else None
            return self._import_ref(node, raw_name, head=node_text(alias_node), imported_name=imported_name)
        if "use_wildcard" in match:
            node = match["use_wildcard"][0]
            prefix_node = node.children[0] if node.child_count else None
            raw_name = node_text(prefix_node) if prefix_node is not None else ""
            return self._import_ref(node, raw_name, head=None, imported_name="*")
        if "use_group_item" in match:
            item_node = match["use_group_item"][0]
            group_prefix = node_text(match["use_group_prefix"][0])
            binding = _group_item_binding(item_node, group_prefix)
            if binding is None:
                return None
            raw_name, imported_name, head = binding
            return self._import_ref(item_node, raw_name, head=head, imported_name=imported_name)
        return None

    @staticmethod
    def _import_ref(
        node: Node, raw_name: str, *, head: str | None, imported_name: str | None = None
    ) -> RawRef:
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
        mapping = {
            "struct_item": Kind.STRUCT,
            "enum_item": Kind.ENUM,
            "trait_item": Kind.TRAIT,
            "const_item": Kind.CONSTANT,
            "static_item": Kind.CONSTANT,
            "type_item": Kind.TYPE_ALIAS,
        }
        if def_node.type in mapping:
            return mapping[def_node.type]
        # function_item: a method if nested in an impl or trait block.
        return Kind.METHOD if self._nearest_enclosing_type(def_node) in _METHOD_CONTAINERS else Kind.FUNCTION

    @staticmethod
    def _nearest_enclosing_type(node: Node) -> str | None:
        current = node.parent
        while current is not None:
            if current.type in _METHOD_CONTAINERS:
                return current.type
            current = current.parent
        return None

    @staticmethod
    def _signature(node: Node, source: bytes) -> str | None:
        if node.type != "function_item":
            return None
        body = node.child_by_field_name("body")
        end = body.start_byte if body else node.end_byte
        return source[node.start_byte : end].decode().strip()

    @staticmethod
    def _doc_comment(node: Node) -> str | None:
        """`///`/`//!` line comments or `/** */`/`/*! */` block comments
        immediately preceding a definition, attributes interspersed."""
        parent = node.parent
        if parent is None:
            return None
        index = None
        for i, child in enumerate(parent.children):
            if child == node:
                index = i
                break
        if index is None:
            return None

        lines: list[str] = []
        for i in range(index - 1, -1, -1):
            sibling = parent.children[i]
            if sibling.type == "line_comment" and sibling.text:
                text = node_text(sibling)
                if text.startswith(("///", "//!")):
                    lines.insert(0, text[3:].strip())
                    continue
                break
            if sibling.type == "block_comment" and sibling.text:
                text = node_text(sibling)
                if text.startswith(("/**", "/*!")):
                    lines.insert(0, text[3:-2].strip())
                    continue
                break
            if sibling.type == "attribute_item":
                continue
            break
        return "\n".join(lines) if lines else None
