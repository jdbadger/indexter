from tree_sitter import Node

from .base import BaseLanguageParser


class RustParser(BaseLanguageParser):
    language = "rust"

    @property
    def query_str(self) -> str:
        return """
            ; Functions (including async and unsafe)
            (function_item
                name: (identifier) @name
                parameters: (parameters) @params
                return_type: (_)? @return_type
                body: (block)? @body
            ) @def

            ; Structs
            (struct_item
                name: (type_identifier) @name
                body: (_)? @struct_body
            ) @def

            ; Enums
            (enum_item
                name: (type_identifier) @name
                body: (enum_variant_list) @enum_body
            ) @def

            ; Traits
            (trait_item
                name: (type_identifier) @name
                body: (declaration_list) @trait_body
            ) @def

            ; Impl blocks
            (impl_item
                type: (type_identifier) @name
                body: (declaration_list) @impl_body
            ) @def

            ; Constants
            (const_item
                name: (identifier) @name
                value: (_) @value
            ) @def

            ; Static variables
            (static_item
                name: (identifier) @name
                value: (_)? @value
            ) @def

            ; Type aliases
            (type_item
                name: (type_identifier) @name
            ) @def

            ; Use declarations (imports)
            (use_declaration
                argument: (use_as_clause
                    path: (scoped_identifier) @path
                    alias: (identifier) @alias
                )
            ) @def

            (use_declaration
                argument: (scoped_identifier) @path
            ) @def

            ; Modules
            (mod_item
                name: (identifier) @name
                body: (declaration_list)? @mod_body
            ) @def
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a single query match for Rust code."""
        def_nodes = match.get("def", [])
        if not def_nodes:
            return None

        node = def_nodes[0]

        # Get the name from various captures
        name_nodes = match.get("name", [])
        path_nodes = match.get("path", [])
        alias_nodes = match.get("alias", [])

        if alias_nodes:
            # use ... as alias
            node_name = alias_nodes[0].text.decode()
        elif name_nodes:
            node_name = name_nodes[0].text.decode()
        elif path_nodes:
            # For use declarations, get the last segment of the path
            node_name = self._get_path_name(path_nodes[0])
        else:
            return None

        content = self._get_content(node, source_bytes)
        node_type = self._get_node_type(node)
        documentation = self._get_documentation(node, source_bytes)
        signature = self._get_signature(node, source_bytes)
        parent_scope = self._get_parent_scope(node)

        metadata = {
            "language": self.language,
            "node_type": node_type,
            "node_name": node_name,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "documentation": documentation,
            "parent_scope": parent_scope,
            "signature": signature,
            "extra": self._get_extra(node, source_bytes),
        }

        return content, metadata

    # Helper methods

    def _get_content(self, node: Node, source: bytes) -> str:
        """Extract the source text of a node."""
        return source[node.start_byte : node.end_byte].decode()

    def _get_node_type(self, node: Node) -> str:
        """Map tree-sitter node type to a normalized type string."""
        ts_type = node.type

        type_map = {
            "function_item": "function",
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
            "impl_item": "impl",
            "mod_item": "module",
            "type_item": "type_alias",
            "const_item": "constant",
            "static_item": "static",
            "use_declaration": "import",
        }

        base_type = type_map.get(ts_type, ts_type)

        # Functions inside impl/trait blocks are methods
        if ts_type == "function_item" and self._get_parent_scope(node):
            return "method"

        return base_type

    def _get_documentation(self, node: Node, source: bytes) -> str | None:
        """Extract doc comments (/// or //!) immediately preceding a node."""
        parent = node.parent
        if not parent:
            return None

        doc_lines = []

        # Find the index of our node and look backwards for adjacent doc comments
        node_index = None
        for i, child in enumerate(parent.children):
            if child == node:
                node_index = i
                break

        if node_index is None:
            return None

        # Look backwards from the node for contiguous doc comments/attributes
        for i in range(node_index - 1, -1, -1):
            child = parent.children[i]
            if child.type == "line_comment" and child.text:
                text = child.text.decode()
                if text.startswith("///") or text.startswith("//!"):
                    doc_lines.insert(0, text[3:].strip())
                else:
                    # Regular comment breaks the chain
                    break
            elif child.type == "block_comment" and child.text:
                text = child.text.decode()
                if text.startswith("/**") or text.startswith("/*!"):
                    cleaned = self._parse_block_comment(text)
                    if cleaned:
                        doc_lines.insert(0, cleaned)
                else:
                    break
            elif child.type == "attribute_item":
                # Attributes can be interspersed with doc comments
                continue
            else:
                # Any other node type breaks the chain
                break

        return "\n".join(doc_lines) if doc_lines else None

    def _parse_block_comment(self, comment: str) -> str:
        """Parse a block doc comment and return cleaned content."""
        # Remove /** or /*! and */
        content = comment
        if content.startswith("/**") or content.startswith("/*!"):
            content = content[3:]
        if content.endswith("*/"):
            content = content[:-2]

        # Clean up each line
        lines = []
        for line in content.split("\n"):
            cleaned = line.strip()
            if cleaned.startswith("*"):
                cleaned = cleaned[1:].strip()
            if cleaned:
                lines.append(cleaned)

        return "\n".join(lines)

    def _get_signature(self, node: Node, source: bytes) -> str | None:
        """Extract function signature (from fn to before the body)."""
        if node.type != "function_item":
            return None

        body = node.child_by_field_name("body")
        if body:
            # Get everything up to the body
            sig = source[node.start_byte : body.start_byte].decode().strip()
        else:
            # Trait method signature (no body)
            sig = source[node.start_byte : node.end_byte].decode().strip()

        return sig

    def _get_parent_scope(self, node: Node) -> str | None:
        """Find the enclosing impl or trait name, if any."""
        current = node.parent
        while current:
            if current.type == "impl_item":
                # impl blocks use 'type' field
                type_node = current.child_by_field_name("type")
                return type_node.text.decode() if type_node and type_node.text else None
            if current.type == "trait_item":
                name_node = current.child_by_field_name("name")
                return name_node.text.decode() if name_node and name_node.text else None
            if current.type == "declaration_list":
                # Go up past the declaration_list
                current = current.parent
                continue
            current = current.parent
        return None

    def _get_extra(self, node: Node, source: bytes) -> dict[str, str]:
        """Extract Rust-specific extra metadata."""
        attributes = self._get_attributes(node, source)
        return {
            "attributes": ",".join(attributes) if attributes else "",
            "is_async": str(self._is_async(node)).lower(),
            "is_unsafe": str(self._is_unsafe(node)).lower(),
            "is_pub": str(self._is_pub(node)).lower(),
        }

    def _get_attributes(self, node: Node, source: bytes) -> list[str]:
        """Extract #[...] attributes immediately preceding a node."""
        parent = node.parent
        if not parent:
            return []

        # Find the index of our node
        node_index = None
        for i, child in enumerate(parent.children):
            if child == node:
                node_index = i
                break

        if node_index is None:
            return []

        # Look backwards for contiguous attributes
        attributes = []
        for i in range(node_index - 1, -1, -1):
            child = parent.children[i]
            if child.type == "attribute_item" and child.text:
                attributes.insert(0, child.text.decode())
            elif child.type in ("line_comment", "block_comment"):
                # Doc comments can be interspersed with attributes
                continue
            else:
                # Any other node type breaks the chain
                break

        return attributes

    def _get_path_name(self, path_node: Node) -> str:
        """Extract the last segment from a scoped_identifier (e.g., std::io::Read -> Read)."""
        # Get the rightmost identifier
        if path_node.type == "scoped_identifier":
            name_node = path_node.child_by_field_name("name")
            if name_node and name_node.text:
                return name_node.text.decode()
        return path_node.text.decode() if path_node.text else ""

    def _is_async(self, node: Node) -> bool:
        """Check if a function is async."""
        if node.type == "function_item":
            for child in node.children:
                if child.type == "function_modifiers":
                    for modifier in child.children:
                        if modifier.type == "async":
                            return True
        return False

    def _is_unsafe(self, node: Node) -> bool:
        """Check if an item is unsafe."""
        if node.type in ("function_item", "impl_item", "trait_item"):
            for child in node.children:
                if child.type == "unsafe":
                    return True
                if child.type == "function_modifiers":
                    for modifier in child.children:
                        if modifier.type == "unsafe":
                            return True
        return False

    def _is_pub(self, node: Node) -> bool:
        """Check if an item has pub visibility."""
        for child in node.children:
            if child.type == "visibility_modifier":
                return True
        return False
