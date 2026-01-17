from tree_sitter import Node

from .base import BaseLanguageParser


class TomlParser(BaseLanguageParser):
    """Parser for TOML files that yields tables and their contents to preserve context."""

    language = "toml"

    @property
    def query_str(self) -> str:
        return """
            ; Standard tables [table]
            (table) @def
            
            ; Array of tables [[table]]
            (table_array_element) @def
            
            ; Top-level pairs (key = value)
            (document
                (pair) @def
            )
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a single query match for TOML code."""
        def_nodes = match.get("def", [])
        if not def_nodes:
            return None

        node = def_nodes[0]

        # Skip nodes with errors
        if node.has_error or self._has_error_descendant(node):
            return None

        # Get node name and path
        node_name, current_path, parent_scope = self._get_node_info(node, source_bytes)

        content = self._get_content(node, source_bytes)
        node_type = self._get_node_type(node)

        node_info = {
            "language": self.language,
            "node_type": node_type,
            "node_name": node_name,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "documentation": None,
            "parent_scope": parent_scope,
            "signature": None,
            "extra": self._get_extra(node, current_path),
        }

        return content, node_info

    # Helper methods

    def _get_content(self, node: Node, source: bytes) -> str:
        """Extract the source text of a node."""
        return source[node.start_byte : node.end_byte].decode()

    def _get_node_type(self, node: Node) -> str:
        """Map tree-sitter node type to a normalized type string."""
        if node.type == "table":
            return "table"
        if node.type == "table_array_element":
            return "table_array"
        if node.type == "pair":
            return "pair"
        return node.type

    def _get_node_info(self, node: Node, source: bytes) -> tuple[str, str, str | None]:
        """Get node name, path, and parent scope.

        Returns:
            Tuple of (node_name, current_path, parent_scope)
        """
        if node.type == "table":
            # Extract the table name from [table.name]
            table_name = self._extract_table_name(node)
            if table_name:
                parts = table_name.split(".")
                node_name = parts[-1]
                current_path = table_name
                parent_scope = ".".join(parts[:-1]) if len(parts) > 1 else None
                return node_name, current_path, parent_scope
            return "unknown", "unknown", None

        if node.type == "table_array_element":
            # Extract the table array name from [[table.name]]
            table_name = self._extract_table_array_name(node)
            if table_name:
                parts = table_name.split(".")
                node_name = parts[-1]
                current_path = table_name
                parent_scope = ".".join(parts[:-1]) if len(parts) > 1 else None
                return node_name, current_path, parent_scope
            return "unknown", "unknown", None

        if node.type == "pair":
            # Extract the key name
            key_node = self._find_child_by_type(node, "bare_key") or self._find_child_by_type(node, "quoted_key")
            if key_node and key_node.text:
                key_name = key_node.text.decode().strip('"').strip("'")
                return key_name, key_name, None
            # Try dotted_key
            dotted_key = self._find_child_by_type(node, "dotted_key")
            if dotted_key and dotted_key.text:
                key_name = dotted_key.text.decode()
                parts = key_name.split(".")
                return parts[-1], key_name, ".".join(parts[:-1]) if len(parts) > 1 else None
            return "unknown", "unknown", None

        return "unknown", "unknown", None

    def _extract_table_name(self, table_node: Node) -> str | None:
        """Extract the table name from a [table] node."""
        # Look for the table header which contains the dotted key or bare key
        for child in table_node.children:
            if child.type in ("[", "]"):
                continue
            if child.type == "dotted_key" and child.text:
                return child.text.decode()
            if child.type == "bare_key" and child.text:
                return child.text.decode()
            if child.type == "quoted_key" and child.text:
                return child.text.decode().strip('"').strip("'")
        return None

    def _extract_table_array_name(self, table_array_node: Node) -> str | None:
        """Extract the table array name from a [[table]] node."""
        # Look for the dotted key or bare key within the header
        for child in table_array_node.children:
            if child.type in ("[[", "]]", "[", "]"):
                continue
            if child.type == "dotted_key" and child.text:
                return child.text.decode()
            if child.type == "bare_key" and child.text:
                return child.text.decode()
            if child.type == "quoted_key" and child.text:
                return child.text.decode().strip('"').strip("'")
        return None

    def _find_child_by_type(self, node: Node, child_type: str) -> Node | None:
        """Find the first child of a given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _has_error_descendant(self, node: Node) -> bool:
        """Check if node has any ERROR descendants or missing nodes."""
        if node.type == "ERROR" or node.is_missing:
            return True
        for child in node.children:
            if self._has_error_descendant(child):
                return True
        return False

    def _get_extra(self, node: Node, current_path: str) -> dict[str, str]:
        """Extract TOML-specific extra metadata."""
        extra = {"path": current_path}

        # Count pairs in table
        if node.type in ("table", "table_array_element"):
            pairs = [c for c in node.children if c.type == "pair"]
            extra["pair_count"] = str(len(pairs))

        return extra
