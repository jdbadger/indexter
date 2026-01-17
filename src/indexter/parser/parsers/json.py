from tree_sitter import Node

from .base import BaseLanguageParser


class JsonParser(BaseLanguageParser):
    """Parser for JSON files that yields objects and arrays to preserve context."""

    language = "json"

    @property
    def query_str(self) -> str:
        return """
            ; Objects
            (object) @def
            
            ; Arrays
            (array) @def
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a single query match for JSON code."""
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
        if node.type == "object":
            return "object"
        if node.type == "array":
            return "array"
        return node.type

    def _get_node_info(self, node: Node, source: bytes) -> tuple[str, str, str | None]:
        """Get node name, path, and parent scope.

        Returns:
            Tuple of (node_name, current_path, parent_scope)
        """
        # Traverse up to find the path
        path_parts = []
        current = node.parent

        while current:
            if current.type == "pair":
                # Get the key from the pair
                key_node = current.child_by_field_name("key")
                if key_node and key_node.text:
                    key_text = key_node.text.decode().strip('"')
                    path_parts.insert(0, key_text)
            elif current.type == "array":
                # Find the index of the current node in the array
                index = self._get_array_index(current, node)
                path_parts.insert(0, f"[{index}]")
            current = current.parent

        # Build the path and determine node name and parent scope
        if path_parts:
            node_name = path_parts[-1]
            current_path = "root." + ".".join(path_parts)
            # Parent scope is the second-to-last part if it exists, otherwise "root"
            if len(path_parts) > 1:
                parent_scope = path_parts[-2]
            else:
                parent_scope = "root"
        else:
            node_name = "root"
            current_path = "root"
            parent_scope = None

        return node_name, current_path, parent_scope

    def _get_array_index(self, array_node: Node, target_node: Node) -> int:
        """Get the index of a node within an array.

        This counts ALL array elements (including primitives), not just objects/arrays.
        """
        index = 0
        for child in array_node.children:
            # Skip structural tokens like commas and brackets
            if child.type in (",", "[", "]"):
                continue
            # Check if this is the target or an ancestor
            if child == target_node or self._is_ancestor(target_node, child):
                return index
            index += 1
        return 0

    def _is_ancestor(self, node: Node, potential_ancestor: Node) -> bool:
        """Check if potential_ancestor is an ancestor of node."""
        current = node.parent
        while current:
            if current == potential_ancestor:
                return True
            current = current.parent
        return False

    def _has_error_descendant(self, node: Node) -> bool:
        """Check if node has any ERROR descendants or missing nodes."""
        if node.type == "ERROR" or node.is_missing:
            return True
        for child in node.children:
            if self._has_error_descendant(child):
                return True
        return False

    def _get_extra(self, node: Node, current_path: str) -> dict[str, str]:
        """Extract JSON-specific extra metadata."""
        extra = {"path": current_path}

        # Add length for arrays
        if node.type == "array":
            # Count actual array elements (skip commas and brackets)
            array_items = [
                c for c in node.children if c.type in ("object", "array", "string", "number", "true", "false", "null")
            ]
            extra["length"] = str(len(array_items))

        return extra
