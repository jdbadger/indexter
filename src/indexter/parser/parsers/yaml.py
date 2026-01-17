from tree_sitter import Node

from .base import BaseLanguageParser


class YamlParser(BaseLanguageParser):
    """Parser for YAML files that yields mappings and sequences to preserve context."""

    language = "yaml"

    @property
    def query_str(self) -> str:
        return """
            ; Block mappings (nested objects)
            (block_mapping) @def
            
            ; Block sequences (arrays)
            (block_sequence) @def
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a single query match for YAML code."""
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
        if node.type == "block_mapping":
            return "mapping"
        if node.type == "block_sequence":
            return "sequence"
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
            if current.type == "block_mapping_pair":
                # Get the key from the pair
                key_node = current.child_by_field_name("key")
                if key_node:
                    # Navigate to the actual scalar inside flow_node
                    key_text = self._extract_key_text(key_node)
                    if key_text:
                        path_parts.insert(0, key_text)
            elif current.type == "block_sequence":
                # Find the index of the current node in the sequence
                index = self._get_sequence_index(current, node)
                path_parts.insert(0, f"[{index}]")
            elif current.type == "block_sequence_item":
                # Get the index of this item in its parent sequence
                parent_seq = current.parent
                if parent_seq and parent_seq.type == "block_sequence":
                    index = self._get_sequence_item_index(parent_seq, current)
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

    def _extract_key_text(self, key_node: Node) -> str | None:
        """Extract text from a key node (which is usually a flow_node containing a scalar)."""
        # Navigate through flow_node to find the scalar
        for child in key_node.children:
            if "scalar" in child.type and child.text:
                return child.text.decode()
        # Fallback: just get the text directly
        return key_node.text.decode() if key_node.text else None

    def _get_sequence_index(self, sequence_node: Node, target_node: Node) -> int:
        """Get the index of a node within a block_sequence.

        This counts block_sequence_item children.
        """
        index = 0
        for child in sequence_node.children:
            if child.type == "block_sequence_item":
                # Check if this item contains the target
                if self._contains_node(child, target_node):
                    return index
                index += 1
        return 0

    def _get_sequence_item_index(self, sequence_node: Node, target_item: Node) -> int:
        """Get the index of a block_sequence_item within its parent sequence."""
        index = 0
        for child in sequence_node.children:
            if child.type == "block_sequence_item":
                if child == target_item:
                    return index
                index += 1
        return 0

    def _contains_node(self, parent: Node, target: Node) -> bool:
        """Check if parent contains target node in its subtree."""
        if parent == target:
            return True
        for child in parent.children:
            if self._contains_node(child, target):
                return True
        return False

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
        """Extract YAML-specific extra metadata."""
        extra = {"path": current_path}

        # Add length for sequences
        if node.type == "block_sequence":
            # Count block_sequence_item children
            items = [c for c in node.children if c.type == "block_sequence_item"]
            extra["length"] = str(len(items))

        return extra
