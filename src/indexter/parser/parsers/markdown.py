from tree_sitter import Node

from .base import BaseLanguageParser


class MarkdownParser(BaseLanguageParser):
    """Parser for Markdown files using tree-sitter."""

    language = "markdown"

    @property
    def query_str(self) -> str:
        return """
            ; ATX headings (# Header style)
            (atx_heading) @def
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a single query match for Markdown headings."""
        def_nodes = match.get("def", [])
        if not def_nodes:
            return None

        node = def_nodes[0]

        # Only skip nodes with actual ERROR children (not just missing optional elements)
        if self._has_error_child(node):
            return None

        # Get heading level and name
        level, heading_name = self._get_heading_info(node, source_bytes)
        if heading_name is None:
            return None

        # Get node type based on level
        node_type = f"Header {level}"

        # Get parent scope by traversing up to find parent section
        parent_scope = self._get_parent_scope(node, source_bytes)

        # Get section content (heading + all content until next same/higher level heading or EOF)
        # This uses tree-sitter's section boundaries which already handle this correctly
        section = node.parent
        if not section or section.type != "section":
            # Fallback: just return the heading line
            content = source_bytes[node.start_byte : node.end_byte].decode()
            end_byte = node.end_byte
            end_line = node.end_point[0]
        else:
            # The section content goes from the heading to the end of this section
            content = source_bytes[node.start_byte : section.end_byte].decode()
            end_byte = section.end_byte
            end_line = section.end_point[0]

        node_info = {
            "language": self.language,
            "node_type": node_type,
            "node_name": heading_name,
            "start_byte": node.start_byte,
            "end_byte": end_byte,
            "start_line": node.start_point[0] + 1,
            "end_line": end_line + 1,
            "documentation": None,
            "parent_scope": parent_scope,
            "signature": None,
            "extra": {},
        }

        return content, node_info

    # Helper methods

    def _get_heading_info(self, node: Node, source: bytes) -> tuple[int, str | None]:
        """Extract heading level and name from an atx_heading node.

        Returns:
            Tuple of (level, heading_name) where level is 1-6
        """
        # Find the marker (atx_h1_marker, atx_h2_marker, etc.)
        marker = None
        inline = None

        for child in node.children:
            if child.type.startswith("atx_h") and child.type.endswith("_marker"):
                marker = child
            elif child.type == "inline":
                inline = child

        if not marker or not inline:
            return 0, None

        # Extract level from marker type (atx_h1_marker -> 1)
        level = int(marker.type[5])  # atx_h{X}_marker

        # Extract heading text
        heading_name = inline.text.decode().strip() if inline.text else ""

        return level, heading_name

    def _get_parent_scope(self, node: Node, source: bytes) -> str | None:
        """Find the parent heading by traversing up the AST.

        Returns the name of the closest ancestor heading with a lower level.
        """
        current_level, _ = self._get_heading_info(node, source)

        # Traverse up to find parent section
        current = node.parent
        while current:
            if current.type == "section":
                # Check if this section has a heading (should be first child of section)
                for child in current.children:
                    if child.type == "atx_heading" and child != node:
                        parent_level, parent_name = self._get_heading_info(child, source)
                        # Only consider headings with lower level (higher hierarchy)
                        if parent_level < current_level:
                            return parent_name
                        break
            current = current.parent

        return None

    def _has_error_child(self, node: Node) -> bool:
        """Check if node has any actual ERROR children.

        Returns True only for ERROR nodes, not for missing optional elements.
        """
        if node.type == "ERROR":
            return True
        for child in node.children:
            if self._has_error_child(child):
                return True
        return False
