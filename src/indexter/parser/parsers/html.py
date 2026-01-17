import re

from tree_sitter import Node

from .base import BaseLanguageParser


class HtmlParser(BaseLanguageParser):
    """
    For parsing and chunking HTML content while maintaining its semantic structure.

    Preserves important elements like tables, lists (<ul>, <ol>), and headers (<h1> through <h6>)

    Includes built-in normalization and stopword removal to clean the text.
    """

    language = "html"

    # Common English stopwords for text cleaning
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
    }

    @property
    def query_str(self) -> str:
        """Return the tree-sitter query string for semantic HTML elements."""
        return """
        ; Headers h1-h6
        (element
            (start_tag
                (tag_name) @tag_name
                (#match? @tag_name "^h[1-6]$")
            )
        ) @header
        
        ; Tables
        (element
            (start_tag
                (tag_name) @tag_name
                (#eq? @tag_name "table")
            )
        ) @table
        
        ; Unordered lists
        (element
            (start_tag
                (tag_name) @tag_name
                (#eq? @tag_name "ul")
            )
        ) @ul
        
        ; Ordered lists
        (element
            (start_tag
                (tag_name) @tag_name
                (#eq? @tag_name "ol")
            )
        ) @ol
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a matched HTML element and extract relevant info.

        Args:
            match: The matched nodes from the query
            source_bytes: The original source code as bytes
        Returns:
            Tuple of (content, node_info_dict) or None to skip
        """
        # Determine which type of element was matched
        node = None
        node_type = None

        if match.get("header"):
            node = match["header"][0]
            tag_name_node = match["tag_name"][0]
            tag_name = tag_name_node.text.decode() if tag_name_node.text else "h1"
            node_type = tag_name  # h1, h2, h3, etc.
        elif match.get("table"):
            node = match["table"][0]
            node_type = "table"
        elif match.get("ul"):
            node = match["ul"][0]
            node_type = "ul"
        elif match.get("ol"):
            node = match["ol"][0]
            node_type = "ol"
        else:
            return None

        # Extract raw content
        content = source_bytes[node.start_byte : node.end_byte].decode()

        # Extract and normalize text content
        text_content = self._extract_text_content(node, source_bytes)
        normalized_text = self._normalize_text(text_content)
        cleaned_text = self._remove_stopwords(normalized_text)

        # Generate node name from cleaned text
        node_name = self._generate_node_name(cleaned_text, node_type)

        # Determine parent scope
        parent_scope = self._get_parent_scope(node, source_bytes)

        # Build extra metadata
        extra = self._get_extra(node, source_bytes, node_type, cleaned_text)

        # Build node info
        node_info = {
            "language": self.language,
            "node_type": node_type,
            "node_name": node_name,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "documentation": None,  # HTML elements don't have documentation
            "parent_scope": parent_scope,
            "signature": None,
            "extra": extra,
        }

        return content, node_info

    def _extract_text_content(self, node: Node, source_bytes: bytes) -> str:
        """Extract text content from an HTML element, excluding tags."""
        text_parts = []

        def traverse(n: Node):
            if n.type == "text" and n.text:
                text_parts.append(n.text.decode())
            for child in n.children:
                traverse(child)

        traverse(node)
        return " ".join(text_parts)

    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase and removing extra whitespace."""
        # Convert to lowercase
        text = text.lower()
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _remove_stopwords(self, text: str) -> str:
        """Remove common stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word not in self.STOPWORDS]
        return " ".join(filtered_words)

    def _generate_node_name(self, cleaned_text: str, node_type: str) -> str:
        """Generate a concise node name from cleaned text."""
        # For headers, use the first few words
        if node_type.startswith("h"):
            words = cleaned_text.split()[:5]  # Take first 5 words
            if words:
                return " ".join(words)
            return node_type

        # For lists and tables, use a descriptive name
        if node_type in ("ul", "ol"):
            return f"{node_type}-list"
        if node_type == "table":
            return "table"

        return node_type

    def _get_parent_scope(self, node: Node, source_bytes: bytes) -> str | None:
        """Get the parent scope for nested HTML elements."""
        parent = node.parent
        while parent:
            if parent.type == "element":
                # Find the tag name in the start tag
                for child in parent.children:
                    if child.type == "start_tag":
                        for grandchild in child.children:
                            if grandchild.type == "tag_name" and grandchild.text:
                                tag_name = grandchild.text.decode()
                                # Only return semantic parent tags
                                if tag_name in (
                                    "article",
                                    "section",
                                    "div",
                                    "main",
                                    "aside",
                                    "nav",
                                    "header",
                                    "footer",
                                ):
                                    return tag_name
                                # Also check for headers as parents
                                if re.match(r"^h[1-6]$", tag_name):
                                    return tag_name
            parent = parent.parent
        return None

    def _get_extra(self, node: Node, source_bytes: bytes, node_type: str, cleaned_text: str) -> dict[str, str]:
        """Extract additional metadata about the HTML element."""
        extra = {}

        # Add cleaned text for reference
        if cleaned_text:
            extra["cleaned_text"] = cleaned_text[:100]  # Limit to 100 chars

        # For lists, count items
        if node_type in ("ul", "ol"):
            item_count = self._count_list_items(node)
            extra["item_count"] = str(item_count)

        # For tables, count rows and columns
        if node_type == "table":
            rows, cols = self._count_table_dimensions(node)
            extra["rows"] = str(rows)
            extra["cols"] = str(cols)

        # Extract id attribute if present
        id_attr = self._get_attribute(node, "id")
        if id_attr:
            extra["id"] = id_attr

        # Extract class attribute if present
        class_attr = self._get_attribute(node, "class")
        if class_attr:
            extra["class"] = class_attr

        return extra

    def _count_list_items(self, node: Node) -> int:
        """Count the number of list items in a list."""
        count = 0

        def traverse(n: Node):
            nonlocal count
            if n.type == "element":
                for child in n.children:
                    if child.type == "start_tag":
                        for grandchild in child.children:
                            if grandchild.type == "tag_name" and grandchild.text and grandchild.text.decode() == "li":
                                count += 1
                                break
            for child in n.children:
                traverse(child)

        traverse(node)
        return count

    def _count_table_dimensions(self, node: Node) -> tuple[int, int]:
        """Count rows and columns in a table."""
        rows = 0
        max_cols = 0

        def traverse(n: Node):
            nonlocal rows, max_cols
            if n.type == "element":
                for child in n.children:
                    if child.type == "start_tag":
                        for grandchild in child.children:
                            if grandchild.type == "tag_name" and grandchild.text:
                                tag = grandchild.text.decode()
                                if tag == "tr":
                                    rows += 1
                                    # Count cells in this row
                                    cols = self._count_cells_in_row(n)
                                    max_cols = max(max_cols, cols)
                                break
            for child in n.children:
                traverse(child)

        traverse(node)
        return rows, max_cols

    def _count_cells_in_row(self, row_node: Node) -> int:
        """Count cells (td or th) in a table row."""
        count = 0

        def traverse(n: Node):
            nonlocal count
            if n.type == "element":
                for child in n.children:
                    if child.type == "start_tag":
                        for grandchild in child.children:
                            if grandchild.type == "tag_name" and grandchild.text:
                                tag = grandchild.text.decode()
                                if tag in ("td", "th"):
                                    count += 1
                                break
            for child in n.children:
                traverse(child)

        traverse(row_node)
        return count

    def _get_attribute(self, node: Node, attr_name: str) -> str | None:
        """Extract an attribute value from an HTML element."""
        for child in node.children:
            if child.type == "start_tag":
                for grandchild in child.children:
                    if grandchild.type == "attribute":
                        # Find attribute_name and quoted_attribute_value children
                        name_node = None
                        value_node = None

                        for attr_child in grandchild.children:
                            if attr_child.type == "attribute_name":
                                name_node = attr_child
                            elif attr_child.type == "quoted_attribute_value":
                                # Get the actual value inside the quotes
                                for quote_child in attr_child.children:
                                    if quote_child.type == "attribute_value":
                                        value_node = quote_child
                                        break

                        if name_node and name_node.text and name_node.text.decode() == attr_name:
                            if value_node and value_node.text:
                                return value_node.text.decode()
                            # If no value found, return empty string for boolean attributes
                            return ""
        return None
