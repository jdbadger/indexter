from tree_sitter import Node

from .base import BaseLanguageParser


class CssParser(BaseLanguageParser):
    """Parser for CSS files that yields rules and at-rules to preserve context."""

    language = "css"

    @property
    def query_str(self) -> str:
        """Return the tree-sitter query string for CSS rules and at-rules."""
        return """
(rule_set
    (selectors) @rule_name
) @rule

(media_statement) @at_rule

(keyframes_statement
    (keyframes_name) @at_rule_name
) @at_rule

(import_statement) @at_rule

(charset_statement) @at_rule

(supports_statement) @at_rule

(at_rule) @at_rule
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a matched CSS rule or at-rule and extract relevant info.

        Args:
            match: The matched nodes from the query
            source_bytes: The original source code as bytes
        Returns:
            Tuple of (content, node_info_dict) or None to skip
        """
        # Check if this is a regular rule or an at-rule
        rule_nodes = match.get("rule", [])
        at_rule_nodes = match.get("at_rule", [])

        if rule_nodes:
            node = rule_nodes[0]
            name_nodes = match.get("rule_name", [])
            if not name_nodes:
                return None

            node_name = name_nodes[0].text.decode().strip()
            node_type = "rule"

        elif at_rule_nodes:
            node = at_rule_nodes[0]
            node_type = "at-rule"

            # Extract the at-rule name based on node type
            if node.type == "media_statement":
                node_name = "@media"
            elif node.type == "keyframes_statement":
                # Get keyframes name if available
                name_nodes = match.get("at_rule_name", [])
                if name_nodes:
                    keyframe_name = name_nodes[0].text.decode().strip()
                    node_name = f"@keyframes {keyframe_name}"
                else:
                    node_name = "@keyframes"
            elif node.type == "import_statement":
                node_name = "@import"
            elif node.type == "charset_statement":
                node_name = "@charset"
            elif node.type == "supports_statement":
                node_name = "@supports"
            elif node.type == "at_rule":
                # Generic at-rule - try to extract name from first child
                for child in node.children:
                    if child.type == "at_keyword":
                        node_name = child.text.decode().strip()
                        break
                else:
                    node_name = "@rule"
            else:
                node_name = f"@{node.type}"
        else:
            return None

        # Extract content
        content = source_bytes[node.start_byte : node.end_byte].decode()

        # Determine parent scope (for nested rules, if applicable)
        parent_scope = self._get_parent_scope(node)

        # Build node info
        node_info = {
            "language": self.language,
            "node_type": node_type,
            "node_name": node_name,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "documentation": None,  # CSS doesn't have standard documentation comments
            "parent_scope": parent_scope,
            "signature": None,
            "extra": self._get_extra(node, node_type),
        }

        return content, node_info

    def _get_parent_scope(self, node: Node) -> str | None:
        """Get the parent scope for nested CSS rules (e.g., media queries, nesting)."""
        parent = node.parent
        while parent:
            # Check for specific at-rule types
            if parent.type == "media_statement":
                return "@media"
            elif parent.type == "supports_statement":
                return "@supports"
            elif parent.type == "keyframes_statement":
                return "@keyframes"
            elif parent.type == "at_rule":
                # Generic at-rule
                for child in parent.children:
                    if child.type == "at_keyword" and child.text:
                        return child.text.decode().strip()
                return "@rule"
            # Check for nested rule sets (CSS nesting)
            elif parent.type == "rule_set":
                # Find selectors child (not using field_by_name since it may not be a field)
                for child in parent.children:
                    if child.type == "selectors" and child.text:
                        return child.text.decode().strip()
            parent = parent.parent
        return None

    def _get_extra(self, node: Node, node_type: str) -> dict[str, str]:
        """Extract additional metadata about the CSS rule."""
        extra = {}

        if node_type == "rule":
            # Count declarations in the rule - find block child
            block = None
            for child in node.children:
                if child.type == "block":
                    block = child
                    break

            if block:
                declarations = [child for child in block.children if child.type == "declaration"]
                extra["declaration_count"] = str(len(declarations))

        elif node_type == "at-rule":
            # Get the value/query part of the at-rule (e.g., media query conditions)
            # For media_statement, look for keyword_query or feature_query
            if node.type == "media_statement":
                for child in node.children:
                    if child.type in ("keyword_query", "feature_query", "binary_query") and child.text:
                        extra["value"] = child.text.decode().strip()
                        break
            # For other at-rules, try to find relevant value children
            elif node.type in ("supports_statement", "import_statement"):
                for child in node.children:
                    if child.type not in ("@supports", "@import", "block", "{", "}"):
                        if not extra.get("value") and child.text:
                            extra["value"] = child.text.decode().strip()

        return extra
