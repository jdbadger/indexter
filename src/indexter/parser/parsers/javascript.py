from tree_sitter import Node

from .base import BaseLanguageParser


class JavaScriptParser(BaseLanguageParser):
    language = "javascript"

    @property
    def query_str(self) -> str:
        return """
            ; Function declarations
            (function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body
            ) @def

            ; Generator function declarations
            (generator_function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body
            ) @def

            ; Arrow functions (assigned to variables)
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @name
                    value: (arrow_function) @arrow_func
                )
            ) @def

            ; Classes
            (class_declaration
                name: (identifier) @name
                body: (class_body) @body
            ) @def

            ; Method definitions
            (method_definition
                name: (property_identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body
            ) @def

            ; Constants (const declarations)
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @name
                    value: (_) @value
                ) 
            ) @def

            ; Import statements
            (import_statement
                source: (string) @source
            ) @def

            ; Export statements
            (export_statement) @def
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a single query match for JavaScript code."""
        def_nodes = match.get("def", [])
        if not def_nodes:
            return None

        node = def_nodes[0]

        # Skip lexical declarations that are inside export statements
        # (they'll be captured by the export pattern)
        if node.type == "lexical_declaration":
            if node.parent and node.parent.type == "export_statement":
                return None

        # For arrow functions, the actual definition is in @arrow_func
        arrow_func_nodes = match.get("arrow_func", [])
        actual_def = arrow_func_nodes[0] if arrow_func_nodes else node

        # Get the name
        name_nodes = match.get("name", [])
        source_nodes = match.get("source", [])
        if name_nodes:
            node_name = name_nodes[0].text.decode()
        elif source_nodes:
            # For imports, use the source string (strip quotes)
            node_name = source_nodes[0].text.decode().strip("'\"")
        elif node.type == "export_statement":
            node_name = self._get_export_name(node, source_bytes)
        else:
            return None

        # Filter lexical declarations: skip non-arrow function declarations
        # unless they're UPPER_CASE constants
        if node.type == "lexical_declaration" and not arrow_func_nodes:
            if not self._is_const_declaration(node):
                return None
            if node_name is None or not self._is_constant(node_name):
                return None

        content = self._get_content(node, source_bytes)
        node_type = self._get_node_type(actual_def, node)
        documentation = self._get_documentation(node, source_bytes)
        signature = self._get_signature(actual_def, source_bytes)
        parent_scope = self._get_parent_scope(actual_def)

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
            "extra": self._get_extra(actual_def, source_bytes),
        }

        return content, metadata

    # Helper methods

    def _get_content(self, node: Node, source: bytes) -> str:
        """Extract the source text of a node."""
        return source[node.start_byte : node.end_byte].decode()

    def _get_node_type(self, actual_def: Node, outer_node: Node) -> str:
        """Map tree-sitter node type to a normalized type string."""
        ts_type = actual_def.type

        if ts_type == "class_declaration":
            return "class"
        if ts_type == "method_definition":
            return "method"
        if ts_type in ("function_declaration", "function", "generator_function_declaration"):
            return "method" if self._get_parent_scope(actual_def) else "function"
        if ts_type == "arrow_function":
            return "method" if self._get_parent_scope(actual_def) else "function"
        if outer_node.type == "lexical_declaration":
            return "constant"
        if outer_node.type == "import_statement":
            return "import"
        if outer_node.type == "export_statement":
            return "export"
        return ts_type

    def _get_documentation(self, node: Node, source: bytes) -> str | None:
        """Extract JSDoc comment preceding a node."""
        # Look for comment nodes that are siblings before this node
        parent = node.parent
        if not parent:
            return None

        for i, child in enumerate(parent.children):
            if child == node:
                # Look backwards for a comment
                if i > 0:
                    prev_sibling = parent.children[i - 1]
                    if prev_sibling.type == "comment" and prev_sibling.text:
                        return self._parse_jsdoc(prev_sibling.text.decode())
                break
        return None

    def _parse_jsdoc(self, comment: str) -> str | None:
        """Parse a JSDoc comment and return cleaned content."""
        if not (comment.startswith("/**") and comment.endswith("*/")):
            return None

        # Remove /** and */
        content = comment[3:-2]

        # Clean up each line
        lines = []
        for line in content.split("\n"):
            cleaned = line.strip()
            # Remove leading asterisks
            if cleaned.startswith("*"):
                cleaned = cleaned[1:].strip()
            if cleaned:
                lines.append(cleaned)

        return "\n".join(lines) if lines else None

    def _get_signature(self, node: Node, source: bytes) -> str | None:
        """Extract function/method signature."""
        if node.type not in (
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
            "generator_function_declaration",
        ):
            return None

        if node.type == "arrow_function":
            # Get parameters and arrow
            params = node.child_by_field_name("parameters") or node.child_by_field_name("parameter")
            if params:
                for child in node.children:
                    if child.type == "=>":
                        return source[params.start_byte : child.end_byte].decode().strip()
            # Fallback: everything before the body
            body = node.child_by_field_name("body")
            if body:
                return source[node.start_byte : body.start_byte].decode().strip()
            return None

        # For regular functions/methods: from start to before body
        body = node.child_by_field_name("body")
        if body:
            return source[node.start_byte : body.start_byte].decode().strip()
        return None

    def _get_parent_scope(self, node: Node) -> str | None:
        """Find the enclosing class name, if any."""
        current = node.parent
        while current:
            if current.type == "class_declaration":
                name_node = current.child_by_field_name("name")
                return name_node.text.decode() if name_node and name_node.text else None
            if current.type == "class_body":
                # Go up to the class_declaration
                current = current.parent
                continue
            current = current.parent
        return None

    def _get_extra(self, node: Node, source: bytes) -> dict[str, str]:
        """Extract JavaScript-specific extra metadata."""
        return {
            "is_async": str(self._is_async(node)).lower(),
            "is_generator": str(self._is_generator(node)).lower(),
            "is_arrow": str(node.type == "arrow_function").lower(),
        }

    def _get_export_name(self, node: Node, source: bytes) -> str | None:
        """Extract the name from an export statement."""
        # Check for default export or named export
        for child in node.children:
            if child.type == "function_declaration":
                name_node = child.child_by_field_name("name")
                return name_node.text.decode() if name_node and name_node.text else "default"
            if child.type == "class_declaration":
                name_node = child.child_by_field_name("name")
                return name_node.text.decode() if name_node and name_node.text else "default"
            if child.type == "identifier" and child.text:
                return child.text.decode()
            if child.type == "lexical_declaration":
                for var_decl in child.children:
                    if var_decl.type == "variable_declarator":
                        name_node = var_decl.child_by_field_name("name")
                        return name_node.text.decode() if name_node and name_node.text else None
            if child.type == "export_clause":
                # Named exports: export { foo, bar }
                names = []
                for export_spec in child.children:
                    if export_spec.type == "export_specifier":
                        name_node = export_spec.child_by_field_name("name")
                        if name_node and name_node.text:
                            names.append(name_node.text.decode())
                return ", ".join(names) if names else None
        return "default"

    def _is_async(self, node: Node) -> bool:
        """Check if a function is async."""
        if node.type in (
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
            "generator_function_declaration",
        ):
            for child in node.children:
                if child.type == "async":
                    return True
        return False

    def _is_generator(self, node: Node) -> bool:
        """Check if a function is a generator."""
        if node.type == "generator_function_declaration":
            return True
        if node.type in ("function_declaration", "function", "method_definition"):
            for child in node.children:
                if child.type == "*":
                    return True
        return False

    def _is_const_declaration(self, node: Node) -> bool:
        """Check if a lexical declaration uses 'const'."""
        for child in node.children:
            if child.type == "const":
                return True
        return False

    def _is_constant(self, name: str) -> bool:
        """Check if a name follows constant naming convention (UPPER_CASE)."""
        return name.isupper() or (name.replace("_", "").isupper() and "_" in name)
