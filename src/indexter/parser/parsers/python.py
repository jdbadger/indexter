from tree_sitter import Node

from .base import BaseLanguageParser


class PythonParser(BaseLanguageParser):
    language = "python"

    @property
    def query_str(self) -> str:
        return """
            ; Functions (including async)
            (function_definition
                name: (identifier) @name
                parameters: (parameters) @params
                return_type: (type)? @return_type
                body: (block) @body
            ) @def

            ; Classes
            (class_definition
                name: (identifier) @name
                superclasses: (argument_list)? @bases
                body: (block) @body
            ) @def

            ; Decorated definitions (functions and classes)
            (decorated_definition
                (decorator)+ @decorators
                definition: [
                    (function_definition
                        name: (identifier) @name
                    ) @inner
                    (class_definition
                        name: (identifier) @name
                    ) @inner
                ]
            ) @def

            ; Module-level constants (uppercase identifiers)
            (module
                (assignment
                    left: (identifier) @name
                    right: (_) @value
                ) @def
            )

            ; Import statements
            (import_statement
                name: (dotted_name) @name
            ) @def

            ; Import from statements  
            (import_from_statement
                module_name: (dotted_name)? @module
            ) @def
        """

    def process_match(self, match: dict[str, list[Node]], source_bytes: bytes) -> tuple[str, dict] | None:
        """Process a single query match for Python code."""
        def_nodes = match.get("def", [])
        if not def_nodes:
            return None

        node = def_nodes[0]

        # Skip function/class definitions that are inside a decorated_definition
        # (those are captured separately via the decorated_definition pattern)
        if node.type in ("function_definition", "class_definition"):
            if node.parent and node.parent.type == "decorated_definition":
                return None

        # For decorated definitions, get the inner node for extracting details
        inner_nodes = match.get("inner", [])
        actual_def = inner_nodes[0] if inner_nodes else node

        # Get the name - check @name first, then @module for import_from_statement
        name_nodes = match.get("name", [])
        module_nodes = match.get("module", [])
        if name_nodes:
            node_name = name_nodes[0].text.decode()
        elif module_nodes:
            node_name = module_nodes[0].text.decode()
        else:
            # Can't process without a name
            return None

        # Skip non-constant assignments (only keep UPPER_CASE identifiers)
        if node.type == "assignment":
            if not self._is_constant(node_name):
                return None

        content = self._get_content(node, source_bytes)
        node_type = self._get_node_type(actual_def.type, actual_def)
        documentation = self._get_documentation(actual_def, source_bytes)
        signature = self._get_signature(actual_def, source_bytes)
        parent_scope = self._get_parent_scope(actual_def)

        node_info = {
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

        return content, node_info

    # Helper methods

    def _get_content(self, node: Node, source: bytes) -> str:
        """Extract the source text of a node."""
        return source[node.start_byte : node.end_byte].decode()

    def _get_node_type(self, ts_type: str, outer_node: Node) -> str:
        """Map tree-sitter node type to a normalized type string."""
        if ts_type == "function_definition":
            # Check if nested within a class to distinguish method from function
            if self._get_parent_scope(outer_node):
                return "method"
            return "function"
        if ts_type == "class_definition":
            return "class"
        if ts_type == "assignment":
            return "constant"
        if ts_type in ("import_statement", "import_from_statement"):
            return "import"
        return ts_type

    def _get_documentation(self, node: Node, source: bytes) -> str | None:
        """Extract docstring from a function or class definition."""
        if node.type not in ("function_definition", "class_definition"):
            return None

        body = node.child_by_field_name("body")
        if not body or not body.children:
            return None

        first_stmt = body.children[0]

        # Direct string literal
        if first_stmt.type == "string":
            return self._strip_docstring(first_stmt.text.decode())

        # String wrapped in expression_statement
        if first_stmt.type == "expression_statement" and first_stmt.children:
            expr = first_stmt.children[0]
            if expr.type == "string":
                return self._strip_docstring(expr.text.decode())

        return None

    def _strip_docstring(self, text: str) -> str:
        """Remove docstring quotes and normalize whitespace."""
        # Handle triple quotes first, then single quotes
        for quote in ('"""', "'''", '"', "'"):
            if text.startswith(quote) and text.endswith(quote):
                text = text[len(quote) : -len(quote)]
                break
        return text.strip()

    def _get_signature(self, node: Node, source: bytes) -> str | None:
        """Extract function/method signature (from 'def' up to the colon before the body)."""
        if node.type != "function_definition":
            return None

        body = node.child_by_field_name("body")
        end = body.start_byte if body else node.end_byte
        return source[node.start_byte : end].decode().rstrip().rstrip(":")

    def _get_parent_scope(self, node: Node) -> str | None:
        """Find the enclosing class name, if any."""
        current = node.parent
        while current:
            if current.type == "class_definition":
                name_node = current.child_by_field_name("name")
                return name_node.text.decode() if name_node and name_node.text else None
            current = current.parent
        return None

    def _get_extra(self, node: Node, source: bytes) -> dict[str, str]:
        """Extract Python-specific extra metadata."""
        decorators = self._get_decorators(node, source)
        return {
            "decorators": ",".join(decorators) if decorators else "",
            "is_async": str(self._is_async(node)).lower(),
        }

    def _get_decorators(self, node: Node, source: bytes) -> list[str]:
        """Extract decorator strings from a decorated_definition node."""
        if node.type != "decorated_definition":
            return []
        return [
            source[child.start_byte : child.end_byte].decode() for child in node.children if child.type == "decorator"
        ]

    def _is_async(self, node: Node) -> bool:
        """Check if a function definition is async."""
        # For decorated definitions, check the inner function
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    return self._is_async(child)
            return False

        if node.type == "function_definition":
            for child in node.children:
                if child.type == "async":
                    return True
        return False

    def _is_constant(self, name: str) -> bool:
        """Check if a name follows Python constant naming convention (UPPER_CASE)."""
        return name.isupper() or (name.replace("_", "").isupper() and "_" in name)
