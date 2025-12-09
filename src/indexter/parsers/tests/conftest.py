"""Shared fixtures for parser tests."""

import subprocess

import pytest

# Sample content for each file type - designed to be parseable and representative

SAMPLE_PYTHON = '''"""A sample Python module for testing."""


def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


class Calculator:
    """A simple calculator class."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''

SAMPLE_JAVASCRIPT = """/**
 * A sample JavaScript module for testing.
 */

function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
    
    subtract(a, b) {
        return a - b;
    }
}

const multiply = (a, b) => a * b;

export { greet, Calculator, multiply };
"""

SAMPLE_JSX = """import React from 'react';

/**
 * A sample React component.
 */
function Greeting({ name }) {
    return (
        <div className="greeting">
            <h1>Hello, {name}!</h1>
        </div>
    );
}

export default Greeting;
"""

SAMPLE_TYPESCRIPT = """/**
 * A sample TypeScript module for testing.
 */

interface Person {
    name: string;
    age: number;
}

function greet(person: Person): string {
    return `Hello, ${person.name}!`;
}

class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
    
    subtract(a: number, b: number): number {
        return a - b;
    }
}

export { Person, greet, Calculator };
"""

SAMPLE_TSX = """import React from 'react';

interface GreetingProps {
    name: string;
}

const Greeting: React.FC<GreetingProps> = ({ name }) => {
    return (
        <div className="greeting">
            <h1>Hello, {name}!</h1>
        </div>
    );
};

export default Greeting;
"""

SAMPLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample HTML</title>
</head>
<body>
    <header>
        <h1>Welcome</h1>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <section id="home">
            <p>This is a sample HTML file for testing.</p>
        </section>
    </main>
</body>
</html>
"""

SAMPLE_CSS = """/* Sample CSS file for testing */

:root {
    --primary-color: #3498db;
    --secondary-color: #2ecc71;
}

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.greeting {
    color: var(--primary-color);
    font-size: 2rem;
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
}
"""

SAMPLE_JSON = """{
    "name": "sample-project",
    "version": "1.0.0",
    "description": "A sample JSON file for testing",
    "dependencies": {
        "lodash": "^4.17.21",
        "express": "^4.18.0"
    },
    "scripts": {
        "start": "node index.js",
        "test": "jest"
    },
    "keywords": ["sample", "test"],
    "author": "Test Author"
}
"""

SAMPLE_YAML = """# Sample YAML configuration file
name: sample-config
version: "1.0"

database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret

services:
  - name: web
    port: 8080
    replicas: 3
  - name: api
    port: 3000
    replicas: 2

settings:
  debug: true
  log_level: info
"""

SAMPLE_YML = """# Docker Compose sample
version: "3.8"

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - db

  db:
    image: postgres:14
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
"""

SAMPLE_TOML = """# Sample TOML configuration file
[project]
name = "sample-project"
version = "1.0.0"
description = "A sample TOML file for testing"
authors = ["Test Author <test@example.com>"]

[dependencies]
lodash = "^4.17.21"
express = "^4.18.0"

[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.black]
line-length = 88
target-version = ["py39", "py310"]

[[services]]
name = "web"
port = 8080
replicas = 3

[[services]]
name = "api"
port = 3000
replicas = 2
"""

SAMPLE_MARKDOWN = """# Sample Markdown Document

This is a sample markdown file for testing the parser.

## Features

- Feature one
- Feature two
- Feature three

## Code Example

```python
def hello():
    print("Hello, World!")
```

## Table

| Name | Description |
|------|-------------|
| Item1 | First item |
| Item2 | Second item |

## Links

[Visit GitHub](https://github.com)
"""

SAMPLE_MKD = """# Alternative Markdown

This uses the .mkd extension.

## Section

Some content here.
"""

SAMPLE_MARKDOWN_FULL = """# Full Markdown Extension

This uses the .markdown extension.

## Another Section

More content here with **bold** and *italic* text.
"""

SAMPLE_RUST = """//! A sample Rust module for testing.

/// Greets a person by name.
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

/// A simple calculator struct.
pub struct Calculator;

impl Calculator {
    /// Creates a new Calculator instance.
    pub fn new() -> Self {
        Calculator
    }
    
    /// Adds two numbers.
    pub fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }
    
    /// Subtracts b from a.
    pub fn subtract(&self, a: i32, b: i32) -> i32 {
        a - b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_greet() {
        assert_eq!(greet("World"), "Hello, World!");
    }
}
"""

SAMPLE_TXT = """This is a plain text file.

It should be parsed by the ChunkParser since .txt
is not a recognized language extension.

The ChunkParser handles any file type that doesn't
have a specific language parser.
"""

SAMPLE_UNKNOWN = """This file has an unknown extension.
It should also be handled by ChunkParser.
"""

SAMPLE_README = """# Test Repository

This is a sample README file without extension.
Should be handled by ChunkParser.
"""

SAMPLE_GITIGNORE = """# Git ignore file
node_modules/
__pycache__/
*.pyc
.env
dist/
build/
"""


@pytest.fixture
def sample_git_repo(tmp_path):
    """
    Create a temporary git repository with sample files of all supported types.

    This fixture creates a realistic project structure with:
    - All file types supported by BaseLanguageParser subclasses
    - Files that should fall back to ChunkParser
    - Nested directory structure
    - Hidden files (dotfiles)
    - Files with multiple extensions

    Returns:
        Path: The path to the temporary git repository
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Create directory structure
    (repo_path / "src").mkdir()
    (repo_path / "src" / "components").mkdir()
    (repo_path / "src" / "utils").mkdir()
    (repo_path / "styles").mkdir()
    (repo_path / "config").mkdir()
    (repo_path / "docs").mkdir()
    (repo_path / "tests").mkdir()

    # Define all files to create with their content
    files = {
        # Python files
        "src/main.py": SAMPLE_PYTHON,
        "src/utils/helpers.py": SAMPLE_PYTHON,
        "tests/test_main.py": SAMPLE_PYTHON,
        # JavaScript files
        "src/index.js": SAMPLE_JAVASCRIPT,
        "src/utils/math.js": SAMPLE_JAVASCRIPT,
        # JSX files
        "src/components/Greeting.jsx": SAMPLE_JSX,
        # TypeScript files
        "src/app.ts": SAMPLE_TYPESCRIPT,
        "src/utils/types.ts": SAMPLE_TYPESCRIPT,
        # TSX files
        "src/components/App.tsx": SAMPLE_TSX,
        # HTML files
        "index.html": SAMPLE_HTML,
        "docs/guide.html": SAMPLE_HTML,
        # CSS files
        "styles/main.css": SAMPLE_CSS,
        "styles/components.css": SAMPLE_CSS,
        # JSON files
        "package.json": SAMPLE_JSON,
        "config/settings.json": SAMPLE_JSON,
        "tsconfig.json": SAMPLE_JSON,
        # YAML files
        "config/config.yaml": SAMPLE_YAML,
        "docker-compose.yml": SAMPLE_YML,
        ".github/workflows/ci.yml": SAMPLE_YML,
        # TOML files
        "pyproject.toml": SAMPLE_TOML,
        "config/app.toml": SAMPLE_TOML,
        "Cargo.toml": SAMPLE_TOML,
        # Markdown files (all extensions)
        "README.md": SAMPLE_MARKDOWN,
        "docs/guide.mkd": SAMPLE_MKD,
        "docs/full.markdown": SAMPLE_MARKDOWN_FULL,
        # Rust files
        "src/lib.rs": SAMPLE_RUST,
        "src/utils/calc.rs": SAMPLE_RUST,
        # Files for ChunkParser (unknown extensions)
        "notes.txt": SAMPLE_TXT,
        "data.csv": "name,value\nitem1,100\nitem2,200",
        "config.xml": '<?xml version="1.0"?>\n<config><setting>value</setting></config>',
        "script.sh": "#!/bin/bash\necho 'Hello World'",
        "Makefile": "all:\n\techo 'Building...'",
        "Dockerfile": "FROM python:3.12\nWORKDIR /app",
        # Dotfiles (hidden files)
        ".gitignore": SAMPLE_GITIGNORE,
        ".env.example": "DATABASE_URL=postgres://localhost/db",
        ".eslintrc.json": '{"extends": "eslint:recommended"}',
        # Files with multiple dots
        "config.prod.json": SAMPLE_JSON,
        "styles.module.css": SAMPLE_CSS,
        "test.spec.ts": SAMPLE_TYPESCRIPT,
        # Edge cases
        "empty.py": "",  # Empty Python file
        "empty.json": "{}",  # Minimal JSON
        "empty.yaml": "",  # Empty YAML
    }

    # Create all files
    for filepath, content in files.items():
        file_path = repo_path / filepath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    # Initialize git repository
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    return repo_path


@pytest.fixture
def sample_files_mapping():
    """
    Return a mapping of file paths to their expected parser types.

    This is useful for parametrized tests that need to verify
    the correct parser is returned for each file type.
    """
    from indexter.parsers.chunk import ChunkParser
    from indexter.parsers.css import CssParser
    from indexter.parsers.html import HtmlParser
    from indexter.parsers.javascript import JavaScriptParser
    from indexter.parsers.json import JsonParser
    from indexter.parsers.markdown import MarkdownParser
    from indexter.parsers.python import PythonParser
    from indexter.parsers.rust import RustParser
    from indexter.parsers.toml import TomlParser
    from indexter.parsers.typescript import TypeScriptParser
    from indexter.parsers.yaml import YamlParser

    return {
        # Python
        "src/main.py": PythonParser,
        "src/utils/helpers.py": PythonParser,
        "tests/test_main.py": PythonParser,
        # JavaScript
        "src/index.js": JavaScriptParser,
        "src/utils/math.js": JavaScriptParser,
        # JSX
        "src/components/Greeting.jsx": JavaScriptParser,
        # TypeScript
        "src/app.ts": TypeScriptParser,
        "src/utils/types.ts": TypeScriptParser,
        # TSX
        "src/components/App.tsx": TypeScriptParser,
        # HTML
        "index.html": HtmlParser,
        "docs/guide.html": HtmlParser,
        # CSS
        "styles/main.css": CssParser,
        "styles/components.css": CssParser,
        # JSON
        "package.json": JsonParser,
        "config/settings.json": JsonParser,
        "tsconfig.json": JsonParser,
        # YAML
        "config/config.yaml": YamlParser,
        "docker-compose.yml": YamlParser,
        ".github/workflows/ci.yml": YamlParser,
        # TOML
        "pyproject.toml": TomlParser,
        "config/app.toml": TomlParser,
        "Cargo.toml": TomlParser,
        # Markdown
        "README.md": MarkdownParser,
        "docs/guide.mkd": MarkdownParser,
        "docs/full.markdown": MarkdownParser,
        # Rust
        "src/lib.rs": RustParser,
        "src/utils/calc.rs": RustParser,
        # ChunkParser (unknown extensions)
        "notes.txt": ChunkParser,
        "data.csv": ChunkParser,
        "config.xml": ChunkParser,
        "script.sh": ChunkParser,
        "Makefile": ChunkParser,
        "Dockerfile": ChunkParser,
        # Dotfiles
        ".gitignore": ChunkParser,
        ".env.example": ChunkParser,
        ".eslintrc.json": JsonParser,
        # Multiple dots
        "config.prod.json": JsonParser,
        "styles.module.css": CssParser,
        "test.spec.ts": TypeScriptParser,
        # Edge cases
        "empty.py": PythonParser,
        "empty.json": JsonParser,
        "empty.yaml": YamlParser,
    }
