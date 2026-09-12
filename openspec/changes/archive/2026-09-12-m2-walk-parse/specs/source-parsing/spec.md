## ADDED Requirements

### Requirement: Parser selection by file extension

The system SHALL select a parser for a file from its lowercased extension, covering Python, JavaScript, TypeScript, Rust, Markdown, JSON, YAML, TOML, HTML, and CSS. Files with no registered extension SHALL fall back to a chunking parser, so every walked file yields something.

#### Scenario: Registered extension selects its language parser

- **WHEN** a file with a registered extension such as `.py`, `.ts`, or `.rs` is parsed
- **THEN** the parser for that language handles it and the resulting nodes carry that language

#### Scenario: Extension matching is case-insensitive

- **WHEN** a file's extension differs only in case, such as `.PY`
- **THEN** the same parser is selected as for the lowercase form

#### Scenario: Unregistered extension falls back to chunking

- **WHEN** a file with an unrecognized extension is parsed
- **THEN** the chunking parser handles it and yields chunk nodes covering the file

#### Scenario: Parser instances are reused across files

- **WHEN** several files of the same language are parsed in sequence
- **THEN** the same parser instance is reused rather than constructed per file

### Requirement: Queries are compiled once per parser

Each language parser SHALL compile its tree-sitter queries when the parser is constructed, not when a file is parsed. Parsing SHALL NOT recompile a query.

#### Scenario: Query compilation happens at construction

- **WHEN** a language parser is constructed and then used to parse several files
- **THEN** its queries are compiled exactly once, regardless of how many files are parsed

#### Scenario: Repeated parses are independent

- **WHEN** the same parser instance parses two different files in sequence
- **THEN** each file's results reflect only its own contents, with no state carried over from the previous parse

### Requirement: Parsed nodes carry the fields the database stores

Parsing a file SHALL yield nodes carrying kind, name, scope path, language, start and end line, start and end byte, signature where the language has one, and docstring where the language has one. Node bodies SHALL NOT be carried on the node, since snippets are read from disk by byte range.

#### Scenario: A function yields a complete node

- **WHEN** a Python file containing a documented function is parsed
- **THEN** the node for it carries kind `function`, the function's name, its 1-indexed start and end lines, its start and end byte offsets, its signature, and its docstring

#### Scenario: Byte ranges address the original source

- **WHEN** a node's start and end byte offsets are used to slice the file's bytes
- **THEN** the slice is exactly the source text of that construct

#### Scenario: Missing optional fields are absent, not fabricated

- **WHEN** a construct has no docstring or no signature
- **THEN** those fields are empty rather than filled with a placeholder

#### Scenario: Node bodies are not carried

- **WHEN** any node is produced
- **THEN** it carries no field holding the construct's source text

### Requirement: Every parsed file yields a file node

Parsing SHALL yield exactly one node of kind `file` per file, spanning the whole file, in addition to any symbol nodes.

#### Scenario: File node accompanies symbol nodes

- **WHEN** a source file containing several symbols is parsed
- **THEN** exactly one node of kind `file` is yielded alongside the symbol nodes, and its byte range spans the whole file

#### Scenario: File with no recognizable symbols

- **WHEN** a source file containing no matching constructs is parsed
- **THEN** the `file` node is still yielded

### Requirement: Kinds come from a closed vocabulary

Every node's kind SHALL be drawn from a fixed set: `file`, `class`, `function`, `method`, `constant`, `interface`, `type_alias`, `enum`, `struct`, `trait`, `section`, `data`, and `chunk`. Parsers SHALL normalize language-specific construct names into this set. The kind `external_module` SHALL NOT be produced by parsing.

#### Scenario: Language constructs normalize to the vocabulary

- **WHEN** files of each supported language are parsed
- **THEN** every node's kind is a member of the fixed set

#### Scenario: Functions inside a class are methods

- **WHEN** a class containing a function definition is parsed
- **THEN** the enclosed function's kind is `method` while a function at file scope is `function`

#### Scenario: Prose and markup headings become sections

- **WHEN** a Markdown file with headings, or an HTML or CSS file with structural blocks, is parsed
- **THEN** those nodes carry kind `section` rather than a language-specific label such as a heading level or at-rule name

#### Scenario: Structured data blocks become data nodes

- **WHEN** a JSON, YAML, or TOML file is parsed
- **THEN** its blocks carry kind `data`

#### Scenario: Fallback chunks are marked as chunks

- **WHEN** a file is handled by the chunking fallback
- **THEN** its nodes carry kind `chunk`

#### Scenario: external_module is reserved

- **WHEN** any file of any supported language is parsed
- **THEN** no node carries kind `external_module`

### Requirement: Imports and exports are not nodes

Parsing SHALL NOT emit nodes for import or export statements. An import SHALL instead be recorded as a reference; an export SHALL be recorded against the symbol it exports rather than as a separate node.

#### Scenario: Python imports produce no nodes

- **WHEN** a Python file containing `import` and `from ... import ...` statements is parsed
- **THEN** no node is produced for either statement

#### Scenario: TypeScript exports produce no separate node

- **WHEN** a TypeScript file containing an exported class is parsed
- **THEN** exactly one node is produced for that class, and no additional node for the export statement

### Requirement: Duplicate suppression for wrapped definitions

Parsing SHALL yield one node per source construct where a language wraps a definition in another node — decorated definitions in Python and export-wrapped declarations in TypeScript.

#### Scenario: A decorated Python function yields one node

- **WHEN** a Python file containing a decorated function is parsed
- **THEN** exactly one node is produced for that function, and its byte range includes the decorator

#### Scenario: An exported TypeScript declaration yields one node

- **WHEN** a TypeScript file containing an exported function declaration is parsed
- **THEN** exactly one node is produced for that function

### Requirement: Parse failures are reported per file, not raised

Parsing SHALL return any errors encountered alongside the nodes and references it produced, rather than raising. A file that fails to parse SHALL NOT prevent other files from being parsed.

#### Scenario: A file with syntax errors still yields what parsed

- **WHEN** a source file containing a syntax error is parsed
- **THEN** nodes for the constructs that did parse are returned, and the result reports the problem rather than raising

#### Scenario: An unexpected parser failure is contained

- **WHEN** a parser raises an unexpected exception while processing a file
- **THEN** the failure is captured in that file's result and parsing other files is unaffected

#### Scenario: Empty file

- **WHEN** an empty or whitespace-only file is parsed
- **THEN** the result is returned without error
