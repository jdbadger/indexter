# reference-extraction Specification

## Purpose

The graph's edges — calls, imports, and inheritance — come from references extracted alongside
nodes, but extraction cannot resolve those references to a target yet: attribute chains must be
reduced to a head identifier, per-language import syntax preserved with enough information for
later resolution, and every reference tied to the innermost node that encloses it, all without
raising when a file's language yields no references at all.

This capability owns that extraction: which languages yield references and which don't, the
fields every reference carries, head-identifier reduction for dotted and member chains,
per-language recording of imports and inheritance as references rather than resolved edges, and
the guarantee that every reference's origin matches a node from the same parse.

## Requirements

### Requirement: References are extracted for the four code languages

Parsing a Python, JavaScript, TypeScript, or Rust file SHALL yield references alongside nodes, covering calls, imports, and inheritance. Parsers for prose, markup, and structured-data languages SHALL yield no references.

#### Scenario: Calls, imports, and inheritance are all captured

- **WHEN** a code file containing an import, a class with a base class, and a function call is parsed
- **THEN** references are yielded for all three, each carrying its kind

#### Scenario: Non-code languages yield no references

- **WHEN** a Markdown, JSON, YAML, TOML, HTML, or CSS file is parsed
- **THEN** no references are yielded

#### Scenario: A file with no references

- **WHEN** a code file containing only definitions with no calls, imports, or base classes is parsed
- **THEN** no references are yielded and parsing succeeds

### Requirement: References carry their origin and target text

Every reference SHALL carry the ID of the node it appears inside, the raw reference text as written, the head identifier of that text, its kind, and the 1-indexed line and column where it appears. References SHALL NOT carry a resolved target — resolution happens later.

#### Scenario: A call inside a method

- **WHEN** a method body calls another function
- **THEN** the reference names the enclosing method's node ID as its origin, and carries the called text, its head, kind `calls`, and its line and column

#### Scenario: A reference at file scope

- **WHEN** a call appears at file scope, outside any symbol
- **THEN** its origin is the file node's ID

#### Scenario: References are unresolved when extracted

- **WHEN** any reference is extracted
- **THEN** it carries no resolved target and no confidence, leaving those for resolution

#### Scenario: Line and column locate the reference

- **WHEN** a reference is extracted
- **THEN** its line is 1-indexed and identifies the line the reference appears on

### Requirement: Attribute chains reduce to a head identifier

A reference written as a dotted or member chain SHALL carry the full chain as its raw text and the leftmost base identifier as its head. When the base of a chain is not an identifier — for example a call, a literal, or a subscript — the reference SHALL be recorded with no head.

#### Scenario: Method call on self

- **WHEN** a method body contains `self.validate(user)`
- **THEN** the reference's raw text is `self.validate` and its head is `self`

#### Scenario: Deeply nested attribute chain

- **WHEN** a call is written `os.path.join(a, b)`
- **THEN** the reference's raw text is `os.path.join` and its head is `os`

#### Scenario: Bare function call

- **WHEN** a call is written `authenticate(user)`
- **THEN** the reference's raw text and head are both `authenticate`

#### Scenario: Chained call has no identifier head

- **WHEN** a call is written `build().run()`
- **THEN** the reference for `run` is recorded with no head

### Requirement: Imports are recorded as references, per-language

Import statements SHALL be recorded as references of kind `imports`, carrying the module or symbol text as written, preserving the information a later resolution step needs — relative-import markers in Python, specifier text in JavaScript and TypeScript, and path segments in Rust.

#### Scenario: Python plain import

- **WHEN** a Python file contains `import os`
- **THEN** a reference of kind `imports` is yielded for `os`

#### Scenario: Python from-import

- **WHEN** a Python file contains `from a.b import thing`
- **THEN** a reference of kind `imports` is yielded carrying enough text to identify both the module `a.b` and the imported name `thing`

#### Scenario: Python relative import

- **WHEN** a Python file contains a relative import such as `from . import sibling` or `from ..pkg import thing`
- **THEN** the reference preserves the leading-dot markers so the target can be resolved relative to the importing file

#### Scenario: JavaScript module specifier

- **WHEN** a JavaScript or TypeScript file imports from a module specifier such as `./utils` or `react`
- **THEN** a reference of kind `imports` is yielded carrying that specifier as written

#### Scenario: Rust use declaration

- **WHEN** a Rust file contains a `use` declaration such as `use crate::auth::Handler`
- **THEN** a reference of kind `imports` is yielded carrying the path as written

### Requirement: Inheritance is recorded as references

Base classes, implemented interfaces, and Rust trait implementations SHALL be recorded as references of kind `inherits`, originating from the deriving type.

#### Scenario: Python base class

- **WHEN** a Python class declares a base class
- **THEN** a reference of kind `inherits` is yielded from the subclass to the base class name

#### Scenario: Multiple bases

- **WHEN** a Python class declares two base classes
- **THEN** two references of kind `inherits` are yielded, one per base

#### Scenario: TypeScript extends and implements

- **WHEN** a TypeScript class extends a class and implements an interface
- **THEN** a reference of kind `inherits` is yielded for each

#### Scenario: Rust trait implementation

- **WHEN** a Rust file contains `impl Display for Foo`
- **THEN** a reference of kind `inherits` is yielded from `Foo` to `Display`

### Requirement: Every reference resolves to an origin node

Each reference SHALL originate from the innermost node that encloses its position in the file, so that no reference is orphaned.

#### Scenario: Reference inside a nested function

- **WHEN** a call appears inside a function nested in another function
- **THEN** its origin is the inner function's node ID

#### Scenario: Reference in a class body but outside any method

- **WHEN** a call appears directly in a class body
- **THEN** its origin is the class's node ID

#### Scenario: Every extracted reference names a node from the same parse

- **WHEN** any file is parsed
- **THEN** every reference's origin ID matches the ID of a node produced by that same parse
