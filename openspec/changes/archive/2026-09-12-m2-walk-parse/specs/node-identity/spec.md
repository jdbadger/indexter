## ADDED Requirements

### Requirement: Deterministic node ID format

Every node SHALL carry a text ID of the form `<relative path>::<scope path>.<name>#<kind>`, where the scope path and its trailing separator are omitted for nodes at file scope. IDs SHALL be derived purely from those components, so the same source always produces the same ID.

#### Scenario: Method inside a class

- **WHEN** a method `login` is parsed inside class `AuthHandler` in `src/auth/handlers.py`
- **THEN** its ID is `src/auth/handlers.py::AuthHandler.login#method`

#### Scenario: Function at file scope

- **WHEN** a function `login` is parsed at file scope in `src/auth/handlers.py`
- **THEN** its ID is `src/auth/handlers.py::login#function`

#### Scenario: The file node's ID

- **WHEN** a file node is produced for `src/auth/handlers.py`
- **THEN** its ID identifies the file itself and carries kind `file`

#### Scenario: IDs are reproducible

- **WHEN** the same file is parsed twice
- **THEN** every node receives the same ID both times

### Requirement: IDs survive formatting and location changes

A node's ID SHALL NOT depend on its line numbers, byte offsets, body text, docstring, or surrounding whitespace. It SHALL change when the symbol is renamed, moved to a different scope, or moved to a different file.

#### Scenario: Inserting code above a symbol

- **WHEN** lines are inserted above a function, shifting its start line
- **THEN** its ID is unchanged

#### Scenario: Reformatting a body

- **WHEN** a function's body or docstring is edited without renaming it
- **THEN** its ID is unchanged

#### Scenario: Renaming a symbol

- **WHEN** a function is renamed
- **THEN** its ID changes

#### Scenario: Moving a symbol into a class

- **WHEN** a file-scope function is moved inside a class
- **THEN** its ID changes to reflect the new scope and kind

### Requirement: Scope paths hold the full ancestor chain

Every node SHALL carry a scope path listing all enclosing named scopes in order, outermost first, not merely the nearest enclosing class.

#### Scenario: Nested Python functions

- **WHEN** a Python file defines function `inner` inside function `outer`
- **THEN** `inner`'s scope path is `outer` and its ID is distinct from a file-scope `inner` in the same file

#### Scenario: Two nested functions with the same name

- **WHEN** a Python file defines an `inner` inside `outer_a` and another `inner` inside `outer_b`
- **THEN** the two receive different IDs

#### Scenario: Method inside a nested class

- **WHEN** a method is defined in a class that is itself defined inside another class
- **THEN** its scope path lists both classes, outermost first

#### Scenario: JavaScript callback inside a method

- **WHEN** a named function expression `cb` appears inside method `m` of class `A`
- **THEN** its scope path is `A` then `m`, rather than `A` alone

#### Scenario: JavaScript object-literal method

- **WHEN** an object literal assigned to `obj` contains a method `handler`
- **THEN** `handler`'s scope path is `obj` rather than empty

#### Scenario: Two object literals with same-named methods

- **WHEN** one file assigns two different object literals, each with a `handler` method
- **THEN** the two methods receive different IDs

### Requirement: Rust trait implementations are distinguished

A Rust `impl` block SHALL contribute a scope segment that includes the implemented trait when one is present, formed from the implemented type and the final segment of the trait path. A plain inherent `impl` SHALL contribute only the type.

#### Scenario: Two trait impls for the same type

- **WHEN** a Rust file contains `impl std::fmt::Display for Foo` and `impl std::fmt::Debug for Foo`, each defining `fmt`
- **THEN** the two `fmt` methods receive different IDs, scoped `Foo<Display>` and `Foo<Debug>` respectively

#### Scenario: Inherent impl

- **WHEN** a Rust file contains `impl Foo` defining `new`
- **THEN** `new`'s scope path is `Foo`

#### Scenario: Trait path is reduced to its last segment

- **WHEN** a trait is implemented by its fully qualified path
- **THEN** only the final path segment appears in the scope segment

### Requirement: Genuine duplicates are disambiguated by line order

When two or more nodes in one file would otherwise receive the same ID, the system SHALL append a `~N` suffix, numbering them in ascending start-line order with the first occurrence left unsuffixed.

#### Scenario: Two same-named functions in one file

- **WHEN** a file defines `login` twice at file scope, both as functions
- **THEN** the earlier receives an unsuffixed ID and the later receives the same ID with `~2` appended

#### Scenario: Numbering follows line order, not emission order

- **WHEN** a parser emits duplicate-identity nodes in an order other than their position in the file
- **THEN** the suffixes still follow ascending start line

#### Scenario: Distinct kinds are not duplicates

- **WHEN** a file defines a class and a function with the same name at the same scope
- **THEN** neither ID is suffixed, because their kinds differ

#### Scenario: Unique names are never suffixed

- **WHEN** every node in a file has a distinct identity
- **THEN** no ID carries a suffix

### Requirement: Nodes are linked to their containing node

Every node SHALL carry the ID of the innermost node that encloses it within the same file. Nodes at file scope SHALL be linked to that file's `file` node, and the `file` node itself SHALL have no parent.

#### Scenario: Method links to its class

- **WHEN** a class containing a method is parsed
- **THEN** the method's parent is the class's ID

#### Scenario: File-scope symbol links to the file node

- **WHEN** a function at file scope is parsed
- **THEN** its parent is the file node's ID

#### Scenario: The file node has no parent

- **WHEN** a file node is produced
- **THEN** it has no parent

#### Scenario: Deeply nested symbol links to its immediate parent

- **WHEN** a function is nested inside another function inside a class
- **THEN** its parent is the enclosing function, not the class or the file
