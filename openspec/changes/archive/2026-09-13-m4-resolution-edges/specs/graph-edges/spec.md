## ADDED Requirements

### Requirement: Containment edges mirror parent links

For every node with a parent, the graph SHALL hold exactly one `contains` edge from the parent's ID to the node's ID, with no line and confidence `exact`. No other `contains` edges SHALL exist.

#### Scenario: Method contained by its class

- **WHEN** a file defining class `Handler` with method `login` is synced
- **THEN** a `contains` edge runs from `Handler`'s node ID to `login`'s node ID

#### Scenario: Top-level symbol contained by its file

- **WHEN** a file defining a top-level function is synced
- **THEN** a `contains` edge runs from the file node to that function

#### Scenario: Removed symbol loses its containment edge

- **WHEN** a method is deleted from a class and the file is synced
- **THEN** no `contains` edge to that method remains

### Requirement: Resolved references produce edges of their kind

A `resolved` reference SHALL produce one edge of the reference's kind (`calls`, `imports`, or `inherits`) from its source to its target, carrying the reference's line and confidence. An `ambiguous` reference SHALL produce one edge per candidate with confidence `ambiguous`. An `external` reference SHALL produce an edge to its external module node only when its kind is `imports`. `too_ambiguous` and `failed` references SHALL produce no edges. The edge source SHALL be the reference's origin node, except for a reference carrying a `for_type`, whose edge source SHALL be the resolved type. Edges that would be identical in source, target, kind, and line SHALL be stored once.

#### Scenario: Call edge carries line and confidence

- **WHEN** a function on line 10 calls a same-file function
- **THEN** a `calls` edge runs from the caller to the callee with line 10 and confidence `exact`

#### Scenario: Ambiguous call fans out

- **WHEN** a reference is `ambiguous` with three candidates
- **THEN** three `calls` edges exist from its origin, one to each candidate, each with confidence `ambiguous`

#### Scenario: Import of an external package

- **WHEN** a file contains `import pydantic`
- **THEN** an `imports` edge runs from the file node to `external::pydantic`

#### Scenario: Call into an external package

- **WHEN** a function calls `pydantic.Field()` through an import of `pydantic`
- **THEN** the reference is `external` with target `external::pydantic` and no `calls` edge is produced for it

#### Scenario: Import edge targets the most specific node

- **WHEN** a file contains `from pkg.auth import login` and `login` resolves to a function
- **THEN** the `imports` edge targets that function rather than `pkg/auth.py`'s file node

#### Scenario: Inheritance edge

- **WHEN** class `Admin` declares base class `User` defined in another file of the repository
- **THEN** an `inherits` edge runs from `Admin` to `User`

#### Scenario: Two calls on one line

- **WHEN** a line contains `f(f(x))`
- **THEN** exactly one `calls` edge from the caller to `f` exists for that line

### Requirement: External modules are nodes

Each external module targeted by at least one reference SHALL exist as a node with ID `external::<name>`, kind `external_module`, name and qualified name equal to `<name>`, an empty file path, and no language, with a full-text row holding its name. An external module node targeted by no reference SHALL be deleted together with its full-text row. External module nodes SHALL NOT be embedded.

#### Scenario: External node created on first import

- **WHEN** a file importing `requests` is synced into a repository that previously imported nothing named `requests`
- **THEN** a node `external::requests` of kind `external_module` exists and is found by a full-text search for `requests`

#### Scenario: External node removed when no longer imported

- **WHEN** the only file importing `requests` is deleted and the repository synced
- **THEN** no node `external::requests` remains

#### Scenario: One node per package across languages

- **WHEN** a Python file imports `yaml` and a JavaScript file imports `yaml`
- **THEN** exactly one `external::yaml` node exists and both files have `imports` edges to it

#### Scenario: External nodes have no vectors

- **WHEN** a sync completes
- **THEN** no vector exists for any node of kind `external_module`

### Requirement: Node degree counts graph edges

Each node's `degree` SHALL equal the number of `calls`, `imports`, and `inherits` edges having that node as source or target, at any confidence. `contains` edges SHALL NOT count toward degree.

#### Scenario: Degree of a called function

- **WHEN** a function is called from three other functions and calls one function itself
- **THEN** its degree is 4

#### Scenario: Containment does not add degree

- **WHEN** a class contains ten methods and has no other edges
- **THEN** its degree is 0

#### Scenario: Degree follows edge changes

- **WHEN** one of a function's three callers is deleted and the repository synced
- **THEN** the function's degree decreases by one

### Requirement: The graph is consistent after every sync

After a sync completes, every edge's source and target, and every non-null reference target and candidate, SHALL name an existing node; the stored edges SHALL equal exactly the edges derived from the current nodes and reference outcomes; and resolution SHALL change only the rows whose derived values differ from the stored ones.

#### Scenario: No dangling edges after a deletion

- **WHEN** a file whose functions are called from other files is deleted and the repository synced
- **THEN** no edge and no reference target names any node from that file

#### Scenario: Unchanged graph rows are not rewritten

- **WHEN** a comment is added inside one function and the repository synced
- **THEN** edges unaffected by that file keep their stored row IDs

#### Scenario: Re-deriving from scratch matches incremental results

- **WHEN** a repository is synced incrementally through a series of edits, and separately indexed from scratch at its final contents
- **THEN** both databases hold the same set of edges with the same confidences and the same reference outcomes

### Requirement: Resolution summary is queryable

The system SHALL provide a summary computed from stored rows giving reference counts by kind and status (with confidence for resolved references), edge counts by kind and confidence, and the number of external module nodes.

#### Scenario: Summary matches stored rows

- **WHEN** the summary is computed for a synced repository
- **THEN** its counts equal the counts obtained by grouping `refs` by kind, status, and confidence and `edges` by kind and confidence
