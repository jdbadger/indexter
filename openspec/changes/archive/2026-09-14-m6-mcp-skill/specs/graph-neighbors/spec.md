## ADDED Requirements

### Requirement: Neighbors validates, then synchronizes, then reads

A neighbors request SHALL take a node ID, a direction (`in`, `out` or `both`; default `both`), a set of edge kinds (any of `calls`, `imports`, `inherits`, `contains`; default all four), a depth (1–3; default 1) and a limit (1–100; default 20). It SHALL reject a blank node ID, an unknown direction, an unknown edge kind, or a depth or limit out of range with an error naming the parameter, the value and the valid values, before synchronizing. Otherwise it SHALL synchronize the repository's index exactly as search does, then read the graph. Like search, it SHALL fail without creating a database when the repository has no index.

#### Scenario: Invalid direction

- **WHEN** neighbors is requested with direction `up`
- **THEN** it fails naming `direction`, `up` and the valid directions, and no sync runs

#### Scenario: Depth out of range

- **WHEN** neighbors is requested with depth 4
- **THEN** it fails naming `depth` and the range 1–3

#### Scenario: Unknown edge kind

- **WHEN** neighbors is requested with edges `["references"]`
- **THEN** it fails naming `edges`, `references` and the four valid kinds

#### Scenario: New call found after an edit

- **WHEN** a function gains a call to another function on disk after the last sync, and neighbors is requested for it with direction `out`
- **THEN** the called function is listed

#### Scenario: Unindexed repository

- **WHEN** neighbors is requested for a repository with no index
- **THEN** it fails with the same not-indexed error as search and no database is created

### Requirement: Unknown node IDs are reported with suggestions

When the node ID is not in the index after synchronizing, neighbors SHALL fail with an error that names the ID and says it may be stale and to search again. When the ID's file part (before `::`) names an indexed file, the error SHALL also list up to 5 current node IDs in that file whose name equals the name in the requested ID, ordered by ID.

#### Scenario: Renamed kind suggests the current ID

- **WHEN** neighbors is requested for `src/a.py::helper#method` and `src/a.py` now contains `src/a.py::helper#function`
- **THEN** the error lists `src/a.py::helper#function`

#### Scenario: Unknown file

- **WHEN** neighbors is requested for an ID whose file is not indexed
- **THEN** the error names the ID, says to search again, and lists no suggestions

### Requirement: Neighbors walks selected edges breadth-first to the requested depth

Neighbors SHALL walk the graph breadth-first from the start node. At each depth it SHALL follow, from every frontier node, edges of the selected kinds in the selected directions — outgoing edges where the frontier node is the source, incoming edges where it is the target. A node already recorded, or the start node, SHALL NOT be recorded again. Each recorded node SHALL carry its depth, the frontier node it was reached from, and the connecting edge's kind, direction, confidence and line; when several edges reach a node first at the same depth, the recorded edge SHALL be the first by edge kind (`calls`, `inherits`, `imports`, `contains`), then confidence (`exact`, `imported`, `unique_name`, `ambiguous`), then the frontier node's ID, then line. The walk SHALL stop after the requested depth, or once 1,000 nodes are recorded.

#### Scenario: Callers only

- **WHEN** neighbors is requested for a function with direction `in` and edges `["calls"]`
- **THEN** exactly the functions and methods with a `calls` edge to it are listed, at depth 1

#### Scenario: Both directions

- **WHEN** a method is called by one function, calls another, and is contained in a class, and neighbors is requested with defaults
- **THEN** the caller, the callee and the class are listed

#### Scenario: Depth two

- **WHEN** `a` calls `b`, `b` calls `c`, and neighbors is requested for `a` with direction `out`, edges `["calls"]` and depth 2
- **THEN** `b` is listed at depth 1 and `c` at depth 2 reached via `b`

#### Scenario: Shallowest depth wins

- **WHEN** `a` calls `b` and `c`, `b` calls `c`, and neighbors is requested for `a` with direction `out` and depth 2
- **THEN** `c` is listed once, at depth 1

#### Scenario: Importers of an external module

- **WHEN** neighbors is requested for `external::pydantic` with direction `in`
- **THEN** every file with an `imports` edge to it is listed

### Requirement: The walk does not pass through hubs, external modules or ambiguous-only nodes

A recorded node SHALL be listed but SHALL NOT be walked from at the next depth when it is an external module, when its degree exceeds 40, or when every edge that reached it at its depth is `ambiguous`. The start node SHALL always be walked from.

#### Scenario: Hub listed but not walked

- **WHEN** `a` calls hub `h` of degree 41, `h` calls `x`, and neighbors is requested for `a` with direction `out` and depth 2
- **THEN** `h` is listed and `x` is not

#### Scenario: Hub as start node

- **WHEN** neighbors is requested for a node of degree 41 with direction `in`
- **THEN** its callers are listed

#### Scenario: Ambiguous-only node not walked

- **WHEN** `a`'s only edge to `b` is an `ambiguous` call, `b` calls `c`, and neighbors is requested for `a` with direction `out` and depth 2
- **THEN** `b` is listed with confidence `ambiguous` and `c` is not listed

### Requirement: Neighbors are ordered, limited and budgeted

Recorded nodes SHALL be ordered by depth, then edge kind (`calls`, `inherits`, `imports`, `contains`), then confidence (`exact`, `imported`, `unique_name`, `ambiguous`), then node ID. The first `limit` SHALL be returned and the remainder counted as omitted. The rendered response SHALL also fit `search_max_chars`: items SHALL be admitted in order until the first that does not fit, and the rest counted as omitted. When the walk stopped at 1,000 nodes, the omitted count SHALL be reported as a lower bound.

#### Scenario: Limit

- **WHEN** a function has 30 callers and neighbors is requested with direction `in` and limit 20
- **THEN** 20 callers are returned and 10 are reported omitted

#### Scenario: Confident edges first

- **WHEN** a function has one `ambiguous` caller and one `exact` caller
- **THEN** the `exact` caller is listed first

#### Scenario: Character budget

- **WHEN** the rendered items would exceed `search_max_chars`
- **THEN** fewer items are returned, every returned item is complete, and the rest are reported omitted

#### Scenario: Deterministic order

- **WHEN** the same neighbors request is made twice with no changes to the repository
- **THEN** both responses render identically

### Requirement: Neighbors responses render as text

The rendered response SHALL begin with the start node's qualified name, kind, `path:start-end`, node ID and the request's direction, edge kinds and depth, and the number of items shown and omitted. Each item SHALL show the relation from the node it was reached from (`calls` / `called by`, `inherits from` / `inherited by`, `imports` / `imported by`, `contains` / `contained in`), the neighbor's qualified name, kind, `path:start-end` (none for an external module), the edge confidence and line when present, `via` and the intermediate node's qualified name when its depth is greater than 1, and the neighbor's node ID. Items SHALL NOT include snippets. A start node with no neighbors SHALL render a line saying none were found.

#### Scenario: Caller item

- **WHEN** `Walker.walk` calls the start node with an `exact` edge at line 188
- **THEN** its item shows `called by Walker.walk`, its kind, `path:start-end`, `exact`, line 188 and its node ID

#### Scenario: Via at depth two

- **WHEN** an item is at depth 2 reached through `b`
- **THEN** the item shows `via b`

#### Scenario: No neighbors

- **WHEN** the start node has no edges of the selected kinds in the selected direction
- **THEN** the response renders the header and a line saying no neighbors were found
