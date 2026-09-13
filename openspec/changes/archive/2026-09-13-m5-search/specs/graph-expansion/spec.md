## ADDED Requirements

### Requirement: Every hit carries its immediate graph context

Each result entry SHALL carry, for its node (the class for a rolled-up class entry): up to 3 callers — sources of incoming `calls` edges — and up to 3 callees — targets of outgoing `calls` edges — each ordered by confidence (`exact`, `imported`, `unique_name`, `ambiguous`) then node ID, each with its node ID, qualified name and edge confidence, together with the total number of callers and callees; and its container, the parent node when the parent is a class, struct, trait, interface or enum.

#### Scenario: Callers and callees

- **WHEN** a hit's function is called from two functions and calls four others
- **THEN** the entry lists both callers with a total of 2, and three callees with a total of 4

#### Scenario: Confident edges listed first

- **WHEN** a hit has one `ambiguous` caller and three `exact` callers
- **THEN** the three listed callers are the `exact` ones and the total is 4

#### Scenario: Method container

- **WHEN** a hit is a method of class `Walker`
- **THEN** the entry's container is `Walker`

#### Scenario: Top-level function has no container

- **WHEN** a hit is a module-level function
- **THEN** the entry has no container

### Requirement: Related nodes are reached by one hop from the top hits

A search SHALL compute a `related` list separate from the ranked entries and SHALL never insert related nodes into the ranking. Seeds SHALL be the best-ranked node of each of the first 5 entries included in the response, a seed at entry position `p` weighing `1 / (60 + p)`. From each seed the search SHALL follow `calls`, `inherits` and `imports` edges in both directions, and `contains` edges in both directions except those with a `file` node at either end. The node across each followed edge SHALL be a candidate unless it is an external module, its degree exceeds 40, or it is already shown in the response as an entry's node, a rolled-up member, or a container.

#### Scenario: Callee of a hit is related

- **WHEN** the top hit calls a function that did not match the query
- **THEN** that function is in `related`

#### Scenario: Caller of a hit is related

- **WHEN** a function that did not match the query calls the top hit
- **THEN** that function is in `related`

#### Scenario: Hub damping

- **WHEN** the top hit calls a function whose degree is 41
- **THEN** that function is not in `related`

#### Scenario: Hits are not repeated

- **WHEN** the top hit calls the second hit
- **THEN** the second hit appears only as an entry, not in `related`

#### Scenario: File containment is not followed

- **WHEN** the top hit is a file node containing functions that did not match
- **THEN** those functions are not in `related` through `contains`

#### Scenario: Only included entries seed

- **WHEN** the character budget admits 2 entries
- **THEN** only those 2 entries' nodes are seeds

#### Scenario: Class members are related to a class hit

- **WHEN** the top hit is a class and one of its methods did not match
- **THEN** that method may appear in `related` with a membership reason

### Requirement: Related nodes are scored, and ambiguous edges never promote alone

Each candidate's score SHALL be the sum, over every (seed, edge) pair reaching it, of the seed's weight times the edge's weight, which SHALL be 0.5 for `ambiguous` edges and 1.0 otherwise. A candidate reached only through `ambiguous` edges SHALL be excluded. Candidates SHALL be ordered by score descending, then by the edge kind of their strongest contribution (`calls`, `inherits`, `imports`, `contains`), then node ID, and the first 5 SHALL form `related`.

#### Scenario: Reached from more hits ranks higher

- **WHEN** node X is called by the first and third hits and node Y only by the first
- **THEN** X is listed before Y

#### Scenario: Ambiguous alone is dropped

- **WHEN** a node is reached from the top hit only through an `ambiguous` `calls` edge
- **THEN** it is not in `related`

#### Scenario: Ambiguous adds weight to a confident path

- **WHEN** node X is reached by an `exact` edge from the second hit and an `ambiguous` edge from the first, and node Y only by an `exact` edge from the second hit
- **THEN** X is listed before Y

#### Scenario: At most five

- **WHEN** the top hits have twelve eligible neighbors
- **THEN** `related` holds 5 nodes

### Requirement: Related nodes say why they are there

Each related node SHALL carry its node ID, qualified name, kind, file path with start and end lines, the confidence of its strongest contributing edge, and a reason derived from that contribution — the highest-weight contribution, ties to the earliest seed, then edge kind order — naming the seed and the relationship from the related node's side (called by, calls, base class of, subclass of, imported by, imports, member of, contains), stating that the seed matched, and noting how many other seeds also reached it.

#### Scenario: Reason names the matching hit

- **WHEN** a related function is called by the top hit `authenticate`
- **THEN** its reason reads that it is called by `authenticate`, which matched

#### Scenario: Several seeds

- **WHEN** a related class is the base class of the first and fourth hits
- **THEN** its reason names the first hit and notes one more

#### Scenario: Inheritance direction

- **WHEN** a related class is derived from the top hit
- **THEN** its reason says it is a subclass of the top hit
