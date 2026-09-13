## ADDED Requirements

### Requirement: Methods roll up into their class

Selection SHALL walk the fused ranking in order, grouping each node under its container when its parent is a class, struct, trait, interface or enum, under itself when it is one of those kinds, and alone otherwise. A node whose group already has an entry SHALL join it; any other node SHALL start a new entry at its position. Selection SHALL stop when `limit` entries exist or the ranking is exhausted. An entry holding the class-like node and at least one member, or at least two members, SHALL be a class entry: the container node with its matched members in rank order and the union of their match reasons. Any other entry SHALL be a hit for its single node.

#### Scenario: Two methods collapse

- **WHEN** the 2nd and 5th ranked nodes are methods of class `Store`
- **THEN** one class entry for `Store` appears at the 2nd position, listing both methods, and the freed slot is filled by the next ranked node

#### Scenario: Class absorbs its own method

- **WHEN** class `Store` ranks 1st and its method `Store.add` ranks 3rd
- **THEN** a single class entry for `Store` lists `Store.add` as a matched member

#### Scenario: One method stays a method hit

- **WHEN** exactly one method of a class is among the selected nodes and the class itself is not
- **THEN** the entry is a hit for that method, with the class as its container

#### Scenario: Limit counts entries, not nodes

- **WHEN** `limit` is 3 and the top 5 ranked nodes are three methods of one class and two functions
- **THEN** the response selects the class entry and both functions

### Requirement: Hits carry real code and identification

Every entry SHALL carry its node's ID, qualified name, kind, file path, start and end lines, match reasons, signature and docstring, and a snippet. Class entries SHALL also carry, per matched member, its qualified name, start and end lines and signature. The snippet SHALL be read from the file on disk after synchronization, decoded as indexing decodes it, and cut by the node's byte range — the best-ranked node's range for a class entry. A snippet longer than `snippet_max_lines` lines SHALL keep the first and last lines with a single marker line stating how many lines were elided, totaling exactly `snippet_max_lines` lines. Lines longer than 240 characters SHALL be cut with a marker. A file that cannot be read SHALL yield an entry with a note in place of the snippet.

#### Scenario: Snippet is the node's source

- **WHEN** a hit is a 12-line function
- **THEN** its snippet is exactly those 12 lines as they are on disk

#### Scenario: Long function is middle-elided

- **WHEN** a hit is a 100-line function and `snippet_max_lines` is 40
- **THEN** its snippet is 40 lines: the first 20, a marker saying 61 lines were elided, and the last 19

#### Scenario: Minified line is cut

- **WHEN** a hit's source contains a 5,000-character line
- **THEN** that line appears in the snippet cut to 240 characters with a marker

#### Scenario: Class entry snippet follows its best node

- **WHEN** a class entry's best-ranked node is its method `Store.add`
- **THEN** the entry's snippet is the source of `Store.add`

#### Scenario: Non-ASCII content

- **WHEN** a hit's file contains multi-byte characters before and inside the node
- **THEN** the snippet contains exactly the node's characters

#### Scenario: File vanished after sync

- **WHEN** a hit's file cannot be read while rendering
- **THEN** the entry is returned with its other fields and a note that the snippet is unavailable

### Requirement: Responses fit a character budget with intact hits

The rendered response SHALL NOT exceed the `search_max_chars` setting (default 20,000), except that the first entry SHALL always be included. Entries SHALL be admitted in rank order, each counted at its rendered size including any file heading it introduces, and admission SHALL stop at the first entry that does not fit; no later entry SHALL be admitted in its place. No entry SHALL be partially rendered. Related nodes SHALL then be admitted in order into the remaining budget, each dropped if it does not fit. The response SHALL report how many selected entries and related nodes were omitted for the budget.

#### Scenario: Budget returns fewer hits

- **WHEN** 10 entries are selected and only the first 6 fit within the budget
- **THEN** the response contains 6 entries, reports 4 omitted, and contains no part of the 7th

#### Scenario: Smaller later hits are not pulled forward

- **WHEN** the 4th entry does not fit but the 5th would
- **THEN** the response contains 3 entries

#### Scenario: Oversized first hit

- **WHEN** the first entry alone exceeds the budget
- **THEN** the response contains that entry only

#### Scenario: Hits take priority over related

- **WHEN** entries use nearly all of the budget
- **THEN** related nodes that do not fit are omitted and counted, and no entry is dropped for them

### Requirement: Responses are grouped by file and rendered deterministically

A response SHALL render as plain text: a header line with the result count, the query, and any omitted count; then entries grouped under a heading per file, files ordered by their best entry's rank and entries within a file by rank; each entry showing its qualified name, kind, `path:start-end`, match reasons, ID, container and caller/callee summary, signature, docstring, matched members for class entries, and snippet; then a `related` section listing each related node's qualified name, kind, `path:start-end` and reason. The same database state, query, filters, limit and settings SHALL render identical text.

#### Scenario: Grouped by file

- **WHEN** entries 1 and 3 are in `a.py` and entry 2 is in `b.py`
- **THEN** the rendering shows `a.py` with entries 1 and 3, then `b.py` with entry 2

#### Scenario: Locations are usable directly

- **WHEN** any entry or related node is rendered
- **THEN** its location appears as `path:start-end` with a repository-relative path

#### Scenario: Stable output

- **WHEN** the same search is run twice with no changes in between
- **THEN** both renderings are identical

#### Scenario: No results

- **WHEN** no node matches the query and filters
- **THEN** the rendering states that there are no results for the query and has no `related` section
