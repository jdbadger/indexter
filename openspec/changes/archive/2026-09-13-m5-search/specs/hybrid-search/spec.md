## ADDED Requirements

### Requirement: Search synchronizes before ranking

A search SHALL validate its query, filters and limit, then synchronize the repository's index exactly as an index sync does, and only then retrieve candidates. There SHALL be no way to search without synchronizing. The response SHALL include the synchronization report.

#### Scenario: Edited file is searchable immediately

- **WHEN** a function is added to a file after the last sync and a search whose query matches that function's name is run
- **THEN** the search's sync indexes the change and the new function is among the results

#### Scenario: Deleted code is not returned

- **WHEN** a file is deleted after the last sync and a search matching its former contents is run
- **THEN** no result names that file

#### Scenario: Unchanged repository

- **WHEN** a search is run on a repository with no changes since the last sync
- **THEN** the included sync report shows no added, changed or removed files and no texts embedded

#### Scenario: Invalid arguments cost no sync

- **WHEN** a search is run with an invalid filter after a file has changed
- **THEN** the search fails with the filter error and the changed file is not synchronized by it

### Requirement: Searching requires an existing index

Searching a repository by path SHALL fail with a typed error that names the repository and tells the user to run `indexter init` when the repository has no database. It SHALL NOT create a database. An empty or whitespace-only query SHALL fail with a typed error.

#### Scenario: Unindexed repository

- **WHEN** a search is run for a repository directory that has never been initialized
- **THEN** it fails with an index-not-found error mentioning `indexter init`, and no database file is created

#### Scenario: Empty query

- **WHEN** a search is run with the query `"   "`
- **THEN** it fails with an empty-query error

### Requirement: Candidates come from vectors and full-text search

A search SHALL retrieve up to 50 vector candidates — the nodes whose vectors are nearest the embedded query, ordered by distance then node ID — and up to 50 keyword candidates — full-text matches ordered by BM25 with the name column weighted highest, then name words and qualified name, then docstring and signature, then body, ties by node ID. The keyword expression SHALL be built from the query's word runs, lowercased, plus the split words of runs containing identifier boundaries, with common English stopwords removed unless no other words remain, each term quoted so that query text is never interpreted as full-text syntax, and terms combined with OR. External module nodes SHALL NOT be candidates.

#### Scenario: Keyword match on a split identifier

- **WHEN** the query is `user by email` and a function named `getUserByEmail` exists
- **THEN** that function is a keyword candidate

#### Scenario: Query syntax is treated as text

- **WHEN** the query is `parse AND NOT "config*`
- **THEN** the search completes without a full-text syntax error

#### Scenario: Stopwords are dropped

- **WHEN** the query is `where is the walker`
- **THEN** the keyword expression contains `walker` and none of `where`, `is`, `the`

#### Scenario: Query made only of stopwords

- **WHEN** the query is `how is it`
- **THEN** the keyword expression contains those words rather than being empty

#### Scenario: Query with no word characters

- **WHEN** the query is `?!`
- **THEN** the keyword candidate list is empty and results come from vector candidates alone

#### Scenario: External modules are never hits

- **WHEN** the query is the name of an imported external package
- **THEN** no result is an `external_module` node

### Requirement: Filters restrict both candidate lists

A search SHALL accept optional `kind`, `language` and `path` filters and apply each inside both the vector and the keyword candidate retrieval, so that filtered results are the best matches among nodes satisfying the filter rather than a filtered subset of unfiltered matches. `kind` and `language` SHALL each accept one value or a list. `kind` values SHALL be node kinds other than `external_module`; `language` values SHALL be languages the parsers emit; nodes without a language SHALL NOT match a language filter. `path` SHALL be a repository-relative path matched at path-component boundaries, ignoring a leading `./` and trailing `/`, treating `""` and `.` as no filter, and converting an absolute path inside the repository to a relative one. An unknown kind or language, or a path outside the repository, SHALL fail with a typed error naming the filter, the value, and — for kind and language — the valid values.

#### Scenario: Kind filter

- **WHEN** a search is run with `kind="class"`
- **THEN** every result is a class

#### Scenario: Several kinds

- **WHEN** a search is run with `kind=["function", "method"]`
- **THEN** every result is a function or a method

#### Scenario: Language filter

- **WHEN** a search is run with `language="rust"` on a repository with Python and Rust files
- **THEN** every result is a Rust node

#### Scenario: Path prefix at a component boundary

- **WHEN** a search is run with `path="src/auth"` on a repository containing `src/auth/login.py` and `src/authz.py`
- **THEN** results may come from `src/auth/login.py` and never from `src/authz.py`

#### Scenario: Filter finds matches outside the unfiltered top 50

- **WHEN** a path filter selects a directory none of whose nodes are among the unfiltered top 50 vector or keyword candidates, but some of whose nodes match the query
- **THEN** the search still returns results from that directory

#### Scenario: Absolute path inside the repository

- **WHEN** a search is run with `path` set to the absolute path of a subdirectory of the repository
- **THEN** it behaves exactly as with that subdirectory's relative path

#### Scenario: Unknown kind

- **WHEN** a search is run with `kind="func"`
- **THEN** it fails with an invalid-filter error that lists the valid kinds

#### Scenario: Path outside the repository

- **WHEN** a search is run with `path="../other"`
- **THEN** it fails with an invalid-filter error naming the path

### Requirement: Rankings are fused with reciprocal rank fusion

A search SHALL score each candidate as the sum, over the candidate lists it appears in, of `1 / (60 + rank)` with 1-based ranks, SHALL multiply the score by 0.5 when the node's file is a test file, and SHALL order candidates by score descending then node ID. A test file SHALL be a path with a component `test`, `tests` or `__tests__`, or a file named `test_*`, `*_test.*`, `*.test.*`, `*.spec.*` or `conftest.py`. Each ranked node SHALL record which lists matched it and its rank in each.

#### Scenario: Both signals beat one

- **WHEN** node A is ranked 3rd by vectors and 3rd by keywords, and node B is ranked 1st by vectors only
- **THEN** A ranks above B

#### Scenario: Match reasons are recorded

- **WHEN** a node appears only in the keyword candidates at rank 2
- **THEN** its match reasons are keyword only, with rank 2

#### Scenario: Test files are demoted

- **WHEN** a node in `tests/test_walker.py` and a node in `src/walker.py` would have equal fused scores
- **THEN** the node in `src/walker.py` ranks above it

#### Scenario: Strong test matches still surface

- **WHEN** a test node is ranked 1st in both lists and a non-test node only 40th in one list
- **THEN** the test node ranks above the non-test node

#### Scenario: Deterministic ties

- **WHEN** two candidates have equal fused scores and neither is a test
- **THEN** they are ordered by node ID, and repeated searches return the same order

### Requirement: Limits are validated

A search SHALL return at most `limit` result entries, where `limit` defaults to the `search_limit` setting and SHALL be between 1 and 50; any other value SHALL fail with a typed error.

#### Scenario: Default limit

- **WHEN** a search matching more than 10 nodes is run with no limit and default settings
- **THEN** at most 10 entries are selected

#### Scenario: Limit out of range

- **WHEN** a search is run with `limit=0` or `limit=51`
- **THEN** it fails with an invalid-limit error
