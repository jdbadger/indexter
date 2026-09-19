## ADDED Requirements

### Requirement: Sync reports its phases to an optional observer

`sync_repo` and `index_repository` SHALL accept an optional progress observer. When none is supplied
they SHALL default to a non-reporting observer and SHALL produce no output on any stream. When one is
supplied, they SHALL report the start and completion of each distinct phase of work — preparing the
embedding model, indexing files, resolving the graph, and embedding — so a caller can describe the
work without inspecting the pipeline's internals. Reporting SHALL NOT alter what is indexed, written,
resolved, or embedded.

#### Scenario: Default is silent

- **WHEN** `sync_repo` is called without a progress observer
- **THEN** nothing is written to stdout or stderr and the sync report is unchanged

#### Scenario: Phases are reported in order

- **WHEN** `sync_repo` is called with an observer on a repository needing a full index
- **THEN** the observer receives the start and completion of the file-indexing, resolution, and embedding phases, in that order

#### Scenario: Skipped phases are not reported

- **WHEN** a sync performs no resolution because none is due
- **THEN** no resolution phase is reported

#### Scenario: Observation does not change results

- **WHEN** the same repository is synced with and without an observer
- **THEN** both runs produce equivalent sync reports and equivalent database contents

### Requirement: The embedding backlog reports determinate progress

When an observer is supplied and there is embedding work to do, the embedding backlog SHALL report
the total number of texts to embed before embedding the first batch, and SHALL report completion
after each batch. When there is no embedding work, it SHALL report no embedding phase and SHALL still
not load the model or the tokenizer.

#### Scenario: Total is known before the first batch

- **WHEN** the embedding backlog begins with work to do and an observer supplied
- **THEN** the observer is given the total count of texts before the first batch is embedded

#### Scenario: Each batch advances the count

- **WHEN** the embedding backlog embeds several batches
- **THEN** the observer's completed count advances after each batch and ends equal to the total

#### Scenario: An empty backlog reports nothing and loads nothing

- **WHEN** the embedding backlog finds no nodes needing vectors
- **THEN** no embedding phase is reported and neither the model nor the tokenizer is loaded

### Requirement: Search-triggered syncs never narrate

The sync performed before a search or a neighbors lookup SHALL supply no progress observer, so that
no progress output is ever produced on the server's standard output, where it would corrupt the
protocol carried on that stream.

#### Scenario: Search sync is silent

- **WHEN** a search triggers a sync that indexes and embeds changed files
- **THEN** nothing is written to stdout or stderr by the progress mechanism

#### Scenario: Neighbors sync is silent

- **WHEN** a neighbors lookup triggers a sync that indexes and embeds changed files
- **THEN** nothing is written to stdout or stderr by the progress mechanism
