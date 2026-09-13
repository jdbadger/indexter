## 1. Settings and package skeleton

- [x] 1.1 Add `search_max_chars: int = 20_000` to `Settings`; test the default and a repository override
- [x] 1.2 Create `src/indexter/search/` (`__init__.py`, `hybrid.py`, `expand.py`, `results.py`, `tests/`) with the ranking constants from design decision 12
- [x] 1.3 Define the `SearchError` hierarchy (`IndexNotFound`, `EmptyQuery`, `InvalidFilter`, `InvalidLimit`) and the frozen response types (`RankedNode`, entry/hit with members and context, `Related`, `SearchResponse` with sync report, omitted counts, and per-stage timings)

## 2. Query and filters (`search/hybrid.py`)

- [x] 2.1 Validate the query (non-blank) and limit (1–50, default `search_limit`)
- [x] 2.2 Normalize `kind` and `language` (single value or list) against the node kinds except `external_module` and the registered parser languages; errors list valid values
- [x] 2.3 Normalize `path`: strip `./` and trailing `/`, `""`/`.` as no filter, absolute-inside-repo to relative, reject absolute-outside and `..` escapes
- [x] 2.4 Build the FTS5 keyword expression: `\w+` runs lowercased, `split_identifier` words for runs with identifier boundaries, frozen stopword list with the all-stopwords fallback, dedupe, 32-term cap, quoted terms joined by `OR`
- [x] 2.5 Tests: every validation and normalization scenario in the hybrid-search spec; keyword expressions for split identifiers, stopwords, stopword-only, no-word-character and syntax-looking queries

## 3. Candidates and fusion (`search/hybrid.py`)

- [x] 3.1 Vector candidates: embed the query, vec0 KNN k=50 with `kind IN`, `language IN` and the path rowid subquery inside the query, ordered by distance then node ID
- [x] 3.2 Keyword candidates: `nodes_fts` match joined to `nodes` with the same filters, excluding `external_module`, top 50 by `bm25` with the design's column weights, then node ID
- [x] 3.3 Test-file predicate per design decision 4, and pure `fuse` (RRF k=60, 1-based ranks, ×0.5 test demotion, ties by node ID, reasons with per-list ranks)
- [x] 3.4 Tests: `fuse` scenarios (both signals beat one, reasons, demotion, strong tests surface, ties); test-path predicate cases; filters applied inside both candidate queries (kind, several kinds, language, nodes without language, component-boundary path, a filtered directory outside the unfiltered top 50); external modules never candidates; a syntax-looking query runs without error

## 4. Selection, snippets and context (`search/results.py`, `search/expand.py`)

- [x] 4.1 Pure `select_entries`: container grouping for class-like parents, join-or-start, stop at `limit`, class entry vs single hit, members in rank order, union of reasons, best-ranked node for the snippet
- [x] 4.2 Snippet reading: one `read_file` per file, UTF-8 byte-range slicing, middle elision to exactly `snippet_max_lines`, 240-character line cut, unavailable-file note
- [x] 4.3 `hit_context`: top 3 callers and callees by confidence order then ID with totals, and the class-like container
- [x] 4.4 Tests: roll-up scenarios (two methods collapse and fill the freed slot, class absorbs its method, single method stays a hit, limit counts entries); snippet scenarios (exact source, 100→40 elision counts, long line, multi-byte content, vanished file); context scenarios (counts, confident-first ordering, container present/absent)

## 5. Expansion (`search/expand.py`)

- [x] 5.1 Follow `calls`/`inherits`/`imports` both directions and `contains` both directions except at `file` nodes, from seeds weighted `1/(60+p)`; exclude externals, degree > 40, and anything already shown
- [x] 5.2 Score with ambiguous edge weight 0.5, drop ambiguous-only candidates, order by score then strongest edge kind then ID, keep 5
- [x] 5.3 Reasons from the strongest contribution, phrased from the related node's side, with `, which matched` and `(+N more)`; carry confidence and location
- [x] 5.4 Tests: every graph-expansion spec scenario over small hand-built node/edge sets, plus an integration check over the M4 fixture repository

## 6. Budget, rendering and orchestration

- [x] 6.1 Per-entry and per-related rendering, file headings, header line with omitted counts, `related` section, and the no-results rendering
- [x] 6.2 Budget admission: first entry always, stop at first misfit, related into the remainder, omitted counts; seeds taken from admitted entries only
- [x] 6.3 `search_repo`: validate → `sync_repo` → candidates → fuse → select → snippets and context → admit entries → expand from admitted seeds → admit related → `SearchResponse` with timings; `render(response)`
- [x] 6.4 `search(repo, …)`: derive the database path, raise `IndexNotFound` (mentioning `indexter init`) without creating a database, otherwise open and delegate
- [x] 6.5 Tests: budget scenarios (fewer hits, no pull-forward, oversized first hit, related yields to hits, only admitted entries seed); grouping by file; stable rendering across repeated searches; sync-before-search scenarios (added function found, deleted file absent, unchanged repo reports no work, invalid filter doesn't sync); unindexed repo and empty query errors
- [x] 6.6 Inline snapshot of the full rendered response for a fixed query over the M4 fixture repository, using a stub embedder with controlled vectors

## 7. Retrieval eval

- [x] 7.1 Write `eval/questions.toml` for `~/dev/indexter`: 15–20 questions phrased without symbol or module names, each with expected files
- [x] 7.2 `eval/run_eval.py`: load and validate questions (missing expected files reported by name); isolate `XDG_DATA_HOME` per index variant under `--cache-dir` and build `Settings` from defaults plus overrides only; record target commit and dirty state
- [x] 7.3 Variants: `composed`, `raw` (harness wraps `compose_file` to embed the node's source slice truncated to the same token budget, FTS unchanged), `vector-only`, `keyword-only`, `no-demotion` (reusing the composed index), and `model:<name>`
- [x] 7.4 Scoring and output: per-question first-correct position, hit@5, mean MRR@10, rendered size, latency median/max; `--variant`, `--repo`, `--show` options
- [x] 7.5 Add a `justfile` with an `eval` recipe (and `list`); exclude `eval/` from packaging and coverage; ignore the eval cache if it is placed in the repository

## 8. Verification

- [x] 8.1 `uv run --group dev ruff check --fix src/indexter` clean
- [x] 8.2 `uv run --group dev ty check src/indexter` clean
- [x] 8.3 `uv run --group test pytest --cov=indexter --cov-fail-under=95 --cov-report=term-missing` green
- [x] 8.4 Run `just eval` against `~/dev/indexter`; record hit@5 and MRR@10 for every default variant, and confirm hybrid beats vector-only and keyword-only
- [x] 8.5 Apply the Q2 rule to `composed` vs `raw` and record the numbers and outcome in design.md's Open Questions
- [x] 8.6 Run the `model:BAAI/bge-small-en-v1.5` variant and record its scores (informational; no default change here)
- [x] 8.7 Real-repo check: warm search latency (no changes, model loaded) on `~/dev/indexter` well under 100 ms, with the per-stage timing breakdown; median and maximum rendered sizes against the 20,000-character budget
- [x] 8.8 Real-repo spot checks with `--show`: a class roll-up occurs, `related` reasons name visible hits, no `related` node has degree > 40, and snippets match the files on disk
- [x] 8.9 Confirm the suite passes on Python 3.11, 3.12, and 3.13
