## Context

After M4 a synced database holds, per repository: nodes with stable IDs, byte ranges, signatures and docstrings; a vector per non-external node, embedded from composed summaries; an FTS5 row per node (name, name words, qualified name, docstring, signature, body; `unicode61` tokenizer); and `contains`/`calls`/`imports`/`inherits` edges with line and confidence, plus `nodes.degree` over the non-`contains` edges. Nothing reads any of it. M5 builds retrieval on top, and M6 wraps it in two MCP tools.

The plan fixes the outline (Search section, settled decisions 5–7): sync first; top 50 vector + top 50 full-text, RRF with k=60; one-hop expansion from the top 5 with hub damping at degree 40; `ambiguous` edges at reduced weight, never promoting alone; roll-up of ≥2 methods into their class; 10 hits and a character cap, degrading to fewer intact hits; results grouped by file with a ≤40-line middle-elided snippet and up to 3 callers/callees plus the container; graph-reached nodes in a separate `related` section. It leaves open the exact query construction, filter semantics, scoring of related nodes, how roll-up interacts with the limit, the rendered shape, and Q2 (composed summaries vs raw code).

Measured on this machine before writing this, against a copy of `~/dev/indexter`'s M4 database (73 files, 2,711 vectors, 2,431 `calls` / 2,632 `contains` / 385 `imports` / 39 `inherits` edges), with a throwaway prototype over 12 natural-language questions with known answer files:

| Measurement | Result |
|---|---|
| Plain KNN, k=50 | 3.6 ms cold, <1 ms warm |
| KNN with `kind IN (…)`, `language IN (…)`, `node_rowid IN (SELECT rowid FROM nodes WHERE …)` | all accepted, <1 ms; the subquery allowlist pre-filters (all 50 results inside it) |
| KNN with `kind LIKE` | rejected by vec0 (as the plan recorded) |
| No-op `sync_repo`, warm, in-process | ~4 ms (37 ms first call) |
| Query embedding, sentence-transformers warm | 4–20 ms, occasional ~250 ms MPS stalls |
| Nodes with degree > 40 | 8 of 2,679 (`Document` 314, `DocumentMetadata` 297, `BaseLanguageParser.parse` 232, …); p95 degree 6, p99 18 |
| Symbol line counts | median 13, p90 41; 274 of 2,638 symbols exceed 40 lines |
| Largest class by members | 33 (a test class) |
| Questions with an answer-file node in the top 5 — vector only, composed / raw | 7 / 5 of 12 |
| — keyword only, `unicode61` / `porter unicode61` | 9 / 9 (Porter fixed one question and broke another: "handled" stems onto every `handle*`) |
| — hybrid RRF, composed / raw vectors | 11 / 9 |
| — hybrid, test-file nodes' fused score × 0.5 | 12; the first correct node moved from rank 10→1, 4→1, 2→1, 2→1, 3→2 |
| Most common top-5 noise before demotion | test methods named after the behavior (`test_should_skip_large_files` for "which files to skip") and README sections |

Constraints: synchronous; sync-on-search with no bypass (decision 5 of the plan); results always return real code read from disk; tests co-located, ≥95% coverage; ranking quality is measured by an eval outside CI, not asserted in unit tests.

## Goals / Non-Goals

**Goals:**
- A search function M6 can expose unchanged: query plus optional `kind`/`language`/`path`/`limit` in, a bounded response with real code out.
- Hybrid ranking that beats either signal alone on the question set, with every constant stated and deterministic tie-breaking.
- Graph context that answers "what calls this / what does this call / what else is involved" without a second round trip for the common case.
- A response that fits an agent's context predictably: a result-count limit and a character budget, never mid-hit truncation.
- An eval that decides Q2 with a rule stated before it runs, and compares the other variants that matter.
- Warm search well under 100 ms on `~/dev/indexter`, excluding the one-time model load.

**Non-Goals:**
- The MCP server, tool schemas, `neighbors`, `alwaysLoad`, skill, model warm-up (M6).
- A CLI `search` command — the CLI stays setup-only.
- Cross-repository search; learned or LLM re-ranking; query expansion with synonyms.
- Changing composition, the embedding model, or the FTS tokenizer as a result of the eval — each is a follow-up change if the eval says so.
- Using `external_module` nodes as hits ("what touches pydantic?" is `neighbors("external::pydantic")` in M6).
- Rolling up Rust impl methods into their `struct` (they aren't the struct's children; see Risks).

## Decisions

### 1. Search always syncs first, on the same connection

`search_repo(conn, repo, query, settings, embedder, *, kind=None, language=None, path=None, limit=None)` validates its arguments, calls `sync_repo(conn, repo, settings, embedder)`, then ranks. `search(repo, query, settings, embedder, **filters)` resolves the database path, fails with `IndexNotFound` ("run `indexter init <repo>`") when no database exists — it never creates one — and otherwise opens it and calls `search_repo`. Settings are the caller's (M6 resolves them per repository with `load_settings`).

*Why validate before syncing:* a malformed filter shouldn't cost a sync, and an error should come back in milliseconds.

*Why not create on demand:* building an index is seconds to minutes and downloads a model; that belongs to `init`, the setup step, not to a tool call that expects a fast answer.

The response carries the `SyncReport`, so a caller can tell a search that also re-indexed from one that didn't, and per-stage timings (sync, query embedding, candidates, fusion and selection, expansion, snippets and rendering).

Errors are typed under `SearchError`: `IndexNotFound`, `EmptyQuery` (no non-whitespace characters), `InvalidFilter` (names the filter, the bad value and the valid values), `InvalidLimit`.

### 2. Two candidate lists, filtered inside each query

**Vector candidates.** The query is embedded with the configured embedder (`embedder.embed([query])`) and the 50 nearest vectors are fetched by vec0 KNN, ordered by distance then node ID. Vectors are unit-normalized, so vec0's L2 order is cosine order.

**Keyword candidates.** The query becomes an FTS5 expression:

1. Take every `\w+` run, lowercased. For a run containing identifier boundaries (`getUserByEmail`, `HTTPServer`), also add its `split_identifier` words, so it matches both the `name` column's single token and the `name_words` column's split form.
2. Drop words in a frozen English stopword list (`a`, `are`, `do`, `does`, `how`, `is`, `the`, `we`, `where`, `which`, …) — unless that leaves nothing, in which case keep them all.
3. Deduplicate preserving order, cap at 32 terms, double-quote each (so `AND`, `NEAR`, `*` and `-` in a query are text, not syntax), and join with `OR`.

The 50 best matches by `bm25(nodes_fts, 0, 10, 5, 5, 3, 3, 1)` — weights for `id` (unindexed), `name`, `name_words`, `qualified_name`, `docstring`, `signature`, `body` — are fetched, ordered by score then node ID, excluding `external_module` nodes. A query with no terms has an empty keyword list and ranks by vectors alone.

*Why OR and not AND:* natural-language questions carry words the code doesn't ("where do we decide which files to skip"); AND over those returns nothing. BM25's IDF already weighs rare terms up.

*Why keep `unicode61`:* measured — Porter stemming produced the same 9/12 as `unicode61`, trading one question for another, and switching it means a schema version bump and a rebuild of every database.

*Why name columns weigh most:* a node whose name contains the query's words is almost always the answer; bodies contain everything.

### 3. Filters: validated, applied inside both candidate queries

- **`kind`**: one kind or a list, each one of the node kinds except `external_module`. Vector side: `kind IN (…)` on vec0's metadata column. Keyword side: joined against `nodes.kind`.
- **`language`**: one language or a list, each one the parsers emit (`python`, `javascript`, `typescript`, `rust`, `markdown`, `json`, `yaml`, `toml`, `html`, `css`). Nodes without a language (chunks of unparsed files) never match a language filter. Vector side: `language IN (…)` (vectors store `''` for no language). Keyword side: joined against `nodes.language`.
- **`path`**: a path relative to the repository root, matched at path-component boundaries: `src/auth` matches `src/auth` and `src/auth/…` but not `src/authz.py`. A leading `./` and trailing `/` are ignored; `""` and `.` mean no filter; an absolute path inside the repository is made relative; an absolute path outside it, or one escaping with `..`, is `InvalidFilter`. Vector side: `node_rowid IN (SELECT rowid FROM nodes WHERE file_path = ? OR file_path >= ? || '/' AND file_path < ? || '0')` (`'0'` is the character after `/`, so the range is exactly the prefix). Keyword side: the same predicate on the join.

*Why filter inside, not after:* post-filtering a top 50 can leave nothing (a `path` filter on a small directory) even when matches exist. Measured: all three forms run inside the KNN under 1 ms.

*Why reject unknown values instead of returning nothing:* an agent that passes `kind="func"` learns the vocabulary from the error; an empty result teaches it that the code doesn't exist.

### 4. Fusion: RRF, test demotion, deterministic ties

Each candidate's score is `Σ 1/(60 + rank)` over the lists it appears in, rank 1-based. A node whose file is a test file then has its score multiplied by **0.5**. The fused ranking orders by score descending, then node ID. Each ranked node records its match reasons: `vector`, `keyword`, or both, with its rank in each list.

A test file is a path with a component `test`, `tests` or `__tests__`, or a basename matching `test_*`, `*_test.*`, `*.test.*`, `*.spec.*`, or `conftest.py`. The rule is frozen in code.

*Why demote tests:* measured — test methods are named after the behavior they check, so they out-rank the implementation for exactly the questions search exists for. Demotion moved the implementation to rank 1 in four of twelve questions with no losses. *Why demote rather than exclude:* "where are the tests for X" should still work, and a strong test match still surfaces; a `path` filter into a test directory affects every candidate equally, so relative order there is unchanged.

*Why a multiplier on RRF rather than on either list:* it applies the same to nodes found by one signal or both, and keeps RRF's rank-only property (no raw distances or BM25 scores are mixed).

### 5. Selection and roll-up

A node's **container** is its parent when the parent is class-like (`class`, `struct`, `trait`, `interface`, `enum`). Selection walks the fused ranking in order and builds entries keyed by *group*: a node with a container groups under the container's ID; a class-like node groups under its own ID; anything else is its own group.

- A node whose group already has an entry joins that entry.
- Otherwise it starts a new entry at this position.
- Selection stops as soon as `limit` entries exist, or the fused ranking is exhausted.

An entry renders as a **class entry** (rolled up) when it holds the class-like node itself plus at least one member, or at least two members; it renders as a plain hit of its single node otherwise. A class entry is the container node, with the matched members listed in rank order; its snippet is the snippet of its best-ranked node (the class when the class itself ranked best, otherwise that member), and its reasons are the union of its nodes' reasons. `limit` defaults to `search_limit` (10) and must be 1–50.

*Why a class absorbs its own method hits even with one method:* the class snippet's byte range contains the method; showing both repeats code and spends budget twice.

*Why group keys rather than a post-pass:* a post-pass over the top 10 would leave fewer than 10 entries after collapsing; building entries until `limit` fills the freed slots from further down the ranking, which is the point of roll-up.

### 6. Snippets are read from disk, bounded, middle-elided

After sync, each selected node's file is read with the walker's `read_file` (same decoding as indexing), encoded as UTF-8, and sliced by the node's byte range — exactly the bytes composition saw. Files are read once per response.

- More than `snippet_max_lines` (40) lines: keep the first `⌈(n−1)/2⌉` and last `⌊(n−1)/2⌋` of `n = snippet_max_lines`, with one marker line `… <k> lines elided …` between them — exactly `n` lines.
- Any line longer than 240 characters is cut to 240 with a trailing `…` (minified code must not blow the budget in one line).
- A file that can't be read at render time (deleted between sync and read) yields no snippet and a `snippet unavailable` note; the hit keeps its other fields.

*Why middle elision:* the head holds the signature and docstring, the tail holds the return — the middle is the least informative part of a long function.

### 7. Per-hit graph context

For each entry's node (the container for a class entry):

- **callers**: sources of incoming `calls` edges, ordered by confidence (`exact`, `imported`, `unique_name`, `ambiguous`) then source ID, at most 3, with the total count.
- **callees**: targets of outgoing `calls` edges, same ordering, at most 3, with the total count.
- **container**: the class-like parent's ID and qualified name, if any.

Each caller and callee carries its ID, qualified name and edge confidence; `ambiguous` ones are marked in the rendering. Two indexed queries per hit (measured: 20 caller lookups in 7 ms, over the highest-degree nodes).

### 8. Expansion: a separate, scored `related` list

**Seeds** are the best-ranked node of each of the first 5 entries that survive the budget (decision 9), so every reason names something the reader can see. A seed at entry position `p` (1-based) has weight `1/(60 + p)`.

**Edges** followed from each seed, both directions: `calls`, `inherits`, `imports`, and `contains` except where either end is a `file` node (a file's members are its listing, and every symbol's file is already shown as its path). The neighbor across the edge is a **candidate** unless it is an `external_module`, has degree > 40 (**hub damping**), or is already part of any returned entry (as a node, a rolled-up member, or a class entry's container).

**Score** of a candidate = Σ over contributing (seed, edge) pairs of `seed weight × edge weight`, edge weight 1.0, or **0.5** for `ambiguous` edges. A candidate whose every contributing edge is `ambiguous` is dropped — ambiguous edges add weight but never promote alone. Candidates are ordered by score, then by the strongest contribution's edge kind (`calls`, `inherits`, `imports`, `contains`), then node ID; the top **5** are `related`.

**Reason** comes from the strongest contribution (highest weight, then earliest seed, then kind order), phrased from the related node's side: `called by <seed>`, `calls <seed>`, `base class of <seed>`, `subclass of <seed>`, `imported by <seed>`, `imports <seed>`, `member of <seed>`, `contains <seed>`, suffixed `, which matched` and, when other seeds also contributed, `(+N more)`. Each related item carries ID, qualified name, kind, `path:start-end`, the reason and the edge confidence — no snippet; `path:start-end` is enough to read it.

*Why score by seed position:* a neighbor of the top hit matters more than a neighbor of the fifth, and a node reached from three hits is a better lead than one reached from one.

*Why hub damping on targets only:* the plan says so; a hub that *matched* is still a good hit, and its neighbors compete on score like any other.

*Why exclude what's already shown:* `related` exists to add leads, not repeat hits; a method's class is already shown as its container.

### 9. Budget and rendering

`search_max_chars` (new setting, default 20,000 — about 5k tokens) caps the rendered response. Entries are admitted in rank order, each measured by its rendered size including its file heading if it opens a new file group; the first entry is always admitted, and admission stops at the first entry that doesn't fit — later, smaller entries are not pulled forward, so the ranking stays honest. The response records how many selected entries were omitted for budget. `related` items are then admitted one by one into what remains, and dropped (counted) if they don't fit.

The rendering is deterministic plain text, grouped by file in order of each file's best entry, entries within a file in rank order (illustrative):

```
5 results for "where do we decide which files to skip" (1 omitted for budget)

## src/indexter/walker/walker.py

### Walker._should_skip — method — src/indexter/walker/walker.py:120-158 — vector, keyword
id: src/indexter/walker/walker.py::Walker._should_skip#method
in: Walker · callers (1): Walker.walk · callees (3 of 4): Walker._is_binary_file, IgnorePatternMatcher.match_file, …
def _should_skip(self, path: Path) -> bool
Decide whether a file should be excluded from indexing.
    def _should_skip(self, path: Path) -> bool:
        …
        … 12 lines elided …
        return False

## related
- IgnorePatternMatcher.match_file — method — src/indexter/walker/walker.py:40-52 — called by Walker._should_skip, which matched
```

A class entry adds a `matched members:` block (qualified name, `path:start-end`, signature per member) before its snippet. The exact layout is pinned by a snapshot test rather than by the spec.

*Why text, not JSON:* the consumer is a language model reading tool output; text is denser, and the budget is measured on what it reads. M6 may add structured content from the same response objects.

*Why stop at the first misfit:* returning hit 7 but not the larger hit 6 would present 7 as the sixth-best result.

### 10. Module layout and types

- `search/hybrid.py`: `search`, `search_repo`, filter normalization, keyword-expression building, candidate queries, `fuse` (pure, over two ranked ID lists and a test-path predicate), errors.
- `search/expand.py`: `hit_context(conn, node_id)` and `expand(conn, seeds, excluded)` — pure SQL reads, reusable by M6's `neighbors`.
- `search/results.py`: `select_entries` (pure roll-up over the fused ranking and a node-row lookup), snippet reading and elision, per-entry rendering, budget admission, `render(response)`.
- Types (frozen dataclasses): `RankedNode` (node ID, score, reasons with ranks), `Hit`/entry (node fields, members, reasons, snippet, context), `Related`, `SearchResponse` (query, normalized filters, limit, entries, related, omitted counts, `SyncReport`, timings).

*Why pure cores:* ranking quality can't be unit-tested against a real model, but fusion, roll-up, elision, budget admission and expansion scoring are deterministic functions of their inputs and are tested exhaustively that way. Integration tests use the M4 fixture repository with `FakeEmbedder` (hash vectors: arbitrary but deterministic order) and a small stub embedder that maps chosen texts to chosen vectors where an order must be controlled.

### 11. The retrieval eval

`eval/questions.toml` holds the target repository (`~/dev/indexter`) and 15–20 questions, each with `text` and one or more `expected` files, phrased without symbol or module names — the case search exists for. `eval/run_eval.py` (run by `just eval`, with `--repo`, `--variant`, `--show` and `--cache-dir` options) indexes the target once per embedding variant, runs every question through `search_repo`, and prints per question and in total:

- **hit@5**: some entry among the first 5 selected entries has a path in `expected`;
- **MRR@10**: reciprocal position of the first such entry within 10, 0 if none;
- rendered size and search latency (p50, max).

**Variants**:

| Variant | What changes |
|---|---|
| `composed` (production) | nothing |
| `raw` | each node's `embed_text` is its raw source slice, truncated to the same token budget; the harness wraps `compose_file` for the duration of the run. FTS rows are identical. |
| `vector-only` / `keyword-only` | fusion over one list |
| `no-demotion` | test demotion factor 1.0 |
| `model:<name>` | `embedding_model` override (384-dimensional models only; e.g. `BAAI/bge-small-en-v1.5`) |

Isolation: the harness sets `XDG_DATA_HOME` to a variant-specific directory under `--cache-dir` (default `~/.cache/indexter-eval`) and builds `Settings` from defaults plus the variant's overrides — never the user's global or repository config — so runs are comparable and never touch real databases. It records the target's git commit and dirty state in its output.

**Q2 decision rule, fixed before the eval runs:** `composed` stays unless `raw`, in hybrid mode, answers at least 2 more questions at hit@5 **and** has an MRR@10 at least as high. The verification task records the numbers and the outcome in this change. If `raw` wins, changing composition is a separate change.

*Why not in CI:* ranking assertions over a real model are flaky across hardware and model revisions, and the target repository isn't part of this one.

*Why wrap `compose_file` in the harness instead of adding a setting:* a raw-embedding mode isn't a product feature; putting it in `Settings` would make it one.

### 12. One new setting; everything else is a constant

`search_max_chars` joins `search_limit` and `snippet_max_lines`. RRF k (60), candidate pool sizes (50/50), seed count (5), hub threshold (40), ambiguous weight (0.5), test demotion (0.5), related count (5), line cut (240), term cap (32) and BM25 column weights are module constants.

*Why constants:* each one is a ranking decision the eval should measure, not a knob users tune per repository; exposing them would make every future ranking change a compatibility question.

## Risks / Trade-offs

- **Twelve prototype questions, and some eval questions will be written by the same person who chose the constants** → the constants come from the plan or the prototype, not from tuning on the eval set; the eval reports per-question results so a reader can see which ones carry a variant's lead.
- **The eval measures one Python repository** → the plan's scale target is local repositories of this size; JS/TS/Rust ranking is covered only by fixture-level tests. A question set for a second repository is additive later.
- **Test demotion is path-heuristic** → repos with unusual test layouts get no demotion (the pre-M5 behavior), never wrong exclusion.
- **FakeEmbedder can't test ranking quality** → unit tests assert mechanics; the eval asserts quality.
- **A hub seed with many low-degree neighbors fills `related` with ties broken by ID** → bounded at 5 items; seeds that share neighbors out-score single-seed neighbors.
- **Rust methods aren't rolled up into their `struct`** → their parent is the file (M4 decision 4); several `impl Foo` methods can each take a slot. Rust grouping by scope path is a later refinement.
- **The first search in a process loads the model** (~2 s, plus sync's own backlog if files changed) → M6 warms the embedder at server start; a changed embedding model triggers a full re-embed on the next search, which is the existing sync behavior.
- **A 20,000-character budget may be wrong for some clients** → it's a setting; the eval reports actual rendered sizes to calibrate the default.
- **Stopword and identifier handling are English- and code-convention-specific** → vectors still carry non-English queries; no worse than no keyword signal.

## Migration Plan

No schema change and no re-index: search reads databases M4 already produces. The new setting has a default, so existing configuration files stay valid. Rollback is removing the `search` package; nothing else depends on it until M6.

## Open Questions

- **Q2 itself** — decided by decision 11's rule during verification, and recorded here. Against `~/dev/indexter` @ `f754484e40c9` (clean), 18 questions: `composed` hit@5=16/18, mean MRR@10=0.806; `raw` hit@5=16/18, mean MRR@10=0.765. `raw` answers 0 more questions at hit@5 (needs ≥2) and has a lower mean MRR@10, so it does not clear the bar. **Outcome: composed remains the production choice.** Full default-variant results from the same run: `vector-only` hit@5=16/18 mrr=0.775, `keyword-only` hit@5=15/18 mrr=0.643, `no-demotion` hit@5=16/18 mrr=0.667. Hybrid (`composed`) is at least as good as both single-signal variants on both hit@5 and mean MRR@10, confirming fusion helps rather than hurts on this repository.
- **Should `ambiguous` method fan-out collapse to a shared base method** (M4's open question) — the eval's `--show` output over questions touching the parsers will show whether it pollutes `related`; decided after M6 exercises `neighbors`.
- **Default model** — the `model:` variant reports whether a different 384-dimensional model is better; switching is a follow-up change. `model:BAAI/bge-small-en-v1.5` scored hit@5=15/18, mean MRR@10=0.691 on the same question set — worse than the default `sentence-transformers/all-MiniLM-L6-v2` (hit@5=16/18, mrr=0.806) on both measures, so no default change is warranted.
- **Receiver type inference** (M4's open question) — if the eval shows answers missing because a call edge wasn't resolved, that is the evidence for it.
